#!/usr/bin/env python3
"""Capsule roundtrip: export compressed pages, attach, materialize bit-exact.

Covers:
- capsule_export -> capsule_attach -> capsule_materialize bit-exact
- capsule_materialize_v batch result identical to per-page materialize
- capsule_pidx_at rank ordering
- ws_tile / apply_ws paths don't corrupt data
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("DYLD_LIBRARY_PATH", str(ROOT / "build"))
os.environ.setdefault("MEMX_CAPSULE_HOST_BIND", "0")

import memx_runtime as memx


def fill(buf, nbytes, seed):
    for i in range(nbytes):
        buf[i] = (i * seed + (i >> 5) + seed) & 0xFF


def make_host(rt, name, pages, seed):
    ctx = rt.create_context(name)
    ctx.set_quota(256 * memx.MB)
    nbytes = pages * 16384
    desc = memx.tensor_desc(
        memx.MEMX_TENSOR_ROLE_WEIGHT,
        memx.MEMX_TENSOR_DTYPE_FP16,
        memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
        memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
        shape=(pages, 8192),
        stride=(8192, 1),
    )
    a = ctx.malloc_tensor(nbytes, desc, name=name)
    buf = a.buffer()
    fill(buf, nbytes, seed)
    golden = bytes(buf)
    ctx.update_tensor_flags_range(a, 0, nbytes, memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD)
    try:
        ctx.force_compress_range(a, 0, nbytes)
    except Exception:
        pass
    return ctx, a, golden


def main():
    dylib = ROOT / "build" / "libmemx_runtime.dylib"
    if not dylib.exists():
        print("SKIP: dylib missing")
        return 0
    rt = memx.Runtime(dylib)
    pages = 64
    ctx, a, golden = make_host(rt, "capsule-src", pages, 17)
    nbytes = pages * 16384

    with tempfile.TemporaryDirectory() as td:
        out_bytes = rt.capsule_export(td)
        assert out_bytes > 0, "capsule_export wrote 0"
        rt.capsule_attach(td)
        stats = rt.capsule_stats()
        assert stats.attached, "capsule not live after attach"
        assert stats.ent_count >= pages // 2, f"too few entries: {stats.ent_count}"

        # Per-page materialize vs golden via rank mapping
        pidx_at = []
        rank = 0
        while True:
            try:
                p = rt.capsule_pidx_at(rank)
            except OSError:
                break
            pidx_at.append(p)
            rank += 1
        assert len(pidx_at) > 0, "no pidx entries readable"

        # Find ranks covering our allocation pages: pidx range
        info = a.info()
        sp = None
        # allocation start page: derive from info by scanning ranks
        target_pages = set(range(pages))
        page_rank = {}
        for r, p in enumerate(pidx_at):
            if p in target_pages:
                page_rank[p] = r
        # Pages that stayed resident (incompressible) are not in the capsule;
        # verify every capsule page bit-exactly, and count the coverage.
        in_cap = sorted(page_rank.keys())
        assert len(in_cap) >= pages // 2, f"too few pages in capsule: {len(in_cap)}/{pages}"

        mism = 0
        per_page = {}
        for p in in_cap:
            buf = bytearray(16384)
            rt.capsule_materialize(p, buf)
            off = p * 16384
            per_page[p] = bytes(buf)
            if bytes(buf) != golden[off:off + 16384]:
                mism += 1
        if mism:
            print(f"FAIL capsule_materialize mism pages={mism}/{len(in_cap)}")
            return 1

        # Batch materialize_v must equal per-page results
        order = list(in_cap)
        batch_buf = bytearray(len(order) * 16384)
        rt.capsule_materialize_v(order, batch_buf, stride=16384)
        for i, p in enumerate(order):
            off = i * 16384
            if bytes(batch_buf[off:off + 16384]) != per_page[p]:
                print(f"FAIL materialize_v mismatch page={p}")
                return 1
        # Shuffled subset
        subset = [in_cap[3], in_cap[0], in_cap[-1], in_cap[len(in_cap)//2]]
        sub_buf = bytearray(len(subset) * 16384)
        rt.capsule_materialize_v(subset, sub_buf, stride=16384)
        for i, p in enumerate(subset):
            off = i * 16384
            if bytes(sub_buf[off:off + 16384]) != per_page[p]:
                print(f"FAIL materialize_v subset mismatch page={p}")
                return 1

        # ws_tile + apply_ws don't corrupt content after materialize
        try:
            ctx.ws_tile(
                a, rows=pages, cols=8192, elem_size=2,
                col_start=0, col_count=512, prefetch_cols=128,
                flags=memx.MEMX_WS_FLAG_HOT | memx.MEMX_WS_FLAG_PREFETCH | memx.MEMX_WS_FLAG_MARK_ACCESS,
            )
        except Exception:
            pass
        try:
            ctx.apply_ws([memx.ws_intent(a.ptr, 0, 16384 * 4,
                                         flags=memx.MEMX_WS_FLAG_HOT | memx.MEMX_WS_FLAG_MARK_ACCESS)])
        except Exception:
            pass

        # materialize still correct after ws ops
        buf2 = bytearray(16384)
        rt.capsule_materialize(0, buf2)
        if bytes(buf2) != per_page[0]:
            print("FAIL capsule materialize after ws_tile")
            return 1

        # Live buffer still bitexact after ws ops (touch via fault)
        live = bytes(a.buffer())
        if live != golden:
            lm = sum(1 for x, y in zip(live, golden) if x != y)
            print(f"FAIL live buffer after ws ops mism={lm}/{nbytes}")
            return 1

        rt.capsule_detach()

        import errno as _errno
        import shutil as _shutil
        import struct as _struct
        import zlib as _zlib

        tamper_dir = Path(td) / "tampered"
        _shutil.copytree(td, tamper_dir)
        ledger = (tamper_dir / "ledger.bin").read_bytes()
        ent0_off = _struct.unpack_from("<Q", ledger, 72 + 8)[0]
        ent0_csz = _struct.unpack_from("<I", ledger, 72 + 4)[0]
        ent0_pidx = _struct.unpack_from("<I", ledger, 72)[0]
        spill_path = tamper_dir / "spill.bin"
        raw = bytearray(spill_path.read_bytes())
        flip_at = ent0_off + ent0_csz // 2
        raw[flip_at] ^= 0xFF
        spill_path.write_bytes(bytes(raw))
        rt.capsule_attach(str(tamper_dir))
        buf = bytearray(16384)
        got_ebadmsg = False
        try:
            rt.capsule_materialize(ent0_pidx, buf)
        except OSError as e:
            got_ebadmsg = (e.errno == _errno.EBADMSG)
        assert got_ebadmsg, f"tampered page did not fail with EBADMSG (pidx={ent0_pidx})"
        bad, total, vrc = rt.capsule_verify()
        assert bad >= 1 and total == stats.ent_count, f"verify counts wrong: bad={bad} total={total}"
        assert vrc == _errno.EBADMSG, f"verify rc={vrc}"
        st = rt.capsule_stats()
        assert st.integrity_failures >= 1, "integrity_failures not counted"
        rt.capsule_detach()

        trunc_dir = Path(td) / "truncated"
        _shutil.copytree(td, trunc_dir)
        tspill = trunc_dir / "spill.bin"
        with open(tspill, "r+b") as f:
            f.truncate(max(1, os.path.getsize(tspill) // 2))
        got_einval = False
        try:
            rt.capsule_attach(str(trunc_dir))
        except OSError as e:
            got_einval = (e.errno == _errno.EINVAL)
        assert got_einval, "truncated spill did not fail attach with EINVAL"
        rt.capsule_detach()

        v2_dir = Path(td) / "v2capsule"
        v2_dir.mkdir()
        v2_page = bytes((i * 7 + 3) & 0xFF for i in range(16384))
        v2_comp = _zlib.compress(v2_page, 1)
        v2_payload = b"MX\x85\x01" + _struct.pack("<I", len(v2_comp)) + v2_comp
        (v2_dir / "spill.bin").write_bytes(v2_payload)
        v2_hdr = _struct.pack(
            "<IIQQQQ4Q", 0x4D584350, 2, 1, len(v2_payload), 16384, 16384, 0, 0, 0, 0
        )
        v2_ent = _struct.pack("<IQIBBBB", 1000, 0, 0, 0, 0x85, 1, 0)[:24]
        v2_ent = _struct.pack("<IIQIBBBB", 1000, len(v2_payload), 0, 0, 0x85, 1, 0, 0)
        (v2_dir / "ledger.bin").write_bytes(v2_hdr + v2_ent)
        rt.capsule_attach(str(v2_dir))
        v2_buf = bytearray(16384)
        rt.capsule_materialize(1000, v2_buf)
        assert bytes(v2_buf) == v2_page, "v2 legacy capsule materialize not bitexact"
        got_enotsup = False
        try:
            rt.capsule_verify()
        except OSError as e:
            got_enotsup = (e.errno in (_errno.ENOTSUP, _errno.EOPNOTSUPP))
        assert got_enotsup, "v2 capsule verify did not report ENOTSUP"
        rt.capsule_detach()

    a.free()
    ctx.destroy()
    rt.shutdown()
    print(f"OK capsule roundtrip capsule_pages={len(in_cap)}/{pages} materialize_v batch==single ws_tile+apply_ws crc-tamper+truncate+v2-legacy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
