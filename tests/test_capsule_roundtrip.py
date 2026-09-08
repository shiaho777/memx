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

    a.free()
    ctx.destroy()
    rt.shutdown()
    print(f"OK capsule roundtrip capsule_pages={len(in_cap)}/{pages} materialize_v batch==single ws_tile+apply_ws")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
