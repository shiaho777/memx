#!/usr/bin/env python3
"""Capsule segments: name allocations, export, attach, materialize by name bit-exact.

Covers:
- memx_runtime_context_name_segment registration
- manifest.bin written on export and loaded on attach
- capsule_segment lookup (rank / pages / nbytes)
- capsule_materialize_segment bit-exact vs golden for each named segment
- unnamed allocations coexist in the same capsule
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
        buf[i] = (i * seed + (i >> 7) + seed) & 0xFF


def make_segment(ctx, name, pages, seed, role=memx.MEMX_TENSOR_ROLE_DATA,
                 dtype=memx.MEMX_TENSOR_DTYPE_INT32):
    nbytes = pages * 16384
    desc = memx.tensor_desc(
        role, dtype, memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
        memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
        shape=(pages, nbytes // 4), stride=(nbytes // 4, 1),
    )
    a = ctx.malloc_tensor(nbytes, desc, name=name)
    buf = a.buffer()
    fill(buf, nbytes, seed)
    golden = bytes(buf)
    ctx.update_tensor_flags_range(a, 0, nbytes,
                                  memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD)
    try:
        ctx.force_compress_range(a, 0, nbytes)
    except Exception:
        pass
    ctx.name_segment(a, name)
    return a, golden


def main():
    dylib = ROOT / "build" / "libmemx_runtime.dylib"
    if not dylib.exists():
        print("SKIP: dylib missing")
        return 0
    rt = memx.Runtime(dylib)
    ctx = rt.create_context("seg-host")
    ctx.set_quota(256 * memx.MB)

    a, golden_rows = make_segment(ctx, "rows", 32, 31)
    b, golden_vocab = make_segment(ctx, "vocab", 24, 77)
    plain = ctx.malloc(16 * 16384, name="unnamed")
    fill(plain.buffer(), 16 * 16384, 13)

    with tempfile.TemporaryDirectory() as td:
        rt.capsule_export(td)
        rt.capsule_attach(td)
        stats = rt.capsule_stats()
        assert stats.attached, "capsule not live after attach"

        rank_r, pages_r, nbytes_r = rt.capsule_segment("rows")
        assert pages_r >= 16, f"rows pages too few: {pages_r}"
        assert nbytes_r == 32 * 16384, f"rows nbytes wrong: {nbytes_r}"
        rank_v, pages_v, nbytes_v = rt.capsule_segment("vocab")
        assert pages_v >= 12, f"vocab pages too few: {pages_v}"
        assert nbytes_v == 24 * 16384, f"vocab nbytes wrong: {nbytes_v}"
        assert pages_r + pages_v <= stats.ent_count, "segment pages exceed capsule entries"

        buf = bytearray(pages_r * 16384)
        rt.capsule_materialize_segment("rows", buf)
        assert bytes(buf[:nbytes_r]) == golden_rows, "rows segment not bitexact"

        buf = bytearray(pages_v * 16384)
        rt.capsule_materialize_segment("vocab", buf)
        assert bytes(buf[:nbytes_v]) == golden_vocab, "vocab segment not bitexact"

        bad, total, rc = rt.capsule_verify()
        assert bad == 0 and rc == 0, f"verify failed: bad={bad} rc={rc}"

        rt.capsule_detach()

    plain.free()
    a.free()
    b.free()
    ctx.destroy()
    rt.shutdown()
    print(f"OK capsule segments rows={pages_r}p vocab={pages_v}p manifest+bitexact+verify_clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
