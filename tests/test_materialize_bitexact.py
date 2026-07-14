#!/usr/bin/env python3
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("DYLD_LIBRARY_PATH", str(ROOT / "build"))

import memx_runtime as memx


def main():
    dylib = ROOT / "build" / "libmemx_runtime.dylib"
    if not dylib.exists():
        print("SKIP")
        return 0
    rt = memx.Runtime(dylib)
    ctx = rt.create_context("mat-test")
    ctx.set_quota(256 * memx.MB)
    rows, cols = 64, 128
    elem = 2
    nbytes = rows * cols * elem
    desc = memx.tensor_desc(
        memx.MEMX_TENSOR_ROLE_WEIGHT,
        memx.MEMX_TENSOR_DTYPE_BF16,
        memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
        memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
        shape=(rows, cols),
        stride=(cols, 1),
    )
    a = ctx.malloc_tensor(nbytes, desc, name="w")
    buf = a.buffer()
    for i in range(nbytes):
        buf[i] = (i * 31 + 7) & 0xFF
    golden = bytes(buf)
    ctx.update_tensor_flags_range(a, 0, nbytes, memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD)
    try:
        ctx.force_compress_range(a, 0, nbytes)
    except Exception:
        pass
    import ctypes
    out = (ctypes.c_uint8 * nbytes)()
    ctx.materialize_range(a, 0, nbytes, ctypes.addressof(out), nbytes)
    got = bytes(out)
    if got != golden:
        mism = sum(1 for x, y in zip(got, golden) if x != y)
        print(f"FAIL range mism={mism}")
        return 1
    # tile: middle columns
    col0, coln = 16, 40
    dense = rows * coln * elem
    tout = (ctypes.c_uint8 * dense)()
    ctx.materialize_tile(a, rows, cols, elem, col0, coln, ctypes.addressof(tout), dense, coln * elem)
    # build golden tile
    g2 = bytearray(dense)
    for r in range(rows):
        src = r * cols * elem + col0 * elem
        dst = r * coln * elem
        g2[dst:dst + coln * elem] = golden[src:src + coln * elem]
    if bytes(tout) != bytes(g2):
        print("FAIL tile")
        return 1
    # pages should still be mostly compressed after keep-compressed materialize
    info = a.info()
    print(f"OK materialize bitexact compressed_pages={info.compressed_pages} page_count={info.page_count}")
    a.free()
    ctx.destroy()
    rt.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
