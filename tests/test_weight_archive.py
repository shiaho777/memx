#!/usr/bin/env python3
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("DYLD_LIBRARY_PATH", str(ROOT / "build"))

import memx_runtime as memx


def main():
    dylib = ROOT / "build" / "libmemx_runtime.dylib"
    if not dylib.exists():
        print("SKIP: dylib missing")
        return 0
    rt = memx.Runtime(dylib)
    ctx = rt.create_context("archive-test")
    ctx.set_quota(256 * memx.MB)
    rows, cols = 128, 256
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
    a = ctx.malloc_tensor(nbytes, desc, name="w0")
    buf = a.buffer()
    for i in range(nbytes):
        buf[i] = (i * 17 + 3) & 0xFF
    golden = bytes(buf)
    ctx.update_tensor_flags_range(a, 0, nbytes, memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD)
    try:
        ctx.force_compress_range(a, 0, nbytes)
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "w0.mxwa")
        n = ctx.export_archive(a, path)
        assert n > 0, "export wrote 0 bytes"
        assert Path(path).stat().st_size > 64
        a.free()
        b = ctx.import_archive(path, desc=desc, name="w0-reload")
        assert b.size == nbytes
        bbuf = b.buffer()
        got = bytes(bbuf)
        if got != golden:
            mism = sum(1 for x, y in zip(got, golden) if x != y)
            print(f"FAIL bitexact mism={mism}/{nbytes}")
            return 1
        ctx.ws_tile(
            b,
            rows=rows,
            cols=cols,
            elem_size=elem,
            col_start=0,
            col_count=32,
            prefetch_cols=16,
            flags=memx.MEMX_WS_FLAG_HOT | memx.MEMX_WS_FLAG_PREFETCH | memx.MEMX_WS_FLAG_MARK_ACCESS,
        )
        b.free()
    ctx.destroy()
    rt.shutdown()
    print("OK weight archive bitexact + ws_tile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
