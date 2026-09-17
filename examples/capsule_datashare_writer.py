import argparse
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
for key, value in {
    "MEMX_NO_SELFTEST": "1",
    "MEMX_CAPSULE_LITE": "1",
    "MEMX_CPU_ONLY": "1",
    "MEMX_CAPSULE_HOST_BIND": "0",
}.items():
    os.environ.setdefault(key, value)

import memx_runtime as memx

PAGE = 16384


def main():
    parser = argparse.ArgumentParser(description="Export an immutable capsule data generation")
    parser.add_argument("--dir", default=f"/tmp/memx_datashare_{os.getpid()}")
    args = parser.parse_args()
    directory = Path(args.dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        parser.error("--dir must be empty; export each immutable generation to a new directory")

    rt = memx.Runtime()
    ctx = rt.create_context("datashare-writer")
    allocations = []
    segments = {}
    try:
        ctx.set_quota(64 * memx.MB)
        for name, shape, dtype, memx_dtype, seed in (
            ("datashare.matrix", (128, 256), "<f4", memx.MEMX_TENSOR_DTYPE_FP32, 31),
            ("datashare.ids", (64, 256), "<i4", memx.MEMX_TENSOR_DTYPE_INT32, 77),
        ):
            nbytes = shape[0] * shape[1] * 4
            desc = memx.tensor_desc(
                memx.MEMX_TENSOR_ROLE_DATA, memx_dtype,
                memx.MEMX_TENSOR_LAYOUT_ROW_MAJOR, memx.MEMX_TENSOR_FLAG_HOT,
                shape=shape, stride=(shape[1], 1),
            )
            allocation = ctx.malloc_tensor(nbytes, desc, name=name)
            allocations.append(allocation)
            view = np.frombuffer(allocation.buffer(), dtype=dtype).reshape(shape)
            index = np.arange(view.size, dtype=np.int64).reshape(shape)
            view[:] = ((index * seed + (index >> 7) + seed) & 255) - 128
            golden = view.tobytes()
            del view
            ctx.name_segment(allocation, name)
            ctx.update_tensor_flags_range(
                allocation, 0, nbytes,
                memx.MEMX_TENSOR_FLAG_READ_MOSTLY | memx.MEMX_TENSOR_FLAG_COLD,
            )
            ctx.force_compress_range(allocation)
            ctx.seal_range(allocation)
            segments[name] = {
                "nbytes": nbytes, "pages": nbytes // PAGE,
                "dtype": dtype, "shape": list(shape), "seed": seed,
                "crc32": zlib.crc32(golden) & 0xFFFFFFFF,
            }

        exported_bytes = rt.capsule_export(str(directory))
        rt.capsule_attach(str(directory))
        try:
            for name, meta in segments.items():
                try:
                    _, pages, nbytes = rt.capsule_segment(name)
                except OSError as exc:
                    raise RuntimeError(
                        f"{name}: missing capsule coverage; export captures only compressed "
                        "pages, not resident/hot/incompressible pages"
                    ) from exc
                if pages != meta["pages"] or nbytes != meta["nbytes"]:
                    raise RuntimeError(
                        f"{name}: incomplete capsule coverage {pages}/{meta['pages']} pages, "
                        f"{nbytes}/{meta['nbytes']} bytes; export captures only compressed "
                        "pages, not resident/hot/incompressible pages"
                    )
                buf = bytearray(nbytes)
                rt.capsule_materialize_segment(name, buf)
                if zlib.crc32(buf) & 0xFFFFFFFF != meta["crc32"]:
                    raise RuntimeError(f"{name}: exported checksum mismatch")
        finally:
            rt.capsule_detach()

        report = {"dir": str(directory), "segments": segments, "exported_bytes": exported_bytes}
        (directory / "checksums.json").write_text(json.dumps(report) + "\n")
        for name, meta in segments.items():
            print(f"SEGMENT {name} pages={meta['pages']} nbytes={meta['nbytes']} crc32=0x{meta['crc32']:08x}")
        print("JSON " + json.dumps(report))
        print(f"OK writer dir={directory} segments={len(segments)}")
    finally:
        for allocation in allocations:
            allocation.free()
        ctx.destroy()
        rt.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
