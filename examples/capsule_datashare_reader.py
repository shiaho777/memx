import argparse
import json
import os
import sys
import zlib
from pathlib import Path

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
KNOWN_SEGMENTS = ("datashare.matrix", "datashare.ids")


def main():
    parser = argparse.ArgumentParser(description="Attach and verify a capsule data generation")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--expect", help="Override checksums.json with the writer's segments JSON")
    args = parser.parse_args()
    if args.expect is not None:
        expected = json.loads(args.expect)
    else:
        expected = json.loads((Path(args.dir) / "checksums.json").read_text())["segments"]
    if set(expected) != set(KNOWN_SEGMENTS):
        parser.error("expected checksums must contain exactly the two known datashare segments")

    rt = memx.Runtime()
    rt.capsule_attach(args.dir)
    try:
        before = rt.capsule_stats()
        if not before.attached:
            raise RuntimeError("capsule not live after attach")
        segments = {}
        for name in KNOWN_SEGMENTS:
            rank, pages, nbytes = rt.capsule_segment(name)
            want = expected[name]
            if nbytes != want["nbytes"] or pages * PAGE != nbytes:
                raise RuntimeError(f"{name}: incomplete capsule coverage or unexpected byte count")
            buf = bytearray(nbytes)
            rt.capsule_materialize_segment(name, buf)
            checksum = zlib.crc32(buf) & 0xFFFFFFFF
            if checksum != want["crc32"]:
                raise RuntimeError(
                    f"{name}: checksum mismatch got=0x{checksum:08x} expected=0x{want['crc32']:08x}"
                )
            segments[name] = {"nbytes": nbytes, "pages": pages, "crc32": checksum}
            print(f"SEGMENT {name} rank={rank} pages={pages} nbytes={nbytes} crc32=0x{checksum:08x}")
        bad, total, rc = rt.capsule_verify()
        if bad or rc:
            raise RuntimeError(f"capsule_verify failed: bad={bad} rc={rc}")
        stats = rt.capsule_stats()
        print("JSON " + json.dumps({"segments": segments}))
        print("STATS " + json.dumps({
            "ent_count": int(stats.ent_count),
            "logical_page_bytes": int(stats.page_bytes),
            "spill_bytes": int(stats.spill_bytes),
            "ledger_bytes": int(stats.ledger_bytes),
            "materialize_pages": int(stats.materialize_pages - before.materialize_pages),
            "materialize_bytes": int(stats.materialize_bytes - before.materialize_bytes),
            "verify_pages": total,
        }))
        print("Capability sizes are not measured process RSS; only requested segments are materialized.")
        print(f"OK reader dir={args.dir} verify_pages={total} bad=0")
    finally:
        rt.capsule_detach()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
