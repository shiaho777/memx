import json
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITER = ROOT / "examples" / "capsule_datashare_writer.py"
READER = ROOT / "examples" / "capsule_datashare_reader.py"
for key, value in {
    "MEMX_NO_SELFTEST": "1",
    "MEMX_CAPSULE_LITE": "1",
    "MEMX_CPU_ONLY": "1",
    "MEMX_CAPSULE_HOST_BIND": "0",
}.items():
    os.environ[key] = value
sys.path.insert(0, str(ROOT / "python"))

import memx_runtime as memx


def run(script, *args, success=True):
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(ROOT / "build"))
    proc = subprocess.run(
        [sys.executable, str(script), *args], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert (proc.returncode == 0) == success, (
        f"{script.name}: rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    )
    return proc


def report(stdout, prefix="JSON "):
    return json.loads(next(line[len(prefix):] for line in stdout.splitlines() if line.startswith(prefix)))


def test_capsule_datashare():
    with tempfile.TemporaryDirectory(prefix="memx-datashare-") as td:
        writer = report(run(WRITER, "--dir", td).stdout)
        assert Path(writer["dir"]) == Path(td).resolve()
        expected = writer["segments"]
        assert set(expected) == {"datashare.matrix", "datashare.ids"}
        golden = {}
        for name, count, seed, fmt in (
            ("datashare.matrix", 32768, 31, "f"),
            ("datashare.ids", 16384, 77, "i"),
        ):
            values = [((i * seed + (i >> 7) + seed) & 255) - 128 for i in range(count)]
            golden[name] = struct.pack(f"<{count}{fmt}", *values)
            assert expected[name]["crc32"] == zlib.crc32(golden[name]) & 0xFFFFFFFF
            assert expected[name]["nbytes"] == len(golden[name])
            assert expected[name]["pages"] * 16384 == len(golden[name])

        for args in ((), ("--expect", json.dumps(expected))):
            output = run(READER, "--dir", td, *args).stdout
            reader = report(output)["segments"]
            for name, meta in reader.items():
                assert meta == {key: expected[name][key] for key in ("nbytes", "pages", "crc32")}
            assert set(reader) == set(expected)
            stats = report(output, "STATS ")
            assert stats["ent_count"] == stats["verify_pages"] == 12
            assert stats["materialize_pages"] == 12
            assert stats["materialize_bytes"] == stats["logical_page_bytes"] == 196608

        wrong = json.loads(json.dumps(expected))
        wrong["datashare.matrix"]["crc32"] ^= 1
        failure = run(READER, "--dir", td, "--expect", json.dumps(wrong), success=False)
        assert "checksum mismatch" in failure.stderr

        rt = memx.Runtime()
        for attempt in range(2):
            rt.capsule_attach(td)
            try:
                assert rt.capsule_stats().attached, f"attach {attempt + 1} failed"
                for name, meta in expected.items():
                    _, pages, nbytes = rt.capsule_segment(name)
                    assert pages == meta["pages"] and nbytes == meta["nbytes"]
                    buf = bytearray(nbytes)
                    rt.capsule_materialize_segment(name, buf)
                    assert buf == golden[name], f"{name}: attach {attempt + 1} not bitexact"
                    assert zlib.crc32(buf) & 0xFFFFFFFF == meta["crc32"]
                assert rt.capsule_verify() == (0, 12, 0)
            finally:
                rt.capsule_detach()
            assert not rt.capsule_stats().attached


if __name__ == "__main__":
    test_capsule_datashare()
    print("OK capsule_datashare cross_process bitexact + checksum_rejection + attach_detach_reattach")
