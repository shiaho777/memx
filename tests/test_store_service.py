#!/usr/bin/env python3
"""memx_stored end-to-end: PUT/GET bitexact, LIST, DROP, COMMIT to a capsule
generation, VERIFY clean, and cross-process materialize_segment from the
committed capsule.

Runs a private server instance on a temp socket + store root.
"""
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("DYLD_LIBRARY_PATH", str(ROOT / "build"))

import memx_runtime as memx
import memx_store


def wait_socket(path, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            try:
                s = memx_store.Store(path).connect()
                s.close()
                return True
            except Exception:
                pass
        time.sleep(0.1)
    return False


def main():
    vessel = ROOT / "build" / "memx_stored"
    if not vessel.exists():
        print("SKIP: memx_stored missing")
        return 0

    with tempfile.TemporaryDirectory() as td:
        sock = os.path.join(td, "store.sock")
        store_root = os.path.join(td, "store")
        env = dict(os.environ)
        env["MEMX_NO_SELFTEST"] = "1"
        proc = subprocess.Popen(
            [str(vessel), "--sock", sock, "--root", store_root],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            assert wait_socket(sock), "server never came up"

            page = bytes((i * 31 + 7) & 0xFF for i in range(16384))
            rows = page * 24
            vocab = bytes(((i * 17 + 3) & 0xFF) for i in range(16384 * 8))

            with memx_store.Store(sock) as st:
                st.put("rows", rows)
                st.put("vocab", vocab)
                segs = st.list()
                names = {s[0] for s in segs}
                assert names == {"rows", "vocab"}, f"list wrong: {segs}"

                got = st.get("rows")
                assert got == rows, "rows GET not bitexact"
                got = st.get("vocab")
                assert got == vocab, "vocab GET not bitexact"

                st.drop("vocab")
                segs = st.list()
                assert {s[0] for s in segs} == {"rows"}, f"drop failed: {segs}"
                st.put("vocab", vocab)

                commit_msg = st.commit()
                assert commit_msg.startswith("OK commit gen-"), commit_msg
                gen = commit_msg.split()[2]
                assert (Path(store_root) / gen / "spill.bin").exists(), "generation dir missing"

                vmsg = st.verify(gen)
                assert vmsg.startswith("OK verify bad=0"), vmsg

                smsg = st.stats()
                assert "segments=2" in smsg, smsg

            rt = memx.Runtime(ROOT / "build" / "libmemx_runtime.dylib")
            rt.capsule_attach(str(Path(store_root) / gen))
            rank, pages, nbytes = rt.capsule_segment("rows")
            assert nbytes == len(rows), f"segment nbytes {nbytes} != {len(rows)}"
            buf = bytearray(pages * 16384)
            rt.capsule_materialize_segment("rows", buf)
            assert bytes(buf[:len(rows)]) == rows, "rows segment not bitexact"
            rank2, pages2, nbytes2 = rt.capsule_segment("vocab")
            assert nbytes2 == len(vocab), f"vocab nbytes {nbytes2} != {len(vocab)}"
            buf2 = bytearray(pages2 * 16384)
            rt.capsule_materialize_segment("vocab", buf2)
            assert bytes(buf2[:len(vocab)]) == vocab, "vocab segment not bitexact"
            rt.capsule_detach()
            rt.shutdown()

            print(f"OK store service {gen} put/get/list/drop/commit/verify bitexact")
            return 0
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
