#!/usr/bin/env python3
"""Session KV persistence gate: put/get roundtrip bitexact, save/load with
multiple keys, list_sessions, drop, commit+verify, checkpoint/restore, and
graceful error when the store service is absent.

Runs a private memx_stored instance on a temp socket + store root, mirroring
tests/test_store_service.py.
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

import memx_store
import memx_session


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


def start_server(vessel, sock, store_root, env):
    proc = subprocess.Popen(
        [str(vessel), "--sock", sock, "--root", store_root],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert wait_socket(sock), "server never came up"
    return proc


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
        env["MEMX_STORE_KEEP_GENS"] = "8"
        proc = start_server(vessel, sock, store_root, env)
        try:

            def page_bytes(seed, npages):
                return bytes((seed + i * 13 + 5) & 0xFF for i in range(16384 * npages))

            tensors = {
                "k0": page_bytes(7, 1),
                "k1": page_bytes(200, 3),
                "k2": page_bytes(5555, 2) + page_bytes(99, 1)[:12345],
            }

            with memx_session.SessionStore(sock) as ss:
                assert ss.save_session("sess-a", tensors) is None
                assert ss.load_session("sess-a") == tensors, "save/load not bitexact"

                assert ss.list_sessions() == ["sess-a"]
                ss.save_session("sess-b", {"kv": b""})
                assert ss.load_session("sess-b") == {"kv": b""}
                assert ss.list_sessions() == ["sess-a", "sess-b"]

                replaced = {"only": page_bytes(1, 1)}
                ss.save_session("sess-a", replaced)
                assert ss.load_session("sess-a") == replaced
                assert ss.list_sessions() == ["sess-a", "sess-b"]

                ss.save_session("sess-c", tensors)
                ss.drop_session("sess-a")
                assert ss.list_sessions() == ["sess-b", "sess-c"]
                try:
                    ss.load_session("sess-a")
                    raise AssertionError("load of dropped session should fail")
                except memx_session.SessionError:
                    pass

                commit_msg = ss.save_session("sess-c", tensors, commit=True)
                assert commit_msg.startswith("OK commit gen-"), commit_msg
                gen = commit_msg.split()[2]
                assert ss.verify(gen).startswith("OK verify bad=0")
                assert ss.load_session("sess-c") == tensors, "post-commit load not bitexact"

                kv = bytes(range(256)) * 100 + b"\x00" * 7
                meta = {"step": 17, "model": "q0.8b", "tokens": 512, "neg": -3, "f": 2.5}
                ss.checkpoint("sess-ckpt", kv, meta)
                got_kv, got_meta = ss.restore("sess-ckpt")
                assert got_kv == kv, "checkpoint kv not bitexact"
                assert got_meta == meta, f"meta mismatch: {got_meta}"
                ss.save_session("sess-ckpt", {"kv": kv})
                try:
                    ss.restore("sess-ckpt")
                    raise AssertionError("restore after plain save should fail")
                except memx_session.SessionError:
                    pass

                try:
                    ss.save_session("bad sess", {"k": b"\x00"})
                    raise AssertionError("invalid session id should fail")
                except memx_session.SessionError:
                    pass

                ss.checkpoint("sess-ckpt", kv, meta, commit=True)
                ss.drop_session("sess-c")
                with memx_store.Store(sock) as raw:
                    names = {row[0] for row in raw.list()}
                    assert not any(name.startswith("sess-a:") for name in names)
                    assert not any(name.startswith("sess-c:") for name in names)
                    assert {name for name in names if name.startswith("sess-ckpt:")} == {
                        "sess-ckpt:kv", "sess-ckpt:__meta__",
                    }
                ss.save_session("edge", {"err": b"ERR arbitrary data\x00", "empty": b""})
                assert ss.load_session("edge") == {"err": b"ERR arbitrary data\x00", "empty": b""}
                ss.save_session("edge", {})
                assert ss.load_session("edge") == {}
                ss.drop_session("edge")
                for invalid in ({"__meta__": b"x"}, {"bad:name": b"x"}, {"x": 7}):
                    try:
                        ss.save_session("sess-b", invalid)
                        raise AssertionError("invalid tensors should fail")
                    except (memx_session.SessionError, TypeError):
                        pass
                    assert ss.load_session("sess-b") == {"kv": b""}

            absent = os.path.join(td, "absent.sock")
            try:
                with memx_session.SessionStore(absent) as ss:
                    ss.save_session("x", {"kv": b"\x00"})
                raise AssertionError("connect to absent server should fail")
            except memx_session.SessionError:
                pass

            with memx_session.SessionStore(sock) as ss:
                assert ss.list_sessions() == ["sess-b", "sess-ckpt"]
                got_kv, got_meta = ss.restore("sess-ckpt")
                assert got_kv == kv and got_meta == meta
            print("OK memx_session bitexact save/load/replace/empty/list/drop/commit+verify/checkpoint")
            return 0
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
