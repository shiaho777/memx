"""Client SDK for memx_stored, the unprivileged capsule store service.

Protocol: length-prefixed frames over a Unix-domain socket.
Each request frame: 4-byte command + payload; response frame: payload.
Commands: PUT (name + page-aligned bytes), GET (name), LIST, DROP (name),
COMIT (commit all segments to a new capsule generation), VERIY <gen>,
STAT, QUIT.
"""

import os
import socket
import struct
from pathlib import Path

PAGE = 16384


class StoreError(Exception):
    pass


class Store:
    def __init__(self, sock_path=None):
        if sock_path is None:
            root = Path(os.environ.get(
                "MEMX_STORE_ROOT",
                Path.home() / "Library/Application Support/MemX/store",
            ))
            sock_path = str(root / "store.sock")
        self.sock_path = sock_path
        self._sock = None

    def connect(self):
        if self._sock is not None:
            return self
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(120.0)
        s.connect(self.sock_path)
        self._sock = s
        return self

    def close(self):
        if self._sock is not None:
            try:
                self._frame(b"QUIT")
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    def _frame(self, payload):
        if self._sock is None:
            raise StoreError("not connected")
        self._sock.sendall(struct.pack(">I", len(payload)) + payload)
        hdr = self._recv_exact(4)
        (rlen,) = struct.unpack(">I", hdr)
        if rlen > 256 * 1024 * 1024:
            raise StoreError(f"response too large: {rlen}")
        return self._recv_exact(rlen) if rlen else b""

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise StoreError("connection closed")
            buf += chunk
        return buf

    def put(self, name, data):
        if not name or len(name) >= 64:
            raise StoreError("name must be 1..63 bytes")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        data = bytes(data)
        if len(data) == 0 or len(data) % PAGE != 0:
            raise StoreError(f"data must be non-empty and page-aligned ({PAGE} bytes)")
        resp = self._frame(b"PUT " + bytes([len(name)]) + name.encode() + data)
        text = resp.decode("utf-8", "replace")
        if not text.startswith("OK"):
            raise StoreError(text)
        return text

    def get(self, name):
        resp = self._frame(b"GET " + name.encode())
        if resp.startswith(b"ERR"):
            raise StoreError(resp.decode("utf-8", "replace"))
        return resp

    def list(self):
        resp = self._frame(b"LIST")
        text = resp.decode("utf-8", "replace")
        if text.strip() == "(empty)":
            return []
        segs = []
        for line in text.strip().splitlines():
            parts = line.split()
            if len(parts) == 3:
                segs.append((parts[0], int(parts[1]), int(parts[2])))
        return segs

    def drop(self, name):
        resp = self._frame(b"DROP" + name.encode())
        text = resp.decode("utf-8", "replace")
        if not text.startswith("OK"):
            raise StoreError(text)
        return text

    def commit(self):
        resp = self._frame(b"COMIT")
        text = resp.decode("utf-8", "replace")
        if not text.startswith("OK"):
            raise StoreError(text)
        return text

    def verify(self, gen):
        resp = self._frame(b"VERIY" + gen.encode())
        text = resp.decode("utf-8", "replace")
        if not text.startswith("OK"):
            raise StoreError(text)
        return text

    def stats(self):
        resp = self._frame(b"STAT")
        text = resp.decode("utf-8", "replace")
        if not text.startswith("OK"):
            raise StoreError(text)
        return text
