"""Named byte blobs and JSON checkpoints backed by memx_stored.

Values use a versioned prefix and page padding; a reserved __meta__ segment
records their original lengths. Saves replace a session, without transactions
or multi-writer locking. COMIT snapshots the entire service, not one session.
A new client can resume while the daemon is running; the service does not
rehydrate committed generations into GET/LIST after a daemon restart.
"""

import json

from memx_store import PAGE, Store, StoreError

META_NAME = "__meta__"
_MAGIC = b"MEMXKV1\x00"


class SessionError(StoreError):
    pass


class SessionStore:
    def __init__(self, sock_path=None):
        self._store = Store(sock_path)

    def connect(self):
        try:
            self._store.connect()
        except OSError as e:
            raise SessionError(f"store service unreachable at {self._store.sock_path}: {e}") from e
        return self

    def close(self):
        self._store.close()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _seg(session_id, name):
        for part in (session_id, name):
            if not isinstance(part, str) or not part or any(
                ord(c) < 33 or ord(c) > 126 or c == ":" for c in part
            ):
                raise SessionError("session and tensor names must be printable ASCII without spaces or ':'")
        seg = f"{session_id}:{name}"
        if len(seg) >= 64:
            raise SessionError(f"segment name out of range: {seg!r}")
        return seg

    @staticmethod
    def _pad(data):
        padded = max(PAGE, -(-len(data) // PAGE) * PAGE)
        return data + b"\x00" * (padded - len(data))

    def _read_meta(self, session_id):
        seg = self._seg(session_id, META_NAME)
        self.connect()
        try:
            blob = self._store.get(seg)
            meta = json.loads(blob.rstrip(b"\x00").decode())
            if not isinstance(meta, dict) or not isinstance(meta.get("lengths"), dict):
                raise ValueError("missing tensor lengths")
            for name, length in meta["lengths"].items():
                self._seg(session_id, name)
                if name == META_NAME or type(length) is not int or length < 0:
                    raise ValueError("invalid tensor length or name")
            return meta
        except (StoreError, ValueError) as e:
            raise SessionError(f"session {session_id!r} missing or invalid: {e}") from e

    def _save(self, session_id, tensors, checkpoint_meta, commit):
        meta_seg = self._seg(session_id, META_NAME)
        if not isinstance(tensors, dict):
            raise TypeError("tensors must be a dict of bytes-like values")
        prepared = {}
        lengths = {}
        for name, data in tensors.items():
            seg = self._seg(session_id, name)
            if name == META_NAME:
                raise SessionError("tensor name __meta__ is reserved")
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise TypeError("tensor data must be bytes-like")
            data = bytes(data)
            lengths[name] = len(data)
            prepared[seg] = self._pad(_MAGIC + data)
        meta = {"lengths": lengths}
        if checkpoint_meta is not None:
            meta["checkpoint_meta"] = checkpoint_meta
        blob = self._pad(json.dumps(meta, sort_keys=True, allow_nan=False).encode())
        self.connect()
        prefix = f"{session_id}:"
        old = {seg for seg, _, _ in self._store.list() if seg.startswith(prefix)}
        for seg, data in prepared.items():
            self._store.put(seg, data)
        self._store.put(meta_seg, blob)
        for seg in sorted(old - prepared.keys() - {meta_seg}):
            self._store.drop(seg)
        return self._store.commit() if commit else None

    def save_session(self, session_id: str, tensors: dict[str, bytes], commit=False):
        """Replace named blobs; optionally return the service COMIT response."""
        return self._save(session_id, tensors, None, commit)

    def _load(self, session_id, meta):
        out = {}
        for name, length in meta["lengths"].items():
            blob = self._store.get(self._seg(session_id, name))
            if not blob.startswith(_MAGIC) or length > len(blob) - len(_MAGIC):
                raise SessionError(f"invalid tensor payload: {session_id}:{name}")
            out[name] = blob[len(_MAGIC):len(_MAGIC) + length]
        return out

    def load_session(self, session_id):
        return self._load(session_id, self._read_meta(session_id))

    def list_sessions(self):
        self.connect()
        return sorted(seg.split(":", 1)[0] for seg, _, _ in self._store.list()
                      if seg.endswith(f":{META_NAME}") and seg.count(":") == 1)

    def drop_session(self, session_id):
        self._read_meta(session_id)
        prefix = f"{session_id}:"
        for seg, _, _ in self._store.list():
            if seg.startswith(prefix):
                self._store.drop(seg)

    def checkpoint(self, session_id, kv_bytes, meta: dict, commit=False):
        if not isinstance(meta, dict):
            raise TypeError("meta must be a dict")
        return self._save(session_id, {"kv": kv_bytes}, meta, commit)

    def restore(self, session_id):
        meta = self._read_meta(session_id)
        if not isinstance(meta.get("checkpoint_meta"), dict) or "kv" not in meta["lengths"]:
            raise SessionError(f"session {session_id!r} has no checkpoint")
        return self._load(session_id, meta)["kv"], meta["checkpoint_meta"]

    def commit(self):
        self.connect()
        return self._store.commit()

    def verify(self, generation):
        self.connect()
        return self._store.verify(generation)


PersistentSession = SessionStore
