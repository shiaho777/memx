"""Bitexact KV-cache residency orchestration without quantization or eviction.

Token storage is contiguous and token-major. token_bytes is the total size of
both K and V for one token across all layers and heads (for example,
2 * layers * heads * head_dim * dtype_itemsize). layers and heads are optional
metadata only; they do not multiply token_bytes. Byte payloads are copied
unchanged; count-only appends announce data already written by the caller via
allocation.buffer(). Residency changes and runtime compression preserve bytes.
"""

import ctypes
import operator

import memx_runtime as rt


PAGE_BYTES = 16384


def _integer(value, name, minimum=0):
    try:
        if isinstance(value, bool):
            raise TypeError
        value = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer >= {minimum}") from None
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class MemXKVCache:
    """Own a fixed-capacity allocation and context, but not the global runtime.

    append(payload) copies whole-token bytes; append(count) or
    append(n_tokens=count) advances residency without modifying data. The hot
    window follows the written tail; optional prefetch stages upcoming capacity.
    An empty, zero-capacity cache has no allocation. Close before runtime shutdown
    and do not use allocation views after close. Calls must be serialized with
    external writers. Configure hot_tokens and prefetch_tokens at construction.
    """

    def __init__(self, token_bytes, capacity_tokens, hot_tokens,
                 prefetch_tokens=0, layers=None, heads=None, name="kv_cache",
                 runtime=None):
        self.token_bytes = _integer(token_bytes, "token_bytes", 1)
        self.capacity_tokens = _integer(capacity_tokens, "capacity_tokens")
        self.hot_tokens = _integer(hot_tokens, "hot_tokens", 1)
        self.prefetch_tokens = _integer(prefetch_tokens, "prefetch_tokens")
        self.layers = None if layers is None else _integer(layers, "layers", 1)
        self.heads = None if heads is None else _integer(heads, "heads", 1)
        self.capacity_bytes = self.capacity_tokens * self.token_bytes
        if max(self.capacity_bytes, self.token_bytes) > (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - PAGE_BYTES:
            raise ValueError("cache size exceeds the runtime address range")
        self.written_tokens = 0
        self.append_calls = 0
        self.seal_calls = 0
        self.window = rt.kv_cache_window()
        self.allocation = None
        self._ctx = None
        self._closed = False
        self._runtime = runtime or rt.Runtime()
        if self.capacity_bytes:
            self._ctx = self._runtime.create_context(name)
            try:
                desc = rt.tensor_desc(
                    role=rt.MEMX_TENSOR_ROLE_KV_CACHE,
                    dtype=rt.MEMX_TENSOR_DTYPE_UINT8,
                    layout=rt.MEMX_TENSOR_LAYOUT_ROW_MAJOR,
                    flags=rt.MEMX_TENSOR_FLAG_SEQUENTIAL,
                    shape=(self.capacity_tokens, self.token_bytes),
                    stride=(self.token_bytes, 1),
                )
                self.allocation = self._ctx.malloc_tensor(self.capacity_bytes, desc, name=name)
                self.window = self._advance_window(0)
            except Exception:
                self.close()
                raise

    def _check_open(self):
        if self._closed:
            raise RuntimeError("KV cache is closed")

    def _advance_window(self, written):
        return self._ctx.advance_kv_window(
            self.allocation, self.token_bytes, written,
            self.hot_tokens, min(self.prefetch_tokens, self.capacity_tokens - written),
        )

    def append(self, tokens_bytes=None, n_tokens=None):
        """Append unchanged bytes or announce a token count; return the window."""
        self._check_open()
        if (tokens_bytes is None) == (n_tokens is None):
            raise ValueError("pass exactly one of tokens_bytes or n_tokens")
        payload = None
        if n_tokens is not None:
            n = _integer(n_tokens, "n_tokens")
        elif isinstance(tokens_bytes, (bytes, bytearray, memoryview)):
            payload = bytes(tokens_bytes)
            n, remainder = divmod(len(payload), self.token_bytes)
            if remainder:
                raise ValueError("payload must contain whole tokens")
        else:
            n = _integer(tokens_bytes, "n_tokens")
        written = self.written_tokens + n
        if written > self.capacity_tokens:
            raise ValueError("KV cache capacity exceeded")
        if n == 0:
            return self.window
        if payload is not None:
            ctypes.memmove(self.allocation.ptr.value + self.written_tokens * self.token_bytes,
                           payload, len(payload))
        window = self._advance_window(written)
        self.written_tokens = written
        self.window = window
        self.append_calls += 1
        return window

    def seal_retired(self):
        """Synchronously seal full retired pages and return the runtime page count.

        Pages shared with the hot window remain resident, including sub-page
        caches. Unwritten capacity is not retired. Incompressible pages may
        remain resident; stats reports actual compression, not an estimate.
        """
        self._check_open()
        length = self.window.managed_length // PAGE_BYTES * PAGE_BYTES
        if not length:
            return 0
        sealed = self._ctx.seal_range(self.allocation, 0, length)
        self.seal_calls += 1
        return sealed

    def stats(self):
        """Return allocation-local compression and page-granular window counts.

        hot_pages includes every page touched by the hot range; retired_pages
        counts only full pages before it. They are disjoint. Token/byte counts
        also include retired bytes sharing a hot page. Compression counts can
        change with background runtime activity and accesses through views.
        """
        self._check_open()
        info = self.allocation.info() if self.allocation is not None else None
        hot_tokens = min(self.written_tokens, self.hot_tokens)
        hot_end = self.window.hot_offset + self.window.hot_length
        hot_pages = ((hot_end + PAGE_BYTES - 1) // PAGE_BYTES -
                     self.window.hot_offset // PAGE_BYTES) if self.window.hot_length else 0
        retired_pages = self.window.managed_length // PAGE_BYTES
        page_count = info.page_count if info is not None else 0
        compressed_pages = info.compressed_pages if info is not None else 0
        return {
            "token_bytes": self.token_bytes,
            "capacity_tokens": self.capacity_tokens,
            "capacity_bytes": self.capacity_bytes,
            "written_tokens": self.written_tokens,
            "written_bytes": self.written_tokens * self.token_bytes,
            "hot_tokens": hot_tokens,
            "hot_bytes": self.window.hot_length,
            "hot_offset": self.window.hot_offset,
            "hot_pages": hot_pages,
            "prefetch_tokens": self.window.prefetch_length // self.token_bytes,
            "prefetch_bytes": self.window.prefetch_length,
            "retired_tokens": self.written_tokens - hot_tokens,
            "retired_bytes": self.window.managed_length,
            "sealable_retired_bytes": retired_pages * PAGE_BYTES,
            "retired_pages": retired_pages,
            "seal_calls": self.seal_calls,
            "append_calls": self.append_calls,
            "page_count": page_count,
            "resident_pages": page_count - compressed_pages,
            "compressed_pages": compressed_pages,
            "compressed_bytes": info.compressed_bytes if info is not None else 0,
            "layers": self.layers,
            "heads": self.heads,
        }

    def close(self):
        if self.allocation is not None:
            self.allocation.free()
            self.allocation = None
        if self._ctx is not None:
            self._ctx.destroy()
            self._ctx = None
        self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
