"""Allocation, sliding residency, page-boundary and bitexact KV-cache tests."""

import ctypes
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import pytest

from memx_kv import MemXKVCache, PAGE_BYTES
import memx_runtime as rt


TB = 64
CAP = 4096
HOT = 256


@pytest.fixture(scope="module")
def runtime():
    runtime = rt.Runtime()
    yield runtime
    runtime.shutdown()


def make(runtime, **kwargs):
    params = dict(token_bytes=TB, capacity_tokens=CAP, hot_tokens=HOT, runtime=runtime)
    params.update(kwargs)
    return MemXKVCache(**params)


def assert_window(cache):
    expected = rt.kv_sliding_window(
        token_bytes=cache.token_bytes,
        hot_tokens=min(cache.written_tokens, cache.hot_tokens),
        prefetch_tokens=cache.prefetch_tokens,
        total_bytes=cache.capacity_bytes,
        start_token=max(0, cache.written_tokens - cache.hot_tokens),
    )
    for field in ("managed_offset", "managed_length", "hot_offset", "hot_length",
                  "prefetch_offset", "prefetch_length"):
        assert getattr(cache.window, field) == getattr(expected, field)


def assert_hot_resident(cache):
    window = cache.window
    if window.hot_length:
        info = cache.allocation.info(window.hot_offset, window.hot_length)
        assert info.compressed_pages == 0
        assert info.tensor_flags & rt.MEMX_TENSOR_FLAG_HOT
        assert info.tensor_flags & rt.MEMX_TENSOR_FLAG_NO_COMPRESS


def test_allocation(runtime):
    with make(runtime, layers=8, heads=4) as cache:
        info = cache.allocation.info()
        assert info.size == CAP * TB
        assert info.tensor_role == rt.MEMX_TENSOR_ROLE_KV_CACHE
        assert info.tensor_dtype == rt.MEMX_TENSOR_DTYPE_UINT8
        assert cache.stats()["layers"] == 8
        assert cache.stats()["heads"] == 4
        assert cache.stats()["capacity_bytes"] == CAP * TB


def test_append_progression(runtime):
    with make(runtime, prefetch_tokens=16) as cache:
        with patch.object(cache._ctx, "advance_kv_window", wraps=cache._ctx.advance_kv_window) as advance:
            for written in range(1, CAP + 1):
                cache.append(1)
                assert cache.written_tokens == written
                assert_window(cache)
            assert advance.call_count == CAP
        assert cache.window.prefetch_length == 0
        assert cache.window.managed_length == (CAP - HOT) * TB
        assert_hot_resident(cache)


def test_payload_and_count_appends(runtime):
    with make(runtime, capacity_tokens=8, hot_tokens=4) as cache:
        payload = bytes(range(256))
        cache.append(payload)
        assert cache.written_tokens == 4
        assert bytes(cache.allocation.buffer()[:len(payload)]) == payload
        ctypes.memset(cache.allocation.ptr.value + len(payload), 17, 2 * TB)
        cache.append(n_tokens=2)
        cache.append(2)
        assert cache.written_tokens == 8
        assert cache.window.hot_length == 4 * TB
        assert bytes(cache.allocation.buffer()[len(payload):6 * TB]) == bytes([17]) * (2 * TB)
        assert_window(cache)


@pytest.mark.parametrize("wrapper", [bytes, bytearray, memoryview])
def test_payload_buffers(runtime, wrapper):
    with make(runtime) as cache:
        payload = bytes(range(256))
        cache.append(tokens_bytes=wrapper(payload))
        assert cache.written_tokens == 4
        assert bytes(cache.allocation.buffer()[:len(payload)]) == payload


def test_seal_retired_bitexact(runtime):
    with make(runtime, hot_tokens=2) as cache:
        count = PAGE_BYTES // TB + 2
        payload = np.resize(np.arange(256, dtype=np.uint8), count * TB).tobytes()
        cache.append(payload)
        assert cache.window.managed_length == PAGE_BYTES
        assert cache.seal_retired() == 1
        assert cache.stats()["compressed_pages"] == 1
        assert_hot_resident(cache)
        assert bytes(cache.allocation.buffer()[:len(payload)]) == payload
        assert cache.seal_retired() == 1
        assert bytes(cache.allocation.buffer()[:len(payload)]) == payload


def test_unaligned_retirement_preserves_hot_page(runtime):
    with make(runtime, hot_tokens=2) as cache:
        cache.append(2 * PAGE_BYTES // TB + 1)
        stats = cache.stats()
        assert stats["hot_offset"] == 2 * PAGE_BYTES - TB
        assert stats["hot_pages"] == 2
        assert stats["retired_pages"] == 1
        assert stats["sealable_retired_bytes"] == PAGE_BYTES
        assert cache.seal_retired() == 1
        assert cache.allocation.info().compressed_pages == 1
        assert_hot_resident(cache)
        cache.append(1)
        assert cache.stats()["retired_pages"] == 2
        cache.seal_retired()
        assert cache.allocation.info().compressed_pages == 2
        assert_hot_resident(cache)


def test_stats_sanity_and_allocation_locality(runtime):
    with make(runtime) as cache, make(runtime) as other:
        other.append(CAP)
        other.seal_retired()
        cache.append(500)
        stats = cache.stats()
        assert stats["written_tokens"] == 500
        assert stats["written_bytes"] == 500 * TB
        assert stats["hot_tokens"] == HOT
        assert stats["hot_bytes"] == HOT * TB
        assert stats["hot_pages"] == 2
        assert stats["retired_tokens"] == 500 - HOT
        assert stats["retired_bytes"] == (500 - HOT) * TB
        assert stats["retired_pages"] == 0
        assert stats["sealable_retired_bytes"] == 0
        assert stats["compressed_pages"] == 0
        assert stats["resident_pages"] == stats["page_count"] == CAP * TB // PAGE_BYTES
        assert stats["append_calls"] == 1


@pytest.mark.parametrize("capacity", [0, 1, 4096])
def test_empty_cache(runtime, capacity):
    with make(runtime, capacity_tokens=capacity) as cache:
        for arg in (0, b""):
            assert cache.append(arg) is cache.window
        assert cache.append(n_tokens=0) is cache.window
        assert cache.seal_retired() == 0
        stats = cache.stats()
        for field in ("written_tokens", "hot_pages", "retired_pages", "append_calls", "seal_calls"):
            assert stats[field] == 0
        if capacity == 0:
            assert cache.allocation is None
            assert stats["page_count"] == 0
            with pytest.raises(ValueError):
                cache.append(1)


@pytest.mark.parametrize("token_bytes", [1, 16, 64, 16385])
def test_single_token_and_large_hot_window(runtime, token_bytes):
    with make(runtime, token_bytes=token_bytes, capacity_tokens=1, hot_tokens=1024) as cache:
        cache.append(bytes([37]) * token_bytes)
        assert_window(cache)
        stats = cache.stats()
        assert stats["hot_tokens"] == 1
        assert stats["hot_pages"] == (token_bytes + PAGE_BYTES - 1) // PAGE_BYTES
        assert stats["retired_pages"] == 0
        assert cache.seal_retired() == 0
        assert_hot_resident(cache)


def test_subpage_retirement(runtime):
    with make(runtime, capacity_tokens=4, hot_tokens=1) as cache:
        cache.append(4)
        assert cache.stats()["retired_tokens"] == 3
        assert cache.stats()["retired_pages"] == 0
        assert cache.stats()["hot_pages"] == 1
        assert cache.seal_retired() == 0
        assert_hot_resident(cache)


@pytest.mark.parametrize("kwargs", [
    dict(token_bytes=0), dict(token_bytes=-1), dict(token_bytes=64.0),
    dict(token_bytes=True), dict(token_bytes=1 << 64), dict(capacity_tokens=-1),
    dict(capacity_tokens=1 << 64), dict(hot_tokens=0), dict(prefetch_tokens=-1),
    dict(layers=0), dict(heads=-2),
])
def test_invalid_constructor(runtime, kwargs):
    before = runtime.stats().total_pages
    with pytest.raises(ValueError):
        make(runtime, **kwargs)
    assert runtime.stats().total_pages == before


def test_invalid_append_and_overflow(runtime):
    with make(runtime, capacity_tokens=2) as cache:
        for kwargs in ({}, dict(n_tokens=-1), dict(tokens_bytes=3.5),
                       dict(tokens_bytes=True), dict(tokens_bytes=b"x"),
                       dict(tokens_bytes=b"x" * TB, n_tokens=1)):
            with pytest.raises(ValueError):
                cache.append(**kwargs)
            assert cache.written_tokens == 0
        cache.append(2)
        for arg in (1, b"x" * TB):
            with pytest.raises(ValueError):
                cache.append(arg)
            assert cache.written_tokens == 2


def test_close_does_not_shutdown_other_cache(runtime):
    with make(runtime) as survivor:
        cache = make(runtime)
        cache.close()
        cache.close()
        for action in (lambda: cache.append(0), cache.stats, cache.seal_retired):
            with pytest.raises(RuntimeError):
                action()
        survivor.append(1)
        assert survivor.stats()["written_tokens"] == 1
