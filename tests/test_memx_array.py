"""Byte-exact gate for the NumPy compressed-residency wrapper.

Covers zero-fill, seal/compress bitexact readback, writes after seal,
copy/materialize independence, invalid inputs, bounds controls,
quota enforcement, context-manager close, and a non-page-aligned
allocation.
"""

import gc
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
os.environ.setdefault("MEMX_CPU_ONLY", "1")
os.environ.setdefault("MEMX_NO_SELFTEST", "1")

import numpy as np

from memx_array import MemXArray


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as e:
        raise AssertionError(f"expected {exc_type.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc_type.__name__}, no exception raised")


def fresh_array(shape, dtype=np.float32, **kwargs):
    return MemXArray.zeros(shape, dtype=dtype, **kwargs)


def test_zeros_shape_and_zero_fill():
    a = fresh_array((128, 64), dtype=np.float32)
    try:
        assert a.shape == (128, 64)
        assert a.dtype == np.float32
        assert a.nbytes == 128 * 64 * 4
        assert a.size == 128 * 64
        assert a.ndim == 2
        assert not a.numpy().any(), "fresh allocation should read as zeros"
        assert a.numpy().base is None, "numpy() must return an independent snapshot"
    finally:
        a.close()


def test_write_read_bitexact_roundtrip():
    a = fresh_array((256, 64), dtype=np.float32)
    try:
        rng = np.random.default_rng(7)
        pattern = rng.standard_normal(256 * 64).astype(np.float32).reshape(256, 64)
        snapshot = a.numpy()
        snapshot[0, 0] = np.float32(123.0)
        assert a.numpy()[0, 0] == 0.0, "snapshot writes must not update MemX"
        a[:] = pattern
        sealed = a.seal()
        assert sealed > 0, "seal did not compress anything"
        got = a.numpy()
        assert got.tobytes() == pattern.tobytes(), "byte-exact mismatch after seal"
        assert got.base is None
    finally:
        a.close()


def test_write_then_seal_then_write():
    n = 512
    a = fresh_array((n, n), dtype=np.float32)
    try:
        pattern = (np.arange(n * n, dtype=np.int64).astype(np.float32) * 0.25).reshape(n, n)
        a[:] = pattern
        a.seal()
        a[7, 100] = -1.5
        assert float(a[7, 100]) == -1.5
        ref = pattern.copy()
        ref[7, 100] = -1.5
        assert np.array_equal(a.numpy(), ref), "write after seal broke content"
        a[100:104, 200:204] = np.float32(3.25)
        ref[100:104, 200:204] = np.float32(3.25)
        assert np.array_equal(a.numpy(), ref), "slice write after seal broke content"
        assert np.array_equal(a[123], ref[123]), "row read after seal broke content"
    finally:
        a.close()


def test_dtypes_supported():
    for dtype in [np.float32, np.float16, np.int32, np.int8, np.uint8]:
        a = fresh_array((1024,), dtype=dtype)
        try:
            assert a.dtype == dtype
            assert not a.numpy().any()
            a[:] = np.arange(1024, dtype=dtype)
            a.seal()
            got = a.numpy()
            assert got.tobytes() == np.arange(1024, dtype=dtype).tobytes(), dtype
        finally:
            a.close()


def test_copy_and_materialize_independent():
    a = fresh_array((128, 32), dtype=np.float32)
    try:
        a[:] = np.arange(128 * 32, dtype=np.float32).reshape(128, 32)
        a.seal()
        snap1 = a.copy()
        snap2 = a.materialize()
        assert isinstance(snap1, np.ndarray) and isinstance(snap2, np.ndarray)
        assert snap1.base is None and snap2.base is None
        a[0, 0] = np.float32(-9.0)
        assert snap1[0, 0] == 0.0 and snap2[0, 0] == 0.0
        as_f64 = np.asarray(a, dtype=np.float64)
        assert as_f64.dtype == np.float64
        assert np.array_equal(as_f64, a.materialize().astype(np.float64))
        expect_raises(ValueError, np.asarray, a, dtype=np.float32, copy=False)
        plain = np.asarray(a)
        assert plain.dtype == np.float32
        plain[1, 1] = np.float32(77.0)
        assert a[1, 1] == 33.0, "np.asarray snapshot must not update MemX"
    finally:
        a.close()


def test_snapshot_lifetime_across_close():
    a = fresh_array((256, 64), dtype=np.float32)
    pattern = np.arange(256 * 64, dtype=np.float32).reshape(256, 64)
    a[:] = pattern
    a.seal()
    snap = a.numpy()
    row = snap[13]
    assert np.array_equal(row, pattern[13])
    a.close()
    a.close()
    assert a.closed
    assert np.array_equal(snap, pattern), "snapshot corrupted after close"
    assert np.array_equal(row, pattern[13]), "snapshot slice corrupted after close"
    assert row.base is snap
    expect_raises(ValueError, a.numpy)
    expect_raises(ValueError, lambda: a[0, 0])
    expect_raises(ValueError, lambda: a.seal())
    expect_raises(ValueError, a.info)
    expect_raises(ValueError, a.copy)
    del snap, row
    gc.collect()
    expect_raises(ValueError, a.stats)


def test_context_manager():
    with fresh_array((32, 8), dtype=np.float32) as a:
        a[:] = 1.5
        a.seal()
        assert np.all(a.numpy() == 1.5)
    assert a.closed
    expect_raises(ValueError, a.numpy)


def test_invalid_inputs():
    expect_raises(TypeError, MemXArray.zeros, "not-a-shape")
    expect_raises(TypeError, MemXArray.zeros, (2, "x"))
    expect_raises(TypeError, MemXArray.zeros, (2, True))
    expect_raises(TypeError, MemXArray.zeros, 1.5)
    with fresh_array(()) as scalar:
        assert scalar.shape == () and scalar.numpy().item() == 0
    expect_raises(ValueError, MemXArray.zeros, (0,))
    expect_raises(ValueError, MemXArray.zeros, (-4,))
    expect_raises(ValueError, MemXArray.zeros, (2, 2, 2, 2, 2))
    expect_raises(ValueError, MemXArray.zeros, (2 ** 32, 2 ** 32))
    expect_raises(TypeError, MemXArray.zeros, ((2 ** 32,) * 4,))
    expect_raises(TypeError, MemXArray.zeros, (8,), dtype=object)
    expect_raises(TypeError, MemXArray.zeros, (8,), dtype=np.dtype("O"))
    expect_raises(TypeError, MemXArray.zeros, (8,), dtype=np.int64)
    expect_raises(TypeError, MemXArray.zeros, (8,), dtype=np.dtype("U8"))
    expect_raises(TypeError, MemXArray.zeros, (8,), dtype=[("a", np.float32)])
    expect_raises(ValueError, MemXArray.zeros, (8,), quota_bytes=0)
    expect_raises(ValueError, MemXArray.zeros, (8,), quota_bytes=-5)
    expect_raises(TypeError, MemXArray.zeros, (8,), quota_bytes=1.5)
    expect_raises(TypeError, MemXArray.zeros, (8,), read_mostly="yes")
    expect_raises(TypeError, MemXArray)


def test_bounds_controls():
    a = fresh_array((256,), dtype=np.float32)
    try:
        n = a.nbytes
        expect_raises(ValueError, a.seal, -1)
        expect_raises(ValueError, a.seal, n + 1)
        expect_raises(ValueError, a.seal, 0, n + 1)
        expect_raises(ValueError, a.seal, 8, -4)
        expect_raises(ValueError, a.prefetch, 0, n + 1)
        expect_raises(ValueError, a.hot_region, 0, n + 1)
        expect_raises(ValueError, a.retire, 0, n + 1)
        expect_raises(TypeError, a.seal, "x")
        a.seal(n)
        a.seal(n, 0)
        a.seal(16, 32)
        assert np.all(a.numpy() == 0)
    finally:
        a.close()


def test_non_page_aligned():
    a = fresh_array((1000,), dtype=np.uint8)
    try:
        assert a.nbytes == 1000
        assert a.nbytes % 16384 != 0
        pattern = (np.arange(1000, dtype=np.uint8) * 37 + 5).astype(np.uint8)
        a[:] = pattern
        a.seal()
        got = a.numpy()
        assert got.tobytes() == pattern.tobytes(), "non-page-aligned mismatch"
        a[999] = 200
        ref = pattern.copy()
        ref[999] = 200
        assert np.array_equal(a.numpy(), ref)
        a.seal(777, 223)
        assert np.array_equal(a.numpy(), ref)
    finally:
        a.close()


def test_quota_enforced():
    quota = 1 << 20
    a = fresh_array((1024,), dtype=np.float32, quota_bytes=quota)
    try:
        assert a.nbytes <= quota
        expect_raises(MemoryError, fresh_array, (1 << 22,), np.float32, context=a._ctx)
    finally:
        a.close()


def test_stats_content():
    a = fresh_array((4096, 64), dtype=np.float32)
    try:
        a[:] = np.arange(4096 * 64, dtype=np.float32).reshape(4096, 64)
        st0 = a.stats()
        assert st0["compressed_pages"] == 0
        assert st0["page_count"] * 16384 >= a.nbytes
        a.seal()
        st1 = a.stats()
        assert st1["compressed_pages"] > 0
        assert st1["resident_pages"] < st1["page_count"]
        assert st1["compressed_bytes"] < st1["nbytes"]
        assert a.compressed_pages() == st1["compressed_pages"]
        assert np.array_equal(a.numpy(), np.arange(4096 * 64, dtype=np.float32).reshape(4096, 64))
    finally:
        a.close()


def test_shared_runtime_isolation():
    a1 = fresh_array((128,), dtype=np.float32)
    try:
        a1[:] = np.float32(1.25)
        a2 = fresh_array((128,), dtype=np.float32)
        assert not a2.numpy().any(), "zero reuse must not leak bytes between arrays"
        a2.close()
    finally:
        a1.close()


def test_zero_reuse_and_borrowed_context():
    from memx_runtime import Runtime

    ctx = Runtime().create_context("array-reuse")
    try:
        first = fresh_array(16387, np.uint8, context=ctx)
        first[:] = 173
        first.seal()
        first.close()
        assert ctx.handle and ctx.stats().allocations_live == 0
        with fresh_array(16387, np.uint8, context=ctx) as second:
            assert second._alloc.ptr.value != 0
            assert second.numpy().tobytes() == bytes(16387)
            second.seal()
            assert second.materialize().tobytes() == bytes(16387)
        assert ctx.stats().allocations_live == 0
    finally:
        ctx.destroy()


def test_zeros_overwrites_dirty_allocator_memory():
    import ctypes
    from unittest.mock import patch
    import memx_array as module
    from memx_runtime import Runtime

    ctx = Runtime().create_context("dirty-array-allocator")
    try:
        desc = module.tensor_desc(
            module.MEMX_TENSOR_ROLE_DATA,
            module.MEMX_TENSOR_DTYPE_UINT8,
            1,
            0,
            shape=(16387,),
        )
        allocation = ctx.malloc_tensor(16387, desc, name="dirty")
        ctypes.memset(allocation.ptr.value, 173, allocation.size)
        with patch.object(ctx, "malloc_tensor", return_value=allocation):
            with fresh_array(16387, np.uint8, context=ctx) as a:
                assert a._alloc is allocation
                assert a.numpy().tobytes() == bytes(16387)
                a.seal()
                assert a.numpy().tobytes() == bytes(16387)
        assert ctx.stats().allocations_live == 0
    finally:
        ctx.destroy()


def test_gc_and_asarray_slice_lifetime():
    import weakref

    a = fresh_array((64, 64))
    a[:] = np.arange(4096, dtype=np.float32).reshape(64, 64)
    ctx = a._ctx
    reference = weakref.ref(a)
    row = np.asarray(a)[7, ::2]
    indexed = a[8, ::-1]
    del a
    gc.collect()
    assert reference() is None and ctx.handle is None
    assert row.tobytes() == np.arange(448, 512, 2, dtype=np.float32).tobytes()
    assert indexed.tobytes() == np.arange(575, 511, -1, dtype=np.float32).tobytes()
    row[:] = -1
    indexed[:] = -2


def test_special_float_bits_and_non_destructive_materialize():
    bits = np.array([0, 0x80000000, 0x7fc01234, 0x7f800000, 0xff800000], dtype=np.uint32)
    pattern = np.tile(bits, 5001).view(np.float32)
    with fresh_array(pattern.shape) as a:
        a[:] = pattern
        a.seal()
        before = a.stats()
        assert before["compressed_pages"] > 0
        for reader in (a.numpy, a.copy, a.materialize):
            assert reader().tobytes() == pattern.tobytes()
            assert a.stats()["compressed_pages"] == before["compressed_pages"]
        expect_raises(IndexError, a.__setitem__, a.size, 0)
        assert a.numpy().tobytes() == pattern.tobytes()
        expect_raises(TypeError, hash, a)
        expect_raises(AttributeError, getattr, a, "__array_interface__")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    if failed:
        print(f"memx_array: {failed} test(s) FAILED")
        return 1
    print("memx_array: OK all byte-exact checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
