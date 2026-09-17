"""MemX storage with copy-only NumPy access and explicit indexed writes.

numpy(), np.asarray(), indexing, copy(), and materialize() return independent
snapshots, never managed-memory views. Modify storage with array[key] = value.
Snapshots and their slices remain valid after close or context destruction.
Caller-supplied contexts must stay alive until the wrapper is closed; runtime
shutdown must not race wrapper operations. Each wrapper serializes its own
operations, not operations performed directly on a supplied context.
"""

import ctypes
import math
import operator
import sys
import threading
import weakref

import numpy as np

from memx_runtime import (
    Runtime,
    tensor_desc,
    MEMX_EPOCH_FINAL,
    MEMX_TENSOR_ROLE_DATA,
    MEMX_TENSOR_DTYPE_FP32,
    MEMX_TENSOR_DTYPE_FP16,
    MEMX_TENSOR_DTYPE_INT32,
    MEMX_TENSOR_DTYPE_INT8,
    MEMX_TENSOR_DTYPE_UINT8,
    MEMX_TENSOR_LAYOUT_ROW_MAJOR,
    MEMX_TENSOR_FLAG_READ_MOSTLY,
    MEMX_TENSOR_FLAG_HOT,
    MEMX_TENSOR_FLAG_COLD,
    MEMX_MATERIALIZE_KEEP_COMPRESSED,
    MEMX_MATERIALIZE_ALLOW_RESIDENT,
)

_NUMPY_TO_DTYPE = {
    np.dtype(np.float32): MEMX_TENSOR_DTYPE_FP32,
    np.dtype(np.float16): MEMX_TENSOR_DTYPE_FP16,
    np.dtype(np.int32): MEMX_TENSOR_DTYPE_INT32,
    np.dtype(np.int8): MEMX_TENSOR_DTYPE_INT8,
    np.dtype(np.uint8): MEMX_TENSOR_DTYPE_UINT8,
}


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None


def _validate(shape, dtype):
    try:
        shape = (_integer(shape, "shape"),)
    except TypeError:
        if isinstance(shape, (str, bytes, bool, np.bool_)):
            raise TypeError("shape must be an integer or sequence of integers")
        shape = tuple(_integer(dim, "dimension") for dim in shape)
    if len(shape) > 4:
        raise ValueError("MemX supports at most four dimensions")
    if any(dim <= 0 for dim in shape):
        raise ValueError("dimensions must be positive; empty arrays are unsupported")
    dtype = np.dtype(dtype)
    if dtype.hasobject or dtype.fields or dtype.subdtype or dtype not in _NUMPY_TO_DTYPE:
        raise TypeError("supported native dtypes: float32, float16, int32, int8, uint8")
    nbytes = math.prod(shape) * dtype.itemsize
    if nbytes > sys.maxsize:
        raise ValueError("array byte size exceeds the platform addressable range")
    return shape, dtype, nbytes


def _release(allocation, context, owned):
    if context.handle:
        allocation.free()
        if owned:
            context.destroy()


class MemXArray:
    """Use zeros() to construct; NumPy reads are independent copies."""

    __hash__ = None

    def __init__(self):
        raise TypeError("use MemXArray.zeros()")

    @classmethod
    def zeros(cls, shape, dtype=np.float32, quota_bytes=None, runtime=None,
              read_mostly=False, context=None):
        shape, dtype, nbytes = _validate(shape, dtype)
        if quota_bytes is not None:
            quota_bytes = _integer(quota_bytes, "quota_bytes")
            if not 0 < quota_bytes <= (1 << 64) - 1:
                raise ValueError("quota_bytes must be in [1, 2**64 - 1]")
        if not isinstance(read_mostly, (bool, np.bool_)):
            raise TypeError("read_mostly must be bool")
        owned = context is None
        if context is not None:
            if not context.handle:
                raise ValueError("context is closed")
            if runtime is not None and runtime is not context.runtime:
                raise ValueError("runtime does not own the supplied context")
        else:
            runtime = Runtime() if runtime is None else runtime
            context = runtime.create_context("memx-array")
        allocation = None
        try:
            if quota_bytes is not None:
                context.set_quota(quota_bytes)
            flags = MEMX_TENSOR_FLAG_READ_MOSTLY if read_mostly else 0
            desc = tensor_desc(
                role=MEMX_TENSOR_ROLE_DATA,
                dtype=_NUMPY_TO_DTYPE[dtype],
                layout=MEMX_TENSOR_LAYOUT_ROW_MAJOR,
                flags=flags,
                shape=shape,
            )
            allocation = context.malloc_tensor(nbytes, desc, name="memx_array")
            ctypes.memset(allocation.ptr.value, 0, nbytes)
            self = object.__new__(cls)
            self._shape = shape
            self._dtype = dtype
            self._nbytes = nbytes
            self._flags = flags
            self._alloc = allocation
            self._ctx = context
            self._lock = threading.RLock()
            self._closed = False
            self._finalizer = weakref.finalize(self, _release, allocation, context, owned)
            return self
        except BaseException:
            if allocation is not None:
                allocation.free()
            if owned:
                context.destroy()
            raise

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        return self._nbytes

    @property
    def size(self):
        return self._nbytes // self._dtype.itemsize

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def closed(self):
        return self._closed

    def _check_live(self):
        if self._closed or not self._ctx.handle or not self._alloc.ptr:
            raise ValueError("MemXArray is closed or its context is unavailable")

    def materialize(self):
        """Copy into NumPy-owned memory without decompressing source pages in place."""
        with self._lock:
            self._check_live()
            result = np.empty(self.shape, dtype=self.dtype)
            self._ctx.materialize_range(
                self._alloc, 0, self.nbytes, result.ctypes.data, result.nbytes,
                flags=MEMX_MATERIALIZE_KEEP_COMPRESSED | MEMX_MATERIALIZE_ALLOW_RESIDENT,
            )
            return result

    def numpy(self):
        """Return an independent snapshot; writes to it do not update MemX."""
        return self.materialize()

    def copy(self):
        return self.materialize()

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("MemXArray cannot provide a zero-copy NumPy array")
        result = self.materialize()
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        return result

    def __getitem__(self, key):
        return self.materialize()[key]

    def _range(self, offset, length):
        self._check_live()
        offset = _integer(offset, "offset")
        if not 0 <= offset <= self.nbytes:
            raise ValueError("offset is outside the logical array bytes")
        length = self.nbytes - offset if length is None else _integer(length, "length")
        if not 0 <= length <= self.nbytes - offset:
            raise ValueError("length is outside the logical array bytes")
        return offset, length

    def _invalidate_snapshot_cache(self):
        self._ctx.begin_epoch(MEMX_EPOCH_FINAL, 0)
        self._ctx.end_epoch(0)

    def __setitem__(self, key, value):
        """Assign via a full snapshot; failed NumPy assignments leave storage unchanged."""
        with self._lock:
            result = self.materialize()
            result[key] = value
            self._check_live()
            ctypes.memmove(self._alloc.ptr.value, result.ctypes.data, self.nbytes)
            self._invalidate_snapshot_cache()

    def seal(self, offset=0, length=None):
        """Synchronously compress a byte range; native controls operate on whole pages."""
        with self._lock:
            offset, length = self._range(offset, length)
            if not length:
                return 0
            sealed = self._ctx.seal_range(self._alloc, offset, length)
            self._invalidate_snapshot_cache()
            return sealed

    def prefetch(self, offset=0, length=None):
        with self._lock:
            offset, length = self._range(offset, length)
            if length:
                self._ctx.prefetch_range(self._alloc, offset, length)

    def hot_region(self, offset=0, length=None):
        """Mark a byte range hot and prefetch it; retire() removes the hot mark."""
        with self._lock:
            offset, length = self._range(offset, length)
            if length:
                self._ctx.update_tensor_flags_range(
                    self._alloc, offset, length, self._flags | MEMX_TENSOR_FLAG_HOT,
                )
                self._ctx.prefetch_range(self._alloc, offset, length)

    def retire(self, offset=0, length=None):
        """Remove the hot mark and mark pages cold; use seal() for synchronous compression."""
        with self._lock:
            offset, length = self._range(offset, length)
            if length:
                self._ctx.update_tensor_flags_range(
                    self._alloc, offset, length, self._flags | MEMX_TENSOR_FLAG_COLD,
                )

    def info(self):
        with self._lock:
            self._check_live()
            return self._alloc.info()

    def compressed_pages(self):
        return int(self.info().compressed_pages)

    def stats(self):
        """Allocation-local counts; compressed bytes exclude shared runtime overhead."""
        info = self.info()
        return {
            "nbytes": self.nbytes,
            "page_count": int(info.page_count),
            "compressed_pages": int(info.compressed_pages),
            "compressed_bytes": int(info.compressed_bytes),
            "resident_pages": int(info.page_count - info.compressed_pages),
        }

    def close(self):
        with self._lock:
            if not self._closed:
                self._finalizer()
                self._closed = True

    free = close

    def __enter__(self):
        with self._lock:
            self._check_live()
            return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
