import ctypes
from pathlib import Path


MB = 1024 * 1024

MEMX_TENSOR_ROLE_UNKNOWN = 0
MEMX_TENSOR_ROLE_WEIGHT = 1
MEMX_TENSOR_ROLE_KV_CACHE = 2
MEMX_TENSOR_ROLE_ACTIVATION = 3
MEMX_TENSOR_ROLE_EMBEDDING = 4
MEMX_TENSOR_ROLE_TEMPORARY = 5

MEMX_TENSOR_DTYPE_UNKNOWN = 0
MEMX_TENSOR_DTYPE_FP16 = 1
MEMX_TENSOR_DTYPE_BF16 = 2
MEMX_TENSOR_DTYPE_FP32 = 3
MEMX_TENSOR_DTYPE_INT8 = 4
MEMX_TENSOR_DTYPE_UINT8 = 5
MEMX_TENSOR_DTYPE_INT32 = 6

MEMX_TENSOR_LAYOUT_UNKNOWN = 0
MEMX_TENSOR_LAYOUT_ROW_MAJOR = 1
MEMX_TENSOR_LAYOUT_COLUMN_MAJOR = 2
MEMX_TENSOR_LAYOUT_BLOCKED = 3
MEMX_TENSOR_LAYOUT_INTERLEAVED = 4

MEMX_TENSOR_FLAG_READ_MOSTLY = 1 << 0
MEMX_TENSOR_FLAG_SEQUENTIAL = 1 << 1
MEMX_TENSOR_FLAG_HOT = 1 << 2
MEMX_TENSOR_FLAG_NO_COMPRESS = 1 << 3
MEMX_TENSOR_FLAG_COLD = 1 << 4

MEMX_RUNTIME_CODEC_DEFAULT = 0
MEMX_RUNTIME_CODEC_TENSOR_FP16_SPLIT = 0x81
MEMX_RUNTIME_CODEC_TENSOR_BITPLANE16 = 0x82
MEMX_RUNTIME_CODEC_TENSOR_SPARSE_BYTE = 0x83
MEMX_RUNTIME_CODEC_TENSOR_FP16_DELTA_SPLIT = 0x84
MEMX_RUNTIME_CODEC_ZLIB = 0x85
MEMX_RUNTIME_CODEC_TENSOR_FP16_ZLIB_SPLIT = 0x86
MEMX_RUNTIME_CODEC_TENSOR_EXP_PACK = 0x87


ROLE_NAMES = {
    MEMX_TENSOR_ROLE_UNKNOWN: "unknown",
    MEMX_TENSOR_ROLE_WEIGHT: "weight",
    MEMX_TENSOR_ROLE_KV_CACHE: "kv_cache",
    MEMX_TENSOR_ROLE_ACTIVATION: "activation",
    MEMX_TENSOR_ROLE_EMBEDDING: "embedding",
    MEMX_TENSOR_ROLE_TEMPORARY: "temporary",
}


CODEC_NAMES = {
    MEMX_RUNTIME_CODEC_DEFAULT: "default",
    MEMX_RUNTIME_CODEC_TENSOR_FP16_SPLIT: "tensor_fp16_split",
    MEMX_RUNTIME_CODEC_TENSOR_BITPLANE16: "tensor_bitplane16",
    MEMX_RUNTIME_CODEC_TENSOR_SPARSE_BYTE: "tensor_sparse_byte",
    MEMX_RUNTIME_CODEC_TENSOR_FP16_DELTA_SPLIT: "tensor_fp16_delta_split",
    MEMX_RUNTIME_CODEC_ZLIB: "tensor_zlib",
    MEMX_RUNTIME_CODEC_TENSOR_FP16_ZLIB_SPLIT: "tensor_fp16_zlib_split",
    MEMX_RUNTIME_CODEC_TENSOR_EXP_PACK: "tensor_exp_pack",
}


class Stats(ctypes.Structure):
    _fields_ = [
        ("compressions", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("bytes_saved", ctypes.c_uint64),
        ("dedup_hits", ctypes.c_uint64),
        ("dedup_bytes_saved", ctypes.c_uint64),
        ("prefetch_count", ctypes.c_uint64),
        ("prefetch_hits", ctypes.c_uint64),
        ("virtual_bytes", ctypes.c_uint64),
        ("pool_used_bytes", ctypes.c_uint64),
        ("total_pages", ctypes.c_uint64),
        ("compressed_pages", ctypes.c_uint64),
        ("resident_pages", ctypes.c_uint64),
        ("pool_capacity_bytes", ctypes.c_uint64),
        ("pool_cursor_bytes", ctypes.c_uint64),
        ("pool_headroom_bytes", ctypes.c_uint64),
        ("free_pages", ctypes.c_uint64),
        ("pool_reclaim_bytes", ctypes.c_uint64),
        ("pool_reclaim_events", ctypes.c_uint64),
        ("tensor_codec_pages", ctypes.c_uint64),
        ("tensor_codec_bytes_saved", ctypes.c_uint64),
        ("tensor_split_pages", ctypes.c_uint64),
        ("tensor_split_bytes_saved", ctypes.c_uint64),
        ("tensor_bitplane_pages", ctypes.c_uint64),
        ("tensor_bitplane_bytes_saved", ctypes.c_uint64),
        ("tensor_sparse_pages", ctypes.c_uint64),
        ("tensor_sparse_bytes_saved", ctypes.c_uint64),
        ("weight_compressed_pages", ctypes.c_uint64),
        ("weight_bytes_saved", ctypes.c_uint64),
        ("kv_cache_compressed_pages", ctypes.c_uint64),
        ("kv_cache_bytes_saved", ctypes.c_uint64),
        ("hot_resident_pages", ctypes.c_uint64),
        ("hot_resident_bytes", ctypes.c_uint64),
        ("no_compress_resident_pages", ctypes.c_uint64),
        ("no_compress_resident_bytes", ctypes.c_uint64),
        ("pool_pressure_percent", ctypes.c_uint32),
        ("_reserved0", ctypes.c_uint32),
        ("running", ctypes.c_int),
        ("tensor_delta_split_pages", ctypes.c_uint64),
        ("tensor_delta_split_bytes_saved", ctypes.c_uint64),
        ("tensor_exp_pack_pages", ctypes.c_uint64),
        ("tensor_exp_pack_bytes_saved", ctypes.c_uint64),
    ]


class ContextStats(ctypes.Structure):
    _fields_ = [
        ("bytes_in_use", ctypes.c_uint64),
        ("peak_bytes_in_use", ctypes.c_uint64),
        ("allocations_live", ctypes.c_uint64),
        ("allocations_total", ctypes.c_uint64),
        ("quota_bytes", ctypes.c_uint64),
        ("allocation_failures_quota", ctypes.c_uint64),
        ("pressure_events", ctypes.c_uint64),
        ("tensor_bytes_in_use", ctypes.c_uint64),
        ("tensor_allocations_live", ctypes.c_uint64),
        ("weight_bytes_in_use", ctypes.c_uint64),
        ("kv_cache_bytes_in_use", ctypes.c_uint64),
        ("hot_bytes_in_use", ctypes.c_uint64),
        ("no_compress_bytes_in_use", ctypes.c_uint64),
    ]


class Pressure(ctypes.Structure):
    _fields_ = [
        ("virtual_capacity_bytes", ctypes.c_uint64),
        ("virtual_used_bytes", ctypes.c_uint64),
        ("virtual_free_bytes", ctypes.c_uint64),
        ("pool_capacity_bytes", ctypes.c_uint64),
        ("pool_cursor_bytes", ctypes.c_uint64),
        ("pool_used_bytes", ctypes.c_uint64),
        ("pool_headroom_bytes", ctypes.c_uint64),
        ("pool_free_extent_bytes", ctypes.c_uint64),
        ("pool_largest_free_extent_bytes", ctypes.c_uint64),
        ("pool_free_extent_count", ctypes.c_uint32),
        ("pool_fragmentation_percent", ctypes.c_uint32),
        ("free_pages", ctypes.c_uint64),
        ("pool_pressure_percent", ctypes.c_uint32),
        ("pool_near_full", ctypes.c_uint32),
    ]


class TensorDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("role", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("layout", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("rank", ctypes.c_uint32),
        ("shape", ctypes.c_uint64 * 4),
        ("stride", ctypes.c_uint64 * 4),
        ("layer_index", ctypes.c_uint32),
        ("head_index", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class AllocationInfo(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("page_count", ctypes.c_uint64),
        ("compressed_pages", ctypes.c_uint64),
        ("compressed_bytes", ctypes.c_uint64),
        ("tensor_role", ctypes.c_uint32),
        ("tensor_dtype", ctypes.c_uint32),
        ("tensor_layout", ctypes.c_uint32),
        ("tensor_flags", ctypes.c_uint32),
        ("primary_codec", ctypes.c_uint32),
        ("_reserved0", ctypes.c_uint32),
        ("tensor_codec_pages", ctypes.c_uint64),
        ("managed", ctypes.c_int),
    ]


class KVCacheWindow(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("_reserved0", ctypes.c_uint32),
        ("managed_offset", ctypes.c_size_t),
        ("managed_length", ctypes.c_size_t),
        ("hot_offset", ctypes.c_size_t),
        ("hot_length", ctypes.c_size_t),
        ("prefetch_offset", ctypes.c_size_t),
        ("prefetch_length", ctypes.c_size_t),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class WeightWindow(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("_reserved0", ctypes.c_uint32),
        ("managed_offset", ctypes.c_size_t),
        ("managed_length", ctypes.c_size_t),
        ("hot_offset", ctypes.c_size_t),
        ("hot_length", ctypes.c_size_t),
        ("prefetch_offset", ctypes.c_size_t),
        ("prefetch_length", ctypes.c_size_t),
        ("reserved", ctypes.c_uint64 * 4),
    ]

MEMX_EPOCH_LOAD = 1
MEMX_EPOCH_COMPRESS = 2
MEMX_EPOCH_INFER = 3
MEMX_EPOCH_FINAL = 4

MEMX_WS_FLAG_NONE = 0
MEMX_WS_FLAG_HOT = 1 << 0
MEMX_WS_FLAG_PREFETCH = 1 << 1
MEMX_WS_FLAG_RETIRE = 1 << 2
MEMX_WS_FLAG_RETIRE_SYNC = 1 << 3
MEMX_WS_FLAG_MARK_ACCESS = 1 << 4
MEMX_WS_FLAG_NO_ASYNC = 1 << 5
MEMX_WS_FLAG_EPHEMERAL = 1 << 6

MEMX_MATERIALIZE_KEEP_COMPRESSED = 1 << 0
MEMX_MATERIALIZE_ALLOW_RESIDENT = 1 << 1

class WSIntent(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ptr", ctypes.c_void_p),
        ("offset", ctypes.c_size_t),
        ("length", ctypes.c_size_t),
        ("prefetch_length", ctypes.c_size_t),
        ("priority", ctypes.c_uint32),
        ("_reserved0", ctypes.c_uint32),
    ]

class WSTile(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ptr", ctypes.c_void_p),
        ("rows", ctypes.c_size_t),
        ("cols", ctypes.c_size_t),
        ("elem_size", ctypes.c_size_t),
        ("col_start", ctypes.c_size_t),
        ("col_count", ctypes.c_size_t),
        ("prefetch_cols", ctypes.c_size_t),
        ("retire_col_start", ctypes.c_size_t),
        ("retire_col_count", ctypes.c_size_t),
    ]





def _default_library_path():
    return Path(__file__).resolve().parents[1] / "build" / "libmemx_runtime.dylib"


def tensor_desc(role, dtype, layout, flags, shape=(), stride=(), layer_index=0, head_index=0):
    desc = TensorDesc()
    desc.struct_size = ctypes.sizeof(TensorDesc)
    desc.role = role
    desc.dtype = dtype
    desc.layout = layout
    desc.flags = flags
    desc.rank = len(shape)
    for i, value in enumerate(shape[:4]):
        desc.shape[i] = value
    for i, value in enumerate(stride[:4]):
        desc.stride[i] = value
    desc.layer_index = layer_index
    desc.head_index = head_index
    return desc


def kv_cache_window(managed=(0, 0), hot=(0, 0), prefetch=(0, 0)):
    window = KVCacheWindow()
    window.struct_size = ctypes.sizeof(KVCacheWindow)
    window.managed_offset, window.managed_length = managed
    window.hot_offset, window.hot_length = hot
    window.prefetch_offset, window.prefetch_length = prefetch
    return window


def weight_window(managed=(0, 0), hot=(0, 0), prefetch=(0, 0)):
    window = WeightWindow()
    window.struct_size = ctypes.sizeof(WeightWindow)
    window.managed_offset, window.managed_length = managed
    window.hot_offset, window.hot_length = hot
    window.prefetch_offset, window.prefetch_length = prefetch
    return window

def ws_intent(ptr, offset, length, prefetch_length=0, flags=None, priority=0):
    it = WSIntent()
    it.struct_size = ctypes.sizeof(WSIntent)
    it.flags = MEMX_WS_FLAG_HOT | MEMX_WS_FLAG_PREFETCH if flags is None else int(flags)
    it.ptr = ptr
    it.offset = int(offset)
    it.length = int(length)
    it.prefetch_length = int(prefetch_length)
    it.priority = int(priority)
    it._reserved0 = 0
    return it



def kv_sliding_window(token_bytes, hot_tokens, prefetch_tokens=0, total_bytes=None, start_token=0):
    hot_bytes = hot_tokens * token_bytes
    prefetch_bytes = prefetch_tokens * token_bytes
    start = start_token * token_bytes
    hot_off = start
    hot_len = hot_bytes
    if total_bytes is not None:
        if hot_off > total_bytes:
            hot_off = total_bytes
            hot_len = 0
        elif hot_off + hot_len > total_bytes:
            hot_len = total_bytes - hot_off
    managed_off = 0
    managed_len = hot_off
    pref_off = hot_off + hot_len
    pref_len = prefetch_bytes
    if total_bytes is not None:
        if pref_off > total_bytes:
            pref_off = total_bytes
            pref_len = 0
        elif pref_off + pref_len > total_bytes:
            pref_len = total_bytes - pref_off
    return kv_cache_window(
        managed=(managed_off, managed_len),
        hot=(hot_off, hot_len),
        prefetch=(pref_off, pref_len),
    )


def role_name(role):
    return ROLE_NAMES.get(role, f"role_{role}")


def codec_name(codec):
    return CODEC_NAMES.get(codec, f"codec_0x{codec:x}")


def _empty_ledger_bucket():
    return {
        "logical_bytes": 0,
        "compressed_pages": 0,
        "compressed_bytes": 0,
        "resident_pages": 0,
        "resident_bytes": 0,
        "tensor_codec_pages": 0,
        "allocations": 0,
    }


def capacity_ledger(allocations, runtime_stats=None):
    items = []
    by_role = {}
    by_codec = {}
    logical_bytes = 0
    compressed_bytes = 0
    resident_bytes = 0
    compressed_pages = 0
    resident_pages = 0
    tensor_codec_pages = 0

    for allocation in allocations:
        info = allocation.info()
        item_resident_pages = info.page_count - info.compressed_pages
        item_resident_bytes = item_resident_pages * 16384
        item_stored_bytes = info.compressed_bytes + item_resident_bytes
        item_ratio = (info.size / item_stored_bytes) if item_stored_bytes else 0.0
        item = {
            "name": getattr(allocation, "name", ""),
            "logical_bytes": info.size,
            "page_count": info.page_count,
            "compressed_pages": info.compressed_pages,
            "compressed_bytes": info.compressed_bytes,
            "resident_pages": item_resident_pages,
            "resident_bytes": item_resident_bytes,
            "tensor_codec_pages": info.tensor_codec_pages,
            "role": info.tensor_role,
            "role_name": role_name(info.tensor_role),
            "dtype": info.tensor_dtype,
            "layout": info.tensor_layout,
            "flags": info.tensor_flags,
            "primary_codec": info.primary_codec,
            "codec_name": codec_name(info.primary_codec),
            "stored_bytes_before_dedup": item_stored_bytes,
            "effective_ratio_before_dedup": item_ratio,
        }
        items.append(item)

        logical_bytes += info.size
        compressed_bytes += info.compressed_bytes
        resident_bytes += item_resident_bytes
        compressed_pages += info.compressed_pages
        resident_pages += item_resident_pages
        tensor_codec_pages += info.tensor_codec_pages

        role_bucket = by_role.setdefault(item["role_name"], _empty_ledger_bucket())
        role_bucket["logical_bytes"] += info.size
        role_bucket["compressed_pages"] += info.compressed_pages
        role_bucket["compressed_bytes"] += info.compressed_bytes
        role_bucket["resident_pages"] += item_resident_pages
        role_bucket["resident_bytes"] += item_resident_bytes
        role_bucket["tensor_codec_pages"] += info.tensor_codec_pages
        role_bucket["allocations"] += 1

        codec_bucket = by_codec.setdefault(item["codec_name"], _empty_ledger_bucket())
        codec_bucket["logical_bytes"] += info.size
        codec_bucket["compressed_pages"] += info.compressed_pages
        codec_bucket["compressed_bytes"] += info.compressed_bytes
        codec_bucket["resident_pages"] += item_resident_pages
        codec_bucket["resident_bytes"] += item_resident_bytes
        codec_bucket["tensor_codec_pages"] += info.tensor_codec_pages
        codec_bucket["allocations"] += 1

    stored_before_dedup = compressed_bytes + resident_bytes
    pool_used = runtime_stats.pool_used_bytes if runtime_stats is not None else compressed_bytes
    physical_estimate = pool_used + resident_bytes
    return {
        "items": items,
        "by_role": by_role,
        "by_codec": by_codec,
        "logical_bytes": logical_bytes,
        "compressed_pages": compressed_pages,
        "compressed_bytes": compressed_bytes,
        "resident_pages": resident_pages,
        "resident_bytes": resident_bytes,
        "tensor_codec_pages": tensor_codec_pages,
        "stored_bytes_before_dedup": stored_before_dedup,
        "pool_used_bytes": pool_used,
        "physical_estimate_bytes": physical_estimate,
        "effective_ratio_before_dedup": (logical_bytes / stored_before_dedup) if stored_before_dedup else 0.0,
        "effective_ratio_physical_est": (logical_bytes / physical_estimate) if physical_estimate else 0.0,
        "dedup_hits": runtime_stats.dedup_hits if runtime_stats is not None else 0,
        "dedup_bytes_saved": runtime_stats.dedup_bytes_saved if runtime_stats is not None else 0,
        "hot_resident_pages": runtime_stats.hot_resident_pages if runtime_stats is not None else 0,
        "hot_resident_bytes": runtime_stats.hot_resident_bytes if runtime_stats is not None else 0,
        "no_compress_resident_pages": runtime_stats.no_compress_resident_pages if runtime_stats is not None else 0,
        "no_compress_resident_bytes": runtime_stats.no_compress_resident_bytes if runtime_stats is not None else 0,
    }


def capacity_projection(ledger, physical_memory_bytes, usable_fraction=0.80):
    ratio = ledger.get("effective_ratio_physical_est", 0.0)
    usable_physical_bytes = int(physical_memory_bytes * usable_fraction)
    projected_logical_bytes = int(usable_physical_bytes * ratio)
    return {
        "physical_memory_bytes": physical_memory_bytes,
        "usable_fraction": usable_fraction,
        "usable_physical_bytes": usable_physical_bytes,
        "effective_ratio_physical_est": ratio,
        "projected_logical_bytes": projected_logical_bytes,
        "projected_logical_gb": projected_logical_bytes / (1024 ** 3),
        "meets_2x": projected_logical_bytes >= physical_memory_bytes * 2,
        "meets_4x": projected_logical_bytes >= physical_memory_bytes * 4,
    }


class Runtime:
    def __init__(self, path=None):
        self.path = Path(path) if path else _default_library_path()
        self.lib = ctypes.CDLL(str(self.path))
        self._bind()

    def _bind(self):
        lib = self.lib
        lib.memx_runtime_context_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.memx_runtime_context_create.restype = ctypes.c_int
        lib.memx_runtime_context_destroy.argtypes = [ctypes.c_void_p]
        lib.memx_runtime_context_destroy.restype = ctypes.c_int
        lib.memx_runtime_context_set_quota.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.memx_runtime_context_set_quota.restype = ctypes.c_int
        lib.memx_runtime_context_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ContextStats)]
        lib.memx_runtime_context_get_stats.restype = ctypes.c_int
        lib.memx_runtime_context_malloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.memx_runtime_context_malloc.restype = ctypes.c_void_p
        lib.memx_runtime_context_malloc_tensor.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(TensorDesc)]
        lib.memx_runtime_context_malloc_tensor.restype = ctypes.c_void_p
        lib.memx_runtime_context_free.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.memx_runtime_context_free.restype = None
        lib.memx_runtime_get_allocation_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(AllocationInfo)]
        lib.memx_runtime_get_allocation_info.restype = ctypes.c_int
        lib.memx_runtime_get_allocation_info_range.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(AllocationInfo)]
        lib.memx_runtime_get_allocation_info_range.restype = ctypes.c_int
        lib.memx_runtime_context_update_tensor_flags.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        lib.memx_runtime_context_update_tensor_flags.restype = ctypes.c_int
        lib.memx_runtime_context_update_tensor_flags_range.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint32]
        lib.memx_runtime_context_update_tensor_flags_range.restype = ctypes.c_int
        lib.memx_runtime_context_prefetch_range.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        lib.memx_runtime_context_prefetch_range.restype = ctypes.c_int
        lib.memx_runtime_context_update_kv_cache_window.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(KVCacheWindow)]
        lib.memx_runtime_context_update_kv_cache_window.restype = ctypes.c_int
        lib.memx_runtime_context_update_weight_window.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(WeightWindow)]
        lib.memx_runtime_context_update_weight_window.restype = ctypes.c_int
        lib.memx_runtime_context_force_compress_range.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
        lib.memx_runtime_context_force_compress_range.restype = ctypes.c_int
        lib.memx_runtime_context_seal_range.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
        lib.memx_runtime_context_seal_range.restype = ctypes.c_int
        lib.memx_runtime_context_seal_range_async.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        lib.memx_runtime_context_seal_range_async.restype = ctypes.c_int
        lib.memx_runtime_seal_flush.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        lib.memx_runtime_seal_flush.restype = ctypes.c_int
        lib.memx_runtime_context_begin_epoch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64]
        lib.memx_runtime_context_begin_epoch.restype = ctypes.c_int
        lib.memx_runtime_context_apply_ws.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.memx_runtime_context_apply_ws.restype = ctypes.c_int
        lib.memx_runtime_context_ws_advance.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint32]
        lib.memx_runtime_context_ws_advance.restype = ctypes.c_int
        if hasattr(lib, "memx_runtime_context_export_archive"):
            lib.memx_runtime_context_export_archive.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
            lib.memx_runtime_context_export_archive.restype = ctypes.c_int
            lib.memx_runtime_context_import_archive.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
            lib.memx_runtime_context_import_archive.restype = ctypes.c_int
            lib.memx_runtime_context_ws_tile.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            lib.memx_runtime_context_ws_tile.restype = ctypes.c_int
            if hasattr(lib, "memx_runtime_context_materialize_range"):
                lib.memx_runtime_context_materialize_range.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32,
                ]
                lib.memx_runtime_context_materialize_range.restype = ctypes.c_int
                lib.memx_runtime_context_materialize_tile.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint32,
                ]
                lib.memx_runtime_context_materialize_tile.restype = ctypes.c_int
        lib.memx_runtime_context_ws_close.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        lib.memx_runtime_context_ws_close.restype = ctypes.c_int
        lib.memx_runtime_context_end_epoch.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.memx_runtime_context_end_epoch.restype = ctypes.c_int
        lib.memx_runtime_context_purge.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.memx_runtime_context_purge.restype = ctypes.c_int
        lib.memx_runtime_context_mark_access_range.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        lib.memx_runtime_context_mark_access_range.restype = ctypes.c_int
        lib.memx_runtime_get_stats.argtypes = [ctypes.POINTER(Stats)]
        lib.memx_runtime_get_stats.restype = ctypes.c_int
        lib.memx_runtime_get_pressure.argtypes = [ctypes.POINTER(Pressure)]
        lib.memx_runtime_get_pressure.restype = ctypes.c_int
        lib.memx_runtime_reclaim.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        lib.memx_runtime_reclaim.restype = ctypes.c_int
        lib.memx_runtime_compact.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        lib.memx_runtime_compact.restype = ctypes.c_int
        lib.memx_runtime_test_set_pool_cursor.argtypes = [ctypes.c_size_t]
        lib.memx_runtime_test_set_pool_cursor.restype = ctypes.c_int
        lib.memx_runtime_shutdown.argtypes = []
        lib.memx_runtime_shutdown.restype = None

    def seal_flush(self):
        out = ctypes.c_uint64(0)
        rc = self.lib.memx_runtime_seal_flush(ctypes.byref(out))
        if rc != 0:
            raise OSError(rc, "memx_runtime_seal_flush failed")
        return out.value

    def create_context(self, name):
        ctx = ctypes.c_void_p()
        rc = self.lib.memx_runtime_context_create(name.encode(), ctypes.byref(ctx))
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_create failed")
        return Context(self, ctx)

    def stats(self):
        out = Stats()
        rc = self.lib.memx_runtime_get_stats(ctypes.byref(out))
        if rc != 0:
            raise OSError(rc, "memx_runtime_get_stats failed")
        return out

    def pressure(self):
        out = Pressure()
        rc = self.lib.memx_runtime_get_pressure(ctypes.byref(out))
        if rc != 0:
            raise OSError(rc, "memx_runtime_get_pressure failed")
        return out

    def reclaim(self):
        reclaimed = ctypes.c_uint64()
        rc = self.lib.memx_runtime_reclaim(ctypes.byref(reclaimed))
        if rc != 0:
            raise OSError(rc, "memx_runtime_reclaim failed")
        return reclaimed.value

    def compact(self):
        reclaimed = ctypes.c_uint64(0)
        rc = self.lib.memx_runtime_compact(ctypes.byref(reclaimed))
        if rc != 0:
            raise OSError(rc, "memx_runtime_compact failed")
        return int(reclaimed.value)

    def test_set_pool_cursor(self, cursor_bytes):
        rc = self.lib.memx_runtime_test_set_pool_cursor(cursor_bytes)
        if rc != 0:
            raise OSError(rc, "memx_runtime_test_set_pool_cursor failed")

    def shutdown(self):
        self.lib.memx_runtime_shutdown()


class Context:
    def __init__(self, runtime, handle):
        self.runtime = runtime
        self.handle = handle

    def set_quota(self, bytes_):
        rc = self.runtime.lib.memx_runtime_context_set_quota(self.handle, bytes_)
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_set_quota failed")

    def stats(self):
        out = ContextStats()
        rc = self.runtime.lib.memx_runtime_context_get_stats(self.handle, ctypes.byref(out))
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_get_stats failed")
        return out

    def malloc_tensor(self, size, desc, name=""):
        ptr = self.runtime.lib.memx_runtime_context_malloc_tensor(self.handle, size, ctypes.byref(desc))
        if not ptr:
            raise MemoryError("memx_runtime_context_malloc_tensor failed")
        return Allocation(self, ctypes.c_void_p(ptr), size, name)

    def malloc(self, size, name=""):
        ptr = self.runtime.lib.memx_runtime_context_malloc(self.handle, size)
        if not ptr:
            raise MemoryError("memx_runtime_context_malloc failed")
        return Allocation(self, ctypes.c_void_p(ptr), size, name)

    def update_kv_cache_window(self, allocation, window):
        rc = self.runtime.lib.memx_runtime_context_update_kv_cache_window(self.handle, allocation.ptr, ctypes.byref(window))
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_update_kv_cache_window failed")

    def advance_kv_window(self, allocation, token_bytes, written_tokens, hot_tokens, prefetch_tokens=1):
        total = getattr(allocation, "size", None)
        start_token = 0
        if written_tokens > hot_tokens:
            start_token = written_tokens - hot_tokens
            hot = hot_tokens
        else:
            hot = written_tokens
        window = kv_sliding_window(
            token_bytes=token_bytes,
            hot_tokens=hot,
            prefetch_tokens=prefetch_tokens,
            total_bytes=total,
            start_token=start_token,
        )
        self.update_kv_cache_window(allocation, window)
        return window

    def update_weight_window(self, allocation, window):
        rc = self.runtime.lib.memx_runtime_context_update_weight_window(self.handle, allocation.ptr, ctypes.byref(window))
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_update_weight_window failed")

    def force_compress_range(self, allocation, offset=0, length=None):
        if length is None:
            length = allocation.size
        out = ctypes.c_uint64(0)
        rc = self.runtime.lib.memx_runtime_context_force_compress_range(
            self.handle, allocation.ptr, offset, length, ctypes.byref(out)
        )
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_force_compress_range failed")
        return out.value

    def seal_range(self, allocation, offset=0, length=None):
        if length is None:
            length = allocation.size
        out = ctypes.c_uint64(0)
        rc = self.runtime.lib.memx_runtime_context_seal_range(
            self.handle, allocation.ptr, offset, length, ctypes.byref(out)
        )
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_seal_range failed")
        return out.value

    def seal_range_async(self, allocation, offset=0, length=None):
        if length is None:
            length = allocation.size
        rc = self.runtime.lib.memx_runtime_context_seal_range_async(
            self.handle, allocation.ptr, offset, length
        )
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_seal_range_async failed")
        return 0

    def seal_flush(self):
        return self.runtime.seal_flush()

    def begin_epoch(self, phase, hot_budget_bytes=0):
        rc = self.runtime.lib.memx_runtime_context_begin_epoch(self.handle, int(phase), int(hot_budget_bytes))
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_begin_epoch failed")

    def apply_ws(self, intents):
        if not intents:
            return
        arr = (WSIntent * len(intents))()
        for i, it in enumerate(intents):
            arr[i] = it
        rc = self.runtime.lib.memx_runtime_context_apply_ws(self.handle, ctypes.byref(arr), len(intents))
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_apply_ws failed")

    def ws_advance(self, allocation, offset, length, prefetch_length=0, flags=None):
        if flags is None:
            flags = MEMX_WS_FLAG_HOT | MEMX_WS_FLAG_PREFETCH | MEMX_WS_FLAG_MARK_ACCESS
        rc = self.runtime.lib.memx_runtime_context_ws_advance(
            self.handle, allocation.ptr, int(offset), int(length), int(prefetch_length), int(flags)
        )
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_ws_advance failed")

    def ws_close(self, allocation, flags=None):
        if flags is None:
            flags = MEMX_WS_FLAG_RETIRE
        rc = self.runtime.lib.memx_runtime_context_ws_close(self.handle, allocation.ptr, int(flags))
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_ws_close failed")

    def end_epoch(self, seal_tracked=0):
        rc = self.runtime.lib.memx_runtime_context_end_epoch(self.handle, int(1 if seal_tracked else 0))
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_end_epoch failed")

    def purge(self, allocation):
        rc = self.runtime.lib.memx_runtime_context_purge(self.handle, allocation.ptr)
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_purge failed")

    def update_tensor_flags(self, allocation, flags):
        rc = self.runtime.lib.memx_runtime_context_update_tensor_flags(self.handle, allocation.ptr, flags)
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_update_tensor_flags failed")

    def update_tensor_flags_range(self, allocation, offset, length, flags):
        rc = self.runtime.lib.memx_runtime_context_update_tensor_flags_range(self.handle, allocation.ptr, offset, length, flags)
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_update_tensor_flags_range failed")

    def prefetch_range(self, allocation, offset, length):
        rc = self.runtime.lib.memx_runtime_context_prefetch_range(self.handle, allocation.ptr, offset, length)
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_prefetch_range failed")

    def mark_access_range(self, allocation, offset, length):
        rc = self.runtime.lib.memx_runtime_context_mark_access_range(self.handle, allocation.ptr, offset, length)
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_mark_access_range failed")



    def materialize_range(self, allocation, offset, length, dst_ptr, dst_cap, flags=None):
        if not hasattr(self.runtime.lib, "memx_runtime_context_materialize_range"):
            raise OSError("materialize_range unavailable")
        if flags is None:
            flags = MEMX_MATERIALIZE_KEEP_COMPRESSED | MEMX_MATERIALIZE_ALLOW_RESIDENT
        rc = self.runtime.lib.memx_runtime_context_materialize_range(
            self.handle, allocation.ptr, int(offset), int(length),
            ctypes.c_void_p(int(dst_ptr)), int(dst_cap), int(flags),
        )
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_materialize_range failed")

    def materialize_tile(self, allocation, rows, cols, elem_size, col_start, col_count, dst_ptr, dst_cap, dst_row_stride=0, flags=None):
        if not hasattr(self.runtime.lib, "memx_runtime_context_materialize_tile"):
            raise OSError("materialize_tile unavailable")
        if flags is None:
            flags = MEMX_MATERIALIZE_KEEP_COMPRESSED | MEMX_MATERIALIZE_ALLOW_RESIDENT
        tile = WSTile()
        tile.struct_size = ctypes.sizeof(WSTile)
        tile.flags = 0
        tile.ptr = allocation.ptr
        tile.rows = int(rows)
        tile.cols = int(cols)
        tile.elem_size = int(elem_size)
        tile.col_start = int(col_start)
        tile.col_count = int(col_count)
        tile.prefetch_cols = 0
        tile.retire_col_start = 0
        tile.retire_col_count = 0
        rc = self.runtime.lib.memx_runtime_context_materialize_tile(
            self.handle, ctypes.byref(tile), ctypes.c_void_p(int(dst_ptr)), int(dst_cap), int(dst_row_stride), int(flags),
        )
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_materialize_tile failed")

    def export_archive(self, allocation, path):
        if not hasattr(self.runtime.lib, "memx_runtime_context_export_archive"):
            raise OSError("export_archive unavailable")
        out = ctypes.c_uint64(0)
        rc = self.runtime.lib.memx_runtime_context_export_archive(
            self.handle, allocation.ptr, path.encode("utf-8"), ctypes.byref(out)
        )
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_export_archive failed")
        return int(out.value)

    def import_archive(self, path, desc=None, name=""):
        if not hasattr(self.runtime.lib, "memx_runtime_context_import_archive"):
            raise OSError("import_archive unavailable")
        out_ptr = ctypes.c_void_p()
        out_size = ctypes.c_size_t(0)
        desc_arg = ctypes.c_void_p()
        desc_ref = None
        if desc is not None:
            if getattr(desc, "struct_size", 0) == 0:
                desc.struct_size = ctypes.sizeof(desc)
            desc_ref = desc
            desc_arg = ctypes.byref(desc)
        rc = self.runtime.lib.memx_runtime_context_import_archive(
            self.handle,
            path.encode("utf-8"),
            desc_arg if desc is not None else None,
            ctypes.byref(out_ptr),
            ctypes.byref(out_size),
        )
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_import_archive failed")
        return Allocation(self, out_ptr, int(out_size.value), name=name)

    def ws_tile(self, allocation, rows, cols, elem_size, col_start, col_count, prefetch_cols=0, retire_col_start=0, retire_col_count=0, flags=None):
        if not hasattr(self.runtime.lib, "memx_runtime_context_ws_tile"):
            raise OSError("ws_tile unavailable")
        if flags is None:
            flags = MEMX_WS_FLAG_HOT | MEMX_WS_FLAG_PREFETCH | MEMX_WS_FLAG_MARK_ACCESS
        tile = WSTile()
        tile.struct_size = ctypes.sizeof(WSTile)
        tile.flags = int(flags)
        tile.ptr = allocation.ptr
        tile.rows = int(rows)
        tile.cols = int(cols)
        tile.elem_size = int(elem_size)
        tile.col_start = int(col_start)
        tile.col_count = int(col_count)
        tile.prefetch_cols = int(prefetch_cols)
        tile.retire_col_start = int(retire_col_start)
        tile.retire_col_count = int(retire_col_count)
        rc = self.runtime.lib.memx_runtime_context_ws_tile(self.handle, ctypes.byref(tile))
        if rc != 0:
            raise OSError(rc, "memx_runtime_context_ws_tile failed")

    def destroy(self):
        if self.handle:
            rc = self.runtime.lib.memx_runtime_context_destroy(self.handle)
            if rc != 0:
                raise OSError(rc, "memx_runtime_context_destroy failed")
            self.handle = None


class Allocation:
    def __init__(self, context, ptr, size, name=""):
        self.context = context
        self.ptr = ptr
        self.size = size
        self.name = name

    def buffer(self):
        return (ctypes.c_uint8 * self.size).from_address(self.ptr.value)

    def torch_tensor(self, dtype, shape, stride=None):
        import torch
        import ctypes

        elements = 1
        for dim in shape:
            elements *= dim
        itemsize = torch.tensor([], dtype=dtype).element_size()
        nbytes = elements * itemsize
        if nbytes > self.size:
            raise ValueError("torch_tensor size exceeds allocation")
        buf = (ctypes.c_uint8 * nbytes).from_address(self.ptr.value)
        tensor = torch.frombuffer(buf, dtype=dtype)
        if stride is not None:
            return torch.as_strided(tensor, tuple(shape), tuple(stride))
        return tensor.reshape(tuple(shape))

    def info(self, offset=0, length=None):
        out = AllocationInfo()
        if length is None:
            rc = self.context.runtime.lib.memx_runtime_get_allocation_info(self.ptr, ctypes.byref(out))
        else:
            rc = self.context.runtime.lib.memx_runtime_get_allocation_info_range(self.ptr, offset, length, ctypes.byref(out))
        if rc != 0:
            raise OSError(rc, "memx_runtime_get_allocation_info failed")
        return out

    def free(self):
        if self.ptr:
            self.context.runtime.lib.memx_runtime_context_free(self.context.handle, self.ptr)
            self.ptr = None
