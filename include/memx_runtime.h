#ifndef MEMX_RUNTIME_H
#define MEMX_RUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct memx_runtime_stats {
    uint64_t compressions;
    uint64_t faults;
    uint64_t bytes_saved;
    uint64_t dedup_hits;
    uint64_t dedup_bytes_saved;
    uint64_t prefetch_count;
    uint64_t prefetch_hits;
    uint64_t virtual_bytes;
    uint64_t pool_used_bytes;
    uint64_t total_pages;
    uint64_t compressed_pages;
    uint64_t resident_pages;
    uint64_t pool_capacity_bytes;
    uint64_t pool_cursor_bytes;
    uint64_t pool_headroom_bytes;
    uint64_t free_pages;
    uint64_t pool_reclaim_bytes;
    uint64_t pool_reclaim_events;
    uint64_t tensor_codec_pages;
    uint64_t tensor_codec_bytes_saved;
    uint64_t tensor_split_pages;
    uint64_t tensor_split_bytes_saved;
    uint64_t tensor_bitplane_pages;
    uint64_t tensor_bitplane_bytes_saved;
    uint64_t tensor_sparse_pages;
    uint64_t tensor_sparse_bytes_saved;
    uint64_t weight_compressed_pages;
    uint64_t weight_bytes_saved;
    uint64_t kv_cache_compressed_pages;
    uint64_t kv_cache_bytes_saved;
    uint64_t hot_resident_pages;
    uint64_t hot_resident_bytes;
    uint64_t no_compress_resident_pages;
    uint64_t no_compress_resident_bytes;
    uint32_t pool_pressure_percent;
    uint32_t _reserved0;
    int running;
    uint64_t tensor_delta_split_pages;
    uint64_t tensor_delta_split_bytes_saved;
    uint64_t tensor_exp_pack_pages;
    uint64_t tensor_exp_pack_bytes_saved;
} memx_runtime_stats_t;

typedef struct memx_runtime_context memx_runtime_context_t;

typedef struct memx_runtime_context_stats {
    uint64_t bytes_in_use;
    uint64_t peak_bytes_in_use;
    uint64_t allocations_live;
    uint64_t allocations_total;
    uint64_t quota_bytes;
    uint64_t allocation_failures_quota;
    uint64_t pressure_events;
    uint64_t tensor_bytes_in_use;
    uint64_t tensor_allocations_live;
    uint64_t weight_bytes_in_use;
    uint64_t kv_cache_bytes_in_use;
    uint64_t hot_bytes_in_use;
    uint64_t no_compress_bytes_in_use;
} memx_runtime_context_stats_t;

typedef struct memx_runtime_pressure {
    uint64_t virtual_capacity_bytes;
    uint64_t virtual_used_bytes;
    uint64_t virtual_free_bytes;
    uint64_t pool_capacity_bytes;
    uint64_t pool_cursor_bytes;
    uint64_t pool_used_bytes;
    uint64_t pool_headroom_bytes;
    uint64_t pool_free_extent_bytes;
    uint64_t pool_largest_free_extent_bytes;
    uint32_t pool_free_extent_count;
    uint32_t pool_fragmentation_percent;
    uint64_t free_pages;
    uint32_t pool_pressure_percent;
    uint32_t pool_near_full;
} memx_runtime_pressure_t;

typedef enum memx_runtime_tensor_role {
    MEMX_TENSOR_ROLE_UNKNOWN = 0,
    MEMX_TENSOR_ROLE_WEIGHT = 1,
    MEMX_TENSOR_ROLE_KV_CACHE = 2,
    MEMX_TENSOR_ROLE_ACTIVATION = 3,
    MEMX_TENSOR_ROLE_EMBEDDING = 4,
    MEMX_TENSOR_ROLE_TEMPORARY = 5
} memx_runtime_tensor_role_t;

typedef enum memx_runtime_tensor_dtype {
    MEMX_TENSOR_DTYPE_UNKNOWN = 0,
    MEMX_TENSOR_DTYPE_FP16 = 1,
    MEMX_TENSOR_DTYPE_BF16 = 2,
    MEMX_TENSOR_DTYPE_FP32 = 3,
    MEMX_TENSOR_DTYPE_INT8 = 4,
    MEMX_TENSOR_DTYPE_UINT8 = 5,
    MEMX_TENSOR_DTYPE_INT32 = 6
} memx_runtime_tensor_dtype_t;

typedef enum memx_runtime_tensor_layout {
    MEMX_TENSOR_LAYOUT_UNKNOWN = 0,
    MEMX_TENSOR_LAYOUT_ROW_MAJOR = 1,
    MEMX_TENSOR_LAYOUT_COLUMN_MAJOR = 2,
    MEMX_TENSOR_LAYOUT_BLOCKED = 3,
    MEMX_TENSOR_LAYOUT_INTERLEAVED = 4
} memx_runtime_tensor_layout_t;

enum {
    MEMX_TENSOR_FLAG_READ_MOSTLY = 1u << 0,
    MEMX_TENSOR_FLAG_SEQUENTIAL = 1u << 1,
    MEMX_TENSOR_FLAG_HOT = 1u << 2,
    MEMX_TENSOR_FLAG_NO_COMPRESS = 1u << 3,
    MEMX_TENSOR_FLAG_COLD = 1u << 4
};

enum {
    MEMX_RUNTIME_CODEC_DEFAULT = 0,
    MEMX_RUNTIME_CODEC_TENSOR_FP16_SPLIT = 0x81,
    MEMX_RUNTIME_CODEC_TENSOR_BITPLANE16 = 0x82,
    MEMX_RUNTIME_CODEC_TENSOR_SPARSE_BYTE = 0x83,
    MEMX_RUNTIME_CODEC_TENSOR_FP16_DELTA_SPLIT = 0x84,
    MEMX_RUNTIME_CODEC_ZLIB = 0x85,
    MEMX_RUNTIME_CODEC_TENSOR_FP16_ZLIB_SPLIT = 0x86,
    MEMX_RUNTIME_CODEC_TENSOR_EXP_PACK = 0x87
};

typedef struct memx_runtime_tensor_desc {
    uint32_t struct_size;
    uint32_t role;
    uint32_t dtype;
    uint32_t layout;
    uint32_t flags;
    uint32_t rank;
    uint64_t shape[4];
    uint64_t stride[4];
    uint32_t layer_index;
    uint32_t head_index;
    uint64_t reserved[4];
} memx_runtime_tensor_desc_t;

typedef struct memx_runtime_allocation_info {
    size_t size;
    uint64_t page_count;
    uint64_t compressed_pages;
    uint64_t compressed_bytes;
    uint32_t tensor_role;
    uint32_t tensor_dtype;
    uint32_t tensor_layout;
    uint32_t tensor_flags;
    uint32_t primary_codec;
    uint32_t _reserved0;
    uint64_t tensor_codec_pages;
    int managed;
} memx_runtime_allocation_info_t;

typedef struct memx_runtime_kv_cache_window {
    uint32_t struct_size;
    uint32_t _reserved0;
    size_t managed_offset;
    size_t managed_length;
    size_t hot_offset;
    size_t hot_length;
    size_t prefetch_offset;
    size_t prefetch_length;
    uint64_t reserved[4];
} memx_runtime_kv_cache_window_t;

typedef struct memx_runtime_weight_window {
    uint32_t struct_size;
    uint32_t _reserved0;
    size_t managed_offset;
    size_t managed_length;
    size_t hot_offset;
    size_t hot_length;
    size_t prefetch_offset;
    size_t prefetch_length;
    uint64_t reserved[4];
} memx_runtime_weight_window_t;

int memx_runtime_init(void);
void memx_runtime_shutdown(void);
int memx_runtime_is_active(void);
int memx_runtime_owns_pointer(const void *ptr);
int memx_runtime_get_stats(memx_runtime_stats_t *out_stats);
int memx_runtime_get_pressure(memx_runtime_pressure_t *out_pressure);
int memx_runtime_reclaim(uint64_t *out_reclaimed_bytes);
int memx_runtime_compact(uint64_t *out_reclaimed_bytes);
int memx_runtime_test_set_pool_cursor(size_t cursor_bytes);
int memx_runtime_get_allocation_info(const void *ptr, memx_runtime_allocation_info_t *out_info);
int memx_runtime_get_allocation_info_range(const void *ptr, size_t offset, size_t length, memx_runtime_allocation_info_t *out_info);
int memx_runtime_prefetch_range(const void *ptr, size_t offset, size_t length);
int memx_runtime_mark_access_range(const void *ptr, size_t offset, size_t length);

int memx_runtime_context_create(const char *name, memx_runtime_context_t **out_ctx);
int memx_runtime_context_destroy(memx_runtime_context_t *ctx);
int memx_runtime_context_get_stats(const memx_runtime_context_t *ctx, memx_runtime_context_stats_t *out_stats);
int memx_runtime_context_set_quota(memx_runtime_context_t *ctx, uint64_t quota_bytes);
int memx_runtime_context_get_quota(const memx_runtime_context_t *ctx, uint64_t *out_quota_bytes);

void *memx_runtime_malloc(size_t size);
void memx_runtime_free(void *ptr);
void *memx_runtime_calloc(size_t nmemb, size_t size);
void *memx_runtime_realloc(void *ptr, size_t size);
int memx_runtime_posix_memalign(void **memptr, size_t alignment, size_t size);
void *memx_runtime_aligned_alloc(size_t alignment, size_t size);
void *memx_runtime_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int memx_runtime_munmap(void *addr, size_t length);

void *memx_runtime_context_malloc(memx_runtime_context_t *ctx, size_t size);
void memx_runtime_context_free(memx_runtime_context_t *ctx, void *ptr);
void *memx_runtime_context_calloc(memx_runtime_context_t *ctx, size_t nmemb, size_t size);
void *memx_runtime_context_realloc(memx_runtime_context_t *ctx, void *ptr, size_t size);
void *memx_runtime_context_malloc_tensor(memx_runtime_context_t *ctx, size_t size, const memx_runtime_tensor_desc_t *desc);
int memx_runtime_context_update_tensor_flags(memx_runtime_context_t *ctx, void *ptr, uint32_t flags);
int memx_runtime_context_update_tensor_flags_range(memx_runtime_context_t *ctx, void *ptr, size_t offset, size_t length, uint32_t flags);
int memx_runtime_context_prefetch_range(memx_runtime_context_t *ctx, const void *ptr, size_t offset, size_t length);
int memx_runtime_context_mark_access_range(memx_runtime_context_t *ctx, const void *ptr, size_t offset, size_t length);
int memx_runtime_context_update_kv_cache_window(memx_runtime_context_t *ctx, void *ptr, const memx_runtime_kv_cache_window_t *window);
int memx_runtime_context_update_weight_window(memx_runtime_context_t *ctx, void *ptr, const memx_runtime_weight_window_t *window);
int memx_runtime_context_force_compress_range(memx_runtime_context_t *ctx, void *ptr, size_t offset, size_t length, uint64_t *out_compressed_pages);
int memx_runtime_context_seal_range(memx_runtime_context_t *ctx, void *ptr, size_t offset, size_t length, uint64_t *out_compressed_pages);
int memx_runtime_context_seal_range_async(memx_runtime_context_t *ctx, void *ptr, size_t offset, size_t length);
int memx_runtime_seal_flush(uint64_t *out_pending);

enum {
    MEMX_EPOCH_LOAD = 1,
    MEMX_EPOCH_COMPRESS = 2,
    MEMX_EPOCH_INFER = 3,
    MEMX_EPOCH_FINAL = 4
};

enum {
    MEMX_WS_FLAG_NONE = 0u,
    MEMX_WS_FLAG_HOT = 1u << 0,
    MEMX_WS_FLAG_PREFETCH = 1u << 1,
    MEMX_WS_FLAG_RETIRE = 1u << 2,
    MEMX_WS_FLAG_RETIRE_SYNC = 1u << 3,
    MEMX_WS_FLAG_MARK_ACCESS = 1u << 4,
    MEMX_WS_FLAG_NO_ASYNC = 1u << 5,
    MEMX_WS_FLAG_EPHEMERAL = 1u << 6
};

enum {
    MEMX_MATERIALIZE_KEEP_COMPRESSED = 1u << 0,
    MEMX_MATERIALIZE_ALLOW_RESIDENT = 1u << 1,
    MEMX_MATERIALIZE_BF16_TO_FP16 = 1u << 2
};

typedef struct memx_runtime_ws_intent {
    uint32_t struct_size;
    uint32_t flags;
    void *ptr;
    size_t offset;
    size_t length;
    size_t prefetch_length;
    uint32_t priority;
    uint32_t _reserved0;
} memx_runtime_ws_intent_t;

int memx_runtime_context_begin_epoch(memx_runtime_context_t *ctx, uint32_t phase, uint64_t hot_budget_bytes);
int memx_runtime_context_apply_ws(memx_runtime_context_t *ctx, const memx_runtime_ws_intent_t *intents, size_t nintents);
int memx_runtime_context_ws_advance(memx_runtime_context_t *ctx, void *ptr, size_t hot_offset, size_t hot_length, size_t prefetch_length, uint32_t flags);
int memx_runtime_context_ws_close(memx_runtime_context_t *ctx, void *ptr, uint32_t flags);
int memx_runtime_context_end_epoch(memx_runtime_context_t *ctx, int seal_tracked);

enum {
    MEMX_ARCHIVE_VERSION = 1
};

typedef struct memx_runtime_ws_tile {
    uint32_t struct_size;
    uint32_t flags;
    void *ptr;
    size_t rows;
    size_t cols;
    size_t elem_size;
    size_t col_start;
    size_t col_count;
    size_t prefetch_cols;
    size_t retire_col_start;
    size_t retire_col_count;
} memx_runtime_ws_tile_t;

int memx_runtime_context_export_archive(memx_runtime_context_t *ctx, void *ptr, const char *path, uint64_t *out_bytes);
int memx_runtime_context_import_archive(memx_runtime_context_t *ctx, const char *path, const memx_runtime_tensor_desc_t *desc_override, void **out_ptr, size_t *out_size);
int memx_runtime_context_ws_tile(memx_runtime_context_t *ctx, const memx_runtime_ws_tile_t *tile);
int memx_runtime_context_materialize_range(memx_runtime_context_t *ctx, const void *ptr, size_t offset, size_t length, void *dst, size_t dst_cap, uint32_t flags);
int memx_runtime_context_materialize_tile(memx_runtime_context_t *ctx, const memx_runtime_ws_tile_t *tile, void *dst, size_t dst_cap, size_t dst_row_stride, uint32_t flags);
int memx_runtime_context_materialize_prefetch_range(memx_runtime_context_t *ctx, const void *ptr, size_t offset, size_t length, uint32_t flags);
int memx_runtime_context_purge(memx_runtime_context_t *ctx, void *ptr);
int memx_runtime_context_posix_memalign(memx_runtime_context_t *ctx, void **memptr, size_t alignment, size_t size);
void *memx_runtime_context_aligned_alloc(memx_runtime_context_t *ctx, size_t alignment, size_t size);
void *memx_runtime_context_mmap(memx_runtime_context_t *ctx, void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int memx_runtime_context_munmap(memx_runtime_context_t *ctx, void *addr, size_t length);

#ifdef __cplusplus
}
#endif

#endif
