#include "memx_runtime.h"

#include <stdint.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MB (1024ULL * 1024)

int main(void) {
    memx_runtime_context_t *ctx = NULL;
    memx_runtime_pressure_t pressure;
    memx_runtime_stats_t stats;
    memx_runtime_context_stats_t ctx_stats;
    memx_runtime_allocation_info_t alloc_info;
    memx_runtime_allocation_info_t cold_range_info;
    memx_runtime_allocation_info_t hot_range_info;
    uint64_t quota = 0;
    uint64_t reclaimed_bytes = 0;

    if (memx_runtime_context_create("smoke", &ctx) != 0 || !ctx) {
        fprintf(stderr, "memx_runtime_context_create failed\n");
        return 1;
    }

    if (memx_runtime_context_set_quota(ctx, 64 * MB) != 0) {
        fprintf(stderr, "memx_runtime_context_set_quota(64MB) failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 2;
    }
    if (memx_runtime_context_get_quota(ctx, &quota) != 0 || quota != 64 * MB) {
        fprintf(stderr, "memx_runtime_context_get_quota mismatch: %llu\n",
                (unsigned long long)quota);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 3;
    }

    size_t size = 128 * MB;
    errno = 0;
    uint8_t *ptr = (uint8_t *)memx_runtime_context_malloc(ctx, size);
    if (ptr || errno != ENOMEM) {
        fprintf(stderr, "quota gate failed: ptr=%p errno=%d\n", (void *)ptr, errno);
        if (ptr) memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 4;
    }

    if (memx_runtime_context_get_stats(ctx, &ctx_stats) != 0) {
        fprintf(stderr, "memx_runtime_context_get_stats after quota fail failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 5;
    }
    if (ctx_stats.allocation_failures_quota == 0) {
        fprintf(stderr, "quota failure counter not incremented\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 6;
    }

    if (memx_runtime_context_set_quota(ctx, 320 * MB) != 0) {
        fprintf(stderr, "memx_runtime_context_set_quota(320MB) failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 7;
    }

    memx_runtime_tensor_desc_t weight_desc = {
        .struct_size = sizeof(weight_desc),
        .role = MEMX_TENSOR_ROLE_WEIGHT,
        .dtype = MEMX_TENSOR_DTYPE_FP16,
        .layout = MEMX_TENSOR_LAYOUT_ROW_MAJOR,
        .flags = MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_SEQUENTIAL | MEMX_TENSOR_FLAG_COLD,
        .rank = 2,
        .shape = {8192, 8192, 0, 0},
        .stride = {8192, 1, 0, 0},
        .layer_index = 3
    };

    ptr = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, size, &weight_desc);
    if (!ptr) {
        fprintf(stderr, "memx_runtime_context_malloc_tensor weight failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 8;
    }

    memset(ptr, 0, size);
    for (size_t off = size / 2; off < size; off += 4096) {
        ptr[off] = (uint8_t)((off / 4096) & 0xFF);
    }

    sleep(3);

    if (memx_runtime_get_stats(&stats) != 0) {
        fprintf(stderr, "memx_runtime_get_stats failed\n");
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 9;
    }
    if (memx_runtime_get_pressure(&pressure) != 0) {
        fprintf(stderr, "memx_runtime_get_pressure failed\n");
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 10;
    }
    if (memx_runtime_reclaim(&reclaimed_bytes) != 0) {
        fprintf(stderr, "memx_runtime_reclaim failed\n");
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 10;
    }
    if (memx_runtime_context_get_stats(ctx, &ctx_stats) != 0) {
        fprintf(stderr, "memx_runtime_context_get_stats failed\n");
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 11;
    }
    if (memx_runtime_get_allocation_info(ptr, &alloc_info) != 0 ||
        alloc_info.tensor_role != MEMX_TENSOR_ROLE_WEIGHT ||
        alloc_info.tensor_dtype != MEMX_TENSOR_DTYPE_FP16 ||
        alloc_info.tensor_layout != MEMX_TENSOR_LAYOUT_ROW_MAJOR ||
        (alloc_info.tensor_flags & MEMX_TENSOR_FLAG_COLD) == 0) {
        fprintf(stderr, "tensor allocation info mismatch role=%u dtype=%u layout=%u flags=0x%x\n",
                alloc_info.tensor_role, alloc_info.tensor_dtype, alloc_info.tensor_layout, alloc_info.tensor_flags);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 11;
    }

    for (size_t off = size / 2; off < size; off += 4096) {
        uint8_t expected = (uint8_t)((off / 4096) & 0xFF);
        if (ptr[off] != expected) {
            fprintf(stderr, "integrity mismatch at %zu: got=%u expected=%u\n",
                    off, (unsigned)ptr[off], (unsigned)expected);
            memx_runtime_context_free(ctx, ptr);
            memx_runtime_context_destroy(ctx);
            memx_runtime_shutdown();
            return 12;
        }
    }

    if (!stats.running || !memx_runtime_owns_pointer(ptr)) {
        fprintf(stderr, "runtime ownership check failed\n");
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 13;
    }
    if (ctx_stats.quota_bytes != 320 * MB) {
        fprintf(stderr, "context quota mismatch after raise: %llu\n",
                (unsigned long long)ctx_stats.quota_bytes);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 14;
    }
    if (ctx_stats.allocations_live != 1 || ctx_stats.bytes_in_use < size) {
        fprintf(stderr, "context accounting mismatch live=%llu in_use=%llu\n",
                (unsigned long long)ctx_stats.allocations_live,
                (unsigned long long)ctx_stats.bytes_in_use);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 15;
    }
    if (ctx_stats.tensor_allocations_live != 1 ||
        ctx_stats.tensor_bytes_in_use != size ||
        ctx_stats.weight_bytes_in_use != size ||
        ctx_stats.kv_cache_bytes_in_use != 0) {
        fprintf(stderr, "tensor accounting mismatch tensors=%llu tensor_mb=%llu weight_mb=%llu kv_mb=%llu\n",
                (unsigned long long)ctx_stats.tensor_allocations_live,
                (unsigned long long)(ctx_stats.tensor_bytes_in_use / MB),
                (unsigned long long)(ctx_stats.weight_bytes_in_use / MB),
                (unsigned long long)(ctx_stats.kv_cache_bytes_in_use / MB));
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 15;
    }

    memx_runtime_tensor_desc_t codec_desc = {
        .struct_size = sizeof(codec_desc),
        .role = MEMX_TENSOR_ROLE_WEIGHT,
        .dtype = MEMX_TENSOR_DTYPE_FP16,
        .layout = MEMX_TENSOR_LAYOUT_ROW_MAJOR,
        .flags = MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD,
        .rank = 2,
        .shape = {4096, 4096, 0, 0},
        .stride = {4096, 1, 0, 0},
        .layer_index = 4
    };
    const size_t codec_size = 32 * MB;
    uint8_t *codec_ptr = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, codec_size, &codec_desc);
    if (!codec_ptr) {
        fprintf(stderr, "memx_runtime_context_malloc_tensor codec failed\n");
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 16;
    }
    for (size_t i = 0; i < codec_size / 2; i++) {
        codec_ptr[i * 2] = (uint8_t)((i * 17 + i / 257) & 0xFF);
        codec_ptr[i * 2 + 1] = (i & 1023) < 1000 ? 0x3C : 0xBC;
    }
    sleep(3);
    if (memx_runtime_get_allocation_info(codec_ptr, &alloc_info) != 0 ||
        alloc_info.tensor_codec_pages == 0 ||
        (alloc_info.primary_codec != MEMX_RUNTIME_CODEC_TENSOR_BITPLANE16 &&
         alloc_info.primary_codec != MEMX_RUNTIME_CODEC_TENSOR_FP16_SPLIT &&
         alloc_info.primary_codec != MEMX_RUNTIME_CODEC_TENSOR_FP16_DELTA_SPLIT &&
         alloc_info.primary_codec != MEMX_RUNTIME_CODEC_ZLIB &&
         alloc_info.primary_codec != MEMX_RUNTIME_CODEC_TENSOR_FP16_ZLIB_SPLIT &&
         alloc_info.primary_codec != MEMX_RUNTIME_CODEC_TENSOR_EXP_PACK)) {
        fprintf(stderr, "tensor codec not used role=%u dtype=%u flags=0x%x primary=0x%x tensor_codec_pages=%llu compressed_pages=%llu\n",
                alloc_info.tensor_role,
                alloc_info.tensor_dtype,
                alloc_info.tensor_flags,
                alloc_info.primary_codec,
                (unsigned long long)alloc_info.tensor_codec_pages,
                (unsigned long long)alloc_info.compressed_pages);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 16;
    }
    for (size_t i = 0; i < codec_size / 2; i += 257) {
        uint8_t expected_lo = (uint8_t)((i * 17 + i / 257) & 0xFF);
        uint8_t expected_hi = (i & 1023) < 1000 ? 0x3C : 0xBC;
        if (codec_ptr[i * 2] != expected_lo || codec_ptr[i * 2 + 1] != expected_hi) {
            fprintf(stderr, "tensor codec mismatch at half=%zu got=%02x%02x expected=%02x%02x\n",
                    i, codec_ptr[i * 2 + 1], codec_ptr[i * 2], expected_hi, expected_lo);
            memx_runtime_context_free(ctx, codec_ptr);
            memx_runtime_context_free(ctx, ptr);
            memx_runtime_context_destroy(ctx);
            memx_runtime_shutdown();
            return 16;
        }
    }

    memx_runtime_tensor_desc_t split_desc = {
        .struct_size = sizeof(split_desc),
        .role = MEMX_TENSOR_ROLE_KV_CACHE,
        .dtype = MEMX_TENSOR_DTYPE_FP16,
        .layout = MEMX_TENSOR_LAYOUT_BLOCKED,
        .flags = MEMX_TENSOR_FLAG_SEQUENTIAL,
        .rank = 4,
        .shape = {1, 8, 256, 128},
        .stride = {262144, 32768, 128, 1},
        .layer_index = 5
    };
    const size_t split_size = 16 * MB;
    uint8_t *split_ptr = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, split_size, &split_desc);
    if (!split_ptr) {
        fprintf(stderr, "memx_runtime_context_malloc_tensor split failed\n");
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 16;
    }
    for (size_t i = 0; i < split_size / 2; i++) {
        split_ptr[i * 2] = (uint8_t)((i * 23 + i / 31) & 0xFF);
        split_ptr[i * 2 + 1] = (i & 511) < 500 ? 0x30 : 0x31;
    }
    sleep(2);
    if (memx_runtime_get_allocation_info(split_ptr, &alloc_info) != 0 ||
        alloc_info.tensor_codec_pages == 0 ||
        (alloc_info.primary_codec != MEMX_RUNTIME_CODEC_TENSOR_FP16_SPLIT &&
         alloc_info.primary_codec != MEMX_RUNTIME_CODEC_TENSOR_FP16_DELTA_SPLIT)) {
        fprintf(stderr, "split tensor codec not used role=%u dtype=%u flags=0x%x primary=0x%x tensor_codec_pages=%llu compressed_pages=%llu\n",
                alloc_info.tensor_role,
                alloc_info.tensor_dtype,
                alloc_info.tensor_flags,
                alloc_info.primary_codec,
                (unsigned long long)alloc_info.tensor_codec_pages,
                (unsigned long long)alloc_info.compressed_pages);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 16;
    }
    for (size_t i = 0; i < split_size / 2; i += 251) {
        uint8_t expected_lo = (uint8_t)((i * 23 + i / 31) & 0xFF);
        uint8_t expected_hi = (i & 511) < 500 ? 0x30 : 0x31;
        if (split_ptr[i * 2] != expected_lo || split_ptr[i * 2 + 1] != expected_hi) {
            fprintf(stderr, "split tensor codec mismatch at half=%zu got=%02x%02x expected=%02x%02x\n",
                    i, split_ptr[i * 2 + 1], split_ptr[i * 2], expected_hi, expected_lo);
            memx_runtime_context_free(ctx, split_ptr);
            memx_runtime_context_free(ctx, codec_ptr);
            memx_runtime_context_free(ctx, ptr);
            memx_runtime_context_destroy(ctx);
            memx_runtime_shutdown();
            return 16;
        }
    }

    if (pressure.virtual_used_bytes == 0 || pressure.free_pages == 0) {
        fprintf(stderr, "pressure telemetry looks invalid used=%llu free_pages=%llu\n",
                (unsigned long long)pressure.virtual_used_bytes,
                (unsigned long long)pressure.free_pages);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 16;
    }
    if (pressure.pool_fragmentation_percent > 100) {
        fprintf(stderr, "fragmentation percent invalid=%u\n",
                (unsigned)pressure.pool_fragmentation_percent);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 16;
    }

    if (memx_runtime_get_stats(&stats) != 0) {
        fprintf(stderr, "memx_runtime_get_stats after reclaim failed\n");
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 16;
    }
    if (stats.tensor_codec_pages == 0 ||
        stats.tensor_codec_bytes_saved == 0 ||
        stats.tensor_split_pages == 0 ||
        stats.tensor_delta_split_pages == 0 ||
        stats.weight_compressed_pages == 0 ||
        stats.kv_cache_compressed_pages == 0 ||
        stats.weight_bytes_saved == 0 ||
        stats.kv_cache_bytes_saved == 0 ||
        stats.dedup_hits == 0 ||
        stats.dedup_bytes_saved == 0) {
        fprintf(stderr, "tensor telemetry missing tensor_pages=%llu tensor_saved=%llu split=%llu delta_split=%llu bitplane=%llu weight_pages=%llu kv_pages=%llu dedup=%llu dedup_bytes=%llu\n",
                (unsigned long long)stats.tensor_codec_pages,
                (unsigned long long)stats.tensor_codec_bytes_saved,
                (unsigned long long)stats.tensor_split_pages,
                (unsigned long long)stats.tensor_delta_split_pages,
                (unsigned long long)stats.tensor_bitplane_pages,
                (unsigned long long)stats.weight_compressed_pages,
                (unsigned long long)stats.kv_cache_compressed_pages,
                (unsigned long long)stats.dedup_hits,
                (unsigned long long)stats.dedup_bytes_saved);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 16;
    }

    printf("memx explicit runtime: active=%d managed=%d compressed=%llu faults=%llu saved_mb=%llu tensor_pages=%llu split=%llu delta_split=%llu bitplane=%llu weight_pages=%llu kv_pages=%llu quota_fail=%llu pressure=%u%% reclaim_mb=%llu frag=%u%% ctx_live=%llu ctx_peak_mb=%llu free_pages=%llu\n",
           stats.running,
           memx_runtime_owns_pointer(ptr),
           (unsigned long long)stats.compressions,
           (unsigned long long)stats.faults,
           (unsigned long long)(stats.bytes_saved / MB),
           (unsigned long long)stats.tensor_codec_pages,
           (unsigned long long)stats.tensor_split_pages,
           (unsigned long long)stats.tensor_delta_split_pages,
           (unsigned long long)stats.tensor_bitplane_pages,
           (unsigned long long)stats.weight_compressed_pages,
           (unsigned long long)stats.kv_cache_compressed_pages,
           (unsigned long long)ctx_stats.allocation_failures_quota,
           (unsigned)pressure.pool_pressure_percent,
           (unsigned long long)(reclaimed_bytes / MB),
           (unsigned)pressure.pool_fragmentation_percent,
           (unsigned long long)ctx_stats.allocations_live,
           (unsigned long long)(ctx_stats.peak_bytes_in_use / MB),
           (unsigned long long)pressure.free_pages);

    memx_runtime_tensor_desc_t kv_desc = {
        .struct_size = sizeof(kv_desc),
        .role = MEMX_TENSOR_ROLE_KV_CACHE,
        .dtype = MEMX_TENSOR_DTYPE_FP16,
        .layout = MEMX_TENSOR_LAYOUT_BLOCKED,
        .flags = MEMX_TENSOR_FLAG_HOT | MEMX_TENSOR_FLAG_NO_COMPRESS,
        .rank = 4,
        .shape = {1, 8, 128, 64},
        .stride = {65536, 8192, 64, 1},
        .layer_index = 3,
        .head_index = 7
    };
    const size_t kv_size = 16 * MB;
    uint8_t *kv = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, kv_size, &kv_desc);
    if (!kv) {
        fprintf(stderr, "memx_runtime_context_malloc_tensor kv failed\n");
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    for (size_t i = 0; i < kv_size / 2; i++) {
        kv[i * 2] = (uint8_t)((i * 7 + i / 19) & 0xFF);
        kv[i * 2 + 1] = (i & 255) < 250 ? 0x32 : 0x33;
    }
    sleep(1);
    if (memx_runtime_get_allocation_info(kv, &alloc_info) != 0 ||
        alloc_info.tensor_role != MEMX_TENSOR_ROLE_KV_CACHE ||
        (alloc_info.tensor_flags & MEMX_TENSOR_FLAG_NO_COMPRESS) == 0 ||
        alloc_info.compressed_pages != 0) {
        fprintf(stderr, "kv tensor policy mismatch role=%u flags=0x%x compressed_pages=%llu\n",
                alloc_info.tensor_role,
                alloc_info.tensor_flags,
                (unsigned long long)alloc_info.compressed_pages);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_context_get_stats(ctx, &ctx_stats) != 0 ||
        ctx_stats.tensor_allocations_live != 4 ||
        ctx_stats.kv_cache_bytes_in_use != kv_size + split_size ||
        ctx_stats.hot_bytes_in_use != kv_size ||
        ctx_stats.no_compress_bytes_in_use != kv_size) {
        fprintf(stderr, "kv tensor accounting mismatch tensors=%llu kv_mb=%llu hot_mb=%llu no_comp_mb=%llu\n",
                (unsigned long long)ctx_stats.tensor_allocations_live,
                (unsigned long long)(ctx_stats.kv_cache_bytes_in_use / MB),
                (unsigned long long)(ctx_stats.hot_bytes_in_use / MB),
                (unsigned long long)(ctx_stats.no_compress_bytes_in_use / MB));
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_get_stats(&stats) != 0 ||
        stats.hot_resident_bytes < kv_size ||
        stats.no_compress_resident_bytes < kv_size) {
        fprintf(stderr, "global hot telemetry mismatch hot_mb=%llu no_comp_mb=%llu kv_mb=%llu\n",
                (unsigned long long)(stats.hot_resident_bytes / MB),
                (unsigned long long)(stats.no_compress_resident_bytes / MB),
                (unsigned long long)(kv_size / MB));
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }

    if (memx_runtime_context_update_tensor_flags_range(ctx, kv, 0, kv_size / 2, MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY) != 0) {
        fprintf(stderr, "memx_runtime_context_update_tensor_flags_range failed\n");
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_context_get_stats(ctx, &ctx_stats) != 0 ||
        ctx_stats.hot_bytes_in_use != kv_size / 2 ||
        ctx_stats.no_compress_bytes_in_use != kv_size / 2 ||
        ctx_stats.kv_cache_bytes_in_use != kv_size + split_size) {
        fprintf(stderr, "kv range policy update accounting mismatch kv_mb=%llu hot_mb=%llu no_comp_mb=%llu\n",
                (unsigned long long)(ctx_stats.kv_cache_bytes_in_use / MB),
                (unsigned long long)(ctx_stats.hot_bytes_in_use / MB),
                (unsigned long long)(ctx_stats.no_compress_bytes_in_use / MB));
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    sleep(3);
    if (memx_runtime_get_allocation_info_range(kv, 0, kv_size / 2, &cold_range_info) != 0 ||
        (cold_range_info.tensor_flags & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY)) != (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY) ||
        cold_range_info.compressed_pages == 0 ||
        cold_range_info.tensor_codec_pages == 0) {
        fprintf(stderr, "kv cold range compression mismatch flags=0x%x compressed=%llu tensor_codec=%llu\n",
                cold_range_info.tensor_flags,
                (unsigned long long)cold_range_info.compressed_pages,
                (unsigned long long)cold_range_info.tensor_codec_pages);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_get_allocation_info_range(kv, kv_size / 2, kv_size / 2, &hot_range_info) != 0 ||
        (hot_range_info.tensor_flags & MEMX_TENSOR_FLAG_HOT) == 0 ||
        (hot_range_info.tensor_flags & MEMX_TENSOR_FLAG_NO_COMPRESS) == 0 ||
        hot_range_info.compressed_pages != 0 ||
        hot_range_info.tensor_codec_pages != 0) {
        fprintf(stderr, "kv hot range policy mismatch flags=0x%x compressed=%llu tensor_codec=%llu\n",
                hot_range_info.tensor_flags,
                (unsigned long long)hot_range_info.compressed_pages,
                (unsigned long long)hot_range_info.tensor_codec_pages);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_get_allocation_info(kv, &alloc_info) != 0 ||
        alloc_info.compressed_pages == 0 ||
        alloc_info.compressed_pages >= alloc_info.page_count ||
        alloc_info.tensor_codec_pages == 0) {
        fprintf(stderr, "kv range hot-to-cold compression mismatch flags=0x%x compressed=%llu/%llu tensor_codec=%llu\n",
                alloc_info.tensor_flags,
                (unsigned long long)alloc_info.compressed_pages,
                (unsigned long long)alloc_info.page_count,
                (unsigned long long)alloc_info.tensor_codec_pages);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }

    if (memx_runtime_context_update_tensor_flags(ctx, kv, MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY) != 0) {
        fprintf(stderr, "memx_runtime_context_update_tensor_flags failed\n");
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_context_get_stats(ctx, &ctx_stats) != 0 ||
        ctx_stats.hot_bytes_in_use != 0 ||
        ctx_stats.no_compress_bytes_in_use != 0 ||
        ctx_stats.kv_cache_bytes_in_use != kv_size + split_size) {
        fprintf(stderr, "kv full policy update accounting mismatch kv_mb=%llu hot_mb=%llu no_comp_mb=%llu\n",
                (unsigned long long)(ctx_stats.kv_cache_bytes_in_use / MB),
                (unsigned long long)(ctx_stats.hot_bytes_in_use / MB),
                (unsigned long long)(ctx_stats.no_compress_bytes_in_use / MB));
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    sleep(2);
    if (memx_runtime_get_allocation_info(kv, &alloc_info) != 0 ||
        (alloc_info.tensor_flags & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY)) != (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY) ||
        alloc_info.compressed_pages == 0 ||
        alloc_info.tensor_codec_pages == 0) {
        fprintf(stderr, "kv hot-to-cold compression mismatch flags=0x%x compressed=%llu tensor_codec=%llu\n",
                alloc_info.tensor_flags,
                (unsigned long long)alloc_info.compressed_pages,
                (unsigned long long)alloc_info.tensor_codec_pages);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    const size_t prefetch_size = 4 * 16384;
    if (memx_runtime_get_allocation_info_range(kv, 0, prefetch_size, &cold_range_info) != 0 ||
        cold_range_info.compressed_pages == 0) {
        fprintf(stderr, "kv prefetch setup mismatch compressed=%llu\n",
                (unsigned long long)cold_range_info.compressed_pages);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_get_stats(&stats) != 0) {
        fprintf(stderr, "memx_runtime_get_stats before prefetch failed\n");
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    uint64_t prefetch_count_before = stats.prefetch_count;
    uint64_t faults_before_prefetch = stats.faults;
    if (memx_runtime_context_prefetch_range(ctx, kv, 0, prefetch_size) != 0) {
        fprintf(stderr, "memx_runtime_context_prefetch_range failed\n");
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_get_allocation_info_range(kv, 0, prefetch_size, &cold_range_info) != 0 ||
        cold_range_info.compressed_pages != 0) {
        fprintf(stderr, "kv prefetch range still compressed=%llu\n",
                (unsigned long long)cold_range_info.compressed_pages);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_get_stats(&stats) != 0 ||
        stats.prefetch_count <= prefetch_count_before ||
        stats.faults != faults_before_prefetch) {
        fprintf(stderr, "kv prefetch stats mismatch prefetch=%llu->%llu faults=%llu->%llu\n",
                (unsigned long long)prefetch_count_before,
                (unsigned long long)stats.prefetch_count,
                (unsigned long long)faults_before_prefetch,
                (unsigned long long)stats.faults);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    uint64_t faults_after_prefetch = stats.faults;
    uint64_t prefetch_hits_before_access = stats.prefetch_hits;
    if (memx_runtime_context_mark_access_range(ctx, kv, 0, prefetch_size) != 0) {
        fprintf(stderr, "memx_runtime_context_mark_access_range failed\n");
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_get_stats(&stats) != 0 || stats.prefetch_hits <= prefetch_hits_before_access) {
        fprintf(stderr, "kv prefetch hit mismatch hits=%llu->%llu\n",
                (unsigned long long)prefetch_hits_before_access,
                (unsigned long long)stats.prefetch_hits);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    for (size_t i = 0; i < prefetch_size / 2; i += 263) {
        uint8_t expected_lo = (uint8_t)((i * 7 + i / 19) & 0xFF);
        uint8_t expected_hi = (i & 255) < 250 ? 0x32 : 0x33;
        if (kv[i * 2] != expected_lo || kv[i * 2 + 1] != expected_hi) {
            fprintf(stderr, "kv prefetch integrity mismatch at half=%zu got=%02x%02x expected=%02x%02x\n",
                    i, kv[i * 2 + 1], kv[i * 2], expected_hi, expected_lo);
            memx_runtime_context_free(ctx, kv);
            memx_runtime_context_free(ctx, split_ptr);
            memx_runtime_context_free(ctx, codec_ptr);
            memx_runtime_context_free(ctx, ptr);
            memx_runtime_context_destroy(ctx);
            memx_runtime_shutdown();
            return 17;
        }
    }
    if (memx_runtime_get_stats(&stats) != 0 || stats.faults != faults_after_prefetch) {
        fprintf(stderr, "kv prefetched read faulted faults=%llu->%llu\n",
                (unsigned long long)faults_after_prefetch,
                (unsigned long long)stats.faults);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    for (size_t i = 0; i < kv_size / 2; i += 263) {
        uint8_t expected_lo = (uint8_t)((i * 7 + i / 19) & 0xFF);
        uint8_t expected_hi = (i & 255) < 250 ? 0x32 : 0x33;
        if (kv[i * 2] != expected_lo || kv[i * 2 + 1] != expected_hi) {
            fprintf(stderr, "kv hot-to-cold integrity mismatch at half=%zu got=%02x%02x expected=%02x%02x\n",
                    i, kv[i * 2 + 1], kv[i * 2], expected_hi, expected_lo);
            memx_runtime_context_free(ctx, kv);
            memx_runtime_context_free(ctx, split_ptr);
            memx_runtime_context_free(ctx, codec_ptr);
            memx_runtime_context_free(ctx, ptr);
            memx_runtime_context_destroy(ctx);
            memx_runtime_shutdown();
            return 17;
        }
    }

    const size_t window_kv_size = 8 * MB;
    memx_runtime_tensor_desc_t window_kv_desc = kv_desc;
    window_kv_desc.shape[2] = 64;
    window_kv_desc.stride[0] = 32768;
    window_kv_desc.layer_index = 9;
    uint8_t *window_kv = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, window_kv_size, &window_kv_desc);
    if (!window_kv) {
        fprintf(stderr, "memx_runtime_context_malloc_tensor window kv failed\n");
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    for (size_t i = 0; i < window_kv_size / 2; i++) {
        window_kv[i * 2] = (uint8_t)((i * 13 + i / 31) & 0xFF);
        window_kv[i * 2 + 1] = (i & 127) < 120 ? 0x34 : 0x35;
    }
    memx_runtime_kv_cache_window_t window = {
        .struct_size = sizeof(window),
        .managed_offset = 0,
        .managed_length = window_kv_size,
        .hot_offset = 0,
        .hot_length = 0,
        .prefetch_offset = 0,
        .prefetch_length = 0
    };
    if (memx_runtime_context_update_kv_cache_window(ctx, window_kv, &window) != 0) {
        fprintf(stderr, "memx_runtime_context_update_kv_cache_window cold failed\n");
        memx_runtime_context_free(ctx, window_kv);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    sleep(3);
    if (memx_runtime_get_allocation_info_range(window_kv, 0, 12 * 16384, &alloc_info) != 0 ||
        alloc_info.compressed_pages == 0 ||
        alloc_info.tensor_codec_pages == 0 ||
        (alloc_info.tensor_flags & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY)) != (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY)) {
        fprintf(stderr, "kv window cold setup mismatch flags=0x%x compressed=%llu tensor_codec=%llu\n",
                alloc_info.tensor_flags,
                (unsigned long long)alloc_info.compressed_pages,
                (unsigned long long)alloc_info.tensor_codec_pages);
        memx_runtime_context_free(ctx, window_kv);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_get_stats(&stats) != 0) {
        fprintf(stderr, "memx_runtime_get_stats before window policy failed\n");
        memx_runtime_context_free(ctx, window_kv);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    prefetch_count_before = stats.prefetch_count;
    window.hot_offset = 4 * 16384;
    window.hot_length = 4 * 16384;
    window.prefetch_offset = 0;
    window.prefetch_length = 4 * 16384;
    if (memx_runtime_context_update_kv_cache_window(ctx, window_kv, &window) != 0) {
        fprintf(stderr, "memx_runtime_context_update_kv_cache_window policy failed\n");
        memx_runtime_context_free(ctx, window_kv);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_get_stats(&stats) != 0 || stats.prefetch_count <= prefetch_count_before) {
        fprintf(stderr, "kv window prefetch count mismatch prefetch=%llu->%llu\n",
                (unsigned long long)prefetch_count_before,
                (unsigned long long)stats.prefetch_count);
        memx_runtime_context_free(ctx, window_kv);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_get_allocation_info_range(window_kv, 0, 4 * 16384, &cold_range_info) != 0 ||
        cold_range_info.compressed_pages != 0 ||
        (cold_range_info.tensor_flags & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY)) != (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY)) {
        fprintf(stderr, "kv window prefetch range mismatch flags=0x%x compressed=%llu\n",
                cold_range_info.tensor_flags,
                (unsigned long long)cold_range_info.compressed_pages);
        memx_runtime_context_free(ctx, window_kv);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_get_allocation_info_range(window_kv, 4 * 16384, 4 * 16384, &hot_range_info) != 0 ||
        hot_range_info.compressed_pages != 0 ||
        (hot_range_info.tensor_flags & MEMX_TENSOR_FLAG_HOT) == 0 ||
        (hot_range_info.tensor_flags & MEMX_TENSOR_FLAG_NO_COMPRESS) == 0) {
        fprintf(stderr, "kv window hot range mismatch flags=0x%x compressed=%llu\n",
                hot_range_info.tensor_flags,
                (unsigned long long)hot_range_info.compressed_pages);
        memx_runtime_context_free(ctx, window_kv);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_get_allocation_info_range(window_kv, 8 * 16384, 8 * 16384, &alloc_info) != 0 ||
        alloc_info.compressed_pages == 0 ||
        alloc_info.tensor_codec_pages == 0 ||
        (alloc_info.tensor_flags & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY)) != (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY)) {
        fprintf(stderr, "kv window cold tail mismatch flags=0x%x compressed=%llu tensor_codec=%llu\n",
                alloc_info.tensor_flags,
                (unsigned long long)alloc_info.compressed_pages,
                (unsigned long long)alloc_info.tensor_codec_pages);
        memx_runtime_context_free(ctx, window_kv);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (memx_runtime_get_stats(&stats) != 0) {
        fprintf(stderr, "memx_runtime_get_stats before window read failed\n");
        memx_runtime_context_free(ctx, window_kv);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    uint64_t window_faults_before = stats.faults;
    for (size_t i = 0; i < (4 * 16384) / 2; i += 263) {
        uint8_t expected_lo = (uint8_t)((i * 13 + i / 31) & 0xFF);
        uint8_t expected_hi = (i & 127) < 120 ? 0x34 : 0x35;
        if (window_kv[i * 2] != expected_lo || window_kv[i * 2 + 1] != expected_hi) {
            fprintf(stderr, "kv window prefetch integrity mismatch at half=%zu got=%02x%02x expected=%02x%02x\n",
                    i, window_kv[i * 2 + 1], window_kv[i * 2], expected_hi, expected_lo);
            memx_runtime_context_free(ctx, window_kv);
            memx_runtime_context_free(ctx, kv);
            memx_runtime_context_free(ctx, split_ptr);
            memx_runtime_context_free(ctx, codec_ptr);
            memx_runtime_context_free(ctx, ptr);
            memx_runtime_context_destroy(ctx);
            memx_runtime_shutdown();
            return 17;
        }
    }
    if (memx_runtime_get_stats(&stats) != 0 || stats.faults != window_faults_before) {
        fprintf(stderr, "kv window prefetched read faulted faults=%llu->%llu\n",
                (unsigned long long)window_faults_before,
                (unsigned long long)stats.faults);
        memx_runtime_context_free(ctx, window_kv);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    memx_runtime_context_free(ctx, window_kv);

    memx_runtime_tensor_desc_t sparse_desc = {
        .struct_size = sizeof(sparse_desc),
        .role = MEMX_TENSOR_ROLE_ACTIVATION,
        .dtype = MEMX_TENSOR_DTYPE_UINT8,
        .layout = MEMX_TENSOR_LAYOUT_ROW_MAJOR,
        .flags = MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY,
        .rank = 1,
        .shape = {4 * 1024 * 1024, 0, 0, 0},
        .stride = {1, 0, 0, 0},
        .layer_index = 10
    };
    const size_t sparse_size = 4 * MB;
    uint8_t *sparse_ptr = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, sparse_size, &sparse_desc);
    if (!sparse_ptr) {
        fprintf(stderr, "memx_runtime_context_malloc_tensor sparse failed\n");
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    for (size_t page = 0; page < sparse_size / 16384; page++) {
        size_t base = page * 16384;
        for (size_t j = 0; j < 32; j++) {
            size_t off = base + 32 + j * 257;
            sparse_ptr[off] = (uint8_t)((j * 7 + page * 11 + 1) & 0xFF);
        }
    }
    sleep(3);
    if (memx_runtime_get_allocation_info(sparse_ptr, &alloc_info) != 0 ||
        alloc_info.compressed_pages == 0 ||
        alloc_info.primary_codec != MEMX_RUNTIME_CODEC_TENSOR_SPARSE_BYTE ||
        alloc_info.tensor_codec_pages == 0 ||
        memx_runtime_get_stats(&stats) != 0 ||
        stats.tensor_sparse_pages == 0 ||
        stats.tensor_sparse_bytes_saved == 0) {
        fprintf(stderr, "sparse tensor codec mismatch primary=0x%x compressed=%llu tensor_codec=%llu sparse_pages=%llu sparse_saved=%llu\n",
                alloc_info.primary_codec,
                (unsigned long long)alloc_info.compressed_pages,
                (unsigned long long)alloc_info.tensor_codec_pages,
                (unsigned long long)stats.tensor_sparse_pages,
                (unsigned long long)stats.tensor_sparse_bytes_saved);
        memx_runtime_context_free(ctx, sparse_ptr);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    for (size_t j = 0; j < 32; j++) {
        size_t off = 32 + j * 257;
        uint8_t expected = (uint8_t)((j * 7 + 1) & 0xFF);
        if (sparse_ptr[off] != expected) {
            fprintf(stderr, "sparse tensor integrity mismatch off=%zu got=%u expected=%u\n",
                    off, (unsigned)sparse_ptr[off], (unsigned)expected);
            memx_runtime_context_free(ctx, sparse_ptr);
            memx_runtime_context_free(ctx, kv);
            memx_runtime_context_free(ctx, split_ptr);
            memx_runtime_context_free(ctx, codec_ptr);
            memx_runtime_context_free(ctx, ptr);
            memx_runtime_context_destroy(ctx);
            memx_runtime_shutdown();
            return 17;
        }
    }
    memx_runtime_context_free(ctx, sparse_ptr);

    memx_runtime_tensor_desc_t pressure_desc = {
        .struct_size = sizeof(pressure_desc),
        .role = MEMX_TENSOR_ROLE_KV_CACHE,
        .dtype = MEMX_TENSOR_DTYPE_FP16,
        .layout = MEMX_TENSOR_LAYOUT_BLOCKED,
        .flags = MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY,
        .rank = 4,
        .shape = {1, 4, 64, 64},
        .stride = {65536, 16384, 64, 1},
        .layer_index = 11
    };
    const size_t pressure_seed_size = 4 * MB;
    uint8_t *pressure_seed = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, pressure_seed_size, &pressure_desc);
    if (!pressure_seed) {
        fprintf(stderr, "memx_runtime_context_malloc_tensor pressure seed failed\n");
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    for (size_t page = 0; page < pressure_seed_size / 16384; page++) {
        uint8_t *page_ptr = pressure_seed + page * 16384;
        for (size_t half = 0; half < 8192; half++) {
            page_ptr[half * 2] = (uint8_t)((half * 5 + page * 17) & 0xFF);
            page_ptr[half * 2 + 1] = (half & 63) < 60 ? (uint8_t)(0x36 + (page & 1)) : 0x37;
        }
    }
    sleep(3);
    if (memx_runtime_get_allocation_info(pressure_seed, &alloc_info) != 0 ||
        alloc_info.compressed_pages == 0 ||
        alloc_info.compressed_bytes == 0) {
        fprintf(stderr, "pressure seed did not compress compressed=%llu bytes=%llu\n",
                (unsigned long long)alloc_info.compressed_pages,
                (unsigned long long)alloc_info.compressed_bytes);
        memx_runtime_context_free(ctx, pressure_seed);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    uint64_t pressure_reclaimable_bytes = alloc_info.compressed_bytes;
    if (memx_runtime_get_stats(&stats) != 0) {
        fprintf(stderr, "memx_runtime_get_stats before pressure seed free failed\n");
        memx_runtime_context_free(ctx, pressure_seed);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    uint64_t reclaim_events_before_free = stats.pool_reclaim_events;
    memx_runtime_context_free(ctx, pressure_seed);
    if (memx_runtime_get_pressure(&pressure) != 0 ||
        memx_runtime_context_get_stats(ctx, &ctx_stats) != 0 ||
        memx_runtime_get_stats(&stats) != 0) {
        fprintf(stderr, "pressure telemetry before recovery failed\n");
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    uint64_t pressure_events_before = ctx_stats.pressure_events;
    if (stats.pool_reclaim_events <= reclaim_events_before_free) {
        fprintf(stderr, "pressure seed free did not reclaim reclaim=%llu->%llu\n",
                (unsigned long long)reclaim_events_before_free,
                (unsigned long long)stats.pool_reclaim_events);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    size_t pressure_cursor = (size_t)((pressure.pool_capacity_bytes * 95ULL + 99ULL) / 100ULL);
    if (pressure_reclaimable_bytes == 0 ||
        memx_runtime_test_set_pool_cursor(pressure_cursor) != 0) {
        fprintf(stderr, "pressure cursor setup failed reclaimable=%llu cursor=%zu capacity=%llu\n",
                (unsigned long long)pressure_reclaimable_bytes,
                pressure_cursor,
                (unsigned long long)pressure.pool_capacity_bytes);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    uint8_t *pressure_rescue = (uint8_t *)memx_runtime_context_malloc(ctx, 64 * 1024);
    if (!pressure_rescue) {
        fprintf(stderr, "pressure recovery allocation failed\n");
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    memset(pressure_rescue, 0xA5, 64 * 1024);
    if (memx_runtime_get_stats(&stats) != 0 ||
        memx_runtime_context_get_stats(ctx, &ctx_stats) != 0 ||
        ctx_stats.pressure_events != pressure_events_before) {
        fprintf(stderr, "pressure recovery telemetry mismatch pressure_events=%llu->%llu\n",
                (unsigned long long)pressure_events_before,
                (unsigned long long)ctx_stats.pressure_events);
        memx_runtime_context_free(ctx, pressure_rescue);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_free(ctx, split_ptr);
        memx_runtime_context_free(ctx, codec_ptr);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    memx_runtime_context_free(ctx, pressure_rescue);

    memx_runtime_context_free(ctx, kv);
    memx_runtime_context_free(ctx, split_ptr);
    memx_runtime_context_free(ctx, codec_ptr);
    memx_runtime_context_free(ctx, ptr);
    if (memx_runtime_context_get_stats(ctx, &ctx_stats) != 0) {
        fprintf(stderr, "memx_runtime_context_get_stats after free failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 17;
    }
    if (ctx_stats.allocations_live != 0 ||
        ctx_stats.bytes_in_use != 0 ||
        ctx_stats.tensor_allocations_live != 0 ||
        ctx_stats.tensor_bytes_in_use != 0 ||
        ctx_stats.weight_bytes_in_use != 0 ||
        ctx_stats.kv_cache_bytes_in_use != 0 ||
        ctx_stats.hot_bytes_in_use != 0 ||
        ctx_stats.no_compress_bytes_in_use != 0) {
        fprintf(stderr, "context accounting did not drain live=%llu in_use=%llu tensors=%llu tensor_bytes=%llu\n",
                (unsigned long long)ctx_stats.allocations_live,
                (unsigned long long)ctx_stats.bytes_in_use,
                (unsigned long long)ctx_stats.tensor_allocations_live,
                (unsigned long long)ctx_stats.tensor_bytes_in_use);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 18;
    }
    if (memx_runtime_context_destroy(ctx) != 0) {
        fprintf(stderr, "memx_runtime_context_destroy failed\n");
        memx_runtime_shutdown();
        return 19;
    }
    memx_runtime_shutdown();
    return 0;
}
