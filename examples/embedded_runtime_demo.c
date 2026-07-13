#include "memx_runtime.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MB (1024ULL * 1024ULL)

typedef struct cache_segment {
    uint8_t *data;
    size_t size;
    uint32_t id;
} cache_segment_t;

static void fill_segment(cache_segment_t *segment) {
    memset(segment->data, 0, segment->size);
    for (size_t off = segment->size / 2; off < segment->size; off += 4096) {
        segment->data[off] = (uint8_t)((segment->id + off / 4096) & 0xFF);
    }
}

static int verify_segment(const cache_segment_t *segment) {
    for (size_t off = segment->size / 2; off < segment->size; off += 4096) {
        uint8_t expected = (uint8_t)((segment->id + off / 4096) & 0xFF);
        if (segment->data[off] != expected) return -1;
    }
    return 0;
}

static void print_runtime_snapshot(
    const memx_runtime_context_stats_t *ctx_stats,
    const memx_runtime_pressure_t *pressure,
    const memx_runtime_stats_t *stats
) {
    printf("runtime snapshot: ctx_live=%llu ctx_used_mb=%llu ctx_peak_mb=%llu quota_mb=%llu "
           "tensor_mb=%llu kv_mb=%llu tensor_pages=%llu split=%llu bitplane=%llu kv_pages=%llu "
           "pool_pressure=%u%% frag=%u%% saved_mb=%llu compressed=%llu faults=%llu reclaim_mb=%llu\n",
           (unsigned long long)ctx_stats->allocations_live,
           (unsigned long long)(ctx_stats->bytes_in_use / MB),
           (unsigned long long)(ctx_stats->peak_bytes_in_use / MB),
           (unsigned long long)(ctx_stats->quota_bytes / MB),
           (unsigned long long)(ctx_stats->tensor_bytes_in_use / MB),
           (unsigned long long)(ctx_stats->kv_cache_bytes_in_use / MB),
           (unsigned long long)stats->tensor_codec_pages,
           (unsigned long long)stats->tensor_split_pages,
           (unsigned long long)stats->tensor_bitplane_pages,
           (unsigned long long)stats->kv_cache_compressed_pages,
           (unsigned)pressure->pool_pressure_percent,
           (unsigned)pressure->pool_fragmentation_percent,
           (unsigned long long)(stats->bytes_saved / MB),
           (unsigned long long)stats->compressions,
           (unsigned long long)stats->faults,
           (unsigned long long)(stats->pool_reclaim_bytes / MB));
}

int main(void) {
    memx_runtime_context_t *ctx = NULL;
    cache_segment_t segments[8] = {0};
    const size_t segment_size = 48 * MB;
    const uint64_t quota_bytes = 192 * MB;
    size_t live_segments = 0;
    size_t evicted_segments = 0;

    memx_runtime_context_stats_t ctx_stats;
    memx_runtime_pressure_t pressure;
    memx_runtime_stats_t stats;
    uint64_t reclaimed_bytes = 0;

    if (memx_runtime_context_create("demo-cache", &ctx) != 0 || !ctx) {
        fprintf(stderr, "memx_runtime_context_create failed\n");
        return 1;
    }

    if (memx_runtime_context_set_quota(ctx, quota_bytes) != 0) {
        fprintf(stderr, "memx_runtime_context_set_quota failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 2;
    }

    for (uint32_t i = 0; i < 8; i++) {
        cache_segment_t segment = {.size = segment_size, .id = i + 1};
        memx_runtime_tensor_desc_t segment_desc = {
            .struct_size = sizeof(segment_desc),
            .role = MEMX_TENSOR_ROLE_KV_CACHE,
            .dtype = MEMX_TENSOR_DTYPE_FP16,
            .layout = MEMX_TENSOR_LAYOUT_BLOCKED,
            .flags = MEMX_TENSOR_FLAG_SEQUENTIAL | MEMX_TENSOR_FLAG_COLD,
            .rank = 4,
            .shape = {1, 8, 768, 512},
            .stride = {3145728, 393216, 512, 1},
            .layer_index = i
        };

        errno = 0;
        segment.data = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, segment.size, &segment_desc);
        if (!segment.data && errno == ENOMEM && live_segments > 0) {
            memx_runtime_context_free(ctx, segments[0].data);
            memmove(&segments[0], &segments[1], (live_segments - 1) * sizeof(segments[0]));
            live_segments--;
            evicted_segments++;
            segment.data = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, segment.size, &segment_desc);
        }

        if (!segment.data) {
            fprintf(stderr, "cache segment allocation failed at segment=%u errno=%d\n", i + 1, errno);
            goto fail;
        }

        fill_segment(&segment);
        segments[live_segments++] = segment;
        usleep(150000);
    }

    sleep(2);

    if (memx_runtime_get_stats(&stats) != 0 ||
        memx_runtime_get_pressure(&pressure) != 0 ||
        memx_runtime_context_get_stats(ctx, &ctx_stats) != 0) {
        fprintf(stderr, "runtime telemetry read failed\n");
        goto fail;
    }

    print_runtime_snapshot(&ctx_stats, &pressure, &stats);

    if (memx_runtime_reclaim(&reclaimed_bytes) != 0) {
        fprintf(stderr, "memx_runtime_reclaim failed\n");
        goto fail;
    }

    if (memx_runtime_get_stats(&stats) != 0 ||
        memx_runtime_get_pressure(&pressure) != 0 ||
        memx_runtime_context_get_stats(ctx, &ctx_stats) != 0) {
        fprintf(stderr, "runtime telemetry read after reclaim failed\n");
        goto fail;
    }

    if (live_segments == 0 || verify_segment(&segments[live_segments - 1]) != 0) {
        fprintf(stderr, "managed segment integrity check failed\n");
        goto fail;
    }

    printf("demo result: live_segments=%zu evicted=%zu quota_failures=%llu pressure_events=%llu "
           "reclaimed_mb=%llu final_live_mb=%llu managed=%d\n",
           live_segments,
           evicted_segments,
           (unsigned long long)ctx_stats.allocation_failures_quota,
           (unsigned long long)ctx_stats.pressure_events,
           (unsigned long long)(reclaimed_bytes / MB),
           (unsigned long long)(ctx_stats.bytes_in_use / MB),
           memx_runtime_owns_pointer(segments[live_segments - 1].data));

    for (size_t i = 0; i < live_segments; i++) {
        memx_runtime_context_free(ctx, segments[i].data);
    }
    memx_runtime_context_destroy(ctx);
    memx_runtime_shutdown();
    return 0;

fail:
    for (size_t i = 0; i < live_segments; i++) {
        if (segments[i].data) memx_runtime_context_free(ctx, segments[i].data);
    }
    if (ctx) {
        memx_runtime_context_destroy(ctx);
    }
    memx_runtime_shutdown();
    return 3;
}
