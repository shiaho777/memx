#include "memx_runtime.h"

#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MB (1024ULL * 1024ULL)
#define PAGE_SZ 16384ULL

typedef struct latency_result {
    const char *name;
    uint64_t ticks;
    uint64_t faults_before;
    uint64_t faults_after;
    uint64_t hits_before;
    uint64_t hits_after;
    uint64_t checksum;
    size_t pages;
} latency_result_t;

static void fill_kv(uint8_t *ptr, size_t size_bytes) {
    for (size_t i = 0; i < size_bytes / 2; i++) {
        size_t page = (i * 2) / PAGE_SZ;
        uint16_t v = (uint16_t)(0x3200 + (page & 15));
        v ^= (uint16_t)((i * 13 + i / 67) & 0x00FF);
        ptr[i * 2] = (uint8_t)(v & 0xFF);
        ptr[i * 2 + 1] = (uint8_t)(v >> 8);
    }
}

static int verify_range(uint8_t *ptr, size_t offset, size_t length) {
    for (size_t off = offset; off < offset + length; off += 263) {
        size_t i = off / 2;
        size_t page = (i * 2) / PAGE_SZ;
        uint16_t expected = (uint16_t)(0x3200 + (page & 15));
        expected ^= (uint16_t)((i * 13 + i / 67) & 0x00FF);
        uint16_t got = (uint16_t)ptr[i * 2] | ((uint16_t)ptr[i * 2 + 1] << 8);
        if (got != expected) {
            fprintf(stderr, "integrity mismatch off=%zu got=0x%04x expected=0x%04x\n", off, got, expected);
            return -1;
        }
    }
    return 0;
}

static latency_result_t measure_pages(const char *name, memx_runtime_context_t *ctx, uint8_t *ptr, size_t offset, size_t pages, int mark_access, double ns_per_tick) {
    (void)ns_per_tick;
    latency_result_t result;
    memset(&result, 0, sizeof(result));
    result.name = name;
    result.pages = pages;
    memx_runtime_stats_t stats;
    memx_runtime_get_stats(&stats);
    result.faults_before = stats.faults;
    result.hits_before = stats.prefetch_hits;
    if (mark_access && memx_runtime_context_mark_access_range(ctx, ptr, offset, pages * PAGE_SZ) != 0) {
        result.name = "mark_failed";
        return result;
    }
    uint64_t t0 = mach_absolute_time();
    for (size_t p = 0; p < pages; p++) {
        size_t base = offset + p * PAGE_SZ;
        volatile uint8_t a = ptr[base];
        volatile uint8_t b = ptr[base + PAGE_SZ / 2];
        result.checksum += a;
        result.checksum += b;
    }
    uint64_t t1 = mach_absolute_time();
    memx_runtime_get_stats(&stats);
    result.faults_after = stats.faults;
    result.hits_after = stats.prefetch_hits;
    result.ticks = t1 - t0;
    return result;
}

static void print_result(const latency_result_t *result, double ns_per_tick) {
    double us_total = ((double)result->ticks * ns_per_tick) / 1000.0;
    double us_page = result->pages ? us_total / (double)result->pages : 0.0;
    printf("  %-10s pages=%zu us_page=%7.2f faults=%llu hits=%llu checksum=%llu\n",
           result->name,
           result->pages,
           us_page,
           (unsigned long long)(result->faults_after - result->faults_before),
           (unsigned long long)(result->hits_after - result->hits_before),
           (unsigned long long)result->checksum);
}

int main(void) {
    memx_runtime_context_t *ctx = NULL;
    mach_timebase_info_data_t timebase;
    double ns_per_tick = 1.0;
    const size_t kv_size = 32 * MB;
    const size_t hot_offset = 0;
    const size_t hot_length = 8 * PAGE_SZ;
    const size_t prefetch_offset = 8 * PAGE_SZ;
    const size_t prefetch_length = 8 * PAGE_SZ;
    const size_t cold_offset = 16 * PAGE_SZ;
    const size_t cold_length = 8 * PAGE_SZ;

    if (memx_runtime_context_create("hot-path-latency", &ctx) != 0 || !ctx) {
        fprintf(stderr, "memx_runtime_context_create failed\n");
        return 1;
    }
    if (memx_runtime_context_set_quota(ctx, 96 * MB) != 0) {
        fprintf(stderr, "memx_runtime_context_set_quota failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 2;
    }

    memx_runtime_tensor_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.struct_size = sizeof(desc);
    desc.role = MEMX_TENSOR_ROLE_KV_CACHE;
    desc.dtype = MEMX_TENSOR_DTYPE_FP16;
    desc.layout = MEMX_TENSOR_LAYOUT_BLOCKED;
    desc.flags = MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY;
    desc.rank = 4;
    desc.shape[0] = 1;
    desc.shape[1] = 8;
    desc.shape[2] = 256;
    desc.shape[3] = 64;
    desc.stride[0] = 131072;
    desc.stride[1] = 16384;
    desc.stride[2] = 64;
    desc.stride[3] = 1;

    uint8_t *kv = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, kv_size, &desc);
    if (!kv) {
        fprintf(stderr, "memx_runtime_context_malloc_tensor failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 3;
    }
    fill_kv(kv, kv_size);
    sleep(3);

    memx_runtime_allocation_info_t cold_info;
    if (memx_runtime_get_allocation_info_range(kv, cold_offset, cold_length, &cold_info) != 0 ||
        cold_info.compressed_pages == 0) {
        fprintf(stderr, "cold setup did not compress compressed=%llu\n", (unsigned long long)cold_info.compressed_pages);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 4;
    }

    memx_runtime_kv_cache_window_t window;
    memset(&window, 0, sizeof(window));
    window.struct_size = sizeof(window);
    window.managed_offset = 0;
    window.managed_length = kv_size;
    window.hot_offset = hot_offset;
    window.hot_length = hot_length;
    window.prefetch_offset = prefetch_offset;
    window.prefetch_length = prefetch_length;
    if (memx_runtime_context_update_kv_cache_window(ctx, kv, &window) != 0) {
        fprintf(stderr, "memx_runtime_context_update_kv_cache_window failed\n");
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 5;
    }

    memx_runtime_allocation_info_t hot_info;
    memx_runtime_allocation_info_t prefetch_info;
    if (memx_runtime_get_allocation_info_range(kv, hot_offset, hot_length, &hot_info) != 0 ||
        memx_runtime_get_allocation_info_range(kv, prefetch_offset, prefetch_length, &prefetch_info) != 0 ||
        hot_info.compressed_pages != 0 ||
        prefetch_info.compressed_pages != 0) {
        fprintf(stderr, "hot/prefetch setup mismatch hot=%llu prefetch=%llu\n",
                (unsigned long long)hot_info.compressed_pages,
                (unsigned long long)prefetch_info.compressed_pages);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 6;
    }

    mach_timebase_info(&timebase);
    ns_per_tick = (double)timebase.numer / (double)timebase.denom;

    latency_result_t hot = measure_pages("hot", ctx, kv, hot_offset, hot_length / PAGE_SZ, 0, ns_per_tick);
    latency_result_t prefetched = measure_pages("prefetch", ctx, kv, prefetch_offset, prefetch_length / PAGE_SZ, 1, ns_per_tick);
    latency_result_t cold = measure_pages("cold", ctx, kv, cold_offset, cold_length / PAGE_SZ, 0, ns_per_tick);

    if (verify_range(kv, hot_offset, hot_length) != 0 ||
        verify_range(kv, prefetch_offset, prefetch_length) != 0 ||
        verify_range(kv, cold_offset, cold_length) != 0) {
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 8;
    }

    memx_runtime_stats_t stats;
    memx_runtime_context_stats_t ctx_stats;
    if (memx_runtime_get_stats(&stats) != 0 ||
        memx_runtime_context_get_stats(ctx, &ctx_stats) != 0) {
        fprintf(stderr, "stats read failed\n");
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 9;
    }

    printf("MemX hot path latency benchmark\n");
    print_result(&hot, ns_per_tick);
    print_result(&prefetched, ns_per_tick);
    print_result(&cold, ns_per_tick);
    printf("  telemetry hot_resident=%llu no_compress=%llu prefetch_hits=%llu faults=%llu ctx_hot_mb=%llu\n",
           (unsigned long long)stats.hot_resident_pages,
           (unsigned long long)stats.no_compress_resident_pages,
           (unsigned long long)stats.prefetch_hits,
           (unsigned long long)stats.faults,
           (unsigned long long)(ctx_stats.hot_bytes_in_use / MB));

    if (hot.faults_after != hot.faults_before ||
        prefetched.faults_after != prefetched.faults_before ||
        prefetched.hits_after <= prefetched.hits_before ||
        cold.faults_after <= cold.faults_before ||
        stats.hot_resident_pages < hot_length / PAGE_SZ ||
        stats.no_compress_resident_pages < hot_length / PAGE_SZ) {
        fprintf(stderr, "latency gates failed hot_faults=%llu prefetch_faults=%llu prefetch_hits=%llu cold_faults=%llu hot_resident=%llu no_compress=%llu\n",
                (unsigned long long)(hot.faults_after - hot.faults_before),
                (unsigned long long)(prefetched.faults_after - prefetched.faults_before),
                (unsigned long long)(prefetched.hits_after - prefetched.hits_before),
                (unsigned long long)(cold.faults_after - cold.faults_before),
                (unsigned long long)stats.hot_resident_pages,
                (unsigned long long)stats.no_compress_resident_pages);
        memx_runtime_context_free(ctx, kv);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 10;
    }

    memx_runtime_context_free(ctx, kv);
    memx_runtime_context_destroy(ctx);
    memx_runtime_shutdown();
    return 0;
}
