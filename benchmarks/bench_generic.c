#include "memx_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <zlib.h>

#define MB (1024ULL * 1024ULL)
#define PAGE_SZ 16384ULL
#define ALLOC_BYTES (32 * MB)

typedef enum { CLASS_ZEROS, CLASS_TEXT, CLASS_STRUCT_I32, CLASS_COUNT } data_class_t;

static const char *class_names[CLASS_COUNT] = { "zeros", "text", "struct_i32" };

static void fill_class(uint8_t *dst, data_class_t cls, size_t bytes) {
    if (cls == CLASS_ZEROS) {
        memset(dst, 0, bytes);
        return;
    }
    if (cls == CLASS_TEXT) {
        static const char words[] =
            "the quick brown fox jumps over the lazy dog "
            "pack my box with five dozen liquor jugs ";
        size_t wl = strlen(words);
        for (size_t i = 0; i < bytes; i++) dst[i] = (uint8_t)words[i % wl];
        return;
    }
    if (cls == CLASS_STRUCT_I32) {
        uint32_t *v = (uint32_t *)dst;
        for (size_t i = 0; i < bytes / 4; i++) {
            uint32_t x = (uint32_t)(i * 7919u);
            v[i] = (x % 251u) * 65599u + (x / 251u);
        }
        return;
    }
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int main(void) {
    printf("MemX generic data benchmark (%.0f MiB per class, descriptor-less + DATA role)\n",
           (double)ALLOC_BYTES / (double)MB);
    printf("  %-12s %10s %12s %12s %12s\n", "class", "ratio", "force_ms", "fault_MB/s", "zlib1_ratio");

    uint8_t *golden = (uint8_t *)malloc(ALLOC_BYTES);
    uint8_t *zbuf = (uint8_t *)malloc((size_t)ALLOC_BYTES + 4096);
    if (!golden || !zbuf) return 1;

    memx_runtime_context_t *ctx = NULL;
    if (memx_runtime_context_create("bench-generic", &ctx) != 0 || !ctx) return 2;
    if (memx_runtime_context_set_quota(ctx, 2ULL * 1024 * MB) != 0) return 3;

    for (int cls = 0; cls < CLASS_COUNT; cls++) {
        fill_class(golden, (data_class_t)cls, ALLOC_BYTES);

        uLongf zlen = (uLongf)ALLOC_BYTES + 4096;
        (void)compress2(zbuf, &zlen, golden, (uLong)ALLOC_BYTES, 1);
        double zlib_ratio = (double)zlen / (double)ALLOC_BYTES;

        uint8_t *ptr = (uint8_t *)memx_runtime_context_malloc(ctx, ALLOC_BYTES);
        if (!ptr) return 4;
        memcpy(ptr, golden, ALLOC_BYTES);

        double t0 = now_ms();
        uint64_t done = 0;
        if (memx_runtime_context_force_compress_range(ctx, ptr, 0, ALLOC_BYTES, &done) != 0) return 5;
        double force_ms = now_ms() - t0;

        memx_runtime_allocation_info_t info;
        (void)memx_runtime_get_allocation_info(ptr, &info);
        double ratio = info.compressed_pages
            ? (double)info.compressed_bytes / (double)ALLOC_BYTES : 0.0;

        volatile uint64_t sink = 0;
        t0 = now_ms();
        for (size_t i = 0; i < ALLOC_BYTES; i += 4096) sink += ptr[i];
        double fault_ms = now_ms() - t0;
        double fault_mbs = fault_ms > 0.0 ? ((double)ALLOC_BYTES / (double)MB) / (fault_ms / 1000.0) : 0.0;

        for (size_t i = 0; i < ALLOC_BYTES; i += PAGE_SZ) {
            if (memcmp(ptr + i, golden + i, PAGE_SZ) != 0) {
                fprintf(stderr, "bitexact failed class=%d off=%zu\n", cls, i);
                return 6;
            }
        }

        printf("  %-12s %10.2fx %12.1f %12.0f %12.2fx%s\n",
               class_names[cls], ratio, force_ms, fault_mbs, zlib_ratio,
               done > 0 ? "" : "  (no compression)");

        memx_runtime_context_free(ctx, ptr);
    }

    memx_runtime_context_destroy(ctx);
    memx_runtime_shutdown();
    free(golden);
    free(zbuf);
    return 0;
}
