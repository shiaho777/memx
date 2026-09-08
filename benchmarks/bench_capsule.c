#include "memx_runtime.h"

#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MB (1024ULL * 1024ULL)
#define PAGE_SZ 16384ULL
#define PAGES 512

static void fill_weights(uint8_t *ptr, size_t size_bytes, uint16_t seed) {
    for (size_t i = 0; i < size_bytes / 2; i++) {
        uint16_t v = (uint16_t)(seed + ((i >> 9) & 0x7F));
        v ^= (uint16_t)((i * 11 + i / 257) & 0x3F);
        ptr[i * 2] = (uint8_t)(v & 0xFF);
        ptr[i * 2 + 1] = (uint8_t)(v >> 8);
    }
}

static double now_ns(mach_timebase_info_data_t *tb) {
    uint64_t t = mach_absolute_time();
    return (double)t * (double)tb->numer / (double)tb->denom;
}

int main(void) {
    memx_runtime_context_t *ctx = NULL;
    const size_t size_bytes = PAGES * PAGE_SZ;

    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);

    if (memx_runtime_context_create("capsule-bench", &ctx) != 0 || !ctx) {
        fprintf(stderr, "context create failed\n");
        return 1;
    }
    if (memx_runtime_context_set_quota(ctx, 256 * MB) != 0) {
        fprintf(stderr, "quota failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 2;
    }

    memx_runtime_tensor_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.struct_size = sizeof(desc);
    desc.role = MEMX_TENSOR_ROLE_WEIGHT;
    desc.dtype = MEMX_TENSOR_DTYPE_FP16;
    desc.layout = MEMX_TENSOR_LAYOUT_ROW_MAJOR;
    desc.flags = MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD;
    desc.rank = 2;
    desc.shape[0] = PAGES;
    desc.shape[1] = PAGE_SZ / 2;
    desc.stride[0] = PAGE_SZ / 2;
    desc.stride[1] = 1;

    uint8_t *w = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, size_bytes, &desc);
    if (!w) {
        fprintf(stderr, "malloc_tensor failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 3;
    }
    fill_weights(w, size_bytes, 0x3A00);
    uint64_t compressed = 0;
    memx_runtime_context_force_compress_range(ctx, w, 0, size_bytes, &compressed);

    char dir[256];
    snprintf(dir, sizeof(dir), "/tmp/memx_capbench_%d", (int)getpid());

    double t0 = now_ns(&tb);
    uint64_t exported = 0;
    int erv = memx_runtime_capsule_export(dir, &exported);
    double t1 = now_ns(&tb);
    if (erv != 0) {
        fprintf(stderr, "capsule_export failed rc=%d\n", erv);
        memx_runtime_context_free(ctx, w);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 4;
    }
    memx_runtime_capsule_detach();

    double t2 = now_ns(&tb);
    erv = memx_runtime_capsule_attach(dir);
    double t3 = now_ns(&tb);
    if (erv != 0) {
        fprintf(stderr, "capsule_attach failed rc=%d\n", erv);
        memx_runtime_context_free(ctx, w);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 5;
    }

    uint8_t *dst = malloc(PAGE_SZ);
    uint8_t *batch = malloc(PAGES * PAGE_SZ);
    if (!dst || !batch) {
        memx_runtime_capsule_detach();
        memx_runtime_context_free(ctx, w);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 6;
    }

    memx_runtime_capsule_stats_t cst;
    memset(&cst, 0, sizeof(cst));
    memx_runtime_capsule_stats(&cst);
    uint64_t ranks = cst.ent_count;

    uint32_t *pidxs = malloc(sizeof(uint32_t) * ranks);
    for (uint64_t r = 0; r < ranks; r++) {
        memx_runtime_capsule_pidx_at(r, &pidxs[r]);
    }

    double t4 = now_ns(&tb);
    for (uint64_t r = 0; r < ranks; r++) {
        memx_runtime_capsule_materialize(pidxs[r], dst, PAGE_SZ);
    }
    double t5 = now_ns(&tb);

    double t6 = now_ns(&tb);
    memx_runtime_capsule_materialize_v(pidxs, (uint32_t)ranks, batch, PAGE_SZ);
    double t7 = now_ns(&tb);

    double export_ms = (t1 - t0) / 1000000.0;
    double attach_ms = (t3 - t2) / 1000000.0;
    double single_ms = (t5 - t4) / 1000000.0;
    double batch_ms = (t7 - t6) / 1000000.0;

    printf("MemX capsule benchmark (%llu pages, %llu bytes)\n",
           (unsigned long long)ranks, (unsigned long long)exported);
    printf("  export      %8.2f ms\n", export_ms);
    printf("  attach      %8.2f ms\n", attach_ms);
    printf("  single x%llu %5.2f ms (%.1f us/page)\n",
           (unsigned long long)ranks, single_ms, ranks ? single_ms * 1000.0 / (double)ranks : 0.0);
    printf("  batch_v x%llu %5.2f ms (%.1f us/page, %.1fx)\n",
           (unsigned long long)ranks, batch_ms, ranks ? batch_ms * 1000.0 / (double)ranks : 0.0,
           (single_ms > 0 && batch_ms > 0) ? single_ms / batch_ms : 0.0);

    int rc = 0;
    for (uint64_t r = 0; r < ranks && rc == 0; r++) {
        memx_runtime_capsule_materialize(pidxs[r], dst, PAGE_SZ);
        if (memcmp(dst, batch + r * PAGE_SZ, PAGE_SZ) != 0) {
            fprintf(stderr, "batch != single at rank=%llu\n", (unsigned long long)r);
            rc = 7;
        }
    }
    if (rc == 0) printf("  batch == single: verified\n");

    free(pidxs);
    free(dst);
    free(batch);
    memx_runtime_capsule_detach();
    memx_runtime_context_free(ctx, w);
    memx_runtime_context_destroy(ctx);
    memx_runtime_shutdown();
    return rc;
}
