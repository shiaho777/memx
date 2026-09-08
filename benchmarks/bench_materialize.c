#include "memx_runtime.h"

#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MB (1024ULL * 1024ULL)
#define PAGE_SZ 16384ULL

// Fills a BF16 weight page with a structured but compressible pattern.
static void fill_weights(uint8_t *ptr, size_t size_bytes) {
    for (size_t i = 0; i < size_bytes / 2; i++) {
        uint16_t v = (uint16_t)(0x3800 + ((i >> 9) & 0x7F));
        v ^= (uint16_t)((i * 7 + i / 311) & 0x1F);
        ptr[i * 2] = (uint8_t)(v & 0xFF);
        ptr[i * 2 + 1] = (uint8_t)(v >> 8);
    }
}

static int verify_weights(const uint8_t *ptr, size_t size_bytes) {
    for (size_t i = 0; i < size_bytes / 2; i++) {
        uint16_t expected = (uint16_t)(0x3800 + ((i >> 9) & 0x7F));
        expected ^= (uint16_t)((i * 7 + i / 311) & 0x1F);
        uint16_t got = (uint16_t)ptr[i * 2] | ((uint16_t)ptr[i * 2 + 1] << 8);
        if (got != expected) {
            fprintf(stderr, "bitexact mismatch half=%zu got=0x%04x expected=0x%04x\n", i, got, expected);
            return -1;
        }
    }
    return 0;
}

static double now_ns(mach_timebase_info_data_t *tb) {
    uint64_t t = mach_absolute_time();
    return (double)t * (double)tb->numer / (double)tb->denom;
}

int main(void) {
    memx_runtime_context_t *ctx = NULL;
    const size_t size_bytes = 64 * MB;
    const size_t strip = 512 * 2;      // one 512-row strip of fp16
    const size_t strip_len = 64 * PAGE_SZ; // 1MB materialize chunk

    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);

    if (memx_runtime_context_create("materialize-bench", &ctx) != 0 || !ctx) {
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
    desc.dtype = MEMX_TENSOR_DTYPE_BF16;
    desc.layout = MEMX_TENSOR_LAYOUT_ROW_MAJOR;
    desc.flags = MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD;
    desc.rank = 2;
    desc.shape[0] = size_bytes / (512 * 2);
    desc.shape[1] = 512;
    desc.stride[0] = 512;
    desc.stride[1] = 1;

    uint8_t *w = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, size_bytes, &desc);
    if (!w) {
        fprintf(stderr, "malloc_tensor failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 3;
    }
    fill_weights(w, size_bytes);
    (void)strip;

    uint64_t compressed = 0;
    if (memx_runtime_context_force_compress_range(ctx, w, 0, size_bytes, &compressed) != 0 || compressed == 0) {
        fprintf(stderr, "force_compress failed compressed=%llu\n", (unsigned long long)compressed);
        memx_runtime_context_free(ctx, w);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 4;
    }

    uint8_t *dst = (uint8_t *)malloc(strip_len);
    if (!dst) {
        memx_runtime_context_free(ctx, w);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 5;
    }

    const uint32_t flags = MEMX_MATERIALIZE_KEEP_COMPRESSED | MEMX_MATERIALIZE_ALLOW_RESIDENT;

    // Warm pass: populate the ND page cache and measure cache-hit path.
    double t0 = now_ns(&tb);
    for (size_t off = 0; off < size_bytes; off += strip_len) {
        if (memx_runtime_context_materialize_range(ctx, w, off, strip_len, dst, strip_len, flags) != 0) {
            fprintf(stderr, "materialize warm failed off=%zu\n", off);
            free(dst);
            memx_runtime_context_free(ctx, w);
            memx_runtime_context_destroy(ctx);
            memx_runtime_shutdown();
            return 6;
        }
    }
    double t1 = now_ns(&tb);

    // Cold pass is the same call on a second allocation (no cache yet).
    uint8_t *w2 = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, size_bytes, &desc);
    if (w2) {
        fill_weights(w2, size_bytes);
        memx_runtime_context_force_compress_range(ctx, w2, 0, size_bytes, &compressed);
    }
    double tc0 = now_ns(&tb);
    if (w2) {
        for (size_t off = 0; off < size_bytes; off += strip_len) {
            memx_runtime_context_materialize_range(ctx, w2, off, strip_len, dst, strip_len, flags);
        }
    }
    double tc1 = now_ns(&tb);

    // Warm re-measure of w (cache warm from first pass).
    double t2 = now_ns(&tb);
    for (size_t off = 0; off < size_bytes; off += strip_len) {
        memx_runtime_context_materialize_range(ctx, w, off, strip_len, dst, strip_len, flags);
    }
    double t3 = now_ns(&tb);

    double cold_ms = (tc1 - tc0) / 1000000.0;
    double warm_ms = (t3 - t2) / 1000000.0;
    double warm1_ms = (t1 - t0) / 1000000.0;
    double mb = (double)size_bytes / (1024.0 * 1024.0);

    printf("MemX materialize benchmark (64MB bf16 weight, keep-compressed)\n");
    printf("  cold  %.2f ms (%.0f MB/s)\n", cold_ms, cold_ms > 0 ? mb * 1000.0 / cold_ms : 0.0);
    printf("  warm1 %.2f ms (%.0f MB/s)\n", warm1_ms, warm1_ms > 0 ? mb * 1000.0 / warm1_ms : 0.0);
    printf("  warm2 %.2f ms (%.0f MB/s)\n", warm_ms, warm_ms > 0 ? mb * 1000.0 / warm_ms : 0.0);

    int rc = 0;
    for (size_t off = 0; off < size_bytes && rc == 0; off += strip_len) {
        memx_runtime_context_materialize_range(ctx, w, off, strip_len, dst, strip_len, flags);
        for (size_t k = 0; k + 1 < strip_len; k += 2) {
            size_t g = (off + k) / 2;
            uint16_t expected = (uint16_t)(0x3800 + ((g >> 9) & 0x7F));
            expected ^= (uint16_t)((g * 7 + g / 311) & 0x1F);
            uint16_t got = (uint16_t)dst[k] | ((uint16_t)dst[k + 1] << 8);
            if (got != expected) {
                fprintf(stderr, "materialize bitexact mismatch off=%zu k=%zu got=0x%04x expected=0x%04x\n",
                        off, k, got, expected);
                rc = 7;
                break;
            }
        }
    }
    if (rc == 0) printf("  bitexact: all strips verified\n");

    free(dst);
    if (w2) memx_runtime_context_free(ctx, w2);
    memx_runtime_context_free(ctx, w);
    memx_runtime_context_destroy(ctx);
    memx_runtime_shutdown();
    return rc;
}
