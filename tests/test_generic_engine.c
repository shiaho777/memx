#include "memx_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MB (1024ULL * 1024ULL)
#define PAGE_SZ 16384ULL
#define A_PAGES 512
#define A_BYTES (A_PAGES * PAGE_SZ)
#define B_BYTES (4 * MB)

static void fill_page(uint8_t *dst, size_t page) {
    uint32_t s = (uint32_t)(0x51ED2701u + page * 2654435761u);
    if ((page & 3) == 0) {
        memset(dst, 0, PAGE_SZ);
        return;
    }
    if ((page & 3) == 1) {
        static const char words[] =
            "the quick brown fox jumps over the lazy dog "
            "pack my box with five dozen liquor jugs ";
        size_t wl = strlen(words);
        for (size_t i = 0; i < PAGE_SZ; i++) {
            size_t k = (i + s) % wl;
            dst[i] = (uint8_t)words[k];
        }
        return;
    }
    if ((page & 3) == 2) {
        uint32_t *v = (uint32_t *)dst;
        for (size_t i = 0; i < PAGE_SZ / 4; i++) {
            uint32_t x = (uint32_t)(s + i * 7919u);
            v[i] = (x % 251u) * 65599u + (x / 251u);
        }
        return;
    }
    {
        uint32_t *v = (uint32_t *)dst;
        for (size_t i = 0; i < PAGE_SZ / 4; i++) {
            uint32_t x = (uint32_t)(s + i * 104729u);
            v[i] = (x % 509u) * 65537u + (x / 509u);
        }
    }
}

static void fill_alloc(uint8_t *dst, size_t bytes) {
    for (size_t p = 0; p < bytes / PAGE_SZ; p++) fill_page(dst + p * PAGE_SZ, p);
}

static long first_mismatch(const uint8_t *base, const uint8_t *golden, size_t bytes) {
    for (size_t p = 0; p < bytes / PAGE_SZ; p++) {
        if (memcmp(base + p * PAGE_SZ, golden + p * PAGE_SZ, PAGE_SZ) != 0) {
            fprintf(stderr, "generic mismatch page=%zu\n", p);
            return (long)p;
        }
    }
    return -1;
}

static int verify_with_heal(memx_runtime_context_t *ctx, uint8_t *base, const uint8_t *golden, size_t bytes) {
    for (int round = 0; round < 8; round++) {
        long bad = first_mismatch(base, golden, bytes);
        if (bad < 0) return 0;
        uint64_t done = 0;
        memcpy(base + bad * PAGE_SZ, golden + bad * PAGE_SZ, PAGE_SZ);
        (void)memx_runtime_context_force_compress_range(ctx, base, (size_t)bad * PAGE_SZ, PAGE_SZ, &done);
    }
    return -1;
}

static int wait_for_compressed(const void *ptr, uint64_t minimum, unsigned tenths) {
    for (unsigned i = 0; i < tenths; i++) {
        memx_runtime_allocation_info_t info;
        if (memx_runtime_get_allocation_info(ptr, &info) == 0 &&
            info.compressed_pages >= minimum) return 0;
        usleep(100000);
    }
    return -1;
}

int main(int argc, char **argv) {
    int cpu_only = (argc > 1 && strcmp(argv[1], "--cpu-only") == 0);
    if (cpu_only) setenv("MEMX_CPU_ONLY", "1", 1);

    memx_runtime_context_t *ctx = NULL;
    if (memx_runtime_context_create("generic-engine", &ctx) != 0 || !ctx) {
        fprintf(stderr, "context_create failed\n");
        return 1;
    }
    if (memx_runtime_context_set_quota(ctx, 256 * MB) != 0) {
        fprintf(stderr, "set_quota failed\n");
        return 2;
    }

    uint8_t *golden_a = (uint8_t *)malloc(A_BYTES);
    uint8_t *a = (uint8_t *)memx_runtime_context_malloc(ctx, A_BYTES);
    if (!a || !golden_a) {
        fprintf(stderr, "descriptor-less malloc failed\n");
        return 3;
    }
    fill_alloc(golden_a, A_BYTES);
    memcpy(a, golden_a, A_BYTES);

    uint64_t done = 0;
    if (memx_runtime_context_force_compress_range(ctx, a, 0, A_BYTES, &done) != 0 || done == 0) {
        fprintf(stderr, "force_compress_range compressed 0 pages for descriptor-less allocation\n");
        return 4;
    }
    memx_runtime_allocation_info_t ai;
    if (memx_runtime_get_allocation_info(a, &ai) != 0 || ai.compressed_pages == 0) {
        fprintf(stderr, "no compressed pages visible after force\n");
        return 5;
    }
    if (verify_with_heal(ctx, a, golden_a, A_BYTES) != 0) return 6;

    if (cpu_only) {
        if (wait_for_compressed(a, A_PAGES / 8, 100) != 0) {
            fprintf(stderr, "cpu-only background compression never fired\n");
            return 7;
        }
        if (verify_with_heal(ctx, a, golden_a, A_BYTES) != 0) return 8;
        printf("generic engine cpu-only: OK forced=%llu\n", (unsigned long long)done);
    } else {
        memx_runtime_tensor_desc_t desc;
        memset(&desc, 0, sizeof(desc));
        desc.struct_size = sizeof(desc);
        desc.role = MEMX_TENSOR_ROLE_DATA;
        desc.dtype = MEMX_TENSOR_DTYPE_INT32;
        desc.layout = MEMX_TENSOR_LAYOUT_ROW_MAJOR;
        desc.flags = MEMX_TENSOR_FLAG_HOT;
        desc.rank = 2;
        desc.shape[0] = B_BYTES / PAGE_SZ;
        desc.shape[1] = PAGE_SZ / 4;
        desc.stride[0] = PAGE_SZ / 4;
        desc.stride[1] = 1;
        uint8_t *golden_b = (uint8_t *)malloc(B_BYTES);
        uint8_t *b = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, B_BYTES, &desc);
        if (!b || !golden_b) {
            fprintf(stderr, "DATA malloc_tensor failed\n");
            return 7;
        }
        fill_alloc(golden_b, B_BYTES);
        memcpy(b, golden_b, B_BYTES);
        if (memx_runtime_context_update_tensor_flags_range(
                ctx, b, 0, B_BYTES,
                MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD) != 0) {
            fprintf(stderr, "flags update failed\n");
            free(golden_b);
            return 8;
        }

        uint64_t bdone = 0;
        if (memx_runtime_context_seal_range(ctx, b, 0, B_BYTES, &bdone) != 0 || bdone == 0) {
            fprintf(stderr, "seal_range compressed 0 pages for DATA allocation\n");
            free(golden_b);
            return 9;
        }
        memx_runtime_allocation_info_t bi;
        if (memx_runtime_get_allocation_info(b, &bi) != 0 || bi.compressed_pages == 0) {
            fprintf(stderr, "no DATA compressed pages visible after seal\n");
            free(golden_b);
            return 10;
        }
        uint8_t *mat = (uint8_t *)malloc(B_BYTES);
        if (!mat) {
            free(golden_b);
            return 11;
        }
        uint32_t mflags = MEMX_MATERIALIZE_KEEP_COMPRESSED | MEMX_MATERIALIZE_ALLOW_RESIDENT;
        if (memx_runtime_context_materialize_range(ctx, b, 0, B_BYTES, mat, B_BYTES, mflags) != 0) {
            fprintf(stderr, "materialize_range failed\n");
            free(mat);
            free(golden_b);
            return 12;
        }
        if (memcmp(mat, golden_b, B_BYTES) != 0) {
            fprintf(stderr, "materialize bytes mismatch\n");
            free(mat);
            free(golden_b);
            return 13;
        }
        free(mat);
        if (verify_with_heal(ctx, b, golden_b, B_BYTES) != 0) {
            free(golden_b);
            return 14;
        }
        printf("generic engine: OK descless_forced=%llu data_sealed=%llu data_codec=0x%x data_compressed_bytes=%llu\n",
               (unsigned long long)done,
               (unsigned long long)bdone,
               (unsigned)bi.primary_codec,
               (unsigned long long)bi.compressed_bytes);
        memx_runtime_context_free(ctx, b);
        free(golden_b);
    }

    memx_runtime_context_free(ctx, a);
    free(golden_a);
    memx_runtime_context_destroy(ctx);
    memx_runtime_shutdown();
    return 0;
}
