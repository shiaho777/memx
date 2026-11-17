#include "memx_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define PAGE 16384
#define NLAYERS 4
#define NKEYS 9
#define NPASSES 3

static size_t pa(size_t n) { return ((n + PAGE - 1) / PAGE) * PAGE; }

static void fill_pattern(uint8_t *p, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)(i * 31u + seed + (i >> 8));
}

typedef struct { uint8_t *data; size_t size; } Hosted;

int main(void) {
    memx_runtime_context_t *ctx = NULL;
    if (memx_runtime_context_create("stream-mat", &ctx) != 0) return 1;

    const size_t sizes[NKEYS] = { 2048, 32768, 8192, 8192, 32768, 2048, 49152, 49152, 49152 };
    Hosted h[NLAYERS][NKEYS];
    uint8_t *golden[NLAYERS][NKEYS];
    size_t total = 0;
    for (int i = 0; i < NLAYERS; i++)
        for (int k = 0; k < NKEYS; k++) total += pa(sizes[k]);
    if (memx_runtime_context_set_quota(ctx, total + 8ULL * 1024 * 1024) != 0) return 2;

    for (int i = 0; i < NLAYERS; i++) {
        for (int k = 0; k < NKEYS; k++) {
            size_t nb = sizes[k], sz = pa(nb);
            memx_runtime_tensor_desc_t d;
            memset(&d, 0, sizeof d);
            d.struct_size = sizeof d;
            d.role = MEMX_TENSOR_ROLE_WEIGHT;
            d.dtype = MEMX_TENSOR_DTYPE_BF16;
            d.layout = MEMX_TENSOR_LAYOUT_ROW_MAJOR;
            d.flags = MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY;
            d.rank = 2; d.shape[0] = 1; d.shape[1] = nb / 2;
            d.stride[0] = nb / 2; d.stride[1] = 1;
            uint8_t *p = memx_runtime_context_malloc_tensor(ctx, sz, &d);
            if (!p) return 3;
            fill_pattern(p, nb, (uint32_t)(i * 97 + k * 13 + 5));
            golden[i][k] = malloc(nb);
            memcpy(golden[i][k], p, nb);
            h[i][k].data = p; h[i][k].size = sz;
        }
    }

    memx_runtime_stats_t st;
    for (int t = 0; t < 100; t++) {
        memx_runtime_get_stats(&st);
        if (st.bytes_saved > 0) break;
        struct timespec ts = { 0, 50 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    memx_runtime_get_stats(&st);
    printf("compressed: saved=%llu bytes\n", (unsigned long long)st.bytes_saved);

    size_t slot_cap = 4 * PAGE;
    uint8_t *slots[2] = { malloc(slot_cap), malloc(slot_cap) };
    int slot_layer[2] = { -1, -1 };
    int slot_key[2] = { -1, -1 };
    long checks = 0, mat_calls = 0;

    for (int pass = 0; pass < NPASSES; pass++) {
        for (int i = 0; i < NLAYERS; i++) {
            for (int k = 0; k < NKEYS; k++) {
                int slot = (slot_layer[0] == i && slot_key[0] == k) ? 0 : 1;
                if (!(slot_layer[slot] == i && slot_key[slot] == k)) {
                    size_t nb = sizes[k];
                    int rc = memx_runtime_context_materialize_range(ctx, h[i][k].data, 0, nb,
                                                                    slots[slot], slot_cap,
                                                                    MEMX_MATERIALIZE_KEEP_COMPRESSED |
                                                                    MEMX_MATERIALIZE_ALLOW_RESIDENT);
                    if (rc != 0) {
                        fprintf(stderr, "FAIL main pass%d layer%d key%d rc=%d nb=%zu\n", pass, i, k, rc, nb);
                        return 4;
                    }
                    slot_layer[slot] = i; slot_key[slot] = k;
                    mat_calls++;
                }
                if (memcmp(slots[slot], golden[i][k], sizes[k]) != 0) {
                    fprintf(stderr, "MISMATCH pass%d layer%d key%d\n", pass, i, k);
                    return 5;
                }
                checks++;
                int ni = i, nk = k + 1;
                if (nk == NKEYS) { nk = 0; ni++; }
                if (ni < NLAYERS) {
                    int other = 1 - slot;
                    if (!(slot_layer[other] == ni && slot_key[other] == nk)) {
                        size_t nb2 = sizes[nk];
                        int rc2 = memx_runtime_context_materialize_range(ctx, h[ni][nk].data, 0, nb2,
                                                                         slots[other], slot_cap,
                                                                         MEMX_MATERIALIZE_KEEP_COMPRESSED |
                                                                         MEMX_MATERIALIZE_ALLOW_RESIDENT);
                        if (rc2 != 0) {
                            memx_runtime_stats_t fs;
                            memx_runtime_get_stats(&fs);
                            fprintf(stderr, "FAIL prefetch pass%d cur=(%d,%d) next=(%d,%d) rc=%d nb=%zu "
                                    "saved=%llu resident=%llu compressed=%llu faults=%llu\n",
                                    pass, i, k, ni, nk, rc2, nb2,
                                    (unsigned long long)fs.bytes_saved,
                                    (unsigned long long)fs.resident_pages,
                                    (unsigned long long)fs.compressed_pages,
                                    (unsigned long long)fs.faults);
                            for (size_t off = 0; off < nb2; off += PAGE) {
                                int prc = -1;
                                for (int t = 0; t < 5; t++) {
                                    prc = memx_runtime_context_materialize_range(ctx, h[ni][nk].data, off, PAGE,
                                                                                 slots[other], slot_cap,
                                                                                 MEMX_MATERIALIZE_KEEP_COMPRESSED |
                                                                                 MEMX_MATERIALIZE_ALLOW_RESIDENT);
                                    if (prc == 0) break;
                                    struct timespec ts2 = { 0, 20 * 1000 * 1000 };
                                    nanosleep(&ts2, NULL);
                                }
                                fprintf(stderr, "  page off=%zu retry_rc=%d\n", off, prc);
                            }
                            return 6;
                        }
                        slot_layer[other] = ni; slot_key[other] = nk;
                        mat_calls++;
                    }
                }
            }
        }
    }

    printf("stream materialize: %ld checks over %d passes OK, mat_calls=%ld\n",
           checks, NPASSES, mat_calls);
    for (int i = 0; i < NLAYERS; i++)
        for (int k = 0; k < NKEYS; k++) free(golden[i][k]);
    free(slots[0]); free(slots[1]);

    {
        size_t big_nb = 64 * PAGE;
        memx_runtime_tensor_desc_t d;
        memset(&d, 0, sizeof d);
        d.struct_size = sizeof d;
        d.role = MEMX_TENSOR_ROLE_WEIGHT;
        d.dtype = MEMX_TENSOR_DTYPE_BF16;
        d.layout = MEMX_TENSOR_LAYOUT_ROW_MAJOR;
        d.flags = MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY;
        d.rank = 2; d.shape[0] = 1; d.shape[1] = big_nb / 2;
        d.stride[0] = big_nb / 2; d.stride[1] = 1;
        uint8_t *bp = memx_runtime_context_malloc_tensor(ctx, big_nb, &d);
        if (!bp) return 7;
        for (size_t j = 0; j < big_nb; j++) bp[j] = (j & 1) ? (uint8_t)(j * 7 + 3) : 0;
        uint8_t *bg = malloc(big_nb);
        memcpy(bg, bp, big_nb);
        struct timespec ts3 = { 1, 0 };
        nanosleep(&ts3, NULL);
        uint8_t *bout = malloc(big_nb);
        for (int r = 0; r < 3; r++) {
            int brc = memx_runtime_context_materialize_range(ctx, bp, 0, big_nb, bout, big_nb,
                                                             MEMX_MATERIALIZE_KEEP_COMPRESSED |
                                                             MEMX_MATERIALIZE_ALLOW_RESIDENT);
            if (brc != 0 || memcmp(bout, bg, big_nb) != 0) {
                fprintf(stderr, "FAIL parallel big round%d rc=%d\n", r, brc);
                return 8;
            }
        }
        printf("parallel big: 64 pages x 3 rounds OK\n");
        free(bg); free(bout);
        memx_runtime_context_free(ctx, bp);
    }

    memx_runtime_context_destroy(ctx);
    memx_runtime_shutdown();
    printf("OK stream materialize\n");
    return 0;
}