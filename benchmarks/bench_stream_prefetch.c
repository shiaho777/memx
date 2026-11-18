#include "memx_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define PAGE 16384
#define NLAYERS 16
#define LSIZE (256 * 1024)
#define ROUNDS 12
#define COMPUTE_US 300

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void busy_us(long us) {
    double t0 = now_s();
    volatile double acc = 1.0;
    while ((now_s() - t0) * 1e6 < us) acc = acc * 1.000001 + 0.5;
    (void)acc;
}

int main(int argc, char **argv) {
    int prefetch = argc > 1 && argv[1][0] == 'p';
    memx_runtime_context_t *ctx = NULL;
    if (memx_runtime_context_create("bench-stream-pf", &ctx) != 0) return 1;
    if (memx_runtime_context_set_quota(ctx, 64ULL * 1024 * 1024) != 0) return 2;

    uint8_t *p[NLAYERS];
    uint8_t *golden[NLAYERS];
    for (int i = 0; i < NLAYERS; i++) {
        memx_runtime_tensor_desc_t d;
        memset(&d, 0, sizeof d);
        d.struct_size = sizeof d;
        d.role = MEMX_TENSOR_ROLE_WEIGHT;
        d.dtype = MEMX_TENSOR_DTYPE_BF16;
        d.layout = MEMX_TENSOR_LAYOUT_ROW_MAJOR;
        d.flags = MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY;
        d.rank = 2; d.shape[0] = 1; d.shape[1] = LSIZE / 2;
        d.stride[0] = LSIZE / 2; d.stride[1] = 1;
        p[i] = memx_runtime_context_malloc_tensor(ctx, LSIZE, &d);
        if (!p[i]) return 3;
        for (size_t j = 0; j < LSIZE; j++) p[i][j] = (j & 1) ? (uint8_t)(j * 7 + i) : 0;
        golden[i] = malloc(LSIZE);
        memcpy(golden[i], p[i], LSIZE);
    }

    struct timespec ts = { 1, 0 };
    nanosleep(&ts, NULL);

    uint8_t *out = malloc(LSIZE);
    int ok = 1;
    double t0 = now_s();
    for (int r = 0; r < ROUNDS; r++) {
        for (int i = 0; i < NLAYERS; i++) {
            if (memx_runtime_context_materialize_range(ctx, p[i], 0, LSIZE, out, LSIZE,
                                                       MEMX_MATERIALIZE_KEEP_COMPRESSED |
                                                       MEMX_MATERIALIZE_ALLOW_RESIDENT) != 0) return 4;
            if (memcmp(out, golden[i], LSIZE) != 0) ok = 0;
            busy_us(COMPUTE_US);
            if (prefetch && i + 2 < NLAYERS) {
                (void)memx_runtime_context_prefetch_range(ctx, p[i + 2], 0, LSIZE);
            }
        }
    }
    double el = now_s() - t0;
    double per_layer_ms = el * 1000.0 / (ROUNDS * NLAYERS);
    memx_runtime_stats_t st;
    memx_runtime_get_stats(&st);
    printf("mode=%s layers=%d rounds=%d wall=%.3fs per_layer=%.3fms prefetch_hits=%llu verify=%s\n",
           prefetch ? "prefetch" : "no-prefetch", NLAYERS, ROUNDS, el, per_layer_ms,
           (unsigned long long)st.prefetch_hits, ok ? "OK" : "FAIL");
    free(out);
    for (int i = 0; i < NLAYERS; i++) free(golden[i]);
    memx_runtime_context_destroy(ctx);
    memx_runtime_shutdown();
    return ok ? 0 : 5;
}