#include "memx_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define PAGE 16384
#define NTENSORS 8
#define TSIZE (256 * 1024)
#define ROUNDS 20

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int compressible = argc > 1 && argv[1][0] == 'c';
    memx_runtime_context_t *ctx = NULL;
    if (memx_runtime_context_create("bench-mat", &ctx) != 0) return 1;
    if (memx_runtime_context_set_quota(ctx, 64ULL * 1024 * 1024) != 0) return 2;

    uint8_t *p[NTENSORS];
    uint8_t *golden[NTENSORS];
    for (int i = 0; i < NTENSORS; i++) {
        memx_runtime_tensor_desc_t d;
        memset(&d, 0, sizeof d);
        d.struct_size = sizeof d;
        d.role = MEMX_TENSOR_ROLE_WEIGHT;
        d.dtype = MEMX_TENSOR_DTYPE_BF16;
        d.layout = MEMX_TENSOR_LAYOUT_ROW_MAJOR;
        d.flags = MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY;
        d.rank = 2; d.shape[0] = 1; d.shape[1] = TSIZE / 2;
        d.stride[0] = TSIZE / 2; d.stride[1] = 1;
        p[i] = memx_runtime_context_malloc_tensor(ctx, TSIZE, &d);
        if (!p[i]) return 3;
        if (compressible) {
            for (size_t j = 0; j < TSIZE; j++) p[i][j] = (uint8_t)((j >> 6) + i);
        } else {
            for (size_t j = 0; j < TSIZE; j++) p[i][j] = (uint8_t)(j * 31 + i * 13 + (j >> 8));
        }
    }
    for (int i = 0; i < NTENSORS; i++) {
        golden[i] = malloc(TSIZE);
        memcpy(golden[i], p[i], TSIZE);
    }

    struct timespec ts = { 1, 0 };
    nanosleep(&ts, NULL);
    memx_runtime_stats_t st;
    memx_runtime_get_stats(&st);

    uint8_t *out = malloc(TSIZE);
    double t0 = now_s();
    long bytes = 0;
    int ok = 1;
    for (int r = 0; r < ROUNDS; r++) {
        for (int i = 0; i < NTENSORS; i++) {
            if (memx_runtime_context_materialize_range(ctx, p[i], 0, TSIZE, out, TSIZE,
                                                       MEMX_MATERIALIZE_KEEP_COMPRESSED |
                                                       MEMX_MATERIALIZE_ALLOW_RESIDENT) != 0) return 4;
            if (memcmp(out, golden[i], TSIZE) != 0) ok = 0;
            bytes += TSIZE;
        }
    }
    double el = now_s() - t0;
    printf("mode=%s saved=%lluKB rounds=%d bytes=%.1fMB wall=%.3fs throughput=%.0fMB/s verify=%s\n",
           compressible ? "compressible" : "incompressible",
           (unsigned long long)(st.bytes_saved / 1024),
           ROUNDS, bytes / 1048576.0, el, bytes / 1048576.0 / el, ok ? "OK" : "FAIL");
    free(out);
    for (int i = 0; i < NTENSORS; i++) free(golden[i]);
    memx_runtime_context_destroy(ctx);
    memx_runtime_shutdown();
    return ok ? 0 : 5;
}