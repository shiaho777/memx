#include "memx_runtime.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MB (1024ULL * 1024ULL)

typedef struct worker_arg {
    memx_runtime_context_t *ctx;
    int thread_id;
    size_t alloc_size;
    int alloc_count;
    int failed;
} worker_arg_t;

static void fill_pattern(uint8_t *ptr, size_t size, int thread_id, int slot) {
    for (size_t off = 0; off < size; off += 256) {
        ptr[off] = (uint8_t)thread_id;
        ptr[off + 1] = (uint8_t)slot;
        ptr[off + 2] = (uint8_t)(off / 256);
    }
}

static int verify_pattern(const uint8_t *ptr, size_t size, int thread_id, int slot) {
    for (size_t off = 0; off < size; off += 256) {
        if (ptr[off] != (uint8_t)thread_id ||
            ptr[off + 1] != (uint8_t)slot ||
            ptr[off + 2] != (uint8_t)(off / 256)) {
            return -1;
        }
    }
    return 0;
}

static void *worker_main(void *opaque) {
    worker_arg_t *arg = (worker_arg_t *)opaque;
    uint8_t **allocs = calloc((size_t)arg->alloc_count, sizeof(*allocs));
    if (!allocs) {
        arg->failed = 1;
        return NULL;
    }

    for (int i = 0; i < arg->alloc_count; i++) {
        allocs[i] = (uint8_t *)memx_runtime_context_malloc(arg->ctx, arg->alloc_size);
        if (!allocs[i]) {
            if (errno != ENOMEM) arg->failed = 1;
            break;
        }
        fill_pattern(allocs[i], arg->alloc_size, arg->thread_id, i);
        usleep(30000);
    }

    sleep(1);

    for (int i = 0; i < arg->alloc_count; i++) {
        if (!allocs[i]) continue;
        if (verify_pattern(allocs[i], arg->alloc_size, arg->thread_id, i) != 0) {
            arg->failed = 1;
            break;
        }
        volatile uint8_t sample = allocs[i][0];
        (void)sample;
    }

    for (int i = 0; i < arg->alloc_count; i++) {
        if (allocs[i]) memx_runtime_context_free(arg->ctx, allocs[i]);
    }
    free(allocs);
    return NULL;
}

int main(void) {
    memx_runtime_context_t *ctx = NULL;
    memx_runtime_context_stats_t ctx_stats;
    memx_runtime_pressure_t pressure;
    memx_runtime_stats_t stats;
    pthread_t threads[4];
    worker_arg_t args[4];

    if (memx_runtime_context_create("stress", &ctx) != 0 || !ctx) {
        fprintf(stderr, "memx_runtime_context_create failed\n");
        return 1;
    }
    if (memx_runtime_context_set_quota(ctx, 768 * MB) != 0) {
        fprintf(stderr, "memx_runtime_context_set_quota failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 2;
    }

    for (int i = 0; i < 4; i++) {
        args[i].ctx = ctx;
        args[i].thread_id = i + 1;
        args[i].alloc_size = 32 * MB;
        args[i].alloc_count = 4;
        args[i].failed = 0;
        pthread_create(&threads[i], NULL, worker_main, &args[i]);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }

    for (int i = 0; i < 4; i++) {
        if (args[i].failed) {
            fprintf(stderr, "worker %d failed\n", i + 1);
            memx_runtime_context_destroy(ctx);
            memx_runtime_shutdown();
            return 3;
        }
    }

    if (memx_runtime_context_get_stats(ctx, &ctx_stats) != 0 ||
        memx_runtime_get_pressure(&pressure) != 0 ||
        memx_runtime_get_stats(&stats) != 0) {
        fprintf(stderr, "runtime telemetry read failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 4;
    }

    printf("context stress: live=%llu total=%llu peak_mb=%llu pressure_events=%llu compressed=%llu faults=%llu frag=%u%%\n",
           (unsigned long long)ctx_stats.allocations_live,
           (unsigned long long)ctx_stats.allocations_total,
           (unsigned long long)(ctx_stats.peak_bytes_in_use / MB),
           (unsigned long long)ctx_stats.pressure_events,
           (unsigned long long)stats.compressions,
           (unsigned long long)stats.faults,
           (unsigned)pressure.pool_fragmentation_percent);

    memx_runtime_context_destroy(ctx);
    memx_runtime_shutdown();
    return 0;
}
