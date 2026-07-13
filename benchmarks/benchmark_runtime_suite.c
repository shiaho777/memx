#include "memx_runtime.h"

#include <errno.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MB (1024ULL * 1024ULL)
#define PAGE_SZ 16384ULL

typedef struct benchmark_case {
    const char *name;
    uint8_t *ptr;
    size_t size_bytes;
} benchmark_case_t;

static long long get_phys_footprint_mb(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) {
        return -1;
    }
    return (long long)(info.phys_footprint / MB);
}

static void fill_sparse_llm(uint8_t *ptr, size_t size_bytes) {
    memset(ptr, 0, size_bytes);
    float *weights = (float *)ptr;
    size_t nfloats = size_bytes / sizeof(float);
    srand(42);
    for (size_t i = nfloats / 2; i < nfloats; i += 128) {
        weights[i] = (float)(rand() % 1024) / 1024.0f;
    }
}

static int verify_sparse_llm(const uint8_t *ptr, size_t size_bytes) {
    const float *weights = (const float *)ptr;
    size_t nfloats = size_bytes / sizeof(float);
    srand(42);
    for (size_t i = 0; i < nfloats / 2; i += 128) {
        if (weights[i] != 0.0f) return -1;
    }
    for (size_t i = nfloats / 2; i < nfloats; i += 128) {
        float expected = (float)(rand() % 1024) / 1024.0f;
        if (weights[i] != expected) return -1;
    }
    return 0;
}

static void fill_dedup_pages(uint8_t *ptr, size_t size_bytes) {
    uint8_t templates[8][PAGE_SZ];
    for (size_t t = 0; t < 8; t++) {
        memset(templates[t], 0, PAGE_SZ);
        for (size_t i = PAGE_SZ / 2; i < PAGE_SZ; i += 64) {
            templates[t][i] = (uint8_t)((t * 31 + i / 64) & 0xFF);
        }
    }
    for (size_t off = 0; off < size_bytes; off += PAGE_SZ) {
        memcpy(ptr + off, templates[(off / PAGE_SZ) % 8], PAGE_SZ);
    }
}

static int verify_dedup_pages(const uint8_t *ptr, size_t size_bytes) {
    for (size_t off = 0; off < size_bytes; off += PAGE_SZ) {
        size_t t = (off / PAGE_SZ) % 8;
        for (size_t i = PAGE_SZ / 2; i < PAGE_SZ; i += 64) {
            uint8_t expected = (uint8_t)((t * 31 + i / 64) & 0xFF);
            if (ptr[off + i] != expected) return -1;
        }
    }
    return 0;
}

static void fill_mixed_cache(uint8_t *ptr, size_t size_bytes) {
    const char *html = "<html><body><div class='cache'>memx</div></body></html>";
    const char *json = "{\"items\":[1,2,3],\"ok\":true}";
    size_t html_len = strlen(html);
    size_t json_len = strlen(json);
    size_t pos = 0;
    srand(7);
    while (pos < size_bytes) {
        size_t mode = (size_t)(rand() % 3);
        if (mode == 0) {
            size_t run = size_bytes - pos > PAGE_SZ ? PAGE_SZ : size_bytes - pos;
            memset(ptr + pos, 0, run);
            for (size_t i = 0; i < run; i += 128) {
                ptr[pos + i] = (uint8_t)((i + pos / PAGE_SZ) & 0xFF);
            }
            pos += run;
        } else if (mode == 1) {
            size_t copy = size_bytes - pos > html_len ? html_len : size_bytes - pos;
            memcpy(ptr + pos, html, copy);
            pos += copy;
        } else {
            size_t copy = size_bytes - pos > json_len ? json_len : size_bytes - pos;
            memcpy(ptr + pos, json, copy);
            pos += copy;
        }
    }
}

static int verify_mixed_cache(const uint8_t *ptr, size_t size_bytes) {
    const char *html = "<html><body><div class='cache'>memx</div></body></html>";
    const char *json = "{\"items\":[1,2,3],\"ok\":true}";
    size_t html_len = strlen(html);
    size_t json_len = strlen(json);
    size_t pos = 0;
    srand(7);
    while (pos < size_bytes) {
        size_t mode = (size_t)(rand() % 3);
        if (mode == 0) {
            size_t run = size_bytes - pos > PAGE_SZ ? PAGE_SZ : size_bytes - pos;
            for (size_t i = 0; i < run; i++) {
                uint8_t expected = (i % 128 == 0) ? (uint8_t)((i + pos / PAGE_SZ) & 0xFF) : 0;
                if (ptr[pos + i] != expected) return -1;
            }
            pos += run;
        } else if (mode == 1) {
            size_t copy = size_bytes - pos > html_len ? html_len : size_bytes - pos;
            if (memcmp(ptr + pos, html, copy) != 0) return -1;
            pos += copy;
        } else {
            size_t copy = size_bytes - pos > json_len ? json_len : size_bytes - pos;
            if (memcmp(ptr + pos, json, copy) != 0) return -1;
            pos += copy;
        }
    }
    return 0;
}

static void print_runtime_stats(const memx_runtime_stats_t *stats, const memx_runtime_pressure_t *pressure) {
    printf("  runtime: compressed=%llu faults=%llu saved_mb=%llu frag=%u%% pressure=%u%% free_pages=%llu\n",
           (unsigned long long)stats->compressions,
           (unsigned long long)stats->faults,
           (unsigned long long)(stats->bytes_saved / MB),
           (unsigned)pressure->pool_fragmentation_percent,
           (unsigned)pressure->pool_pressure_percent,
           (unsigned long long)pressure->free_pages);
}

static int measure_random_faults(const benchmark_case_t *bc, double ns_per_tick) {
    size_t npages = bc->size_bytes / PAGE_SZ;
    double samples_us[128];
    size_t collected = 0;

    srand(1234);
    for (size_t i = 0; i < 512 && collected < 128; i++) {
        size_t page = (size_t)rand() % npages;
        uint64_t t0 = mach_absolute_time();
        volatile uint8_t value = bc->ptr[page * PAGE_SZ];
        uint64_t t1 = mach_absolute_time();
        (void)value;

        double us = ((double)(t1 - t0) * ns_per_tick) / 1000.0;
        if (us > 1.0) {
            samples_us[collected++] = us;
        }
    }

    if (collected == 0) {
        printf("  faults: no distinct cold-fault samples captured\n");
        return 0;
    }

    for (size_t i = 0; i + 1 < collected; i++) {
        for (size_t j = i + 1; j < collected; j++) {
            if (samples_us[j] < samples_us[i]) {
                double tmp = samples_us[i];
                samples_us[i] = samples_us[j];
                samples_us[j] = tmp;
            }
        }
    }

    double p50 = samples_us[collected / 2];
    double p95 = samples_us[(collected * 95 / 100) < collected ? (collected * 95 / 100) : (collected - 1)];
    double max = samples_us[collected - 1];
    printf("  faults: samples=%zu p50=%.1f us p95=%.1f us max=%.1f us\n", collected, p50, p95, max);
    return 0;
}

static int measure_sequential_sweep(const benchmark_case_t *bc, double ns_per_tick) {
    uint64_t t0 = mach_absolute_time();
    for (size_t off = 0; off < bc->size_bytes; off += PAGE_SZ) {
        volatile uint8_t value = bc->ptr[off];
        (void)value;
    }
    uint64_t t1 = mach_absolute_time();

    double total_ns = (double)(t1 - t0) * ns_per_tick;
    double throughput_mb_s = ((double)bc->size_bytes / MB) / (total_ns / 1e9);
    double page_us = total_ns / ((double)(bc->size_bytes / PAGE_SZ) * 1000.0);
    printf("  sequential: %.1f MB/s, %.1f us/page\n", throughput_mb_s, page_us);
    return 0;
}

static int run_case(
    memx_runtime_context_t *ctx,
    const char *name,
    size_t size_bytes,
    void (*fill)(uint8_t *, size_t),
    int (*verify)(const uint8_t *, size_t),
    double ns_per_tick
) {
    benchmark_case_t bc = {.name = name, .ptr = NULL, .size_bytes = size_bytes};
    memx_runtime_stats_t stats;
    memx_runtime_pressure_t pressure;
    memx_runtime_context_stats_t ctx_stats;

    errno = 0;
    bc.ptr = (uint8_t *)memx_runtime_context_malloc(ctx, bc.size_bytes);
    if (!bc.ptr) {
        fprintf(stderr, "%s: memx allocation failed errno=%d\n", bc.name, errno);
        return 1;
    }

    fill(bc.ptr, bc.size_bytes);
    long long fp_before = get_phys_footprint_mb();
    printf("\n[%s]\n", bc.name);
    printf("  allocated=%llu MB footprint_before_wait=%lld MB\n",
           (unsigned long long)(bc.size_bytes / MB),
           fp_before);

    sleep(3);

    if (memx_runtime_get_stats(&stats) != 0 ||
        memx_runtime_get_pressure(&pressure) != 0 ||
        memx_runtime_context_get_stats(ctx, &ctx_stats) != 0) {
        fprintf(stderr, "%s: runtime telemetry read failed\n", bc.name);
        memx_runtime_context_free(ctx, bc.ptr);
        return 2;
    }

    print_runtime_stats(&stats, &pressure);
    printf("  context: live=%llu used_mb=%llu peak_mb=%llu quota_fail=%llu\n",
           (unsigned long long)ctx_stats.allocations_live,
           (unsigned long long)(ctx_stats.bytes_in_use / MB),
           (unsigned long long)(ctx_stats.peak_bytes_in_use / MB),
           (unsigned long long)ctx_stats.allocation_failures_quota);

    if (measure_random_faults(&bc, ns_per_tick) != 0 ||
        measure_sequential_sweep(&bc, ns_per_tick) != 0) {
        memx_runtime_context_free(ctx, bc.ptr);
        return 3;
    }

    if (verify(bc.ptr, bc.size_bytes) != 0) {
        fprintf(stderr, "%s: integrity check failed\n", bc.name);
        memx_runtime_context_free(ctx, bc.ptr);
        return 4;
    }

    memx_runtime_context_free(ctx, bc.ptr);
    return 0;
}

int main(void) {
    memx_runtime_context_t *ctx = NULL;
    memx_runtime_stats_t stats;
    memx_runtime_pressure_t pressure;
    mach_timebase_info_data_t timebase;
    double ns_per_tick;
    uint64_t reclaimed = 0;

    if (memx_runtime_context_create("benchmark-suite", &ctx) != 0 || !ctx) {
        fprintf(stderr, "memx_runtime_context_create failed\n");
        return 1;
    }
    if (memx_runtime_context_set_quota(ctx, 1536 * MB) != 0) {
        fprintf(stderr, "memx_runtime_context_set_quota failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 2;
    }

    mach_timebase_info(&timebase);
    ns_per_tick = (double)timebase.numer / (double)timebase.denom;

    printf("MemX runtime benchmark suite\n");
    printf("quota=%llu MB page_size=%llu KB\n",
           (unsigned long long)(1536),
           (unsigned long long)(PAGE_SZ / 1024));

    if (run_case(ctx, "Sparse LLM Weights", 512 * MB, fill_sparse_llm, verify_sparse_llm, ns_per_tick) != 0 ||
        run_case(ctx, "Deduplicated Page Cache", 256 * MB, fill_dedup_pages, verify_dedup_pages, ns_per_tick) != 0 ||
        run_case(ctx, "Mixed Web/Object Cache", 256 * MB, fill_mixed_cache, verify_mixed_cache, ns_per_tick) != 0) {
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 3;
    }

    if (memx_runtime_reclaim(&reclaimed) != 0 ||
        memx_runtime_get_stats(&stats) != 0 ||
        memx_runtime_get_pressure(&pressure) != 0) {
        fprintf(stderr, "runtime summary read failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 4;
    }

    printf("\nsummary: compressed=%llu faults=%llu saved_mb=%llu reclaim_mb=%llu frag=%u%% pressure=%u%%\n",
           (unsigned long long)stats.compressions,
           (unsigned long long)stats.faults,
           (unsigned long long)(stats.bytes_saved / MB),
           (unsigned long long)(reclaimed / MB),
           (unsigned)pressure.pool_fragmentation_percent,
           (unsigned)pressure.pool_pressure_percent);

    memx_runtime_context_destroy(ctx);
    memx_runtime_shutdown();
    return 0;
}
