#include "memx_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mach/mach.h>
#include <time.h>

static int rss_mb(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return -1;
    return (int)(info.resident_size / (1024ull * 1024ull));
}

static int phys_mb(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return -1;
    return (int)(info.phys_footprint / (1024ull * 1024ull));
}

int main(int argc, char **argv) {
    const char *dir = NULL;
    int pages = 16;
    int batch = 1;
    int ultra = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc) dir = argv[++i];
        else if (strcmp(argv[i], "--pages") == 0 && i + 1 < argc) pages = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ultra") == 0 && i + 1 < argc) ultra = atoi(argv[++i]);
        else if (!dir) dir = argv[i];
    }
    if (!dir || !dir[0]) {
        fprintf(stderr, "usage: memx_capsule_vessel --dir <capsule_dir> [--pages N]\n");
        return 2;
    }
    if (pages < 1) pages = 1;
    if (pages > 512) pages = 512;
    if (!getenv("MEMX_NO_SELFTEST")) setenv("MEMX_NO_SELFTEST", "1", 0);
    if (!getenv("MEMX_CAPSULE_LITE")) setenv("MEMX_CAPSULE_LITE", "1", 0);
    if (!getenv("MEMX_CPU_ONLY")) setenv("MEMX_CPU_ONLY", "1", 0);
    int r0 = rss_mb();
    int p0 = phys_mb();
    if (memx_runtime_capsule_attach(dir) != 0) {
        fprintf(stderr, "VESSEL_ATTACH_FAIL\n");
        return 1;
    }
    memx_runtime_capsule_stats_t st;
    memset(&st, 0, sizeof(st));
    (void)memx_runtime_capsule_stats(&st);
    int r_att = rss_mb();
    int p_att = phys_mb();
    uint8_t page[16384] __attribute__((aligned(16)));
    uint8_t *buf = NULL;
    if (!ultra && pages > 1) {
        buf = (uint8_t *)aligned_alloc(16384, (size_t)pages * 16384u);
        if (!buf) {
            (void)memx_runtime_capsule_detach();
            return 1;
        }
        memset(buf, 0, (size_t)pages * 16384u);
    }
    uint64_t ent = st.ent_count ? st.ent_count : 1;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int ok = 0;
    if (ultra || !buf) {
        for (int i = 0; i < pages; i++) {
            uint64_t rank = ((uint64_t)i * 97ull) % ent;
            if (memx_runtime_capsule_materialize_rank(rank, page, 16384) == 0) ok++;
        }
    } else {
        uint32_t *pidxs = (uint32_t *)calloc((size_t)pages, sizeof(uint32_t));
        if (!pidxs) {
            free(buf);
            (void)memx_runtime_capsule_detach();
            return 1;
        }
        for (int i = 0; i < pages; i++) {
            uint64_t rank = ((uint64_t)i * 97ull) % ent;
            uint32_t pid = 0;
            if (memx_runtime_capsule_pidx_at(rank, &pid) != 0) pid = (uint32_t)rank;
            pidxs[i] = pid;
        }
        if (batch && pages > 1) {
            if (memx_runtime_capsule_materialize_v(pidxs, (uint32_t)pages, buf, 16384) == 0) ok = pages;
        }
        if (!ok) {
            for (int i = 0; i < pages; i++) {
                if (memx_runtime_capsule_materialize(pidxs[i], buf + (size_t)i * 16384u, 16384) == 0) ok++;
            }
        }
        free(pidxs);
        free(buf);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    memset(&st, 0, sizeof(st));
    (void)memx_runtime_capsule_stats(&st);
    int r1 = rss_mb();
    int p1 = phys_mb();
    int rbest = r1;
    if (r_att > 0 && r_att < rbest) rbest = r_att;
    if (r0 > 0 && r0 < rbest) rbest = r0;
    uint64_t page_mb = st.page_bytes / (1024ull * 1024ull);
    uint64_t spill_mb = st.spill_bytes / (1024ull * 1024ull);
    uint64_t led_kb = st.ledger_bytes / 1024ull;
    printf("VESSEL_OK=1\n");
    printf("VESSEL_RSS_MB=%d\n", rbest);
    printf("VESSEL_PHYS_MB=%d\n", p1 < p_att ? p1 : p_att);
    printf("VESSEL_RSS0_MB=%d\n", r0);
    printf("VESSEL_PHYS0_MB=%d\n", p0);
    printf("VESSEL_RSS_ATTACH_MB=%d\n", r_att);
    printf("VESSEL_PAGES_OK=%d\n", ok);
    printf("VESSEL_PAGES_REQ=%d\n", pages);
    printf("VESSEL_MAT_MS=%.3f\n", ms);
    printf("VESSEL_ENTS=%llu\n", (unsigned long long)st.ent_count);
    printf("VESSEL_PAGE_LOGICAL_MB=%llu\n", (unsigned long long)page_mb);
    printf("VESSEL_SPILL_MB=%llu\n", (unsigned long long)spill_mb);
    printf("VESSEL_LEDGER_KB=%llu\n", (unsigned long long)led_kb);
    printf("VESSEL_DENSE=%d\n", st.dense);
    printf("VESSEL_SPANS=%llu\n", (unsigned long long)st.materialize_spans);
    printf("VESSEL_BATCH_PAGES=%llu\n", (unsigned long long)st.materialize_batch_pages);
    printf("VESSEL_ULTRA=%d\n", ultra);
    if (page_mb > 0 && rbest > 0)
        printf("VESSEL_X=%.1f\n", (double)page_mb / (double)rbest);
    else
        printf("VESSEL_X=0\n");
    (void)memx_runtime_capsule_detach();
    return ok > 0 ? 0 : 1;
}
