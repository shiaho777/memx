// Probe 1 v2: Cache Hierarchy - with memory barriers to prevent optimization
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/mach_vm.h>

#define GB (1024ULL*1024*1024)
#define MB (1024ULL*1024)
#define KB (1024ULL)

static double ns_per_tick;
static void init_time(void) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    ns_per_tick = (double)info.numer / (double)info.denom;
}
#define NS(ticks) ((double)(ticks) * ns_per_tick)

// Force the pointer chase result to be used
static volatile uintptr_t sink = 0;

int main(void) {
    init_time();
    printf("=== PROBE 1v2: Cache Hierarchy (barrier-enforced) ===\n");
    printf("Page size: %lu bytes\n\n", (unsigned long)vm_page_size);

    size_t sizes[] = {
        1*KB, 2*KB, 4*KB, 8*KB, 16*KB, 32*KB, 48*KB, 64*KB, 96*KB, 128*KB,
        192*KB, 256*KB, 384*KB, 512*KB, 768*KB,
        1*MB, 1.5*MB, 2*MB, 3*MB, 4*MB, 6*MB, 8*MB, 10*MB, 12*MB, 16*MB,
        24*MB, 32*MB, 48*MB, 64*MB, 96*MB, 128*MB, 192*MB, 256*MB, 384*MB, 512*MB,
        768*MB, (size_t)(1*GB), (size_t)(1.5*GB)
    };
    int nsizes = sizeof(sizes)/sizeof(sizes[0]);

    printf("%-12s %10s\n", "WorkingSet", "Latency(ns)");
    printf("-------------------------------\n");

    for (int i = 0; i < nsizes; i++) {
        size_t sz = sizes[i];
        size_t stride = 64;
        size_t nentries = sz / stride;
        if (nentries < 2) continue;

        void *buf = mmap(NULL, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
        if (buf == MAP_FAILED) continue;

        // Build pointer chain
        size_t *idx = malloc(nentries * sizeof(size_t));
        for (size_t j = 0; j < nentries; j++) idx[j] = j;
        for (size_t j = nentries - 1; j > 0; j--) {
            size_t k = rand() % (j + 1);
            size_t tmp = idx[j]; idx[j] = idx[k]; idx[k] = tmp;
        }

        char *base = (char*)buf;
        for (size_t j = 0; j < nentries; j++) {
            size_t curr = idx[j];
            size_t next = idx[(j + 1) % nentries];
            void **slot = (void**)(base + curr * stride);
            *slot = (void*)(base + next * stride);
        }
        free(idx);

        // Touch all pages first
        for (size_t j = 0; j < sz; j += vm_page_size) {
            ((volatile char*)buf)[j] = 0;
        }

        size_t chase_len = nentries < 200000 ? nentries : 200000;

        // Warmup
        volatile void **p = (volatile void**)buf;
        for (int w = 0; w < 3; w++) {
            for (size_t j = 0; j < chase_len; j++) {
                p = (volatile void**)*p;
            }
        }

        // Measure with asm barrier
        uint64_t t0 = mach_absolute_time();
        for (size_t j = 0; j < chase_len; j++) {
            p = (volatile void**)*p;
            __asm__ __volatile__("" : : "r"(p) : "memory");
        }
        sink = (uintptr_t)p;
        uint64_t t1 = mach_absolute_time();
        double latency = NS(t1 - t0) / chase_len;

        if (sz >= MB) printf("%-8lluMB %10.1f\n", (unsigned long long)(sz/MB), latency);
        else printf("%-8lluKB %10.1f\n", (unsigned long long)(sz/KB), latency);

        munmap(buf, sz);
    }

    printf("\nDone. sink=%lu\n", (unsigned long)sink);
    return 0;
}
