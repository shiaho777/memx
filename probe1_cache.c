// Probe 1: Map the EXACT cache hierarchy on this Apple Silicon
// Pointer-chasing to measure true latency at each level
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

int main(void) {
    init_time();
    printf("=== PROBE 1: Cache Hierarchy Deep Map ===\n");
    printf("Page size: %lu bytes\n\n", (unsigned long)vm_page_size);

    // Pointer-chasing: each cache-line-sized slot holds a pointer to the next
    // This defeats prefetchers and measures TRUE access latency
    size_t sizes[] = {
        1*KB, 2*KB, 4*KB, 8*KB, 16*KB, 32*KB, 48*KB, 64*KB, 96*KB, 128*KB,
        192*KB, 256*KB, 384*KB, 512*KB, 768*KB,
        1*MB, 1.5*MB, 2*MB, 3*MB, 4*MB, 6*MB, 8*MB, 10*MB, 12*MB, 16*MB,
        24*MB, 32*MB, 48*MB, 64*MB, 96*MB, 128*MB, 192*MB, 256*MB, 384*MB, 512*MB,
        768*MB, 1*GB, 1.5*GB, 2*GB
    };
    int nsizes = sizeof(sizes)/sizeof(sizes[0]);

    printf("%-12s %10s\n", "WorkingSet", "Latency(ns)");
    printf("-------------------------------\n");

    for (int i = 0; i < nsizes; i++) {
        size_t sz = sizes[i];
        size_t stride = 64; // cache line size on Apple Silicon
        size_t nentries = sz / stride;
        if (nentries < 2) continue;

        void *buf = mmap(NULL, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
        if (buf == MAP_FAILED) continue;

        // Build pointer chain (Fisher-Yates shuffle)
        size_t *idx = malloc(nentries * sizeof(size_t));
        for (size_t j = 0; j < nentries; j++) idx[j] = j;
        for (size_t j = nentries - 1; j > 0; j--) {
            size_t k = rand() % (j + 1);
            size_t tmp = idx[j]; idx[j] = idx[k]; idx[k] = tmp;
        }

        // Each slot: [next_ptr, padding...]
        char *base = (char*)buf;
        for (size_t j = 0; j < nentries; j++) {
            size_t curr = idx[j];
            size_t next = idx[(j + 1) % nentries];
            void **slot = (void**)(base + curr * stride);
            *slot = (void*)(base + next * stride);
        }
        free(idx);

        // Chase
        volatile void **p = (volatile void**)buf;
        size_t chase_len = nentries < 200000 ? nentries : 200000;

        // Warmup 5 rounds
        for (int w = 0; w < 5; w++) {
            for (size_t j = 0; j < chase_len; j++) {
                p = (volatile void**)*p;
            }
        }

        // Measure 3 rounds, take median
        double times[3];
        for (int r = 0; r < 3; r++) {
            uint64_t t0 = mach_absolute_time();
            for (size_t j = 0; j < chase_len; j++) {
                p = (volatile void**)*p;
            }
            uint64_t t1 = mach_absolute_time();
            times[r] = NS(t1 - t0) / chase_len;
        }
        // Simple median
        if (times[0] > times[1]) { double tmp=times[0]; times[0]=times[1]; times[1]=tmp; }
        if (times[1] > times[2]) { double tmp=times[1]; times[1]=times[2]; times[2]=tmp; }
        double median = times[1];

        if (sz >= MB) printf("%-8zuMB %10.1f\n", sz/MB, median);
        else printf("%-8zuKB %10.1f\n", sz/KB, median);

        munmap(buf, sz);
    }

    printf("\n=== Interpretation ===\n");
    printf("L1D cache: ~128KB (performance cores), latency ~1-2ns\n");
    printf("L2 cache:  ~12MB shared per cluster, latency ~3-5ns\n");
    printf("DRAM:      beyond L2, latency ~10-30ns (local) or ~100+ns (swapped)\n");
    printf("Watch for JUMPS in latency - those are the real boundaries!\n");

    return 0;
}
