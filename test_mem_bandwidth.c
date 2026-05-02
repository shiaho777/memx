#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/vm_statistics.h>
#include <mach/mach_vm.h>

#define GB (1024ULL * 1024 * 1024)
#define MB (1024ULL * 1024)
#define KB (1024ULL)

static uint64_t get_ticks(void) {
    return mach_absolute_time();
}

static double ticks_to_ns(uint64_t ticks) {
    static double ns_per_tick = 0.0;
    if (ns_per_tick == 0.0) {
        mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        ns_per_tick = (double)info.numer / (double)info.denom;
    }
    return ticks * ns_per_tick;
}

// Measure sequential read bandwidth
double measure_seq_read(void *buf, size_t size, size_t stride) {
    volatile char *p = (volatile char *)buf;
    uint64_t start = get_ticks();
    char sink = 0;
    for (size_t i = 0; i < size; i += stride) {
        sink += p[i];
    }
    uint64_t end = get_ticks();
    // prevent optimization
    if (sink == 127) printf("impossible\n");
    return ticks_to_ns(end - start);
}

// Measure random read latency
double measure_random_read(void *buf, size_t size, size_t count) {
    volatile char *p = (volatile char *)buf;
    // Pre-generate random offsets
    size_t *offsets = malloc(count * sizeof(size_t));
    for (size_t i = 0; i < count; i++) {
        offsets[i] = ((size_t)rand() * rand()) % size;
    }

    uint64_t start = get_ticks();
    char sink = 0;
    for (size_t i = 0; i < count; i++) {
        sink += p[offsets[i]];
    }
    uint64_t end = get_ticks();
    if (sink == 127) printf("impossible\n");
    free(offsets);
    return ticks_to_ns(end - start);
}

// Measure write bandwidth
double measure_seq_write(void *buf, size_t size, size_t stride) {
    char *p = (char *)buf;
    uint64_t start = get_ticks();
    for (size_t i = 0; i < size; i += stride) {
        p[i] = (char)i;
    }
    uint64_t end = get_ticks();
    return ticks_to_ns(end - start);
}

// Test with different page types
int main() {
    printf("=== Apple Silicon Memory Deep Probe ===\n");
    uint64_t memsize = 0;
    size_t len = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &len, NULL, 0);
    printf("Page size: %lu bytes\n", (unsigned long)vm_page_size);
    printf("Total memory: %.1f GB\n", (double)memsize / GB);

    size_t test_sizes[] = {
        4 * KB, 16 * KB, 64 * KB, 256 * KB,
        1 * MB, 4 * MB, 16 * MB, 64 * MB,
        256 * MB, 512 * MB, 1 * GB
    };
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("\n=== Sequential Read Bandwidth (stride=64, cache-line) ===\n");
    printf("%-12s %12s %12s\n", "Size", "Time(ms)", "GB/s");
    for (int i = 0; i < num_sizes; i++) {
        size_t sz = test_sizes[i];
        void *buf = mmap(NULL, sz, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON, -1, 0);
        if (buf == MAP_FAILED) { printf("mmap failed for %zu\n", sz); continue; }
        // Touch all pages first
        memset(buf, 0xAA, sz);
        double ns = measure_seq_read(buf, sz, 64);
        double ms = ns / 1e6;
        double gbps = (double)sz / GB / (ns / 1e9);
        if (sz >= MB) printf("%-12zu %12.3f %12.2f\n", sz/MB, ms, gbps);
        else printf("%-12zu %12.3f %12.2f\n", sz/KB, ms, gbps);
        munmap(buf, sz);
    }

    printf("\n=== Random Read Latency ===\n");
    printf("%-12s %12s %12s\n", "Size", "Avg(ns)", "Samples");
    for (int i = 0; i < num_sizes; i++) {
        size_t sz = test_sizes[i];
        void *buf = mmap(NULL, sz, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON, -1, 0);
        if (buf == MAP_FAILED) continue;
        memset(buf, 0x55, sz);
        size_t count = 1000000;
        double ns = measure_random_read(buf, sz, count);
        double avg_ns = ns / count;
        if (sz >= MB) printf("%-12zu %12.1f %12zu\n", sz/MB, avg_ns, count);
        else printf("%-12zu %12.1f %12zu\n", sz/KB, avg_ns, count);
        munmap(buf, sz);
    }

    printf("\n=== Sequential Write Bandwidth (stride=64) ===\n");
    printf("%-12s %12s %12s\n", "Size", "Time(ms)", "GB/s");
    for (int i = 0; i < num_sizes; i++) {
        size_t sz = test_sizes[i];
        void *buf = mmap(NULL, sz, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON, -1, 0);
        if (buf == MAP_FAILED) continue;
        double ns = measure_seq_write(buf, sz, 64);
        double ms = ns / 1e6;
        double gbps = (double)sz / GB / (ns / 1e9);
        if (sz >= MB) printf("%-12zu %12.3f %12.2f\n", sz/MB, ms, gbps);
        else printf("%-12zu %12.3f %12.2f\n", sz/KB, ms, gbps);
        munmap(buf, sz);
    }

    // Now test the KEY experiment: purgable memory behavior
    printf("\n=== Purgable Memory Compression Test ===\n");
    mach_vm_address_t paddr = 0;
    size_t psize = 100 * vm_page_size; // 100 pages
    kern_return_t kr = mach_vm_allocate(mach_task_self(), &paddr, psize,
                                         VM_FLAGS_ANYWHERE | VM_FLAGS_PURGABLE);
    if (kr == KERN_SUCCESS) {
        printf("Purgable allocation: SUCCESS (%zu pages)\n", psize / vm_page_size);
        // Fill with compressible data
        memset((void*)paddr, 0x42, psize);
        printf("Filled with compressible data (0x42)\n");

        // Set volatile - OS can reclaim
        int state = VM_PURGABLE_VOLATILE;
        kr = vm_purgable_control(mach_task_self(), (vm_offset_t)paddr,
                                 VM_PURGABLE_SET_STATE, &state);
        printf("Set VOLATILE: %s\n", kr == KERN_SUCCESS ? "SUCCESS" : "FAILED");

        // Check if still resident
        printf("Waiting 2s for OS to potentially reclaim...\n");
        usleep(2000000);

        // Set non-volatile and check
        state = VM_PURGABLE_NONVOLATILE;
        int old_state = 0;
        kr = vm_purgable_control(mach_task_self(), (vm_offset_t)paddr,
                                 VM_PURGABLE_SET_STATE, &state);
        printf("Set NONVOLATILE: %s (old_state=%d)\n",
               kr == KERN_SUCCESS ? "SUCCESS" : "FAILED", old_state);
        // VM_PURGABLE_EMPTY means content was reclaimed!
        if (old_state == VM_PURGABLE_EMPTY) {
            printf("CONTENT WAS RECLAIMED BY OS (expected for volatile)\n");
        } else {
            printf("Content survived (old_state=%d)\n", old_state);
        }

        mach_vm_deallocate(mach_task_self(), paddr, psize);
    }

    // Test: measure malloc vs mmap allocation latency
    printf("\n=== Allocation Latency: malloc vs mmap ===\n");
    size_t alloc_sizes[] = {4096, 16384, 65536, 262144, 1048576, 4194304};
    int nalloc = sizeof(alloc_sizes)/sizeof(alloc_sizes[0]);
    printf("%-12s %12s %12s\n", "Size", "malloc(ns)", "mmap(ns)");
    for (int i = 0; i < nalloc; i++) {
        size_t sz = alloc_sizes[i];
        // malloc
        uint64_t s = get_ticks();
        void *p = malloc(sz);
        uint64_t e = get_ticks();
        double malloc_ns = ticks_to_ns(e - s);
        free(p);
        // mmap
        s = get_ticks();
        void *q = mmap(NULL, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
        e = get_ticks();
        double mmap_ns = ticks_to_ns(e - s);
        munmap(q, sz);
        printf("%-12zu %12.0f %12.0f\n", sz, malloc_ns, mmap_ns);
    }

    printf("\n=== Test Complete ===\n");
    return 0;
}
