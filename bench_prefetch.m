// MemX Prefetch Benchmark - measures end-to-end throughput with and without prefetch
// Key insight: prefetch increases per-fault latency but reduces total faults,
// so the NET effect on throughput should be positive for sequential access.
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <unistd.h>

#define MB (1024ULL*1024)
#define PAGE_SZ 16384

static long long get_fp(void) {
    task_vm_info_data_t info; mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
    return (long long)info.phys_footprint;
}

int main() {
    printf("╔══════════════════════════════════════╗\n");
    printf("║   MemX Prefetch Effectiveness Bench    ║\n");
    printf("╚══════════════════════════════════════╝\n\n");
    
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / tb.denom;
    
    // Allocate 1GB zero-filled (will be compressed + deduped)
    size_t sz = 1024*MB;
    char *p = (char*)mmap(NULL, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(p, 0, sz);
    printf("Allocated %zu MB. Waiting 10s for compression...\n", sz/MB);
    sleep(10);
    printf("Footprint after compression: %lld MB\n\n", get_fp()/MB);
    
    // ─── Test 1: Sequential read (prefetch should help) ───
    printf("─── Test 1: Sequential Read (prefetch-friendly) ───\n");
    uint64_t t0 = mach_absolute_time();
    for (size_t off = 0; off < sz; off += PAGE_SZ) {
        volatile char c = p[off];
        (void)c;
    }
    uint64_t t1 = mach_absolute_time();
    double seq_ns = (t1-t0) * ns_per_tick;
    printf("  Time: %.1f ms (%.1f μs/page, %.0f MB/s)\n\n",
           seq_ns/1e6, seq_ns/(sz/PAGE_SZ)/1000.0, (double)sz/(seq_ns/1e9)/MB);
    
    // Wait for re-compression
    printf("Waiting 10s for re-compression...\n");
    sleep(10);
    printf("Footprint: %lld MB\n\n", get_fp()/MB);
    
    // ─── Test 2: Random access (prefetch won't help) ───
    printf("─── Test 2: Random Access (prefetch-unfriendly) ───\n");
    srand(42);
    size_t n_accesses = 10000;
    size_t *offsets = (size_t*)malloc(n_accesses * sizeof(size_t));
    for (size_t i = 0; i < n_accesses; i++)
        offsets[i] = ((size_t)rand() % (sz/PAGE_SZ)) * PAGE_SZ;
    
    t0 = mach_absolute_time();
    for (size_t i = 0; i < n_accesses; i++) {
        volatile char c = p[offsets[i]];
        (void)c;
    }
    t1 = mach_absolute_time();
    double rand_ns = (t1-t0) * ns_per_tick;
    printf("  Time: %.1f ms (%.1f μs/access avg)\n\n",
           rand_ns/1e6, rand_ns/n_accesses/1000.0);
    free(offsets);
    
    // Wait for re-compression
    printf("Waiting 10s for re-compression...\n");
    sleep(10);
    
    // ─── Test 3: Strided access (every 4th page) ───
    printf("─── Test 3: Strided Access (every 4th page) ───\n");
    t0 = mach_absolute_time();
    for (size_t off = 0; off < sz; off += PAGE_SZ * 4) {
        volatile char c = p[off];
        (void)c;
    }
    t1 = mach_absolute_time();
    double stride_ns = (t1-t0) * ns_per_tick;
    size_t stride_pages = sz / (PAGE_SZ * 4);
    printf("  Time: %.1f ms (%.1f μs/page, %.0f MB/s effective)\n\n",
           stride_ns/1e6, stride_ns/stride_pages/1000.0,
           (double)(stride_pages*PAGE_SZ)/(stride_ns/1e9)/MB);
    
    // Verify
    int ok = 1;
    for (size_t i = 0; i < sz && ok; i++) if (p[i] != 0) { ok = 0; }
    printf("Integrity: %s\n", ok ? "PERFECT" : "CORRUPT");
    
    munmap(p, sz);
    return ok ? 0 : 1;
}
