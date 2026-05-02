// MemX Fault Latency Benchmark
// Measures end-to-end latency from SIGSEGV → decompress → return to user code
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
    printf("║   MemX Fault Latency Measurement      ║\n");
    printf("╚══════════════════════════════════════╝\n\n");
    
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / tb.denom;
    
    // Allocate 1GB zero-filled (will be compressed)
    size_t sz = 1024*MB;
    char *p = (char*)mmap(NULL, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(p, 0, sz);
    printf("Allocated %zu MB zero-filled. Footprint: %lld MB\n", sz/MB, get_fp()/MB);
    
    // Wait for compression
    printf("Waiting 10s for GPU compression...\n");
    sleep(10);
    printf("After compression: %lld MB\n\n", get_fp()/MB);
    
    // Now access compressed pages and measure latency
    printf("Measuring decompression latency...\n");
    
    int n_samples = 1000;
    double latencies_us[1000];
    int valid = 0;
    
    for (int i = 0; i < n_samples && valid < 100; i++) {
        // Pick a random page that might be compressed
        size_t offset = ((size_t)rand() % (sz / PAGE_SZ)) * PAGE_SZ;
        
        uint64_t t0 = mach_absolute_time();
        volatile char c = p[offset];  // This triggers fault if compressed
        uint64_t t1 = mach_absolute_time();
        (void)c;
        
        double ns = (t1 - t0) * ns_per_tick;
        double us = ns / 1000.0;
        
        // Only count if it was actually a fault (latency > 1us)
        if (us > 1.0) {
            latencies_us[valid++] = us;
        }
    }
    
    if (valid > 0) {
        // Sort
        for (int i = 0; i < valid-1; i++)
            for (int j = i+1; j < valid; j++)
                if (latencies_us[j] < latencies_us[i]) { double tmp = latencies_us[i]; latencies_us[i] = latencies_us[j]; latencies_us[j] = tmp; }
        
        double p50 = latencies_us[valid/2];
        double p90 = latencies_us[valid*9/10];
        double p99 = latencies_us[valid*99/100 < valid ? valid*99/100 : valid-1];
        double min_l = latencies_us[0];
        double max_l = latencies_us[valid-1];
        double avg = 0;
        for (int i = 0; i < valid; i++) avg += latencies_us[i];
        avg /= valid;
        
        printf("  Samples: %d decompression faults measured\n", valid);
        printf("  Min:     %.1f μs\n", min_l);
        printf("  P50:     %.1f μs\n", p50);
        printf("  P90:     %.1f μs\n", p90);
        printf("  P99:     %.1f μs\n", p99);
        printf("  Max:     %.1f μs\n", max_l);
        printf("  Avg:     %.1f μs\n", avg);
    } else {
        printf("  No decompression faults measured (all pages still resident?)\n");
    }
    
    // Also measure sequential access pattern (more realistic)
    printf("\nSequential decompression sweep:\n");
    uint64_t t_start = mach_absolute_time();
    for (size_t off = 0; off < sz; off += PAGE_SZ) {
        volatile char c = p[off];
        (void)c;
    }
    uint64_t t_end = mach_absolute_time();
    double total_ns = (t_end - t_start) * ns_per_tick;
    double total_ms = total_ns / 1e6;
    size_t npages = sz / PAGE_SZ;
    double per_page_us = total_ns / npages / 1000.0;
    
    printf("  %zu pages in %.1f ms (%.1f μs/page avg)\n", npages, total_ms, per_page_us);
    printf("  Throughput: %.0f MB/s\n", (double)sz / (total_ns / 1e9) / MB);
    
    // Verify
    int ok = 1;
    for (size_t i = 0; i < sz && ok; i++) if (p[i] != 0) { ok = 0; printf("MISMATCH at %zu\n", i); }
    printf("\n  Integrity: %s\n", ok ? "PERFECT" : "CORRUPT");
    
    munmap(p, sz);
    return ok ? 0 : 1;
}
