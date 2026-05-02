// Real-world memory savings test
// Verifies that MemX actually reduces PHYSICAL memory usage
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/sysctl.h>

static long long get_phys_memory(void) {
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    int64_t total = 0; size_t len = sizeof(total);
    sysctl(mib, 2, &total, &len, NULL, 0);
    return total;
}

// Get process RSS in bytes
static long long get_rss(void) {
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) {
        // macOS: use ps
        char cmd[128];
        snprintf(cmd, sizeof(cmd), "ps -o rss= -p %d", getpid());
        FILE *p = popen(cmd, "r");
        long long rss_kb = 0;
        fscanf(p, "%lld", &rss_kb);
        pclose(p);
        return rss_kb * 1024;
    }
    long long rss, vmsize;
    fscanf(f, "%lld %lld", &vmsize, &rss);
    fclose(f);
    return rss * 4096;
}

// Get system free memory
static long long get_free_memory(void) {
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    int64_t total = 0; size_t len = sizeof(total);
    sysctl(mib, 2, &total, &len, NULL, 0);
    
    // Use vm_stat
    FILE *p = popen("vm_stat | grep 'Pages free'", "r");
    if (!p) return -1;
    long long free_pages = 0;
    fscanf(p, "Pages free: %lld.", &free_pages);
    pclose(p);
    return free_pages * 16384;  // 16KB pages
}

int main() {
    printf("══════════════════════════════════════════════════\n");
    printf("  MemX Real-World Memory Savings Test\n");
    printf("══════════════════════════════════════════════════\n\n");
    
    long long total_mem = get_phys_memory();
    printf("  Total physical RAM: %lld MB\n", total_mem / (1024*1024));
    
    // Phase 1: Baseline
    long long rss0 = get_rss();
    long long free0 = get_free_memory();
    printf("  Baseline RSS: %lld MB, Free: %lld MB\n\n", 
           rss0/(1024*1024), free0/(1024*1024));
    
    // Phase 2: Allocate 1GB in 64KB chunks
    printf("  Allocating 1024 MB (16384 chunks x 64KB)...\n");
    int N = 16384;
    void **ptrs = malloc(N * sizeof(void*));
    int count = 0;
    for (int i = 0; i < N; i++) {
        ptrs[i] = malloc(65536);
        if (!ptrs[i]) break;
        count++;
    }
    printf("  Allocated: %d chunks = %d MB\n", count, count * 64 / 1024);
    
    long long rss1 = get_rss();
    long long free1 = get_free_memory();
    printf("  After alloc RSS: %lld MB, Free: %lld MB\n", 
           rss1/(1024*1024), free1/(1024*1024));
    printf("  RSS increase: %lld MB\n\n", (rss1-rss0)/(1024*1024));
    
    // Phase 3: Write realistic data (compressible patterns)
    printf("  Writing realistic data patterns...\n");
    for (int i = 0; i < count; i++) {
        unsigned char *p = (unsigned char*)ptrs[i];
        // Mix of patterns: JSON-like, code-like, zeros, gradients
        int pattern = i % 4;
        if (pattern == 0) {
            // Repeating (like JSON) - highly compressible
            for (int j = 0; j < 65536; j++) p[j] = "Hello World! "[j % 13];
        } else if (pattern == 1) {
            // Semi-random (like source code) - moderately compressible
            for (int j = 0; j < 65536; j++) p[j] = (j * 7 + i * 13) & 0xFF;
        } else if (pattern == 2) {
            // Zeros - maximally compressible
            memset(p, 0, 65536);
        } else {
            // Sequential (like database) - compressible
            for (int j = 0; j < 65536; j++) p[j] = j & 0xFF;
        }
    }
    
    long long rss2 = get_rss();
    long long free2 = get_free_memory();
    printf("  After write RSS: %lld MB, Free: %lld MB\n\n",
           rss2/(1024*1024), free2/(1024*1024));
    
    // Phase 4: Wait for GPU compression
    printf("  Waiting for GPU compression (10s)...\n");
    for (int t = 0; t < 10; t++) {
        sleep(1);
        long long rss = get_rss();
        long long free = get_free_memory();
        printf("    [%2ds] RSS: %lld MB, Free: %lld MB\n", 
               t+1, rss/(1024*1024), free/(1024*1024));
    }
    
    long long rss3 = get_rss();
    long long free3 = get_free_memory();
    printf("\n  After compression RSS: %lld MB, Free: %lld MB\n",
           rss3/(1024*1024), free3/(1024*1024));
    printf("  RSS saved by compression: %lld MB\n", (rss2-rss3)/(1024*1024));
    printf("  Free memory recovered: %lld MB\n\n", (free3-free2)/(1024*1024));
    
    // Phase 5: Verify data integrity
    printf("  Verifying data integrity...\n");
    int mismatches = 0;
    for (int i = 0; i < count && mismatches < 5; i++) {
        unsigned char *p = (unsigned char*)ptrs[i];
        int pattern = i % 4;
        if (pattern == 0) {
            for (int j = 0; j < 65536; j++)
                if (p[j] != "Hello World! "[j % 13]) { mismatches++; break; }
        } else if (pattern == 1) {
            for (int j = 0; j < 65536; j++)
                if (p[j] != ((j * 7 + i * 13) & 0xFF)) { mismatches++; break; }
        } else if (pattern == 2) {
            for (int j = 0; j < 65536; j++)
                if (p[j] != 0) { mismatches++; break; }
        } else {
            for (int j = 0; j < 65536; j++)
                if (p[j] != (j & 0xFF)) { mismatches++; break; }
        }
    }
    printf("  Result: %s (%d mismatches)\n\n",
           mismatches == 0 ? "✅ PERFECT" : "❌ MISMATCH", mismatches);
    
    // Summary
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  REAL-WORLD MEMORY SAVINGS                       ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  Allocated:     %4d MB                          ║\n", count*64/1024);
    printf("║  RSS before:    %4lld MB                          ║\n", rss2/(1024*1024));
    printf("║  RSS after:     %4lld MB                          ║\n", rss3/(1024*1024));
    printf("║  RSS saved:     %4lld MB (%.0f%%)                 ║\n", 
           (rss2-rss3)/(1024*1024),
           rss2 > 0 ? 100.0*(rss2-rss3)/rss2 : 0);
    printf("║  Integrity:     %s                          ║\n",
           mismatches == 0 ? "✅ PERFECT" : "❌ FAIL  ");
    printf("╚══════════════════════════════════════════════════╝\n");
    
    // Cleanup
    for (int i = 0; i < count; i++) free(ptrs[i]);
    free(ptrs);
    
    return 0;
}
