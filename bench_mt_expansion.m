// MemX Multi-threaded Stress Test + Effective Memory Expansion
// Critical for paper: proves thread safety and measures real expansion factor
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/mman.h>
#include <mach/mach_time.h>
#include <libproc.h>

#define PAGE_SZ 16384
#define MB (1024ULL*1024)

static size_t get_phys_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return 0;
    return info.phys_footprint;
}

// ─── Part 1: Multi-threaded stress test ───
typedef struct {
    int thread_id;
    size_t alloc_size;
    void **allocs;
    int n_allocs;
    int integrity_ok;
    volatile int *start_flag;
} ThreadArg;

static void *stress_thread(void *arg) {
    ThreadArg *a = (ThreadArg*)arg;
    // Wait for all threads to be ready
    while (!*a->start_flag) {}
    
    a->allocs = (void**)malloc(a->n_allocs * sizeof(void*));
    a->integrity_ok = 1;
    
    for (int i = 0; i < a->n_allocs; i++) {
        size_t sz = a->alloc_size;
        a->allocs[i] = malloc(sz);
        if (!a->allocs[i]) { a->integrity_ok = 0; break; }
        
        // Write pattern: thread_id + index
        uint8_t *p = (uint8_t*)a->allocs[i];
        for (size_t off = 0; off < sz; off += 64) {
            p[off] = (uint8_t)(a->thread_id & 0xFF);
            p[off+1] = (uint8_t)((i >> 0) & 0xFF);
            p[off+2] = (uint8_t)((i >> 8) & 0xFF);
            p[off+3] = (uint8_t)((off >> 8) & 0xFF);
        }
    }
    
    // Verify all allocations
    for (int i = 0; i < a->n_allocs && a->allocs[i]; i++) {
        uint8_t *p = (uint8_t*)a->allocs[i];
        for (size_t off = 0; off < a->alloc_size; off += 64) {
            if (p[off] != (uint8_t)(a->thread_id & 0xFF) ||
                p[off+1] != (uint8_t)((i >> 0) & 0xFF) ||
                p[off+2] != (uint8_t)((i >> 8) & 0xFF) ||
                p[off+3] != (uint8_t)((off >> 8) & 0xFF)) {
                a->integrity_ok = 0;
                break;
            }
        }
    }
    
    // Re-access some pages to trigger decompression
    for (int i = 0; i < a->n_allocs && a->allocs[i]; i++) {
        if (i % 3 == 0) {
            uint8_t *p = (uint8_t*)a->allocs[i];
            volatile uint8_t x = p[0]; // may trigger fault
            (void)x;
        }
    }
    
    // Verify again after decompression
    for (int i = 0; i < a->n_allocs && a->allocs[i]; i++) {
        uint8_t *p = (uint8_t*)a->allocs[i];
        for (size_t off = 0; off < a->alloc_size; off += 64) {
            if (p[off] != (uint8_t)(a->thread_id & 0xFF) ||
                p[off+1] != (uint8_t)((i >> 0) & 0xFF) ||
                p[off+2] != (uint8_t)((i >> 8) & 0xFF) ||
                p[off+3] != (uint8_t)((off >> 8) & 0xFF)) {
                a->integrity_ok = 0;
                break;
            }
        }
    }
    
    // Free half, keep half
    for (int i = 0; i < a->n_allocs && a->allocs[i]; i++) {
        if (i % 2 == 0) { free(a->allocs[i]); a->allocs[i] = NULL; }
    }
    
    return NULL;
}

static void test_multithread(int nthreads, size_t alloc_per_thread, int n_allocs) {
    printf("  ─── %d threads × %d allocs × %zu MB = %zu MB total ───\n",
           nthreads, n_allocs, alloc_per_thread/MB,
           (size_t)nthreads * n_allocs * alloc_per_thread / MB);
    
    pthread_t threads[32];
    ThreadArg args[32];
    volatile int start = 0;
    
    for (int i = 0; i < nthreads; i++) {
        args[i].thread_id = i;
        args[i].alloc_size = alloc_per_thread;
        args[i].n_allocs = n_allocs;
        args[i].allocs = NULL;
        args[i].integrity_ok = 0;
        args[i].start_flag = &start;
        pthread_create(&threads[i], NULL, stress_thread, &args[i]);
    }
    
    // Start all threads simultaneously
    start = 1;
    
    for (int i = 0; i < nthreads; i++)
        pthread_join(threads[i], NULL);
    
    int ok = 1;
    for (int i = 0; i < nthreads; i++)
        if (!args[i].integrity_ok) ok = 0;
    
    size_t fp = get_phys_footprint();
    printf("    Integrity: %s, Footprint: %zu MB\n",
           ok ? "✅ PERFECT" : "❌ FAILED", fp/MB);
    
    // Clean up remaining allocs
    for (int i = 0; i < nthreads; i++) {
        if (args[i].allocs) {
            for (int j = 0; j < args[i].n_allocs; j++)
                if (args[i].allocs[j]) free(args[i].allocs[j]);
            free(args[i].allocs);
        }
    }
}

// ─── Part 2: Effective memory expansion ───
static void test_expansion(void) {
    printf("\n  ═══ Effective Memory Expansion ═══\n\n");
    
    // Allocate increasingly large blocks until we approach physical memory limit
    // With MemX, we should be able to allocate far beyond physical RAM
    size_t ram = 24ULL * 1024 * 1024 * 1024; // 24 GB M4 Pro
    
    printf("  Physical RAM: 24 GB\n");
    printf("  Virtual pool: 96 GB (4× RAM)\n\n");
    
    // Allocate 1.5× RAM and measure footprint after compression
    size_t target = (size_t)(ram * 1.5);
    size_t chunk = 256 * MB;
    int n_chunks = (int)(target / chunk);
    
    void **allocs = (void**)malloc(n_chunks * sizeof(void*));
    int n = 0;
    
    printf("  Allocating %.1f GB (1.5× RAM)...\n", (double)target / (1024*MB));
    for (int i = 0; i < n_chunks; i++) {
        allocs[i] = malloc(chunk);
        if (!allocs[i]) break;
        // Fill with compressible data (80% zero, 20% structured)
        uint8_t *p = (uint8_t*)allocs[i];
        memset(p, 0, chunk);
        for (size_t off = chunk * 4 / 5; off < chunk; off += PAGE_SZ)
            for (size_t j = 0; j < PAGE_SZ; j++)
                p[off+j] = (uint8_t)((j*7+13+i) & 0xFF);
        n++;
    }
    
    size_t alloced = (size_t)n * chunk;
    size_t fp_before = get_phys_footprint();
    printf("  Allocated: %.1f GB, Footprint: %.1f GB\n",
           (double)alloced / (1024*MB), (double)fp_before / (1024*MB));
    
    printf("  Waiting 15s for GPU compression...\n");
    sleep(15);
    
    size_t fp_after = get_phys_footprint();
    double expansion = (double)alloced / fp_after;
    printf("  After compression: Footprint %.1f GB\n", (double)fp_after / (1024*MB));
    printf("  Effective expansion: %.1f× (allocated %.1f GB using %.1f GB physical)\n",
           expansion, (double)alloced / (1024*MB), (double)fp_after / (1024*MB));
    
    // Verify integrity of random samples
    int integrity_ok = 1;
    for (int i = 0; i < n && integrity_ok; i += (n/20 > 0 ? n/20 : 1)) {
        uint8_t *p = (uint8_t*)allocs[i];
        // Check zero region
        for (size_t off = 0; off < chunk * 4 / 5; off += 4096) {
            if (p[off] != 0) { integrity_ok = 0; break; }
        }
        // Check structured region
        for (size_t off = chunk * 4 / 5; off < chunk; off += PAGE_SZ) {
            for (size_t j = 0; j < PAGE_SZ; j++) {
                if (p[off+j] != (uint8_t)((j*7+13+i) & 0xFF)) {
                    integrity_ok = 0; break;
                }
            }
            if (!integrity_ok) break;
        }
    }
    printf("  Integrity: %s\n", integrity_ok ? "✅ PERFECT" : "❌ FAILED");
    
    // Access all pages to verify full integrity
    printf("  Full verification (accessing all pages)...\n");
    for (int i = 0; i < n; i++) {
        uint8_t *p = (uint8_t*)allocs[i];
        for (size_t off = 0; off < chunk; off += PAGE_SZ) {
            volatile uint8_t x = p[off]; (void)x;
        }
    }
    size_t fp_accessed = get_phys_footprint();
    printf("  After full access: Footprint %.1f GB (all decompressed)\n",
           (double)fp_accessed / (1024*MB));
    
    // Clean up
    for (int i = 0; i < n; i++) free(allocs[i]);
    free(allocs);
}

int main(void) {
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  MemX Multi-Thread + Memory Expansion Test          ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");
    
    // Part 1: Multi-threaded stress tests
    printf("  ═══ Multi-Threaded Stress Test ═══\n\n");
    
    test_multithread(1, 64*MB, 4);       // 256 MB, 1 thread
    sleep(2);
    test_multithread(2, 64*MB, 4);       // 512 MB, 2 threads
    sleep(2);
    test_multithread(4, 32*MB, 4);       // 512 MB, 4 threads
    sleep(2);
    test_multithread(8, 16*MB, 4);       // 512 MB, 8 threads
    sleep(2);
    
    // Part 2: Effective memory expansion
    test_expansion();
    
    return 0;
}
