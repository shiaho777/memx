// Test with memx_vm standalone (known working) to compare physical savings
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <signal.h>
#include <mach/mach.h>
#include <pthread.h>

#define PAGE_SZ 16384

static long long get_fp(void) {
    task_vm_info_data_t info; mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
    return (long long)info.phys_footprint;
}

// Simple CPU "compressor" - just copy 1/22 of the data (simulating 22x compression)
static void cpu_compress_page(const uint8_t *src, uint8_t *dst, uint32_t *cs) {
    // Simulate 22x compression by keeping first 744 bytes
    *cs = PAGE_SZ / 22;
    memcpy(dst, src, *cs);
}

static void cpu_decompress_page(const uint8_t *src, uint32_t cs, uint8_t *dst) {
    // Simulate decompression - just copy what we have and zero-fill
    memcpy(dst, src, cs);
    memset(dst + cs, 0, PAGE_SZ - cs);
}

typedef struct { uint8_t state; uint32_t comp_size; uint64_t pool_offset; } PageMeta;
#define PAGE_NONE 0
#define PAGE_RESIDENT 1
#define PAGE_COMPRESSED 2

static void *vmem; static size_t vmem_sz; static size_t npages;
static PageMeta *meta;
static uint8_t *pool; static uint64_t pool_next;
static struct sigaction old_segv;

static void fault_handler(int sig, siginfo_t *info, void *ctx) {
    uintptr_t fa = (uintptr_t)info->si_addr;
    uintptr_t vs = (uintptr_t)vmem, ve = vs + vmem_sz;
    if (fa < vs || fa >= ve) { raise(sig); return; }
    size_t pi = (fa - vs) / PAGE_SZ;
    uint8_t *pa = (uint8_t*)vmem + pi * PAGE_SZ;
    PageMeta *m = &meta[pi];
    mprotect(pa, PAGE_SZ, PROT_READ|PROT_WRITE);
    if (m->state == PAGE_NONE) { memset(pa, 0, PAGE_SZ); m->state = PAGE_RESIDENT; }
    else if (m->state == PAGE_COMPRESSED) { cpu_decompress_page(pool+m->pool_offset, m->comp_size, pa); m->state=PAGE_RESIDENT; }
}

static void *bg_compressor(void *arg) {
    const size_t BATCH = 256;  // Much larger batch
    while (1) {
        size_t tc[BATCH]; size_t nc = 0;
        for (size_t i=0; i<npages && nc<BATCH; i++)
            if (meta[i].state == PAGE_RESIDENT) tc[nc++] = i;
        if (nc == 0) { sleep(1); continue; }
        
        for (size_t i=0; i<nc; i++) {
            uint32_t cs;
            uint8_t *src = (uint8_t*)vmem + tc[i]*PAGE_SZ;
            cpu_compress_page(src, pool+pool_next, &cs);
            meta[tc[i]].state = PAGE_COMPRESSED;
            meta[tc[i]].comp_size = cs;
            meta[tc[i]].pool_offset = pool_next;
            pool_next += cs;
            mprotect(src, PAGE_SZ, PROT_NONE);
        }
        usleep(10000);  // 10ms between batches
    }
    return NULL;
}

int main() {
    printf("═══ CPU-Only Memory Expansion Test ═══\n\n");
    
    // Setup
    vmem_sz = 512ULL * 1024 * 1024;  // 512MB
    npages = vmem_sz / PAGE_SZ;
    vmem = mmap(NULL, vmem_sz, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    pool = mmap(NULL, vmem_sz/2, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    meta = (PageMeta*)mmap(NULL, npages*sizeof(PageMeta), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(meta, 0, npages*sizeof(PageMeta));
    
    struct sigaction sa; memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = fault_handler; sa.sa_flags = SA_SIGINFO|SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old_segv);
    
    pthread_t tid; pthread_create(&tid, NULL, bg_compressor, NULL);
    
    long long fp0 = get_fp();
    printf("Baseline: %lld MB\n", fp0/(1024*1024));
    
    // Write data
    printf("Writing 512MB...\n");
    for (size_t i=0; i<npages; i++) {
        uint8_t *p = (uint8_t*)vmem + i*PAGE_SZ;
        mprotect(p, PAGE_SZ, PROT_READ|PROT_WRITE);
        memset(p, i & 0xFF, PAGE_SZ);
        meta[i].state = PAGE_RESIDENT;
    }
    
    long long fp1 = get_fp();
    printf("After write: %lld MB (+%lld MB)\n\n", fp1/(1024*1024), (fp1-fp0)/(1024*1024));
    
    // Wait for compression
    printf("Compressing...\n");
    for (int t=0; t<10; t++) {
        sleep(1);
        long long fp = get_fp();
        size_t compressed = 0;
        for (size_t i=0; i<npages; i++) if (meta[i].state==PAGE_COMPRESSED) compressed++;
        printf("  [%2ds] footprint: %lld MB, compressed: %zu/%zu pages (%.0f%%), pool_used: %lu KB\n",
               t+1, fp/(1024*1024), compressed, npages, 100.0*compressed/npages,
               (unsigned long)(pool_next/1024));
    }
    
    long long fp2 = get_fp();
    printf("\n═══ Result ═══\n");
    printf("Peak:     %lld MB\n", fp1/(1024*1024));
    printf("Current:  %lld MB\n", fp2/(1024*1024));
    printf("Saved:    %lld MB (%.0f%%)\n", (fp1-fp2)/(1024*1024), 100.0*(fp1-fp2)/fp1);
    printf("Pool:     %llu KB\n", (unsigned long long)(pool_next/1024));
    
    // Verify
    printf("\nVerifying...\n");
    int ok = 1;
    for (size_t i=0; i<100 && ok; i++) {
        uint8_t *p = (uint8_t*)vmem + i*PAGE_SZ;
        // This will trigger fault handler if compressed
        mprotect(p, PAGE_SZ, PROT_READ|PROT_WRITE);  // ensure accessible
        if (meta[i].state == PAGE_COMPRESSED) {
            cpu_decompress_page(pool+meta[i].pool_offset, meta[i].comp_size, p);
            meta[i].state = PAGE_RESIDENT;
        }
        for (size_t j=0; j<PAGE_SZ; j++) if (p[j] != (i & 0xFF)) { ok=0; break; }
    }
    printf("Integrity: %s\n", ok ? "✅ PERFECT" : "❌ FAIL");
    
    return 0;
}
