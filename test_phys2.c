// Diagnose WHY physical memory isn't being saved in the dylib
// Test each component separately
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <mach/mach.h>

static long long get_fp(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
    return (long long)info.phys_footprint;
}

int main() {
    printf("═══ Component-by-Component Diagnosis ═══\n\n");
    long long fp0 = get_fp();
    printf("Baseline: %lld MB\n\n", fp0/(1024*1024));
    
    // Test 1: Allocate 256MB via mmap PROT_NONE, then touch, then mprotect back
    printf("Test 1: mmap + touch + mprotect(PROT_NONE)\n");
    size_t sz = 256ULL * 1024 * 1024;
    void *mem = mmap(NULL, sz, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    printf("  After mmap(PROT_NONE): %lld MB\n", (get_fp()-fp0)/(1024*1024));
    
    mprotect(mem, sz, PROT_READ|PROT_WRITE);
    memset(mem, 0xAB, sz);
    long long fp1 = get_fp();
    printf("  After touch: %lld MB (+%lld MB)\n", fp1/(1024*1024), (fp1-fp0)/(1024*1024));
    
    mprotect(mem, sz, PROT_NONE);
    long long fp2 = get_fp();
    printf("  After mprotect(PROT_NONE): %lld MB (released: %lld MB)\n\n",
           fp2/(1024*1024), (fp1-fp2)/(1024*1024));
    
    // Test 2: Pool allocation - how much does a 48GB mmap cost?
    printf("Test 2: Large pool mmap (48GB, PROT_READ|PROT_WRITE, untouched)\n");
    size_t pool_sz = 48ULL * 1024 * 1024 * 1024;
    void *pool = mmap(NULL, pool_sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    long long fp3 = get_fp();
    printf("  After pool mmap: %lld MB (+%lld MB)\n", fp3/(1024*1024), (fp3-fp2)/(1024*1024));
    
    // Write 23MB to pool (simulating compressed data)
    memset(pool, 0, 23 * 1024 * 1024);
    long long fp4 = get_fp();
    printf("  After writing 23MB to pool: %lld MB (+%lld MB)\n", fp4/(1024*1024), (fp4-fp3)/(1024*1024));
    
    // Test 3: MADV_FREE on unused pool pages
    printf("\nTest 3: MADV_FREE on pool pages beyond 23MB\n");
    madvise(pool + 23*1024*1024, pool_sz - 23*1024*1024, MADV_FREE);
    long long fp5 = get_fp();
    printf("  After MADV_FREE: %lld MB (released: %lld MB)\n\n", fp5/(1024*1024), (fp4-fp5)/(1024*1024));
    
    // Test 4: Simulate the full cycle
    printf("Test 4: Full cycle - allocate, compress to pool, release original\n");
    // Re-touch the 256MB region
    mprotect(mem, sz, PROT_READ|PROT_WRITE);
    memset(mem, 0xAB, sz);
    long long fp6 = get_fp();
    printf("  Re-touch 256MB: %lld MB\n", fp6/(1024*1024));
    
    // "Compress" - just copy 1MB to pool (simulating 256x compression)
    memcpy(pool + 24*1024*1024, mem, 1024*1024);
    // Release original
    mprotect(mem, sz, PROT_NONE);
    long long fp7 = get_fp();
    printf("  After compress+release: %lld MB (saved: %lld MB)\n\n",
           fp7/(1024*1024), (fp6-fp7)/(1024*1024));
    
    printf("═══ Summary ═══\n");
    printf("mprotect(PROT_NONE) works: %s\n", (fp1-fp2) > 200*1024*1024 ? "✅ YES" : "❌ NO");
    printf("Large mmap without touch: %s\n", (fp3-fp2) < 10*1024*1024 ? "✅ FREE" : "❌ COSTLY");
    printf("Pool write cost: %lld MB for 23MB data\n", (fp4-fp3)/(1024*1024));
    printf("Full cycle net saving: %lld MB\n", (fp6-fp7)/(1024*1024));
    
    munmap(mem, sz);
    munmap(pool, pool_sz);
    return 0;
}
