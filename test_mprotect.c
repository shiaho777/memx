// Test: Does mprotect(PROT_NONE) actually release physical memory on macOS?
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <mach/mach.h>

static long long get_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
    return (long long)info.phys_footprint;
}

int main() {
    printf("═══ mprotect(PROT_NONE) Physical Memory Release Test ═══\n\n");
    
    long long fp0 = get_footprint();
    printf("Baseline: %lld MB\n", fp0/(1024*1024));
    
    // Allocate 256MB via mmap
    size_t size = 256 * 1024 * 1024;
    void *mem = mmap(NULL, size, PROT_READ|PROT_WRITE,
                     MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) { printf("mmap failed\n"); return 1; }
    
    // Touch all pages
    memset(mem, 0xAB, size);
    long long fp1 = get_footprint();
    printf("After mmap+touch (256MB): %lld MB (delta: %lld MB)\n",
           fp1/(1024*1024), (fp1-fp0)/(1024*1024));
    
    // Now mprotect to PROT_NONE - should release physical pages
    mprotect(mem, size, PROT_NONE);
    long long fp2 = get_footprint();
    printf("After mprotect(PROT_NONE): %lld MB (delta: %lld MB)\n",
           fp2/(1024*1024), (fp2-fp0)/(1024*1024));
    printf("Physical memory released: %lld MB\n\n", (fp1-fp2)/(1024*1024));
    
    // Try MADV_DONTNEED instead
    mprotect(mem, size, PROT_READ|PROT_WRITE);  // need access for madvise
    memset(mem, 0xAB, size);  // re-touch
    long long fp3 = get_footprint();
    printf("Re-touch: %lld MB\n", fp3/(1024*1024));
    
    madvise(mem, size, MADV_DONTNEED);
    long long fp4 = get_footprint();
    printf("After madvise(DONTNEED): %lld MB (released: %lld MB)\n\n",
           fp4/(1024*1024), (fp3-fp4)/(1024*1024));
    
    // Try MADV_PAGEOUT (macOS 13+)
    memset(mem, 0xAB, size);
    long long fp5 = get_footprint();
    printf("Re-touch: %lld MB\n", fp5/(1024*1024));
    
    int rc = madvise(mem, size, MADV_PAGEOUT);
    long long fp6 = get_footprint();
    printf("After madvise(PAGEOUT): rc=%d, footprint: %lld MB (released: %lld MB)\n\n",
           rc, fp6/(1024*1024), (fp5-fp6)/(1024*1024));
    
    // Try decommission (macOS specific)
    // Actually, let's try the combination: PROT_NONE + madvise
    mprotect(mem, size, PROT_NONE);
    madvise(mem, size, MADV_FREE);  // MADV_FREE tells kernel pages are reclaimable
    long long fp7 = get_footprint();
    printf("PROT_NONE + MADV_FREE: %lld MB (released: %lld MB)\n",
           fp7/(1024*1024), (fp5-fp7)/(1024*1024));
    
    // Summary
    printf("\n═══ Summary ═══\n");
    printf("mprotect(PROT_NONE) released: %lld MB / 256 MB\n", (fp1-fp2)/(1024*1024));
    printf("madvise(DONTNEED) released:   %lld MB / 256 MB\n", (fp3-fp4)/(1024*1024));
    printf("madvise(PAGEOUT) released:    %lld MB / 256 MB\n", (fp5-fp6)/(1024*1024));
    printf("PROT_NONE+MADV_FREE released: %lld MB / 256 MB\n", (fp5-fp7)/(1024*1024));
    
    munmap(mem, size);
    return 0;
}
