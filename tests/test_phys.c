// Precise physical memory measurement using mach_task_info
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mach/mach.h>

static long long get_phys_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO,
                                  (task_info_t)&info, &count);
    if (kr != KERN_SUCCESS) return -1;
    // phys_footprint is the actual physical memory used
    return (long long)info.phys_footprint;
}

int main() {
    printf("═══ Precise Physical Memory Test ═══\n\n");
    
    long long fp0 = get_phys_footprint();
    printf("Baseline footprint: %lld MB\n", fp0 / (1024*1024));
    
    // Allocate 512MB
    int N = 8192;
    void **ptrs = malloc(N * sizeof(void*));
    for (int i = 0; i < N; i++) ptrs[i] = malloc(65536);
    
    long long fp1 = get_phys_footprint();
    printf("After alloc (512MB): %lld MB (delta: %lld MB)\n", 
           fp1/(1024*1024), (fp1-fp0)/(1024*1024));
    
    // Write data
    for (int i = 0; i < N; i++) {
        unsigned char *p = ptrs[i];
        int pat = i % 4;
        if (pat == 0) for (int j=0;j<65536;j++) p[j] = "Hello World! "[j%13];
        else if (pat == 1) for (int j=0;j<65536;j++) p[j] = (j*7+i*13)&0xFF;
        else if (pat == 2) memset(p, 0, 65536);
        else for (int j=0;j<65536;j++) p[j] = j&0xFF;
    }
    
    long long fp2 = get_phys_footprint();
    printf("After write:        %lld MB (delta: %lld MB)\n\n",
           fp2/(1024*1024), (fp2-fp0)/(1024*1024));
    
    // Wait for compression
    printf("Waiting for GPU compression...\n");
    for (int t = 0; t < 15; t++) {
        sleep(1);
        long long fp = get_phys_footprint();
        printf("  [%2ds] footprint: %lld MB (saved: %lld MB)\n",
               t+1, fp/(1024*1024), (fp2-fp)/(1024*1024));
    }
    
    long long fp3 = get_phys_footprint();
    printf("\n═══ Result ═══\n");
    printf("Allocated:     512 MB\n");
    printf("Footprint max: %lld MB\n", fp2/(1024*1024));
    printf("Footprint now: %lld MB\n", fp3/(1024*1024));
    printf("Saved:         %lld MB (%.0f%%)\n", 
           (fp2-fp3)/(1024*1024),
           fp2 > 0 ? 100.0*(fp2-fp3)/fp2 : 0);
    
    // Verify
    int ok = 1;
    for (int i = 0; i < N && ok; i++) {
        unsigned char *p = ptrs[i];
        int pat = i % 4;
        if (pat == 0) { for (int j=0;j<65536;j++) if (p[j]!="Hello World! "[j%13]) { ok=0; break; } }
        else if (pat == 1) { for (int j=0;j<65536;j++) if (p[j]!=((j*7+i*13)&0xFF)) { ok=0; break; } }
        else if (pat == 2) { for (int j=0;j<65536;j++) if (p[j]!=0) { ok=0; break; } }
        else { for (int j=0;j<65536;j++) if (p[j]!=(j&0xFF)) { ok=0; break; } }
    }
    printf("Integrity: %s\n", ok ? "✅ PERFECT" : "❌ FAIL");
    
    for (int i = 0; i < N; i++) free(ptrs[i]);
    free(ptrs);
    return 0;
}
