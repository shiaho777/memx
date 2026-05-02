// Test real-world workload memory savings
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>

#define MB (1024ULL*1024)

static long long get_fp(void) {
    task_vm_info_data_t info; mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
    return (long long)info.phys_footprint;
}

int main() {
    printf("═══ Real Workload Memory Test ═══\n\n");
    long long fp_base = get_fp();
    printf("Baseline footprint: %lld MB\n\n", fp_base/MB);

    // 1GB zero-filled (sparse LLM weights) - use mmap so MemX can intercept
    printf("1. Allocating 1GB zero-filled (simulated sparse weights)...\n");
    size_t sz1 = 1024*MB;
    void *p1 = mmap(NULL, sz1, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p1 == MAP_FAILED) p1 = NULL;
    // mmap already provides zero-filled pages, no memset needed
    printf("   ptr=%p, footprint: %lld MB\n", p1, get_fp()/MB);

    // 512MB repetitive text (JSON)
    printf("2. Allocating 512MB repetitive text (JSON-like)...\n");
    size_t sz2 = 512*MB;
    char *p2 = (char*)malloc(sz2);
    if (p2) {
        const char *pat = "{\"name\":\"user_12345\",\"email\":\"test@example.com\",\"data\":[1,2,3,4,5,6,7,8,9,10],\"active\":true}";
        size_t plen = strlen(pat);
        for (size_t i = 0; i < sz2; i += plen)
            memcpy(p2+i, pat, (i+plen<=sz2) ? plen : sz2-i);
    }
    printf("   ptr=%p, footprint: %lld MB\n", p2, get_fp()/MB);

    // 512MB mixed heap (various sizes, compressible data)
    printf("3. Allocating 512MB mixed heap (64-320KB chunks)...\n");
    void *ptrs[8192]; int nptrs = 0; size_t sz3 = 0;
    srand(42);
    for (int i = 0; i < 8192; i++) {
        size_t s = 65536 + (rand() % (256*1024));
        ptrs[i] = malloc(s);
        if (!ptrs[i]) break;
        nptrs++;
        if (i%2 == 0) memset(ptrs[i], 0, s);
        else { for (size_t j=0; j<s/4; j++) ((int*)ptrs[i])[j] = i*1000+(int)j; }
        sz3 += s;
    }
    printf("   %d chunks, %llu MB, footprint: %lld MB\n", nptrs, (unsigned long long)(sz3/MB), get_fp()/MB);

    size_t total_alloc = sz1 + sz2 + sz3;
    long long fp_peak = get_fp();
    printf("\n   Total allocated: %llu MB\n", (unsigned long long)(total_alloc/MB));
    printf("   Current footprint: %lld MB\n\n", fp_peak/MB);

    // Wait for compression
    printf("Waiting 15s for GPU compression...\n");
    sleep(15);
    long long fp_compressed = get_fp();
    printf("   Footprint after compression: %lld MB\n", fp_compressed/MB);
    double saved_pct = (double)(total_alloc - fp_compressed + fp_base) / total_alloc * 100;
    if (saved_pct < 0) saved_pct = 0;
    printf("   Effective savings: %.0f%%\n\n", saved_pct);

    // Verify integrity
    printf("Verifying data integrity...\n");
    int ok = 1;
    if (p1) { for (size_t i = 0; i < sz1; i++) if (((char*)p1)[i] != 0) { ok = 0; printf("p1 mismatch at %zu\n", i); break; } }
    if (ok && p2) {
        const char *pat = "{\"name\":\"user_12345\",\"email\":\"test@example.com\",\"data\":[1,2,3,4,5,6,7,8,9,10],\"active\":true}";
        size_t plen = strlen(pat);
        for (size_t i = 0; i < sz2 && ok; i += plen) {
            size_t cm = (i+plen<=sz2) ? plen : sz2-i;
            if (memcmp(p2+i, pat, cm) != 0) { ok = 0; printf("p2 mismatch at %zu\n", i); }
        }
    }
    if (ok) {
        srand(42);
        for (int i = 0; i < nptrs && ok; i++) {
            size_t s = 65536 + (rand() % (256*1024));
            if (i%2 == 0) { for (size_t j=0; j<s; j++) if (((char*)ptrs[i])[j] != 0) { ok = 0; printf("ptrs[%d] mismatch at %zu\n", i, j); break; } }
            else { for (size_t j=0; j<s/4; j++) if (((int*)ptrs[i])[j] != (int)(i*1000+j)) { ok = 0; printf("ptrs[%d] mismatch at %zu\n", i, j*4); break; } }
        }
    }
    printf("   Integrity: %s\n\n", ok ? "PERFECT" : "CORRUPT");

    if (p1) munmap(p1, sz1); free(p2);
    for (int i = 0; i < nptrs; i++) free(ptrs[i]);

    printf("═══ Summary ═══\n");
    printf("Allocated:  %llu MB\n", (unsigned long long)(total_alloc/MB));
    printf("Compressed: %lld MB (%.0f%% saved)\n", fp_compressed/MB, saved_pct);
    printf("Integrity:  %s\n", ok ? "PERFECT" : "CORRUPT");

    return 0;
}
