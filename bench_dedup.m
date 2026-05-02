// MemX Dedup Benchmark - measures pool savings from page deduplication
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <mach/mach.h>
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
    printf("║   MemX Page Deduplication Benchmark    ║\n");
    printf("╚══════════════════════════════════════╝\n\n");
    
    long long base = get_fp();
    
    // Test 1: 1GB all zeros (maximum dedup - all pages identical)
    printf("─── Test 1: 1GB All Zeros (max dedup) ───\n");
    size_t sz1 = 1024*MB;
    char *p1 = (char*)mmap(NULL, sz1, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(p1, 0, sz1);
    sleep(8);
    printf("  Footprint: %lld MB (baseline %lld MB)\n", get_fp()/MB, base/MB);
    int ok1 = 1;
    for (size_t i = 0; i < sz1 && ok1; i++) if (p1[i] != 0) { ok1 = 0; }
    printf("  Integrity: %s\n\n", ok1 ? "PERFECT" : "CORRUPT");
    
    // Test 2: 1GB repeated 16KB pattern (each page = same 16KB block)
    printf("─── Test 2: 1GB Repeated Pattern ───\n");
    size_t sz2 = 1024*MB;
    char *p2 = (char*)mmap(NULL, sz2, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    // Create one page of pattern, repeat it
    char template_page[PAGE_SZ];
    for (size_t i = 0; i < PAGE_SZ; i++) template_page[i] = (char)(i * 7 + 13);
    for (size_t off = 0; off < sz2; off += PAGE_SZ)
        memcpy(p2 + off, template_page, PAGE_SZ);
    sleep(8);
    printf("  Footprint: %lld MB\n", get_fp()/MB);
    int ok2 = 1;
    for (size_t off = 0; off < sz2 && ok2; off += PAGE_SZ)
        if (memcmp(p2+off, template_page, PAGE_SZ) != 0) { ok2 = 0; }
    printf("  Integrity: %s\n\n", ok2 ? "PERFECT" : "CORRUPT");
    
    // Test 3: 1GB unique pages (no dedup possible)
    printf("─── Test 3: 1GB Unique Pages (no dedup) ───\n");
    size_t sz3 = 1024*MB;
    char *p3 = (char*)mmap(NULL, sz3, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    srand(99);
    for (size_t off = 0; off < sz3; off += PAGE_SZ) {
        // Each page has unique seed-based content
        unsigned int seed = (unsigned int)(off / PAGE_SZ);
        for (size_t j = 0; j < PAGE_SZ/4; j++) ((unsigned int*)(p3+off))[j] = rand_r(&seed);
    }
    sleep(8);
    printf("  Footprint: %lld MB\n\n", get_fp()/MB);
    
    // Test 4: 512MB with 10 unique templates (moderate dedup)
    printf("─── Test 4: 512MB, 10 Templates (moderate dedup) ───\n");
    size_t sz4 = 512*MB;
    char *p4 = (char*)mmap(NULL, sz4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    char templates[10][PAGE_SZ];
    for (int t = 0; t < 10; t++)
        for (size_t j = 0; j < PAGE_SZ; j++) templates[t][j] = (char)(t*25 + j*3);
    for (size_t off = 0; off < sz4; off += PAGE_SZ)
        memcpy(p4+off, templates[(off/PAGE_SZ) % 10], PAGE_SZ);
    sleep(8);
    printf("  Footprint: %lld MB\n", get_fp()/MB);
    int ok4 = 1;
    for (size_t off = 0; off < sz4 && ok4; off += PAGE_SZ)
        if (memcmp(p4+off, templates[(off/PAGE_SZ)%10], PAGE_SZ) != 0) { ok4 = 0; }
    printf("  Integrity: %s\n\n", ok4 ? "PERFECT" : "CORRUPT");
    
    printf("═══ Summary ═══\n");
    printf("  All integrity checks: %s\n", (ok1&&ok2&&ok4) ? "PERFECT" : "CORRUPT");
    return (ok1&&ok2&&ok4) ? 0 : 1;
}
