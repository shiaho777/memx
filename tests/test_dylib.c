// Test program for libmemx.dylib
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main() {
    printf("=== MemX Dylib Test ===\n\n");
    
    // Allocate large chunks (should go through compressed pool)
    printf("1. Allocating 512MB in 64KB chunks...\n");
    void *ptrs[8192];
    int count = 0;
    for (int i = 0; i < 8192; i++) {
        ptrs[i] = malloc(65536);  // 64KB each
        if (!ptrs[i]) break;
        count++;
    }
    printf("   Allocated %d chunks (%d MB)\n", count, count * 64 / 1024);
    
    // Write data to some chunks
    printf("2. Writing data to first 100 chunks...\n");
    for (int i = 0; i < 100 && i < count; i++) {
        memset(ptrs[i], i & 0xFF, 65536);
    }
    
    // Wait for compression
    printf("3. Sleeping 5s for GPU compression...\n");
    sleep(5);
    
    // Verify
    printf("4. Verifying data integrity...\n");
    int mismatches = 0;
    for (int i = 0; i < 100 && i < count; i++) {
        unsigned char *p = (unsigned char*)ptrs[i];
        for (int j = 0; j < 65536; j++) {
            if (p[j] != (unsigned char)(i & 0xFF)) {
                mismatches++;
                break;
            }
        }
    }
    printf("   Result: %s (%d mismatches)\n",
           mismatches == 0 ? "✅ PERFECT" : "❌ MISMATCH", mismatches);
    
    // Free all
    printf("5. Freeing all...\n");
    for (int i = 0; i < count; i++) free(ptrs[i]);
    
    printf("\nDone!\n");
    return 0;
}
