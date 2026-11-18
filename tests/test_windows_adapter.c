/*
 * End-to-end test for the Windows reference adapter: first-touch writes,
 * background compression (zero + fp16-split paths), fault-driven decompress
 * bitexact, second-generation rewrite, re-compression, re-verification.
 * Page-size agnostic — runs at whatever the core was compiled with.
 */
#include "../platform/windows/memx_windows.h"
#include "../core/memx_core.h"

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_PAGES 48

static void fill_fp16(uint8_t *dst, uint32_t seed) {
    for (uint32_t i = 0; i < PAGE_SZ / 2; i++) {
        uint16_t h = (uint16_t)(0x3C00 + ((i * 37 + seed) & 0x7F));
        dst[i * 2] = (uint8_t)(h & 0xFF);
        dst[i * 2 + 1] = (uint8_t)(h >> 8);
    }
}

static int wait_compressed(uint64_t minimum, double seconds) {
    ULONGLONG deadline = GetTickCount64() + (ULONGLONG)(seconds * 1000.0);
    for (;;) {
        if (memx_win_compressed_pages() >= minimum) return 0;
        if (GetTickCount64() >= deadline) return -1;
        Sleep(20);
    }
}

int main(void) {
    size_t bytes = (size_t)TEST_PAGES * PAGE_SZ;
    int rc = memx_win_init(bytes, MEMX_ROLE_WEIGHT, MEMX_DTYPE_FP16,
                           MEMX_FLAG_READ_MOSTLY | MEMX_FLAG_COLD);
    if (rc == -2) {
        printf("windows adapter: SKIP (hardware page > %u-byte core page)\n",
               (unsigned)PAGE_SZ);
        return 0;
    }
    if (rc != 0) {
        printf("windows adapter: FAILED (init rc=%d)\n", rc);
        return 1;
    }
    uint8_t *r = memx_win_region();

    uint8_t *golden = (uint8_t *)malloc(bytes);
    for (uint32_t p = 0; p < TEST_PAGES; p++) {
        if ((p & 1) == 0) fill_fp16(golden + (size_t)p * PAGE_SZ, 0x11 + p);
        else memset(golden + (size_t)p * PAGE_SZ, 0, PAGE_SZ);
    }
    memcpy(r, golden, bytes);

    if (wait_compressed(TEST_PAGES, 30.0) != 0) {
        printf("windows adapter: FAILED (gen1 compress; got %llu)\n",
               (unsigned long long)memx_win_compressed_pages());
        return 1;
    }
    if (memcmp(r, golden, bytes) != 0) {
        printf("windows adapter: FAILED (gen1 readback not bitexact)\n");
        return 1;
    }

    for (uint32_t p = 0; p < TEST_PAGES; p++) {
        if ((p & 1) == 0) fill_fp16(golden + (size_t)p * PAGE_SZ, 0x77 + p);
        else fill_fp16(golden + (size_t)p * PAGE_SZ, 0x99 + p);
    }
    memcpy(r, golden, bytes);

    if (wait_compressed(TEST_PAGES, 30.0) != 0) {
        printf("windows adapter: FAILED (gen2 compress; got %llu)\n",
               (unsigned long long)memx_win_compressed_pages());
        return 1;
    }
    if (memcmp(r, golden, bytes) != 0) {
        printf("windows adapter: FAILED (gen2 readback not bitexact)\n");
        return 1;
    }

    uint64_t faults = memx_win_faults();
    uint64_t compressed = memx_win_compressed_pages();
    memx_win_shutdown();
    free(golden);

    printf("windows adapter: OK (%u-byte pages, %d pages, compressed=%llu, faults=%llu)\n",
           (unsigned)PAGE_SZ, TEST_PAGES,
           (unsigned long long)compressed, (unsigned long long)faults);
    return 0;
}