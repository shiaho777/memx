/*
 * End-to-end test for the POSIX reference adapter: first-touch writes,
 * background compression (zero + fp16-split paths), fault-driven decompress
 * bitexact, second-generation rewrite, re-compression, re-verification.
 * Page-size agnostic — runs at whatever the core was compiled with.
 */
#include "../platform/posix/memx_posix.h"
#include "../core/memx_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEST_PAGES 48

static void fill_fp16(uint8_t *dst, uint32_t seed) {
    for (uint32_t i = 0; i < PAGE_SZ / 2; i++) {
        uint16_t h = (uint16_t)(0x3C00 + ((i * 37 + seed) & 0x7F));
        dst[i * 2] = (uint8_t)(h & 0xFF);
        dst[i * 2 + 1] = (uint8_t)(h >> 8);
    }
}

static int wait_compressed(uint64_t minimum, double seconds) {
    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += (time_t)seconds;
    for (;;) {
        if (memx_posix_compressed_pages() >= minimum) return 0;
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec > deadline.tv_sec ||
            (now.tv_sec == deadline.tv_sec && now.tv_nsec >= deadline.tv_nsec))
            return -1;
        struct timespec nap = {0, 20000000L};
        nanosleep(&nap, NULL);
    }
}

int main(void) {
    size_t bytes = (size_t)TEST_PAGES * PAGE_SZ;
    int rc = memx_posix_init(bytes, MEMX_ROLE_WEIGHT, MEMX_DTYPE_FP16,
                             MEMX_FLAG_READ_MOSTLY | MEMX_FLAG_COLD);
    if (rc == -2) {
        printf("posix adapter: SKIP (hardware page > %u-byte core page; needs a matching platform)\n",
               (unsigned)PAGE_SZ);
        return 0;
    }
    if (rc != 0) {
        printf("posix adapter: FAILED (init rc=%d)\n", rc);
        return 1;
    }
    uint8_t *r = memx_posix_region();

    uint8_t *golden = malloc(bytes);
    /* gen 1: even pages fp16 pattern, odd pages zero */
    for (uint32_t p = 0; p < TEST_PAGES; p++) {
        if ((p & 1) == 0) fill_fp16(golden + (size_t)p * PAGE_SZ, 0x11 + p);
        else memset(golden + (size_t)p * PAGE_SZ, 0, PAGE_SZ);
    }
    memcpy(r, golden, bytes);

    if (wait_compressed(TEST_PAGES, 30.0) != 0) {
        printf("posix adapter: FAILED (gen1 compress; got %llu)\n",
               (unsigned long long)memx_posix_compressed_pages());
        return 1;
    }

    if (memcmp(r, golden, bytes) != 0) {
        printf("posix adapter: FAILED (gen1 readback not bitexact)\n");
        return 1;
    }

    /* gen 2: rewrite everything with a different pattern */
    for (uint32_t p = 0; p < TEST_PAGES; p++) {
        if ((p & 1) == 0) fill_fp16(golden + (size_t)p * PAGE_SZ, 0x77 + p);
        else fill_fp16(golden + (size_t)p * PAGE_SZ, 0x99 + p);
    }
    memcpy(r, golden, bytes);

    if (wait_compressed(TEST_PAGES, 30.0) != 0) {
        printf("posix adapter: FAILED (gen2 compress; got %llu)\n",
               (unsigned long long)memx_posix_compressed_pages());
        return 1;
    }
    if (memcmp(r, golden, bytes) != 0) {
        printf("posix adapter: FAILED (gen2 readback not bitexact)\n");
        return 1;
    }

    uint64_t faults = memx_posix_faults();
    uint64_t compressed = memx_posix_compressed_pages();
    memx_posix_shutdown();
    free(golden);

    printf("posix adapter: OK (%u-byte pages, %d pages, compressed=%llu, faults=%llu)\n",
           (unsigned)PAGE_SZ, TEST_PAGES,
           (unsigned long long)compressed, (unsigned long long)faults);
    return 0;
}
