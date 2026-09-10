#include "../core/memx_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

static uint32_t xs_state = 0x9E3779B9u;

static uint32_t xs(void) {
    uint32_t x = xs_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xs_state = x;
    return x;
}

static uint8_t page[PAGE_SZ];
static uint8_t comp[PAGE_SZ];
static uint8_t back[PAGE_SZ];

static void fill_pattern(int kind, uint32_t seed) {
    xs_state = seed;
    if (kind == 0) {
        memset(page, 0, PAGE_SZ);
    } else if (kind == 1) {
        for (uint32_t i = 0; i < PAGE_SZ; i++) page[i] = (uint8_t)(i * 3 + seed);
    } else if (kind == 2) {
        for (uint32_t i = 0; i < PAGE_SZ / 2; i++) {
            uint16_t h = (uint16_t)(((i * 37 + seed) & 0xFF) | (((seed >> 3) & 0x1F) << 8));
            memcpy(page + i * 2, &h, 2);
        }
    } else if (kind == 3) {
        memset(page, 0, PAGE_SZ);
        for (int k = 0; k < 64; k++) {
            uint32_t off = xs() % PAGE_SZ;
            page[off] = (uint8_t)(xs() | 1u);
        }
    } else {
        for (uint32_t i = 0; i < PAGE_SZ; i += 4) {
            uint32_t r = xs();
            memcpy(page + i, &r, 4);
        }
    }
}

static int hook_inflate(const uint8_t *src, uint32_t src_len, uint8_t *dst, uint32_t *dst_len) {
    uLongf dl = *dst_len;
    if (uncompress(dst, &dl, src, src_len) != Z_OK) return -1;
    *dst_len = (uint32_t)dl;
    return 0;
}

static int hook_uncompress(const uint8_t *src, uint32_t src_len, uint8_t *dst, uint32_t expected_len) {
    uLongf dl = expected_len;
    if (uncompress(dst, &dl, src, src_len) != Z_OK || dl != expected_len) return -1;
    return 0;
}

static void make_zlib_payload(uint8_t *dst, uint32_t *out_len, const uint8_t *src_page) {
    uLongf zlen = PAGE_SZ;
    compress2(dst + 8, &zlen, src_page, PAGE_SZ, 1);
    dst[0] = 0x4D;
    dst[1] = 0x58;
    dst[2] = MEMX_CODEC_ZLIB;
    dst[3] = 1;
    dst[4] = (uint8_t)(zlen & 0xFF);
    dst[5] = (uint8_t)((zlen >> 8) & 0xFF);
    dst[6] = (uint8_t)((zlen >> 16) & 0xFF);
    dst[7] = (uint8_t)((zlen >> 24) & 0xFF);
    *out_len = (uint32_t)(8 + zlen);
}

int main(void) {
    int failures = 0;

    fill_pattern(2, 0x1234);
    uint8_t zlib_payload[PAGE_SZ + 64];
    uint32_t zlib_len = 0;
    make_zlib_payload(zlib_payload, &zlib_len, page);

    memset(back, 0xCC, PAGE_SZ);
    cpu_decompress(zlib_payload, zlib_len, back);
    int all_zero = 1;
    for (uint32_t i = 0; i < PAGE_SZ; i++) if (back[i]) { all_zero = 0; break; }
    if (!all_zero) {
        printf("FAIL: zlib payload decoded without hooks registered\n");
        failures++;
    }

    memx_core_hooks_t hooks;
    hooks.inflate = hook_inflate;
    hooks.uncompress = hook_uncompress;
    memx_core_set_hooks(&hooks);

    memset(back, 0xCC, PAGE_SZ);
    cpu_decompress(zlib_payload, zlib_len, back);
    if (memcmp(back, page, PAGE_SZ) != 0) {
        printf("FAIL: zlib payload did not roundtrip with hooks\n");
        failures++;
    }

    static const uint8_t ZERO_PAGE[8] = {0x4D, 0x58, 0x03, 0x00, 0xFD, 0x00, 0x00, 0x40};
    memset(back, 0xCC, PAGE_SZ);
    cpu_decompress(ZERO_PAGE, 8, back);
    if (!page_is_all_zero(back)) {
        printf("FAIL: zero-page constant did not decode to zeros\n");
        failures++;
    }

    for (int kind = 0; kind <= 4; kind++) {
        fill_pattern(kind, 0xA5 + kind);
        uint32_t csz;

        csz = tensor_fp16_split_compress(page, comp, PAGE_SZ);
        if (csz > 0) {
            memset(back, 0xCC, PAGE_SZ);
            cpu_decompress(comp, csz, back);
            if (memcmp(back, page, PAGE_SZ) != 0) {
                printf("FAIL: fp16_split roundtrip kind=%d\n", kind);
                failures++;
            }
        }

        csz = tensor_fp16_delta_split_compress(page, comp, PAGE_SZ);
        if (csz > 0) {
            memset(back, 0xCC, PAGE_SZ);
            cpu_decompress(comp, csz, back);
            if (memcmp(back, page, PAGE_SZ) != 0) {
                printf("FAIL: fp16_delta_split roundtrip kind=%d\n", kind);
                failures++;
            }
        }

        csz = tensor_bitplane16_compress(page, comp, PAGE_SZ);
        if (csz > 0) {
            memset(back, 0xCC, PAGE_SZ);
            cpu_decompress(comp, csz, back);
            if (memcmp(back, page, PAGE_SZ) != 0) {
                printf("FAIL: bitplane16 roundtrip kind=%d\n", kind);
                failures++;
            }
        }

        csz = tensor_sparse_byte_compress(page, comp, PAGE_SZ);
        if (kind == 3) {
            if (csz == 0) {
                printf("FAIL: sparse_byte rejected a sparse page kind=%d\n", kind);
                failures++;
            } else {
                memset(back, 0xCC, PAGE_SZ);
                cpu_decompress(comp, csz, back);
                if (memcmp(back, page, PAGE_SZ) != 0) {
                    printf("FAIL: sparse_byte roundtrip kind=%d\n", kind);
                    failures++;
                }
            }
        }
    }

    {
        uint8_t rle_src[300];
        uint8_t rle_dst[600];
        uint8_t rle_back[300];
        for (int i = 0; i < 300; i++) rle_src[i] = (uint8_t)(i / 37);
        uint32_t rl = rle8_encode(rle_src, 300, rle_dst, sizeof(rle_dst));
        if (rl == 0 || rle8_decode(rle_dst, rl, rle_back, 300) != 0 || memcmp(rle_src, rle_back, 300) != 0) {
            printf("FAIL: rle8 roundtrip\n");
            failures++;
        }
    }

    {
        PageMeta m;
        memset(&m, 0, sizeof(m));
        m.state = MEMX_PAGE_COMPRESSING;
        m.dirty = 0;
        m.write_seq = 7;
        if (!page_compress_meta_stable(&m, 7)) { printf("FAIL: meta_stable seq match\n"); failures++; }
        if (page_compress_meta_stable(&m, 8)) { printf("FAIL: meta_stable seq mismatch\n"); failures++; }
        m.dirty = 1;
        if (page_compress_meta_stable(&m, 7)) { printf("FAIL: meta_stable dirty\n"); failures++; }
        m.state = MEMX_PAGE_RESIDENT;
        m.dirty = 0;
        if (page_compress_meta_stable(&m, 7)) { printf("FAIL: meta_stable state\n"); failures++; }
    }

    {
        PageMeta m;
        memset(&m, 0, sizeof(m));
        m.tensor_role = MEMX_ROLE_WEIGHT;
        m.tensor_flags = MEMX_FLAG_READ_MOSTLY;
        if (!page_wants_write_protect(&m)) { printf("FAIL: wants_write_protect weight\n"); failures++; }
        if (page_stable_need(&m) != 0) { printf("FAIL: stable_need cold weight\n"); failures++; }
        m.tensor_flags = 0;
        if (page_stable_need(&m) != 1) { printf("FAIL: stable_need weight\n"); failures++; }
        m.tensor_role = MEMX_ROLE_UNKNOWN;
        m.tensor_flags = 0;
        if (page_stable_need(&m) != 2) { printf("FAIL: stable_need unknown\n"); failures++; }
        if (fault_stream_for_role(MEMX_ROLE_KV_CACHE) != 0) { printf("FAIL: stream kv\n"); failures++; }
        if (fault_stream_for_role(MEMX_ROLE_WEIGHT) != 1) { printf("FAIL: stream weight\n"); failures++; }
        if (fault_stream_for_role(MEMX_ROLE_UNKNOWN) != 2) { printf("FAIL: stream other\n"); failures++; }
    }

    {
        fill_pattern(0, 1);
        if (!page_is_all_zero(page)) { printf("FAIL: page_is_all_zero\n"); failures++; }
        fill_pattern(1, 2);
        uint8_t copy[PAGE_SZ];
        memcpy(copy, page, PAGE_SZ);
        if (!page_bytes_equal(page, copy)) { printf("FAIL: page_bytes_equal same\n"); failures++; }
        copy[PAGE_SZ / 2] ^= 0xFF;
        if (page_bytes_equal(page, copy)) { printf("FAIL: page_bytes_equal diff\n"); failures++; }
    }

    if (failures) {
        printf("core kernel test: FAILED (%d)\n", failures);
        return 1;
    }
    printf("core kernel test: OK (codec roundtrips, hook vtable, state helpers, freestanding core)\n");
    return 0;
}
