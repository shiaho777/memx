#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PAGE_SZ 16384U
#define PAGE_COUNT 2048U

typedef struct codec_result {
    const char *name;
    uint64_t compressed_bytes;
    uint64_t encode_ticks;
    uint64_t decode_ticks;
    uint32_t wins;
    uint32_t failed;
} codec_result_t;

typedef struct dataset {
    const char *name;
    uint8_t *pages;
    uint32_t page_count;
} dataset_t;

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static uint32_t rle8_encode(const uint8_t *src, uint32_t len, uint8_t *dst, uint32_t cap) {
    uint32_t ip = 0, op = 0;
    while (ip < len) {
        uint8_t value = src[ip];
        uint32_t run = 1;
        while (ip + run < len && src[ip + run] == value && run < 255) run++;
        if (op + 2 > cap) return 0;
        dst[op++] = (uint8_t)run;
        dst[op++] = value;
        ip += run;
    }
    return op;
}

static int rle8_decode(const uint8_t *src, uint32_t len, uint8_t *dst, uint32_t out_len) {
    uint32_t ip = 0, op = 0;
    while (ip + 1 < len && op < out_len) {
        uint32_t run = src[ip++];
        uint8_t value = src[ip++];
        if (run == 0 || run > out_len - op) return -1;
        memset(dst + op, value, run);
        op += run;
    }
    return (ip == len && op == out_len) ? 0 : -1;
}

static uint32_t codec_split_rle_encode(const uint8_t *src, uint8_t *dst) {
    const uint32_t half_count = PAGE_SZ / 2;
    uint8_t *lo = dst + 16;
    uint8_t hi[PAGE_SZ / 2];
    uint8_t rle[PAGE_SZ];
    for (uint32_t i = 0; i < half_count; i++) {
        lo[i] = src[i * 2];
        hi[i] = src[i * 2 + 1];
    }
    uint32_t hi_rle = rle8_encode(hi, half_count, rle, sizeof(rle));
    if (hi_rle == 0 || 16 + half_count + hi_rle >= PAGE_SZ) return 0;
    dst[0] = 'S';
    dst[1] = 'R';
    dst[2] = (uint8_t)(half_count & 0xFF);
    dst[3] = (uint8_t)((half_count >> 8) & 0xFF);
    dst[4] = (uint8_t)(hi_rle & 0xFF);
    dst[5] = (uint8_t)((hi_rle >> 8) & 0xFF);
    memcpy(dst + 16 + half_count, rle, hi_rle);
    return 16 + half_count + hi_rle;
}

static int codec_split_rle_decode(const uint8_t *src, uint32_t size, uint8_t *dst) {
    if (size < 16 || src[0] != 'S' || src[1] != 'R') return -1;
    uint32_t half_count = (uint32_t)src[2] | ((uint32_t)src[3] << 8);
    uint32_t hi_rle = (uint32_t)src[4] | ((uint32_t)src[5] << 8);
    if (half_count != PAGE_SZ / 2 || 16 + half_count + hi_rle > size) return -1;
    uint8_t hi[PAGE_SZ / 2];
    if (rle8_decode(src + 16 + half_count, hi_rle, hi, half_count) != 0) return -1;
    const uint8_t *lo = src + 16;
    for (uint32_t i = 0; i < half_count; i++) {
        dst[i * 2] = lo[i];
        dst[i * 2 + 1] = hi[i];
    }
    return 0;
}

static uint32_t codec_delta_split_encode(const uint8_t *src, uint8_t *dst) {
    const uint32_t half_count = PAGE_SZ / 2;
    uint8_t lo_delta[PAGE_SZ / 2];
    uint8_t hi[PAGE_SZ / 2];
    uint8_t lo_rle_tmp[PAGE_SZ];
    uint8_t hi_rle_tmp[PAGE_SZ];
    for (uint32_t i = 0; i < half_count; i++) {
        uint8_t lo = src[i * 2];
        uint8_t prev = i == 0 ? 0 : src[(i - 1) * 2];
        lo_delta[i] = (uint8_t)(lo - prev);
        hi[i] = src[i * 2 + 1];
    }
    uint32_t lo_rle = rle8_encode(lo_delta, half_count, lo_rle_tmp, sizeof(lo_rle_tmp));
    uint32_t hi_rle = rle8_encode(hi, half_count, hi_rle_tmp, sizeof(hi_rle_tmp));
    if (lo_rle == 0 || hi_rle == 0 || 24 + lo_rle + hi_rle >= PAGE_SZ) return 0;
    dst[0] = 'D';
    dst[1] = 'S';
    dst[2] = (uint8_t)(half_count & 0xFF);
    dst[3] = (uint8_t)((half_count >> 8) & 0xFF);
    dst[4] = (uint8_t)(lo_rle & 0xFF);
    dst[5] = (uint8_t)((lo_rle >> 8) & 0xFF);
    dst[6] = (uint8_t)(hi_rle & 0xFF);
    dst[7] = (uint8_t)((hi_rle >> 8) & 0xFF);
    memcpy(dst + 24, lo_rle_tmp, lo_rle);
    memcpy(dst + 24 + lo_rle, hi_rle_tmp, hi_rle);
    return 24 + lo_rle + hi_rle;
}

static int codec_delta_split_decode(const uint8_t *src, uint32_t size, uint8_t *dst) {
    if (size < 24 || src[0] != 'D' || src[1] != 'S') return -1;
    uint32_t half_count = (uint32_t)src[2] | ((uint32_t)src[3] << 8);
    uint32_t lo_rle = (uint32_t)src[4] | ((uint32_t)src[5] << 8);
    uint32_t hi_rle = (uint32_t)src[6] | ((uint32_t)src[7] << 8);
    if (half_count != PAGE_SZ / 2 || 24 + lo_rle + hi_rle > size) return -1;
    uint8_t lo_delta[PAGE_SZ / 2];
    uint8_t hi[PAGE_SZ / 2];
    if (rle8_decode(src + 24, lo_rle, lo_delta, half_count) != 0) return -1;
    if (rle8_decode(src + 24 + lo_rle, hi_rle, hi, half_count) != 0) return -1;
    uint8_t lo = 0;
    for (uint32_t i = 0; i < half_count; i++) {
        lo = (uint8_t)(lo + lo_delta[i]);
        dst[i * 2] = lo;
        dst[i * 2 + 1] = hi[i];
    }
    return 0;
}

static uint32_t codec_xor_rle_encode(const uint8_t *src, uint8_t *dst) {
    uint8_t delta[PAGE_SZ];
    uint8_t rle[PAGE_SZ * 2];
    delta[0] = src[0];
    for (uint32_t i = 1; i < PAGE_SZ; i++) delta[i] = src[i] ^ src[i - 1];
    uint32_t rle_size = rle8_encode(delta, PAGE_SZ, rle, sizeof(rle));
    if (rle_size == 0 || 8 + rle_size >= PAGE_SZ) return 0;
    dst[0] = 'X';
    dst[1] = 'R';
    dst[2] = (uint8_t)(rle_size & 0xFF);
    dst[3] = (uint8_t)((rle_size >> 8) & 0xFF);
    memcpy(dst + 8, rle, rle_size);
    return 8 + rle_size;
}

static int codec_xor_rle_decode(const uint8_t *src, uint32_t size, uint8_t *dst) {
    if (size < 8 || src[0] != 'X' || src[1] != 'R') return -1;
    uint32_t rle_size = (uint32_t)src[2] | ((uint32_t)src[3] << 8);
    if (8 + rle_size > size) return -1;
    uint8_t delta[PAGE_SZ];
    if (rle8_decode(src + 8, rle_size, delta, PAGE_SZ) != 0) return -1;
    dst[0] = delta[0];
    for (uint32_t i = 1; i < PAGE_SZ; i++) dst[i] = delta[i] ^ dst[i - 1];
    return 0;
}

static uint32_t codec_bitplane16_encode(const uint8_t *src, uint8_t *dst) {
    uint8_t planes[16][PAGE_SZ / 16];
    memset(planes, 0, sizeof(planes));
    for (uint32_t i = 0; i < PAGE_SZ / 2; i++) {
        uint16_t v = (uint16_t)src[i * 2] | ((uint16_t)src[i * 2 + 1] << 8);
        for (uint32_t b = 0; b < 16; b++) {
            if (v & (1u << b)) planes[b][i >> 3] |= (uint8_t)(1u << (i & 7));
        }
    }
    uint32_t op = 34;
    dst[0] = 'B';
    dst[1] = 'P';
    for (uint32_t b = 0; b < 16; b++) {
        uint32_t sz = rle8_encode(planes[b], PAGE_SZ / 16, dst + op, PAGE_SZ - op);
        if (sz == 0) return 0;
        dst[2 + b * 2] = (uint8_t)(sz & 0xFF);
        dst[3 + b * 2] = (uint8_t)((sz >> 8) & 0xFF);
        op += sz;
        if (op >= PAGE_SZ) return 0;
    }
    return op;
}

static int codec_bitplane16_decode(const uint8_t *src, uint32_t size, uint8_t *dst) {
    if (size < 34 || src[0] != 'B' || src[1] != 'P') return -1;
    uint8_t planes[16][PAGE_SZ / 16];
    uint32_t ip = 34;
    for (uint32_t b = 0; b < 16; b++) {
        uint32_t sz = (uint32_t)src[2 + b * 2] | ((uint32_t)src[3 + b * 2] << 8);
        if (ip + sz > size) return -1;
        if (rle8_decode(src + ip, sz, planes[b], PAGE_SZ / 16) != 0) return -1;
        ip += sz;
    }
    for (uint32_t i = 0; i < PAGE_SZ / 2; i++) {
        uint16_t v = 0;
        for (uint32_t b = 0; b < 16; b++) {
            if (planes[b][i >> 3] & (uint8_t)(1u << (i & 7))) v |= (uint16_t)(1u << b);
        }
        dst[i * 2] = (uint8_t)(v & 0xFF);
        dst[i * 2 + 1] = (uint8_t)(v >> 8);
    }
    return 0;
}

static void fill_fp16_weights(uint8_t *pages, uint32_t page_count) {
    uint32_t rng = 0xC0FFEEu;
    for (uint32_t p = 0; p < page_count; p++) {
        uint8_t *page = pages + (uint64_t)p * PAGE_SZ;
        for (uint32_t i = 0; i < PAGE_SZ / 2; i++) {
            uint16_t mant = (uint16_t)(xorshift32(&rng) & 0x03FFu);
            uint16_t exp = (uint16_t)(14u + ((i + p) % 5u));
            uint16_t sign = (uint16_t)((xorshift32(&rng) & 1u) << 15);
            uint16_t v = sign | (uint16_t)(exp << 10) | mant;
            page[i * 2] = (uint8_t)(v & 0xFF);
            page[i * 2 + 1] = (uint8_t)(v >> 8);
        }
    }
}

static void fill_bf16_weights(uint8_t *pages, uint32_t page_count) {
    uint32_t rng = 0xBEEFu;
    for (uint32_t p = 0; p < page_count; p++) {
        uint8_t *page = pages + (uint64_t)p * PAGE_SZ;
        for (uint32_t i = 0; i < PAGE_SZ / 2; i++) {
            page[i * 2] = (uint8_t)(xorshift32(&rng) & 0x7F);
            page[i * 2 + 1] = (uint8_t)(0x3A + ((i + p) % 8));
        }
    }
}

static void fill_kv_smooth(uint8_t *pages, uint32_t page_count) {
    for (uint32_t p = 0; p < page_count; p++) {
        uint8_t *page = pages + (uint64_t)p * PAGE_SZ;
        uint16_t base = (uint16_t)(0x3000 + (p % 7));
        for (uint32_t i = 0; i < PAGE_SZ / 2; i++) {
            uint16_t v = (uint16_t)(base + ((i / 64) & 3));
            v ^= (uint16_t)((i * 13 + p) & 0x003F);
            page[i * 2] = (uint8_t)(v & 0xFF);
            page[i * 2 + 1] = (uint8_t)(v >> 8);
        }
    }
}

static void fill_kv_random(uint8_t *pages, uint32_t page_count) {
    uint32_t rng = 0x12345678u;
    for (uint64_t i = 0; i < (uint64_t)page_count * PAGE_SZ; i++) pages[i] = (uint8_t)xorshift32(&rng);
}

static void fill_prefix_repeat(uint8_t *pages, uint32_t page_count) {
    uint8_t template_page[PAGE_SZ];
    fill_kv_smooth(template_page, 1);
    for (uint32_t p = 0; p < page_count; p++) {
        uint8_t *page = pages + (uint64_t)p * PAGE_SZ;
        memcpy(page, template_page, PAGE_SZ);
        if (p % 16 == 0) page[(p * 97) % PAGE_SZ] ^= (uint8_t)p;
    }
}

static void run_codec(dataset_t *dataset, codec_result_t *result,
                      uint32_t (*encode)(const uint8_t *, uint8_t *),
                      int (*decode)(const uint8_t *, uint32_t, uint8_t *)) {
    uint8_t *compressed = malloc(PAGE_SZ * 2);
    uint8_t *roundtrip = malloc(PAGE_SZ);
    if (!compressed || !roundtrip) {
        result->failed = dataset->page_count;
        free(compressed);
        free(roundtrip);
        return;
    }
    for (uint32_t p = 0; p < dataset->page_count; p++) {
        const uint8_t *page = dataset->pages + (uint64_t)p * PAGE_SZ;
        uint64_t t0 = mach_absolute_time();
        uint32_t size = encode(page, compressed);
        uint64_t t1 = mach_absolute_time();
        if (size == 0 || size >= PAGE_SZ) continue;
        int ok = decode(compressed, size, roundtrip);
        uint64_t t2 = mach_absolute_time();
        if (ok != 0 || memcmp(page, roundtrip, PAGE_SZ) != 0) {
            result->failed++;
            continue;
        }
        result->compressed_bytes += size;
        result->encode_ticks += t1 - t0;
        result->decode_ticks += t2 - t1;
        result->wins++;
    }
    free(compressed);
    free(roundtrip);
}

static void print_result(const dataset_t *dataset, const codec_result_t *result, double ns_per_tick) {
    uint64_t logical = (uint64_t)dataset->page_count * PAGE_SZ;
    uint64_t stored = result->compressed_bytes + (uint64_t)(dataset->page_count - result->wins) * PAGE_SZ;
    double ratio = stored ? (double)logical / (double)stored : 0.0;
    double saved = logical ? 100.0 * (double)(logical - stored) / (double)logical : 0.0;
    double enc_us = result->wins ? ((double)result->encode_ticks * ns_per_tick) / ((double)result->wins * 1000.0) : 0.0;
    double dec_us = result->wins ? ((double)result->decode_ticks * ns_per_tick) / ((double)result->wins * 1000.0) : 0.0;
    printf("  %-14s ratio=%5.2fx saved=%5.1f%% wins=%4u/%u enc=%6.2fus dec=%6.2fus failed=%u\n",
           result->name,
           ratio,
           saved,
           result->wins,
           dataset->page_count,
           enc_us,
           dec_us,
           result->failed);
}

int main(void) {
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    double ns_per_tick = (double)timebase.numer / (double)timebase.denom;

    dataset_t datasets[] = {
        {.name = "fp16-weights", .pages = calloc(PAGE_COUNT, PAGE_SZ), .page_count = PAGE_COUNT},
        {.name = "bf16-weights", .pages = calloc(PAGE_COUNT, PAGE_SZ), .page_count = PAGE_COUNT},
        {.name = "kv-smooth", .pages = calloc(PAGE_COUNT, PAGE_SZ), .page_count = PAGE_COUNT},
        {.name = "kv-random", .pages = calloc(PAGE_COUNT, PAGE_SZ), .page_count = PAGE_COUNT},
        {.name = "prefix-repeat", .pages = calloc(PAGE_COUNT, PAGE_SZ), .page_count = PAGE_COUNT}
    };
    const size_t dataset_count = sizeof(datasets) / sizeof(datasets[0]);

    for (size_t i = 0; i < dataset_count; i++) {
        if (!datasets[i].pages) {
            fprintf(stderr, "allocation failed for %s\n", datasets[i].name);
            return 1;
        }
    }

    fill_fp16_weights(datasets[0].pages, datasets[0].page_count);
    fill_bf16_weights(datasets[1].pages, datasets[1].page_count);
    fill_kv_smooth(datasets[2].pages, datasets[2].page_count);
    fill_kv_random(datasets[3].pages, datasets[3].page_count);
    fill_prefix_repeat(datasets[4].pages, datasets[4].page_count);

    printf("MemX tensor codec benchmark pages=%u page_kb=%u\n", PAGE_COUNT, PAGE_SZ / 1024);
    for (size_t i = 0; i < dataset_count; i++) {
        codec_result_t results[] = {
            {.name = "split-rle"},
            {.name = "delta-split"},
            {.name = "xor-rle"},
            {.name = "bitplane16"}
        };
        printf("\n[%s]\n", datasets[i].name);
        run_codec(&datasets[i], &results[0], codec_split_rle_encode, codec_split_rle_decode);
        run_codec(&datasets[i], &results[1], codec_delta_split_encode, codec_delta_split_decode);
        run_codec(&datasets[i], &results[2], codec_xor_rle_encode, codec_xor_rle_decode);
        run_codec(&datasets[i], &results[3], codec_bitplane16_encode, codec_bitplane16_decode);
        for (size_t r = 0; r < sizeof(results) / sizeof(results[0]); r++) print_result(&datasets[i], &results[r], ns_per_tick);
    }

    for (size_t i = 0; i < dataset_count; i++) free(datasets[i].pages);
    return 0;
}
