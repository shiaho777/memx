#ifndef MEMX_CORE_H
#define MEMX_CORE_H

#include <stdint.h>
#include <stddef.h>

#if !defined(MEMX_FORCE_SCALAR) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#define MEMX_HAS_NEON 1
#else
#define MEMX_HAS_NEON 0
#endif

#define MEMX_CORE_PAGE_SZ 16384

#define MEMX_PAGE_NONE       0
#define MEMX_PAGE_RESIDENT   1
#define MEMX_PAGE_COMPRESSED 2
#define MEMX_PAGE_HOT        3
#define MEMX_PAGE_COMPRESSING 4

#define PAGE_NONE       MEMX_PAGE_NONE
#define PAGE_RESIDENT   MEMX_PAGE_RESIDENT
#define PAGE_COMPRESSED MEMX_PAGE_COMPRESSED
#define PAGE_HOT        MEMX_PAGE_HOT
#define PAGE_COMPRESSING MEMX_PAGE_COMPRESSING

#define MEMX_ROLE_UNKNOWN    0
#define MEMX_ROLE_WEIGHT     1
#define MEMX_ROLE_KV_CACHE   2
#define MEMX_ROLE_ACTIVATION 3
#define MEMX_ROLE_EMBEDDING  4
#define MEMX_ROLE_TEMPORARY  5
#define MEMX_ROLE_DATA       6

#define MEMX_DTYPE_UNKNOWN 0
#define MEMX_DTYPE_FP16    1
#define MEMX_DTYPE_BF16    2
#define MEMX_DTYPE_FP32    3
#define MEMX_DTYPE_INT8    4
#define MEMX_DTYPE_UINT8   5
#define MEMX_DTYPE_INT32   6

#define MEMX_FLAG_READ_MOSTLY (1u << 0)
#define MEMX_FLAG_SEQUENTIAL  (1u << 1)
#define MEMX_FLAG_HOT         (1u << 2)
#define MEMX_FLAG_NO_COMPRESS (1u << 3)
#define MEMX_FLAG_COLD        (1u << 4)

#define MEMX_CODEC_DEFAULT             0
#define MEMX_CODEC_TENSOR_FP16_SPLIT    0x81
#define MEMX_CODEC_TENSOR_BITPLANE16    0x82
#define MEMX_CODEC_TENSOR_SPARSE_BYTE   0x83
#define MEMX_CODEC_TENSOR_FP16_DELTA_SPLIT 0x84
#define MEMX_CODEC_ZLIB                 0x85
#define MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT  0x86
#define MEMX_CODEC_TENSOR_EXP_PACK      0x87

#define PAGE_SZ MEMX_CORE_PAGE_SZ

typedef struct {
    uint8_t  state;
    uint8_t  codec;
    uint8_t  preferred_codec;
    uint8_t  codec_fail_streak;
    uint8_t  dirty;
    uint8_t  stable_ticks;
    uint32_t write_seq;
    uint32_t comp_size;
    uint64_t pool_offset;
    uint8_t  prefetched;
    uint8_t  cooldown;
    uint8_t  in_res_list;
    uint8_t  in_hot_list;
    uint16_t tensor_role;
    uint16_t tensor_dtype;
    uint16_t tensor_layout;
    uint32_t tensor_flags;
    uint32_t tensor_layer;
    uint32_t tensor_head;
    size_t   alloc_size;
    uintptr_t owner_tag;
} PageMeta;

typedef struct memx_core_hooks {
    int (*inflate)(const uint8_t *src, uint32_t src_len, uint8_t *dst, uint32_t *dst_len);
    int (*uncompress)(const uint8_t *src, uint32_t src_len, uint8_t *dst, uint32_t expected_len);
} memx_core_hooks_t;

void memx_core_set_hooks(const memx_core_hooks_t *hooks);
const memx_core_hooks_t *memx_core_get_hooks(void);

uint64_t fnv1a_word(const uint8_t *data, uint32_t len);
int page_is_all_zero(const uint8_t *src);
int page_bytes_equal(const uint8_t *a, const uint8_t *b);

uint32_t rle8_encode(const uint8_t *src, uint32_t len, uint8_t *dst, uint32_t cap);
int rle8_decode(const uint8_t *src, uint32_t len, uint8_t *dst, uint32_t out_len);

void memx_interleave_lo_hi(const uint8_t *lo, const uint8_t *hi, uint8_t *dst, uint32_t half_count);

uint32_t tensor_fp16_split_compress(const uint8_t *src, uint8_t *dst, uint32_t cap);
uint32_t tensor_fp16_delta_split_compress(const uint8_t *src, uint8_t *dst, uint32_t cap);
uint32_t tensor_bitplane16_compress(const uint8_t *src, uint8_t *dst, uint32_t cap);
uint32_t tensor_sparse_byte_compress(const uint8_t *src, uint8_t *dst, uint32_t cap);

int tensor_fp16_split_eligible(const PageMeta *m);
int tensor_sparse_byte_eligible(const PageMeta *m);

int page_wants_write_protect(const PageMeta *m);
int fault_stream_for_role(uint16_t role);
int page_stable_need(const PageMeta *m);

int page_compress_meta_stable(const PageMeta *m, uint32_t seq0);
int page_compress_content_ok(const PageMeta *m, uint32_t seq0, const uint8_t *snap, const uint8_t *live);

void wait_decompress_complete(PageMeta *m);

void cpu_decompress(const uint8_t *src, uint32_t cs, uint8_t *dst);

#endif
