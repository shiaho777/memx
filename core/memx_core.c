#include "memx_core.h"

#if MEMX_HAS_NEON
#include <arm_neon.h>
#endif

void *memcpy(void *, const void *, size_t);
void *memset(void *, int, size_t);
int memcmp(const void *, const void *, size_t);

static const memx_core_hooks_t *g_core_hooks;

void memx_core_set_hooks(const memx_core_hooks_t *hooks) { g_core_hooks = hooks; }
const memx_core_hooks_t *memx_core_get_hooks(void) { return g_core_hooks; }

uint64_t fnv1a_word(const uint8_t *data, uint32_t len) {
    uint64_t h = 14695981039346656037ULL;
    uint32_t i = 0;
    for (; i + 8 <= len; i += 8) {
        uint64_t w;
        memcpy(&w, data + i, 8);
        h ^= w;
        h *= 1099511628211ULL;
    }
    for (; i < len; i++) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

uint32_t rle8_encode(const uint8_t *src, uint32_t len, uint8_t *dst, uint32_t cap) {
    uint32_t ip = 0, op = 0;
    while (ip < len) {
        uint8_t value = src[ip];
        uint32_t run = 1;
        uint32_t remain = len - ip;
        if (remain > 255) remain = 255;
        while (run < remain && src[ip + run] == value) run++;
        if (op + 2 > cap) return 0;
        dst[op++] = (uint8_t)run;
        dst[op++] = value;
        ip += run;
    }
    return op;
}

int rle8_decode(const uint8_t *src, uint32_t len, uint8_t *dst, uint32_t out_len) {
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

int tensor_fp16_split_eligible(const PageMeta *m) {
    if (!m) return 0;
    uint16_t dtype = m->tensor_dtype;
    uint16_t role = m->tensor_role;
    if (dtype != MEMX_DTYPE_FP16 && dtype != MEMX_DTYPE_BF16) return 0;
    if (role != MEMX_ROLE_WEIGHT &&
        role != MEMX_ROLE_KV_CACHE &&
        role != MEMX_ROLE_ACTIVATION &&
        role != MEMX_ROLE_EMBEDDING) return 0;
    return 1;
}

int tensor_sparse_byte_eligible(const PageMeta *m) {
    if (!m) return 0;
    uint16_t dtype = m->tensor_dtype;
    uint16_t role = m->tensor_role;
    if (role != MEMX_ROLE_WEIGHT &&
        role != MEMX_ROLE_KV_CACHE &&
        role != MEMX_ROLE_ACTIVATION &&
        role != MEMX_ROLE_EMBEDDING &&
        role != MEMX_ROLE_TEMPORARY &&
        role != MEMX_ROLE_DATA) return 0;
    if (dtype != MEMX_DTYPE_FP16 &&
        dtype != MEMX_DTYPE_BF16 &&
        dtype != MEMX_DTYPE_FP32 &&
        dtype != MEMX_DTYPE_INT8 &&
        dtype != MEMX_DTYPE_UINT8 &&
        dtype != MEMX_DTYPE_INT32) return 0;
    return 1;
}

int page_is_all_zero(const uint8_t *src) {
    if (!src) return 0;
    uint64_t w0, w1;
    memcpy(&w0, src, 8);
    memcpy(&w1, src + PAGE_SZ - 8, 8);
    if (w0 | w1) return 0;
#if MEMX_HAS_NEON
    {
        uint32_t i = 8;
        for (; i + 64 <= PAGE_SZ - 8; i += 64) {
            uint8x16_t a = vld1q_u8(src + i);
            uint8x16_t b = vld1q_u8(src + i + 16);
            uint8x16_t c = vld1q_u8(src + i + 32);
            uint8x16_t d = vld1q_u8(src + i + 48);
            uint8x16_t o = vorrq_u8(vorrq_u8(a, b), vorrq_u8(c, d));
            if (vmaxvq_u8(o) != 0) return 0;
        }
        for (; i < PAGE_SZ - 8; i += 8) {
            uint64_t w;
            memcpy(&w, src + i, 8);
            if (w) return 0;
        }
        return 1;
    }
#else
    for (size_t k = 8; k < PAGE_SZ - 8; k += 8) {
        uint64_t w;
        memcpy(&w, src + k, 8);
        if (w) return 0;
    }
    return 1;
#endif
}

int page_bytes_equal(const uint8_t *a, const uint8_t *b) {
    if (a == b) return 1;
    if (!a || !b) return 0;
#if MEMX_HAS_NEON
    {
        uint32_t i = 0;
        for (; i + 64 <= PAGE_SZ; i += 64) {
            uint8x16_t a0 = vld1q_u8(a + i);
            uint8x16_t b0 = vld1q_u8(b + i);
            uint8x16_t a1 = vld1q_u8(a + i + 16);
            uint8x16_t b1 = vld1q_u8(b + i + 16);
            uint8x16_t a2 = vld1q_u8(a + i + 32);
            uint8x16_t b2 = vld1q_u8(b + i + 32);
            uint8x16_t a3 = vld1q_u8(a + i + 48);
            uint8x16_t b3 = vld1q_u8(b + i + 48);
            uint8x16_t d0 = veorq_u8(a0, b0);
            uint8x16_t d1 = veorq_u8(a1, b1);
            uint8x16_t d2 = veorq_u8(a2, b2);
            uint8x16_t d3 = veorq_u8(a3, b3);
            uint8x16_t o = vorrq_u8(vorrq_u8(d0, d1), vorrq_u8(d2, d3));
            if (vmaxvq_u8(o) != 0) return 0;
        }
        for (; i + 8 <= PAGE_SZ; i += 8) {
            uint64_t wa, wb;
            memcpy(&wa, a + i, 8);
            memcpy(&wb, b + i, 8);
            if (wa != wb) return 0;
        }
        return 1;
    }
#else
    return memcmp(a, b, PAGE_SZ) == 0;
#endif
}

int page_compress_meta_stable(const PageMeta *m, uint32_t seq0) {
    return m->state == MEMX_PAGE_COMPRESSING && !m->dirty && m->write_seq == seq0;
}

int page_compress_content_ok(const PageMeta *m, uint32_t seq0, const uint8_t *snap, const uint8_t *live) {
    if (!page_compress_meta_stable(m, seq0)) return 0;
    return page_bytes_equal(snap, live);
}

int page_wants_write_protect(const PageMeta *m) {
    if (!m) return 0;
    if (m->tensor_flags & MEMX_FLAG_NO_COMPRESS) return 0;
    if (m->tensor_flags & (MEMX_FLAG_READ_MOSTLY | MEMX_FLAG_COLD | MEMX_FLAG_SEQUENTIAL))
        return 1;
    if (m->tensor_role == MEMX_ROLE_WEIGHT ||
        m->tensor_role == MEMX_ROLE_EMBEDDING ||
        m->tensor_role == MEMX_ROLE_KV_CACHE)
        return 1;
    return 0;
}

int fault_stream_for_role(uint16_t role) {
    if (role == MEMX_ROLE_KV_CACHE) return 0;
    if (role == MEMX_ROLE_WEIGHT || role == MEMX_ROLE_EMBEDDING) return 1;
    return 2;
}

int page_stable_need(const PageMeta *m) {
    if (!m) return 1;
    if (m->tensor_flags & MEMX_FLAG_HOT) return 0;
    if (m->tensor_flags & MEMX_FLAG_NO_COMPRESS) return 0;
    if (m->tensor_flags & MEMX_FLAG_SEQUENTIAL) return 4;
    if (m->tensor_role == MEMX_ROLE_KV_CACHE) {
        if (m->tensor_flags & MEMX_FLAG_COLD) return 1;
        return 1;
    }
    if (m->tensor_role == MEMX_ROLE_WEIGHT || m->tensor_role == MEMX_ROLE_EMBEDDING) {
        if (m->tensor_flags & (MEMX_FLAG_COLD | MEMX_FLAG_READ_MOSTLY)) return 0;
        return 1;
    }
    if (m->tensor_role == MEMX_ROLE_ACTIVATION) return 1;
    return 2;
}

uint32_t tensor_fp16_split_compress(const uint8_t *src, uint8_t *dst, uint32_t cap) {
    if (!src || !dst || cap < 16) return 0;
    const uint32_t half_count = PAGE_SZ / 2;
    uint8_t *lo = dst + 16;
    uint8_t hi_tmp[PAGE_SZ / 2];
    if (16 + half_count > cap) return 0;
#if MEMX_HAS_NEON
    {
        uint32_t i = 0;
        for (; i + 16 <= half_count; i += 16) {
            uint8x16x2_t z = vld2q_u8(src + i * 2);
            vst1q_u8(lo + i, z.val[0]);
            vst1q_u8(hi_tmp + i, z.val[1]);
        }
        for (; i < half_count; i++) { lo[i] = src[i * 2]; hi_tmp[i] = src[i * 2 + 1]; }
    }
#else
    for (uint32_t i = 0; i < half_count; i++) {
        lo[i] = src[i * 2];
        hi_tmp[i] = src[i * 2 + 1];
    }
#endif
    uint8_t rle_tmp[PAGE_SZ];
    uint32_t hi_rle = rle8_encode(hi_tmp, half_count, rle_tmp, sizeof(rle_tmp));
    if (hi_rle == 0 || 16 + half_count + hi_rle >= PAGE_SZ - 32) return 0;
    dst[0] = 0x4D;
    dst[1] = 0x58;
    dst[2] = MEMX_CODEC_TENSOR_FP16_SPLIT;
    dst[3] = 0;
    dst[4] = (uint8_t)(half_count & 0xFF);
    dst[5] = (uint8_t)((half_count >> 8) & 0xFF);
    dst[6] = (uint8_t)((half_count >> 16) & 0xFF);
    dst[7] = (uint8_t)((half_count >> 24) & 0xFF);
    dst[8] = (uint8_t)(hi_rle & 0xFF);
    dst[9] = (uint8_t)((hi_rle >> 8) & 0xFF);
    dst[10] = (uint8_t)((hi_rle >> 16) & 0xFF);
    dst[11] = (uint8_t)((hi_rle >> 24) & 0xFF);
    dst[12] = 0;
    dst[13] = 0;
    dst[14] = 0;
    dst[15] = 0;
    memcpy(dst + 16 + half_count, rle_tmp, hi_rle);
    return 16 + half_count + hi_rle;
}

uint32_t tensor_fp16_delta_split_compress(const uint8_t *src, uint8_t *dst, uint32_t cap) {
    if (!src || !dst || cap < 24) return 0;
    const uint32_t half_count = PAGE_SZ / 2;
    uint8_t lo_delta[PAGE_SZ / 2];
    uint8_t hi_tmp[PAGE_SZ / 2];
    uint8_t lo_bytes[PAGE_SZ / 2];
#if MEMX_HAS_NEON
    {
        uint32_t i = 0;
        for (; i + 16 <= half_count; i += 16) {
            uint8x16x2_t z = vld2q_u8(src + i * 2);
            vst1q_u8(lo_bytes + i, z.val[0]);
            vst1q_u8(hi_tmp + i, z.val[1]);
        }
        for (; i < half_count; i++) {
            lo_bytes[i] = src[i * 2];
            hi_tmp[i] = src[i * 2 + 1];
        }
    }
#else
    for (uint32_t i = 0; i < half_count; i++) {
        lo_bytes[i] = src[i * 2];
        hi_tmp[i] = src[i * 2 + 1];
    }
#endif
    {
        uint8_t prev = 0;
        for (uint32_t i = 0; i < half_count; i++) {
            uint8_t lo = lo_bytes[i];
            lo_delta[i] = (uint8_t)(lo - prev);
            prev = lo;
        }
    }
    uint8_t lo_rle_tmp[PAGE_SZ];
    uint8_t hi_rle_tmp[PAGE_SZ];
    uint32_t lo_rle = rle8_encode(lo_delta, half_count, lo_rle_tmp, sizeof(lo_rle_tmp));
    uint32_t hi_rle = rle8_encode(hi_tmp, half_count, hi_rle_tmp, sizeof(hi_rle_tmp));
    if (lo_rle == 0 || hi_rle == 0 || 24 + lo_rle + hi_rle >= PAGE_SZ - 32 || 24 + lo_rle + hi_rle > cap) return 0;
    dst[0] = 0x4D;
    dst[1] = 0x58;
    dst[2] = MEMX_CODEC_TENSOR_FP16_DELTA_SPLIT;
    dst[3] = 0;
    dst[4] = (uint8_t)(half_count & 0xFF);
    dst[5] = (uint8_t)((half_count >> 8) & 0xFF);
    dst[6] = (uint8_t)((half_count >> 16) & 0xFF);
    dst[7] = (uint8_t)((half_count >> 24) & 0xFF);
    dst[8] = (uint8_t)(lo_rle & 0xFF);
    dst[9] = (uint8_t)((lo_rle >> 8) & 0xFF);
    dst[10] = (uint8_t)((lo_rle >> 16) & 0xFF);
    dst[11] = (uint8_t)((lo_rle >> 24) & 0xFF);
    dst[12] = (uint8_t)(hi_rle & 0xFF);
    dst[13] = (uint8_t)((hi_rle >> 8) & 0xFF);
    dst[14] = (uint8_t)((hi_rle >> 16) & 0xFF);
    dst[15] = (uint8_t)((hi_rle >> 24) & 0xFF);
    memset(dst + 16, 0, 8);
    memcpy(dst + 24, lo_rle_tmp, lo_rle);
    memcpy(dst + 24 + lo_rle, hi_rle_tmp, hi_rle);
    return 24 + lo_rle + hi_rle;
}

uint32_t tensor_bitplane16_compress(const uint8_t *src, uint8_t *dst, uint32_t cap) {
    if (!src || !dst || cap < 34) return 0;
    uint8_t planes[16][PAGE_SZ / 16];
#if MEMX_HAS_NEON
    {
        const uint16x8_t lane_w = {1, 2, 4, 8, 16, 32, 64, 128};
        for (uint32_t i = 0; i < PAGE_SZ / 2; i += 8) {
            uint16x8_t v = vld1q_u16((const uint16_t *)(src + i * 2));
            uint32_t byte_i = i >> 3;
            for (uint32_t b = 0; b < 16; b++) {
                uint16x8_t tst = vtstq_u16(v, vdupq_n_u16((uint16_t)(1u << b)));
                planes[b][byte_i] = (uint8_t)vaddvq_u16(vandq_u16(tst, lane_w));
            }
        }
    }
#else
    memset(planes, 0, sizeof(planes));
    for (uint32_t i = 0; i < PAGE_SZ / 2; i += 8) {
        uint16_t v[8];
        memcpy(v, src + i * 2, 16);
        uint32_t byte_i = i >> 3;
        for (uint32_t b = 0; b < 16; b++) {
            uint16_t m = (uint16_t)(1u << b);
            uint8_t byte = 0;
            if (v[0] & m) byte |= 1u << 0;
            if (v[1] & m) byte |= 1u << 1;
            if (v[2] & m) byte |= 1u << 2;
            if (v[3] & m) byte |= 1u << 3;
            if (v[4] & m) byte |= 1u << 4;
            if (v[5] & m) byte |= 1u << 5;
            if (v[6] & m) byte |= 1u << 6;
            if (v[7] & m) byte |= 1u << 7;
            planes[b][byte_i] = byte;
        }
    }
#endif
    uint32_t op = 36;
    dst[0] = 0x4D;
    dst[1] = 0x58;
    dst[2] = MEMX_CODEC_TENSOR_BITPLANE16;
    dst[3] = 0;
    for (uint32_t b = 0; b < 16; b++) {
        uint32_t sz = rle8_encode(planes[b], PAGE_SZ / 16, dst + op, cap - op);
        if (sz == 0 || sz > 65535) return 0;
        dst[4 + b * 2] = (uint8_t)(sz & 0xFF);
        dst[5 + b * 2] = (uint8_t)((sz >> 8) & 0xFF);
        op += sz;
        if (op >= PAGE_SZ - 32) return 0;
    }
    return op;
}

uint32_t tensor_sparse_byte_compress(const uint8_t *src, uint8_t *dst, uint32_t cap) {
    if (!src || !dst || cap < 8) return 0;
    uint32_t count = 0;
    uint32_t i = 0;
    for (; i + 8 <= PAGE_SZ; i += 8) {
        uint64_t w;
        memcpy(&w, src + i, 8);
        if (w) {
            for (uint32_t b = 0; b < 8; b++) if (src[i + b]) count++;
        }
    }
    for (; i < PAGE_SZ; i++) if (src[i]) count++;
    uint32_t need = 8 + count * 3;
    if (count == 0 || need >= PAGE_SZ - 32 || need > cap) return 0;
    dst[0] = 0x4D;
    dst[1] = 0x58;
    dst[2] = MEMX_CODEC_TENSOR_SPARSE_BYTE;
    dst[3] = 0;
    dst[4] = (uint8_t)(count & 0xFF);
    dst[5] = (uint8_t)((count >> 8) & 0xFF);
    dst[6] = 0;
    dst[7] = 0;
    uint32_t op = 8;
    i = 0;
    for (; i + 8 <= PAGE_SZ; i += 8) {
        uint64_t w;
        memcpy(&w, src + i, 8);
        if (!w) continue;
        for (uint32_t b = 0; b < 8; b++) {
            uint8_t value = src[i + b];
            if (!value) continue;
            uint32_t off = i + b;
            dst[op++] = (uint8_t)(off & 0xFF);
            dst[op++] = (uint8_t)((off >> 8) & 0xFF);
            dst[op++] = value;
        }
    }
    for (; i < PAGE_SZ; i++) {
        uint8_t value = src[i];
        if (!value) continue;
        dst[op++] = (uint8_t)(i & 0xFF);
        dst[op++] = (uint8_t)((i >> 8) & 0xFF);
        dst[op++] = value;
    }
    return op;
}

void memx_interleave_lo_hi(const uint8_t *lo, const uint8_t *hi, uint8_t *dst, uint32_t half_count) {
#if MEMX_HAS_NEON
    uint32_t i = 0;
    for (; i + 16 <= half_count; i += 16) {
        uint8x16_t vlo = vld1q_u8(lo + i);
        uint8x16_t vhi = vld1q_u8(hi + i);
        uint8x16x2_t z = { vlo, vhi };
        vst2q_u8(dst + i * 2, z);
    }
    for (; i < half_count; i++) { dst[i * 2] = lo[i]; dst[i * 2 + 1] = hi[i]; }
#else
    for (uint32_t i = 0; i < half_count; i++) { dst[i * 2] = lo[i]; dst[i * 2 + 1] = hi[i]; }
#endif
}

void wait_decompress_complete(PageMeta *m) {
    if (!m) return;
    for (;;) {
        uint8_t st = __atomic_load_n(&m->state, __ATOMIC_ACQUIRE);
        uint32_t cs = __atomic_load_n(&m->comp_size, __ATOMIC_ACQUIRE);
        if (st != MEMX_PAGE_HOT && st != MEMX_PAGE_COMPRESSED) return;
        if (st == MEMX_PAGE_HOT && cs == 0) return;
        if (st == MEMX_PAGE_COMPRESSED) return;
#if defined(__aarch64__)
        __asm__ __volatile__("yield" ::: "memory");
#endif
    }
}

void cpu_decompress(const uint8_t *src, uint32_t cs, uint8_t *dst) {
    if (cs>=PAGE_SZ||src[0]!=0x4D||src[1]!=0x58){memcpy(dst,src,cs<PAGE_SZ?cs:PAGE_SZ);if(cs<PAGE_SZ)memset(dst+cs,0,PAGE_SZ-cs);return;}
    uint8_t ver=src[2];
    if(ver==MEMX_CODEC_TENSOR_FP16_SPLIT){
        if(cs<16){memset(dst,0,PAGE_SZ);return;}
        uint32_t half_count=(uint32_t)src[4]|((uint32_t)src[5]<<8)|((uint32_t)src[6]<<16)|((uint32_t)src[7]<<24);
        uint32_t hi_rle=(uint32_t)src[8]|((uint32_t)src[9]<<8)|((uint32_t)src[10]<<16)|((uint32_t)src[11]<<24);
        if(half_count!=PAGE_SZ/2||16+half_count+hi_rle>cs){memset(dst,0,PAGE_SZ);return;}
        uint8_t hi[PAGE_SZ/2];
        if(rle8_decode(src+16+half_count,hi_rle,hi,half_count)!=0){memset(dst,0,PAGE_SZ);return;}
        memx_interleave_lo_hi(src+16, hi, dst, half_count);
        return;
    }
    if(ver==MEMX_CODEC_TENSOR_FP16_DELTA_SPLIT){
        if(cs<24){memset(dst,0,PAGE_SZ);return;}
        uint32_t half_count=(uint32_t)src[4]|((uint32_t)src[5]<<8)|((uint32_t)src[6]<<16)|((uint32_t)src[7]<<24);
        uint32_t lo_rle=(uint32_t)src[8]|((uint32_t)src[9]<<8)|((uint32_t)src[10]<<16)|((uint32_t)src[11]<<24);
        uint32_t hi_rle=(uint32_t)src[12]|((uint32_t)src[13]<<8)|((uint32_t)src[14]<<16)|((uint32_t)src[15]<<24);
        if(half_count!=PAGE_SZ/2||24+lo_rle+hi_rle>cs){memset(dst,0,PAGE_SZ);return;}
        uint8_t lo_delta[PAGE_SZ/2];
        uint8_t hi[PAGE_SZ/2];
        if(rle8_decode(src+24,lo_rle,lo_delta,half_count)!=0||rle8_decode(src+24+lo_rle,hi_rle,hi,half_count)!=0){memset(dst,0,PAGE_SZ);return;}
        uint8_t lo_bytes[PAGE_SZ/2];
        {
            uint8_t lo=0;
            uint32_t i=0;
            for(; i+4<=half_count; i+=4){
                lo=(uint8_t)(lo+lo_delta[i]); lo_bytes[i]=lo;
                lo=(uint8_t)(lo+lo_delta[i+1]); lo_bytes[i+1]=lo;
                lo=(uint8_t)(lo+lo_delta[i+2]); lo_bytes[i+2]=lo;
                lo=(uint8_t)(lo+lo_delta[i+3]); lo_bytes[i+3]=lo;
            }
            for(; i<half_count; i++){lo=(uint8_t)(lo+lo_delta[i]);lo_bytes[i]=lo;}
        }
        memx_interleave_lo_hi(lo_bytes, hi, dst, half_count);
        return;
    }
    if(ver==MEMX_CODEC_TENSOR_BITPLANE16){
        if(cs<36){memset(dst,0,PAGE_SZ);return;}
        uint8_t planes[16][PAGE_SZ/16];
        uint32_t ip=36;
        for(uint32_t b=0;b<16;b++){
            uint32_t sz=(uint32_t)src[4+b*2]|((uint32_t)src[5+b*2]<<8);
            if(ip+sz>cs||rle8_decode(src+ip,sz,planes[b],PAGE_SZ/16)!=0){memset(dst,0,PAGE_SZ);return;}
            ip+=sz;
        }
#if MEMX_HAS_NEON
        {
            const uint8x8_t bit_sel = {1, 2, 4, 8, 16, 32, 64, 128};
            for(uint32_t i=0;i<PAGE_SZ/2;i+=8){
                uint32_t byte_i=i>>3;
                uint16x8_t acc=vdupq_n_u16(0);
                for(uint32_t b=0;b<16;b++){
                    uint8x8_t pv=vdup_n_u8(planes[b][byte_i]);
                    uint16x8_t m=vmovl_s8(vreinterpret_s8_u8(vtst_u8(pv,bit_sel)));
                    acc=vorrq_u16(acc,vandq_u16(m,vdupq_n_u16((uint16_t)(1u<<b))));
                }
                vst1q_u16((uint16_t*)(dst+i*2),acc);
            }
        }
#else
        for(uint32_t i=0;i<PAGE_SZ/2;i+=8){
            uint32_t byte_i=i>>3;
            uint16_t out[8]={0,0,0,0,0,0,0,0};
            for(uint32_t b=0;b<16;b++){
                uint8_t p=planes[b][byte_i];
                uint16_t bit=(uint16_t)(1u<<b);
                if(p&1u) out[0]|=bit;
                if(p&2u) out[1]|=bit;
                if(p&4u) out[2]|=bit;
                if(p&8u) out[3]|=bit;
                if(p&16u) out[4]|=bit;
                if(p&32u) out[5]|=bit;
                if(p&64u) out[6]|=bit;
                if(p&128u) out[7]|=bit;
            }
            memcpy(dst+i*2,out,16);
        }
#endif
        return;
    }
    if(ver==MEMX_CODEC_TENSOR_SPARSE_BYTE){
        if(cs<8){memset(dst,0,PAGE_SZ);return;}
        uint32_t count=(uint32_t)src[4]|((uint32_t)src[5]<<8);
        if(8+count*3!=cs){memset(dst,0,PAGE_SZ);return;}
        memset(dst,0,PAGE_SZ);
        uint32_t ip=8;
        for(uint32_t i=0;i<count;i++){
            uint32_t off=(uint32_t)src[ip]|((uint32_t)src[ip+1]<<8);
            uint8_t value=src[ip+2];
            ip+=3;
            if(off>=PAGE_SZ){memset(dst,0,PAGE_SZ);return;}
            dst[off]=value;
        }
        return;
    }
    if(ver==MEMX_CODEC_TENSOR_EXP_PACK){
        if(cs<28){memset(dst,0,PAGE_SZ);return;}
        uint32_t half_count=(uint32_t)src[4]|((uint32_t)src[5]<<8)|((uint32_t)src[6]<<16)|((uint32_t)src[7]<<24);
        uint32_t sign_zlen=(uint32_t)src[8]|((uint32_t)src[9]<<8)|((uint32_t)src[10]<<16)|((uint32_t)src[11]<<24);
        uint32_t exp_zlen=(uint32_t)src[12]|((uint32_t)src[13]<<8)|((uint32_t)src[14]<<16)|((uint32_t)src[15]<<24);
        uint32_t mant_zlen=(uint32_t)src[16]|((uint32_t)src[17]<<8)|((uint32_t)src[18]<<16)|((uint32_t)src[19]<<24);
        int is_bf16 = (src[20] == 2);
        uint8_t flags = src[21];
        uint32_t mant_len=(uint32_t)src[24]|((uint32_t)src[25]<<8)|((uint32_t)src[26]<<16)|((uint32_t)src[27]<<24);
        uint32_t sign_bytes=(half_count+7u)/8u;
        uint32_t expect_mant = (flags & 4u)
            ? (is_bf16 ? ((half_count * 7u + 7u) / 8u) : ((half_count * 10u + 7u) / 8u))
            : (is_bf16 ? half_count : (half_count * 2u));
        if(half_count!=PAGE_SZ/2||mant_len!=expect_mant||28+sign_zlen+exp_zlen+mant_zlen>cs){memset(dst,0,PAGE_SZ);return;}
        uint8_t sign_raw[PAGE_SZ/16];
        uint8_t exp_raw[PAGE_SZ/2];
        uint8_t mant_raw[PAGE_SZ];
        const uint8_t *p=src+28;
        const memx_core_hooks_t *hk = g_core_hooks;
        if(!hk || !hk->uncompress){memset(dst,0,PAGE_SZ);return;}
        if(flags&1u){
            if(sign_zlen!=sign_bytes){memset(dst,0,PAGE_SZ);return;}
            memcpy(sign_raw,p,sign_bytes);
        }else{
            if(hk->uncompress(p,sign_zlen,sign_raw,sign_bytes)!=0){memset(dst,0,PAGE_SZ);return;}
        }
        p+=sign_zlen;
        if(hk->uncompress(p,exp_zlen,exp_raw,half_count)!=0){memset(dst,0,PAGE_SZ);return;}
        p+=exp_zlen;
        if(flags&2u){
            if(mant_zlen!=mant_len){memset(dst,0,PAGE_SZ);return;}
            memcpy(mant_raw,p,mant_len);
        }else{
            if(hk->uncompress(p,mant_zlen,mant_raw,mant_len)!=0){memset(dst,0,PAGE_SZ);return;}
        }
        if(flags & 4u){
#if MEMX_HAS_NEON
            const uint8x8_t bit_sel = {1, 2, 4, 8, 16, 32, 64, 128};
            const uint16x8_t exp_mask5 = vdupq_n_u16(0x1Fu);
            for(uint32_t i=0;i<half_count;i+=8){
                uint8x8_t sv=vdup_n_u8(sign_raw[i>>3]);
                uint16x8_t sign16=vandq_u16(vmovl_s8(vreinterpret_s8_u8(vtst_u8(sv,bit_sel))),vdupq_n_u16(0x8000u));
                uint16x8_t exp16=vmovl_u8(vld1_u8(exp_raw+i));
                uint16_t mw[8];
                if(is_bf16){
                    uint64_t packed=0;
                    memcpy(&packed,mant_raw+(size_t)(i>>3)*7,7);
                    for(int j=0;j<8;j++) mw[j]=(uint16_t)((packed>>(7*j))&0x7Fu);
                }else{
                    uint64_t w0; uint16_t w1=0;
                    memcpy(&w0,mant_raw+(size_t)(i>>3)*10,8);
                    memcpy(&w1,mant_raw+(size_t)(i>>3)*10+8,2);
                    mw[0]=(uint16_t)(w0&0x3FFu);
                    mw[1]=(uint16_t)((w0>>10)&0x3FFu);
                    mw[2]=(uint16_t)((w0>>20)&0x3FFu);
                    mw[3]=(uint16_t)((w0>>30)&0x3FFu);
                    mw[4]=(uint16_t)((w0>>40)&0x3FFu);
                    mw[5]=(uint16_t)((w0>>50)&0x3FFu);
                    mw[6]=(uint16_t)(((w0>>60)|((uint64_t)w1<<4))&0x3FFu);
                    mw[7]=(uint16_t)((w1>>6)&0x3FFu);
                }
                uint16x8_t mant16=vld1q_u16(mw);
                uint16x8_t h;
                if(is_bf16){
                    h=vorrq_u16(sign16,vorrq_u16(vshlq_n_u16(exp16,7),mant16));
                }else{
                    h=vorrq_u16(sign16,vorrq_u16(vshlq_n_u16(vandq_u16(exp16,exp_mask5),10),mant16));
                }
                vst1q_u16((uint16_t*)(dst+i*2),h);
            }
#else
            uint32_t bit_pos = 0;
            for(uint32_t i=0;i<half_count;i++){
                uint16_t sign=(sign_raw[i>>3]>>(i&7u))&1u;
                uint16_t h;
                if(is_bf16){
                    uint16_t m=0;
                    for(int b=0;b<7;b++){
                        if(mant_raw[bit_pos>>3]&(1u<<(bit_pos&7u))) m|=(uint16_t)(1u<<b);
                        bit_pos++;
                    }
                    h=(uint16_t)((sign<<15)|(((uint16_t)exp_raw[i])<<7)|(m&0x7Fu));
                }else{
                    uint16_t m=0;
                    for(int b=0;b<10;b++){
                        if(mant_raw[bit_pos>>3]&(1u<<(bit_pos&7u))) m|=(uint16_t)(1u<<b);
                        bit_pos++;
                    }
                    h=(uint16_t)((sign<<15)|(((uint16_t)(exp_raw[i]&0x1Fu))<<10)|(m&0x3FFu));
                }
                dst[i*2]=(uint8_t)(h&0xFF);
                dst[i*2+1]=(uint8_t)((h>>8)&0xFF);
            }
#endif
        }else{
            for(uint32_t i=0;i<half_count;i++){
                uint16_t sign=(sign_raw[i>>3]>>(i&7u))&1u;
                uint16_t h;
                if(is_bf16){
                    h=(uint16_t)((sign<<15)|(((uint16_t)exp_raw[i])<<7)|(mant_raw[i]&0x7Fu));
                }else{
                    uint16_t m=(uint16_t)mant_raw[i*2]|((uint16_t)mant_raw[i*2+1]<<8);
                    h=(uint16_t)((sign<<15)|(((uint16_t)(exp_raw[i]&0x1Fu))<<10)|(m&0x3FFu));
                }
                dst[i*2]=(uint8_t)(h&0xFF);
                dst[i*2+1]=(uint8_t)((h>>8)&0xFF);
            }
        }
        return;
    }
    if(ver==MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT){
        if(cs<24){memset(dst,0,PAGE_SZ);return;}
        uint32_t half_count=(uint32_t)src[4]|((uint32_t)src[5]<<8)|((uint32_t)src[6]<<16)|((uint32_t)src[7]<<24);
        if(half_count!=PAGE_SZ/2){memset(dst,0,PAGE_SZ);return;}
        const memx_core_hooks_t *hk = g_core_hooks;
        if(!hk || !hk->inflate){memset(dst,0,PAGE_SZ);return;}
        if(src[3]>=2){
            uint32_t lo_zlen=(uint32_t)src[8]|((uint32_t)src[9]<<8)|((uint32_t)src[10]<<16)|((uint32_t)src[11]<<24);
            uint32_t hi_zlen=(uint32_t)src[12]|((uint32_t)src[13]<<8)|((uint32_t)src[14]<<16)|((uint32_t)src[15]<<24);
            if(lo_zlen==0||hi_zlen==0||24+lo_zlen+hi_zlen>cs){memset(dst,0,PAGE_SZ);return;}
            uint8_t lo[PAGE_SZ/2];
            uint8_t hi[PAGE_SZ/2];
            uint32_t loLen=half_count, hiLen=half_count;
            if(hk->inflate(src+24,lo_zlen,lo,&loLen)!=0||loLen!=half_count){memset(dst,0,PAGE_SZ);return;}
            if(hk->inflate(src+24+lo_zlen,hi_zlen,hi,&hiLen)!=0||hiLen!=half_count){memset(dst,0,PAGE_SZ);return;}
            memx_interleave_lo_hi(lo, hi, dst, half_count);
            return;
        }
        uint32_t hi_zlen=(uint32_t)src[8]|((uint32_t)src[9]<<8)|((uint32_t)src[10]<<16)|((uint32_t)src[11]<<24);
        if(hi_zlen==0||16+half_count+hi_zlen>cs){memset(dst,0,PAGE_SZ);return;}
        uint8_t hi[PAGE_SZ/2];
        uint32_t destLen=half_count;
        if(hk->inflate(src+16+half_count,hi_zlen,hi,&destLen)!=0||destLen!=half_count){memset(dst,0,PAGE_SZ);return;}
        memx_interleave_lo_hi(src+16, hi, dst, half_count);
        return;
    }
    if(ver==MEMX_CODEC_ZLIB){
        if(cs<9){memset(dst,0,PAGE_SZ);return;}
        uint32_t zlen=(uint32_t)src[4]|((uint32_t)src[5]<<8)|((uint32_t)src[6]<<16)|((uint32_t)src[7]<<24);
        if(zlen==0||8+zlen>cs){memset(dst,0,PAGE_SZ);return;}
        const memx_core_hooks_t *hk = g_core_hooks;
        if(!hk || !hk->inflate){memset(dst,0,PAGE_SZ);return;}
        uint32_t destLen=PAGE_SZ;
        if(hk->inflate(src+8,zlen,dst,&destLen)!=0||destLen!=PAGE_SZ){memset(dst,0,PAGE_SZ);}
        return;
    }
    uint32_t ip=4,op=0;
    while(ip<cs&&op<PAGE_SZ){uint8_t b=src[ip];
    if(b==0xFD&&ver>=2&&ip+3<cs){uint8_t vb=src[ip+1];uint32_t rl=(uint32_t)src[ip+2]|((uint32_t)src[ip+3]<<8);ip+=4;memset(dst+op,vb,rl<PAGE_SZ-op?rl:PAGE_SZ-op);op+=rl<PAGE_SZ-op?rl:PAGE_SZ-op;}
    else if(b==0xFF&&ip+4<cs){ip++;uint32_t o=(uint32_t)src[ip]|((uint32_t)src[ip+1]<<8);ip+=2;uint32_t m=(uint32_t)src[ip]|((uint32_t)src[ip+1]<<8);ip+=2;uint32_t s=op-o;for(uint32_t i=0;i<m&&op<PAGE_SZ;i++)dst[op++]=dst[s+i];}
    else if(b==0xFE&&ip+1<cs){ip++;dst[op++]=src[ip++];}
    else{dst[op++]=b;ip++;}}
    if(op<PAGE_SZ)memset(dst+op,0,PAGE_SZ-op);
    if(op>0){uint8_t acc=dst[0];for(uint32_t i=1;i<op;i++){acc+=dst[i];dst[i]=acc;}}
}
