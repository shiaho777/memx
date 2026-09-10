/*
 * Reference embedder for memx_core — the "kernel integration sample".
 *
 * Implements the EMBEDDING.md contract with ZERO OS services: the "operating
 * system" is simulated with plain memory (a per-page protection byte instead
 * of mprotect, explicit access checks instead of signal delivery). Everything
 * the engine does — compression, the commit handshake, fault-driven
 * decompression — runs purely on the core API plus this adapter.
 *
 * An OS port replaces the ~150 lines of sim_os_* glue below with its real
 * fault delivery and protection primitives; nothing else changes.
 */
#include "../core/memx_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIM_PAGES 64

typedef enum {
    SIM_PROT_NONE = 0,
    SIM_PROT_RO,
    SIM_PROT_RW,
} sim_prot_t;

typedef struct {
    uint8_t data[PAGE_SZ];
    sim_prot_t prot;
    uint8_t blob[PAGE_SZ];
    uint32_t blob_sz;
    uint8_t snap[PAGE_SZ];
} sim_page_t;

static sim_page_t sim_pages[SIM_PAGES];
static PageMeta sim_meta[SIM_PAGES];
static int sim_failures = 0;
static uint32_t sim_seq_next = 1;

/* ── the "OS" surface an embedder implements ─────────────────────── */

static void os_set_prot(uint32_t p, sim_prot_t prot) {
    sim_pages[p].prot = prot;
}

static void os_barrier(void) {
    __sync_synchronize();
}

/* Write-fault protocol from EMBEDDING.md: mark dirty, bump write_seq
 * (ordered), grant RW, then CAS any in-flight compression out of the way. */
static void os_write_fault(uint32_t p) {
    PageMeta *m = &sim_meta[p];
    m->dirty = 1;
    m->stable_ticks = 0;
    os_barrier();
    m->write_seq++;
    os_barrier();
    os_set_prot(p, SIM_PROT_RW);
    for (;;) {
        uint8_t st = m->state;
        if (st == MEMX_PAGE_COMPRESSED || st == MEMX_PAGE_COMPRESSING) {
            if (__sync_bool_compare_and_swap(&m->state, st, MEMX_PAGE_HOT)) {
                m->cooldown = 12;
                break;
            }
            continue;
        }
        break;
    }
}

/* Embedder-side write: access check stands in for the MMU. */
static void os_write(uint32_t p, const uint8_t *src, size_t n) {
    if (sim_pages[p].prot != SIM_PROT_RW) os_write_fault(p);
    memcpy(sim_pages[p].data, src, n ? n : PAGE_SZ);
}

/* Read on a protected page triggers the decompress-install protocol:
 * CAS COMPRESSED->HOT, fetch blob, decode via the core, install RW. */
static int os_read_fault(uint32_t p, uint8_t *out_page) {
    PageMeta *m = &sim_meta[p];
    if (!__sync_bool_compare_and_swap(&m->state, MEMX_PAGE_COMPRESSED, MEMX_PAGE_HOT))
        return 0;
    uint8_t scratch[PAGE_SZ];
    uint32_t csz = m->comp_size;
    if (csz == 0 || csz > PAGE_SZ) return 0;
    cpu_decompress(sim_pages[p].blob, csz, scratch);
    memcpy(sim_pages[p].data, scratch, PAGE_SZ);
    os_set_prot(p, SIM_PROT_RW);
    m->pool_offset = 0;
    os_barrier();
    m->comp_size = 0;
    return 1;
}

static const uint8_t *os_read(uint32_t p) {
    if (sim_pages[p].prot == SIM_PROT_NONE && sim_meta[p].state == MEMX_PAGE_COMPRESSED) {
        uint8_t scratch[PAGE_SZ];
        if (!os_read_fault(p, scratch)) return NULL;
    }
    return sim_pages[p].data;
}

/* ── the single-threaded compressor the embedder drives ──────────── */

static int sim_compress_one(uint32_t p) {
    PageMeta *m = &sim_meta[p];
    if (m->state != MEMX_PAGE_RESIDENT || m->dirty) return 0;
    if (page_stable_need(m) > 0 && m->stable_ticks < page_stable_need(m)) {
        m->stable_ticks++;
        return 0;
    }
    uint32_t seq0 = m->write_seq;
    if (!__sync_bool_compare_and_swap(&m->state, MEMX_PAGE_RESIDENT, MEMX_PAGE_COMPRESSING))
        return 0;
    if (m->dirty || m->write_seq != seq0) {
        m->state = MEMX_PAGE_RESIDENT;
        return 0;
    }

    /* revoke write, barrier, snapshot (the protection handshake) */
    if (page_wants_write_protect(m)) os_set_prot(p, SIM_PROT_RO);
    os_barrier();
    if (m->dirty || m->write_seq != seq0) {
        m->state = MEMX_PAGE_RESIDENT;
        if (page_wants_write_protect(m)) os_set_prot(p, SIM_PROT_RW);
        return 0;
    }
    memcpy(sim_pages[p].snap, sim_pages[p].data, PAGE_SZ);
    os_barrier();
    if (!page_compress_content_ok(m, seq0, sim_pages[p].snap, sim_pages[p].data)) {
        m->state = MEMX_PAGE_RESIDENT;
        if (page_wants_write_protect(m)) os_set_prot(p, SIM_PROT_RW);
        return 0;
    }

    /* encode with the core's zlib-free codecs */
    uint32_t csz = 0;
    if (page_is_all_zero(sim_pages[p].snap)) {
        static const uint8_t ZERO_PAGE[8] = {0x4D, 0x58, 0x03, 0x00, 0xFD, 0x00, 0x00, 0x40};
        memcpy(sim_pages[p].blob, ZERO_PAGE, 8);
        csz = 8;
    } else if (tensor_fp16_split_eligible(m)) {
        csz = tensor_fp16_split_compress(sim_pages[p].snap, sim_pages[p].blob, PAGE_SZ);
    }
    if (csz == 0) csz = tensor_sparse_byte_compress(sim_pages[p].snap, sim_pages[p].blob, PAGE_SZ);
    if (csz == 0) {
        /* incompressible: store nothing, back off */
        m->cooldown = 16;
        m->state = MEMX_PAGE_RESIDENT;
        if (page_wants_write_protect(m)) os_set_prot(p, SIM_PROT_RW);
        return 0;
    }

    /* commit: re-verify under the revoked protection, drop to NONE, CAS */
    os_barrier();
    if (!page_compress_content_ok(m, seq0, sim_pages[p].snap, sim_pages[p].data)) {
        m->state = MEMX_PAGE_RESIDENT;
        if (page_wants_write_protect(m)) os_set_prot(p, SIM_PROT_RW);
        return 0;
    }
    os_set_prot(p, SIM_PROT_NONE);
    os_barrier();
    if (m->dirty || m->write_seq != seq0 || m->state != MEMX_PAGE_COMPRESSING) {
        m->state = MEMX_PAGE_RESIDENT;
        os_set_prot(p, SIM_PROT_RW);
        return 0;
    }
    if (!__sync_bool_compare_and_swap(&m->state, MEMX_PAGE_COMPRESSING, MEMX_PAGE_COMPRESSED)) {
        os_set_prot(p, SIM_PROT_RW);
        return 0;
    }
    sim_pages[p].blob_sz = csz;
    m->comp_size = csz;
    return 1;
}

/* ── tests ────────────────────────────────────────────────────────── */

static void sim_reset(void) {
    memset(sim_pages, 0, sizeof(sim_pages));
    memset(sim_meta, 0, sizeof(sim_meta));
    for (uint32_t i = 0; i < SIM_PAGES; i++) {
        sim_meta[i].state = MEMX_PAGE_RESIDENT;
        sim_meta[i].tensor_role = MEMX_ROLE_WEIGHT;
        sim_meta[i].tensor_dtype = MEMX_DTYPE_FP16;
        sim_meta[i].tensor_flags = MEMX_FLAG_READ_MOSTLY | MEMX_FLAG_COLD;
        sim_pages[i].prot = SIM_PROT_RW;
    }
}

static void fill_fp16(uint8_t *dst, uint32_t seed) {
    for (uint32_t i = 0; i < PAGE_SZ / 2; i++) {
        uint16_t h = (uint16_t)(0x3C00 + ((i * 37 + seed) & 0x7F));
        memcpy(dst + i * 2, &h, 2);
    }
}

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); sim_failures++; } \
} while (0)

int main(void) {
    /* 1. roundtrip: write -> compress -> read fault -> bitexact */
    sim_reset();
    fill_fp16(sim_pages[0].data, 7);
    fill_fp16(sim_pages[1].data, 9);
    memset(sim_pages[2].data, 0, PAGE_SZ);
    CHECK(sim_compress_one(0) == 1, "page0 compressed");
    CHECK(sim_compress_one(1) == 1, "page1 compressed");
    CHECK(sim_compress_one(2) == 1, "zero page compressed");
    CHECK(sim_meta[0].state == MEMX_PAGE_COMPRESSED, "page0 state");
    CHECK(sim_pages[0].prot == SIM_PROT_NONE, "page0 revoked");
    CHECK(sim_pages[0].blob_sz < PAGE_SZ - 32, "page0 actually compressed");

    fill_fp16(sim_pages[0].snap, 7);
    const uint8_t *got = os_read(0);
    CHECK(got != NULL, "read served");
    CHECK(memcmp(got, sim_pages[0].snap, PAGE_SZ) == 0, "page0 roundtrip bitexact");
    CHECK(sim_meta[0].state == MEMX_PAGE_HOT, "page0 hot after install");
    fill_fp16(sim_pages[1].snap, 9);
    CHECK(memcmp(os_read(1), sim_pages[1].snap, PAGE_SZ) == 0, "page1 bitexact");
    CHECK(page_is_all_zero(os_read(2)), "zero page bitexact");

    /* 2. torn write: a writer lands between snapshot and commit */
    sim_reset();
    fill_fp16(sim_pages[3].data, 11);
    PageMeta *m = &sim_meta[3];
    CHECK(__sync_bool_compare_and_swap(&m->state, MEMX_PAGE_RESIDENT, MEMX_PAGE_COMPRESSING),
          "entered compressing");
    uint32_t seq0 = m->write_seq;
    memcpy(sim_pages[3].snap, sim_pages[3].data, PAGE_SZ);
    /* concurrent writer: full fault protocol */
    os_write_fault(3);
    fill_fp16(sim_pages[3].data, 99);
    /* commit must refuse */
    CHECK(!page_compress_meta_stable(m, seq0), "torn write detected via meta");
    CHECK(m->state == MEMX_PAGE_HOT, "writer CASed out of compressing");
    fill_fp16(sim_pages[3].snap, 99);
    CHECK(memcmp(sim_pages[3].data, sim_pages[3].snap, PAGE_SZ) == 0, "writer's bytes intact");

    /* 3. write_seq discipline: bump without CAS must still block commit */
    sim_reset();
    fill_fp16(sim_pages[4].data, 13);
    m = &sim_meta[4];
    m->state = MEMX_PAGE_COMPRESSING;
    seq0 = m->write_seq;
    memcpy(sim_pages[4].snap, sim_pages[4].data, PAGE_SZ);
    m->write_seq++;  /* writer bumped but CAS race pending */
    m->dirty = 1;
    CHECK(!page_compress_meta_stable(m, seq0), "seq bump blocks commit");
    CHECK(!page_compress_content_ok(m, seq0, sim_pages[4].snap, sim_pages[4].data),
          "content check also blocked");

    /* 4. dirty-clear retry path: after quiesce the page compresses again */
    sim_reset();
    fill_fp16(sim_pages[5].data, 21);
    m = &sim_meta[5];
    m->dirty = 1;
    CHECK(sim_compress_one(5) == 0, "dirty page not compressed");
    m->dirty = 0;
    m->stable_ticks = 0;
    /* READ_MOSTLY|COLD weight needs 0 stable ticks */
    CHECK(sim_compress_one(5) == 1, "clean page compresses");
    fill_fp16(sim_pages[5].snap, 21);
    CHECK(memcmp(os_read(5), sim_pages[5].snap, PAGE_SZ) == 0, "page5 roundtrip");

    /* 5. sparse page takes the sparse codec */
    sim_reset();
    memset(sim_pages[6].data, 0, PAGE_SZ);
    for (uint32_t k = 0; k < PAGE_SZ; k += 517) sim_pages[6].data[k] = (uint8_t)(k ^ 0x5A);
    sim_meta[6].tensor_dtype = MEMX_DTYPE_UINT8;
    CHECK(sim_compress_one(6) == 1, "sparse page compressed");
    CHECK(sim_pages[6].blob_sz < 512, "sparse page tiny");
    uint8_t expect[PAGE_SZ];
    memset(expect, 0, PAGE_SZ);
    for (uint32_t k = 0; k < PAGE_SZ; k += 517) expect[k] = (uint8_t)(k ^ 0x5A);
    CHECK(memcmp(os_read(6), expect, PAGE_SZ) == 0, "sparse page bitexact");

    /* 6. incompressible page backs off and stays resident */
    sim_reset();
    uint32_t rs = 0x9E3779B9u;
    for (uint32_t k = 0; k < PAGE_SZ; k += 4) {
        rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
        memcpy(sim_pages[7].data + k, &rs, 4);
    }
    sim_meta[7].tensor_dtype = MEMX_DTYPE_UINT8;
    uint8_t golden[PAGE_SZ];
    memcpy(golden, sim_pages[7].data, PAGE_SZ);
    CHECK(sim_compress_one(7) == 0, "random page rejected");
    CHECK(sim_meta[7].state == MEMX_PAGE_RESIDENT, "random page resident");
    CHECK(sim_meta[7].cooldown == 16, "backoff set");
    CHECK(memcmp(sim_pages[7].data, golden, PAGE_SZ) == 0, "random page intact");

    if (sim_failures) {
        printf("kernel sim: FAILED (%d)\n", sim_failures);
        return 1;
    }
    printf("kernel sim: OK (%u-byte pages; %d pages; fault+commit+install contract holds)\n",
           (unsigned)PAGE_SZ, SIM_PAGES);
    return 0;
}
