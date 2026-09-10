/*
 * Minimal POSIX reference adapter for memx_core.
 *
 * Implements the EMBEDDING.md contract on plain POSIX: SIGSEGV/sigaction fault
 * delivery (the portable baseline — userfaultfd is a Linux-specific refinement),
 * mprotect for the protection handshake, one compressor thread, a bump arena
 * for compressed blobs. The macOS runtime (libmemx3.m) is the full-featured
 * sibling; this adapter exists to prove the contract ports to any POSIX OS
 * (Linux, BSDs) and builds at any core page size (4K on Linux CI, 16K here).
 *
 * Deliberately simple: one region, one global descriptor, O(npages) scan per
 * compressor pass, no pool reclaim. It is the seed for per-OS runtimes, not
 * the end product.
 */
#include "memx_posix.h"
#include "memx_core.h"

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    uint8_t *region;
    size_t region_bytes;
    size_t npages;
    PageMeta *meta;
    uint8_t *arena;
    size_t arena_next;
    pthread_mutex_t mu;
    volatile int running;
    pthread_t thr;
    uint64_t faults;
    uint16_t role;
    uint16_t dtype;
    uint32_t tflags;
    struct sigaction old_segv;
    struct sigaction old_bus;
    int handler_installed;
} posix_zone_t;

static posix_zone_t Z;

static void *page_at(size_t p) {
    return Z.region + p * PAGE_SZ;
}

static void posix_barrier(void) {
    __sync_synchronize();
}

/* ── fault handler: the embedder's fault-delivery surface ────────── */

static void posix_fault_handler(int sig, siginfo_t *info, void *uctx) {
    (void)sig;
    (void)uctx;
    if (!Z.region) return;
    uintptr_t fa = (uintptr_t)info->si_addr;
    uintptr_t base = (uintptr_t)Z.region;
    if (fa < base || fa >= base + Z.region_bytes) return;
    size_t p = (fa - base) / PAGE_SZ;
    PageMeta *m = &Z.meta[p];
    void *pa = page_at(p);
    __sync_fetch_and_add(&Z.faults, 1);

    for (int attempt = 0; attempt < 64; attempt++) {
        uint8_t st = m->state;
        if (st == MEMX_PAGE_NONE) {
            mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
            uint8_t old = __sync_val_compare_and_swap(&m->state, MEMX_PAGE_NONE, MEMX_PAGE_RESIDENT);
            if (old == MEMX_PAGE_NONE) {
                memset(pa, 0, PAGE_SZ);
                m->dirty = 1;
                m->stable_ticks = 0;
            }
            return;
        }
        if (st == MEMX_PAGE_RESIDENT || st == MEMX_PAGE_HOT) {
            if (st == MEMX_PAGE_HOT && m->comp_size != 0) {
                wait_decompress_complete(m);
                continue;
            }
            int is_write = (!info || info->si_code == SEGV_ACCERR || info->si_code == BUS_ADRERR);
            if (is_write) {
                m->dirty = 1;
                m->stable_ticks = 0;
                __sync_fetch_and_add(&m->write_seq, 1);
                posix_barrier();
            }
            mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
            if (is_write && st == MEMX_PAGE_RESIDENT) {
                uint8_t old = __sync_val_compare_and_swap(&m->state, MEMX_PAGE_RESIDENT, MEMX_PAGE_HOT);
                if (old == MEMX_PAGE_RESIDENT) m->cooldown = 8;
            } else if (is_write && st == MEMX_PAGE_HOT && m->cooldown < 3) {
                m->cooldown = 3;
            }
            return;
        }
        if (st == MEMX_PAGE_COMPRESSED) {
            pthread_mutex_lock(&Z.mu);
            if (m->state != MEMX_PAGE_COMPRESSED) {
                pthread_mutex_unlock(&Z.mu);
                continue;
            }
            if (!__sync_bool_compare_and_swap(&m->state, MEMX_PAGE_COMPRESSED, MEMX_PAGE_HOT)) {
                pthread_mutex_unlock(&Z.mu);
                continue;
            }
            uint64_t off = m->pool_offset;
            uint32_t csz = m->comp_size;
            uint8_t payload[PAGE_SZ];
            int ok = (csz > 0 && csz <= PAGE_SZ && off + csz <= Z.arena_next);
            if (ok) memcpy(payload, Z.arena + off, csz);
            pthread_mutex_unlock(&Z.mu);
            if (!ok) return;

            uint8_t scratch[PAGE_SZ];
            cpu_decompress(payload, csz, scratch);
            mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
            memcpy(pa, scratch, PAGE_SZ);
            posix_barrier();
            m->comp_size = 0;
            m->pool_offset = 0;
            if ((Z.tflags & MEMX_FLAG_READ_MOSTLY) != 0)
                mprotect(pa, PAGE_SZ, PROT_READ);
            return;
        }
        if (st == MEMX_PAGE_COMPRESSING) {
            m->dirty = 1;
            m->stable_ticks = 0;
            __sync_fetch_and_add(&m->write_seq, 1);
            posix_barrier();
            mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
            if (__sync_bool_compare_and_swap(&m->state, MEMX_PAGE_COMPRESSING, MEMX_PAGE_HOT)) {
                m->cooldown = 12;
                return;
            }
            continue;
        }
        return;
    }
}

/* ── the compressor thread the embedder drives ───────────────────── */

static int compress_one(size_t p) {
    PageMeta *m = &Z.meta[p];
    if (m->state != MEMX_PAGE_RESIDENT || m->dirty) return 0;
    if (m->cooldown > 0) {
        m->cooldown--;
        return 0;
    }
    if (m->stable_ticks < 1) {
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

    void *pa = page_at(p);
    int wants_wp = page_wants_write_protect(m);
    if (wants_wp) mprotect(pa, PAGE_SZ, PROT_READ);
    posix_barrier();
    if (m->dirty || m->write_seq != seq0) {
        m->state = MEMX_PAGE_RESIDENT;
        if (wants_wp) mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
        return 0;
    }
    static __thread uint8_t snap[PAGE_SZ];
    memcpy(snap, pa, PAGE_SZ);
    posix_barrier();
    if (!page_compress_content_ok(m, seq0, snap, pa)) {
        m->state = MEMX_PAGE_RESIDENT;
        if (wants_wp) mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
        return 0;
    }

    uint32_t csz = 0;
    static __thread uint8_t blob[PAGE_SZ];
    if (page_is_all_zero(snap)) {
        static const uint8_t ZERO_PAGE[8] = {0x4D, 0x58, 0x03, 0x00, 0xFD, 0x00, 0x00, 0x40};
        memcpy(blob, ZERO_PAGE, 8);
        csz = 8;
    } else if (tensor_fp16_split_eligible(m)) {
        csz = tensor_fp16_split_compress(snap, blob, PAGE_SZ);
    }
    if (csz == 0) csz = tensor_sparse_byte_compress(snap, blob, PAGE_SZ);
    if (csz == 0) {
        m->cooldown = 16;
        m->state = MEMX_PAGE_RESIDENT;
        if (wants_wp) mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
        return 0;
    }

    posix_barrier();
    if (!page_compress_content_ok(m, seq0, snap, pa)) {
        m->state = MEMX_PAGE_RESIDENT;
        if (wants_wp) mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
        return 0;
    }
    pthread_mutex_lock(&Z.mu);
    if (Z.arena_next + csz > (size_t)(Z.npages * PAGE_SZ)) {
        pthread_mutex_unlock(&Z.mu);
        m->state = MEMX_PAGE_RESIDENT;
        if (wants_wp) mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
        return 0;
    }
    uint64_t off = Z.arena_next;
    memcpy(Z.arena + off, blob, csz);
    Z.arena_next += csz;
    m->pool_offset = off;
    m->comp_size = csz;
    m->codec = 0;
    posix_barrier();
    if (m->dirty || m->write_seq != seq0 || m->state != MEMX_PAGE_COMPRESSING) {
        Z.arena_next = off;
        m->pool_offset = 0;
        m->comp_size = 0;
        pthread_mutex_unlock(&Z.mu);
        m->state = MEMX_PAGE_RESIDENT;
        mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
        return 0;
    }
    uint8_t old = __sync_val_compare_and_swap(&m->state, MEMX_PAGE_COMPRESSING, MEMX_PAGE_COMPRESSED);
    pthread_mutex_unlock(&Z.mu);
    if (old != MEMX_PAGE_COMPRESSING) {
        if (m->state == MEMX_PAGE_RESIDENT || m->state == MEMX_PAGE_HOT) {
            m->pool_offset = 0;
            m->comp_size = 0;
            mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
        }
        return 0;
    }
    mprotect(pa, PAGE_SZ, PROT_NONE);
    return 1;
}

static void *compressor_main(void *arg) {
    (void)arg;
    while (Z.running) {
        uint64_t did = 0;
        for (size_t p = 0; p < Z.npages && Z.running; p++) {
            PageMeta *m = &Z.meta[p];
            if (m->state == MEMX_PAGE_HOT) {
                if (m->dirty) {
                    m->dirty = 0;
                    if (m->cooldown < 8) m->cooldown = 8;
                    m->stable_ticks = 0;
                } else if (m->cooldown > 0) {
                    m->cooldown--;
                } else if (__sync_bool_compare_and_swap(&m->state, MEMX_PAGE_HOT, MEMX_PAGE_RESIDENT)) {
                    m->stable_ticks = 0;
                }
                continue;
            }
            if (m->state == MEMX_PAGE_RESIDENT && m->dirty) {
                m->dirty = 0;
                m->stable_ticks = 0;
                if (m->cooldown < 2) m->cooldown = 2;
                continue;
            }
            did += (uint64_t)compress_one(p);
        }
        if (did == 0) {
            struct timespec ts = {0, 2000000L};
            nanosleep(&ts, NULL);
        }
    }
    return NULL;
}

/* ── public surface ───────────────────────────────────────────────── */

int memx_posix_init(size_t region_bytes, uint16_t tensor_role, uint16_t tensor_dtype,
                   uint32_t tensor_flags) {
    memset(&Z, 0, sizeof(Z));
    if (region_bytes < PAGE_SZ || (region_bytes % PAGE_SZ) != 0) return -1;
    long hw = sysconf(_SC_PAGESIZE);
    if (hw <= 0 || ((size_t)hw > PAGE_SZ || (PAGE_SZ % (size_t)hw) != 0)) return -2;
    Z.region_bytes = region_bytes;
    Z.npages = region_bytes / PAGE_SZ;
    Z.role = tensor_role;
    Z.dtype = tensor_dtype;
    Z.tflags = tensor_flags;

    Z.region = mmap(NULL, region_bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (Z.region == MAP_FAILED) return -1;
    Z.meta = calloc(Z.npages, sizeof(PageMeta));
    Z.arena = calloc(Z.npages, PAGE_SZ);
    if (!Z.meta || !Z.arena) return -1;
    for (size_t p = 0; p < Z.npages; p++) {
        Z.meta[p].state = MEMX_PAGE_NONE;
        Z.meta[p].tensor_role = tensor_role;
        Z.meta[p].tensor_dtype = tensor_dtype;
        Z.meta[p].tensor_flags = tensor_flags;
    }
    pthread_mutex_init(&Z.mu, NULL);

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = posix_fault_handler;
    sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGSEGV, &sa, &Z.old_segv) != 0) return -1;
    if (sigaction(SIGBUS, &sa, &Z.old_bus) != 0) {
        sigaction(SIGSEGV, &Z.old_segv, NULL);
        return -1;
    }
    Z.handler_installed = 1;

    Z.running = 1;
    if (pthread_create(&Z.thr, NULL, compressor_main, NULL) != 0) {
        Z.running = 0;
        return -1;
    }
    return 0;
}

void memx_posix_shutdown(void) {
    if (Z.running) {
        Z.running = 0;
        pthread_join(Z.thr, NULL);
    }
    if (Z.handler_installed) {
        sigaction(SIGSEGV, &Z.old_segv, NULL);
        sigaction(SIGBUS, &Z.old_bus, NULL);
        Z.handler_installed = 0;
    }
    pthread_mutex_destroy(&Z.mu);
    if (Z.region) munmap(Z.region, Z.region_bytes);
    free(Z.meta);
    free(Z.arena);
    memset(&Z, 0, sizeof(Z));
}

uint8_t *memx_posix_region(void) {
    return Z.region;
}

size_t memx_posix_region_bytes(void) {
    return Z.region_bytes;
}

uint64_t memx_posix_compressed_pages(void) {
    uint64_t n = 0;
    for (size_t p = 0; p < Z.npages; p++) {
        if (Z.meta[p].state == MEMX_PAGE_COMPRESSED) n++;
    }
    return n;
}

uint64_t memx_posix_faults(void) {
    return Z.faults;
}

void memx_posix_dump_states(void) {
    int cnt[6] = {0, 0, 0, 0, 0, 0};
    for (size_t p = 0; p < Z.npages; p++) {
        int st = Z.meta[p].state;
        if (st < 0 || st > 4) st = 5;
        cnt[st]++;
    }
    fprintf(stderr, "[posix] none=%d res=%d comp=%d hot=%d compressing=%d other=%d arena=%zu\n",
            cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cnt[5], Z.arena_next);
}
