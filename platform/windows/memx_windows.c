/*
 * Windows reference adapter for memx_core.
 *
 * Implements the EMBEDDING.md contract on Win32: vectored exception handling
 * (EXCEPTION_ACCESS_VIOLATION) for fault delivery, VirtualProtect for the
 * protection handshake, one compressor thread, a bump arena for compressed
 * blobs. Mirrors platform/posix/memx_posix.c; the core archive is shared.
 *
 * Notes:
 * - Reserve the region with MEM_RESERVE|MEM_COMMIT + PAGE_NOACCESS; faults
 *   arrive as access violations, not commit faults.
 * - Hardware page size is 4K on x86_64; a 16K logical page is a multiple of
 *   it, so VirtualProtect granularity is fine. Init returns -2 when the
 *   hardware page exceeds the logical page (same contract as POSIX).
 * - CRITICAL_SECTION is recursive on the same thread, which keeps the
 *   VEH-handler/compressor interplay safe.
 * - Builds with both MSVC (_MSC_VER) and mingw-w64 GCC (__sync builtins).
 */
#include "memx_windows.h"
#include "memx_core.h"

#include <windows.h>
#ifdef _MSC_VER
#include <intrin.h>
#define MEMX_TLS __declspec(thread)
#define MEMX_CAS8(p, newv, oldv) ((uint8_t)_InterlockedCompareExchange8((volatile char *)(p), (char)(newv), (char)(oldv)))
#define MEMX_INC32(p) InterlockedIncrement((volatile LONG *)(p))
#else
#define MEMX_TLS __thread
#define MEMX_CAS8(p, newv, oldv) ((uint8_t)__sync_val_compare_and_swap((volatile uint8_t *)(p), (uint8_t)(oldv), (uint8_t)(newv)))
#define MEMX_INC32(p) __sync_fetch_and_add((volatile uint32_t *)(p), 1)
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t *region;
    size_t region_bytes;
    size_t npages;
    PageMeta *meta;
    uint8_t *arena;
    size_t arena_next;
    CRITICAL_SECTION mu;
    volatile LONG running;
    HANDLE thr;
    volatile uint32_t faults;
    uint16_t role;
    uint16_t dtype;
    uint32_t tflags;
    PVOID veh;
    int mu_ready;
} win_zone_t;

static win_zone_t Z;

static void *page_at(size_t p) {
    return Z.region + p * PAGE_SZ;
}

static void win_barrier(void) {
    MemoryBarrier();
}

static int grant_rw(void *pa) {
    DWORD old;
    return VirtualProtect(pa, PAGE_SZ, PAGE_READWRITE, &old) ? 0 : -1;
}

static int grant_ro(void *pa) {
    DWORD old;
    return VirtualProtect(pa, PAGE_SZ, PAGE_READONLY, &old) ? 0 : -1;
}

static int grant_none(void *pa) {
    DWORD old;
    return VirtualProtect(pa, PAGE_SZ, PAGE_NOACCESS, &old) ? 0 : -1;
}

/* ── fault delivery: vectored exception handler ─────────────────── */

static LONG CALLBACK win_fault_handler(PEXCEPTION_POINTERS ep) {
    if (!Z.region) return EXCEPTION_CONTINUE_SEARCH;
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION)
        return EXCEPTION_CONTINUE_SEARCH;
    uintptr_t fa = (uintptr_t)ep->ExceptionRecord->ExceptionInformation[1];
    uintptr_t base = (uintptr_t)Z.region;
    if (fa < base || fa >= base + Z.region_bytes) return EXCEPTION_CONTINUE_SEARCH;
    /* ExceptionInformation[0]: 0=read, 1=write, 8=DEP */
    int is_write = (ep->ExceptionRecord->ExceptionInformation[0] == 1);
    size_t p = (fa - base) / PAGE_SZ;
    PageMeta *m = &Z.meta[p];
    void *pa = page_at(p);
    MEMX_INC32(&Z.faults);

    for (int attempt = 0; attempt < 64; attempt++) {
        uint8_t st = m->state;
        if (st == MEMX_PAGE_NONE) {
            grant_rw(pa);
            uint8_t old = MEMX_CAS8(&m->state, MEMX_PAGE_RESIDENT, MEMX_PAGE_NONE);
            if (old == MEMX_PAGE_NONE) {
                memset(pa, 0, PAGE_SZ);
                m->dirty = 1;
                m->stable_ticks = 0;
            }
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if (st == MEMX_PAGE_RESIDENT || st == MEMX_PAGE_HOT) {
            if (st == MEMX_PAGE_HOT && m->comp_size != 0) {
                wait_decompress_complete(m);
                continue;
            }
            if (is_write) {
                m->dirty = 1;
                m->stable_ticks = 0;
                MEMX_INC32(&m->write_seq);
                win_barrier();
            }
            grant_rw(pa);
            if (is_write && st == MEMX_PAGE_RESIDENT) {
                uint8_t old = MEMX_CAS8(&m->state, MEMX_PAGE_HOT, MEMX_PAGE_RESIDENT);
                if (old == MEMX_PAGE_RESIDENT) m->cooldown = 8;
            } else if (is_write && st == MEMX_PAGE_HOT && m->cooldown < 3) {
                m->cooldown = 3;
            }
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if (st == MEMX_PAGE_COMPRESSED) {
            EnterCriticalSection(&Z.mu);
            if (m->state != MEMX_PAGE_COMPRESSED) {
                LeaveCriticalSection(&Z.mu);
                continue;
            }
            uint8_t prev = MEMX_CAS8(&m->state, MEMX_PAGE_HOT, MEMX_PAGE_COMPRESSED);
            if (prev != MEMX_PAGE_COMPRESSED) {
                LeaveCriticalSection(&Z.mu);
                continue;
            }
            uint64_t off = m->pool_offset;
            uint32_t csz = m->comp_size;
            uint8_t payload[PAGE_SZ];
            int ok = (csz > 0 && csz <= PAGE_SZ && off + csz <= Z.arena_next);
            if (ok) memcpy(payload, Z.arena + off, csz);
            LeaveCriticalSection(&Z.mu);
            if (!ok) return EXCEPTION_CONTINUE_EXECUTION;

            uint8_t scratch[PAGE_SZ];
            cpu_decompress(payload, csz, scratch);
            grant_rw(pa);
            memcpy(pa, scratch, PAGE_SZ);
            win_barrier();
            m->comp_size = 0;
            m->pool_offset = 0;
            if ((Z.tflags & MEMX_FLAG_READ_MOSTLY) != 0) grant_ro(pa);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if (st == MEMX_PAGE_COMPRESSING) {
            m->dirty = 1;
            m->stable_ticks = 0;
            MEMX_INC32(&m->write_seq);
            win_barrier();
            grant_rw(pa);
            uint8_t old = MEMX_CAS8(&m->state, MEMX_PAGE_HOT, MEMX_PAGE_COMPRESSING);
            if (old == MEMX_PAGE_COMPRESSING) {
                m->cooldown = 12;
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            continue;
        }
        return EXCEPTION_CONTINUE_SEARCH;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

/* ── compressor thread ───────────────────────────────────────────── */

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
    uint8_t old = MEMX_CAS8(&m->state, MEMX_PAGE_COMPRESSING, MEMX_PAGE_RESIDENT);
    if (old != MEMX_PAGE_RESIDENT) return 0;
    if (m->dirty || m->write_seq != seq0) {
        m->state = MEMX_PAGE_RESIDENT;
        return 0;
    }

    void *pa = page_at(p);
    int wants_wp = page_wants_write_protect(m);
    if (wants_wp) grant_ro(pa);
    win_barrier();
    if (m->dirty || m->write_seq != seq0) {
        m->state = MEMX_PAGE_RESIDENT;
        if (wants_wp) grant_rw(pa);
        return 0;
    }
    static MEMX_TLS uint8_t snap[PAGE_SZ];
    memcpy(snap, pa, PAGE_SZ);
    win_barrier();
    if (!page_compress_content_ok(m, seq0, snap, pa)) {
        m->state = MEMX_PAGE_RESIDENT;
        if (wants_wp) grant_rw(pa);
        return 0;
    }

    uint32_t csz = 0;
    static MEMX_TLS uint8_t blob[PAGE_SZ];
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
        if (wants_wp) grant_rw(pa);
        return 0;
    }

    win_barrier();
    if (!page_compress_content_ok(m, seq0, snap, pa)) {
        m->state = MEMX_PAGE_RESIDENT;
        if (wants_wp) grant_rw(pa);
        return 0;
    }
    EnterCriticalSection(&Z.mu);
    if (Z.arena_next + csz > (size_t)(Z.npages * PAGE_SZ)) {
        LeaveCriticalSection(&Z.mu);
        m->state = MEMX_PAGE_RESIDENT;
        if (wants_wp) grant_rw(pa);
        return 0;
    }
    uint64_t off = Z.arena_next;
    memcpy(Z.arena + off, blob, csz);
    Z.arena_next += csz;
    m->pool_offset = off;
    m->comp_size = csz;
    m->codec = 0;
    win_barrier();
    if (m->dirty || m->write_seq != seq0 || m->state != MEMX_PAGE_COMPRESSING) {
        Z.arena_next = off;
        m->pool_offset = 0;
        m->comp_size = 0;
        LeaveCriticalSection(&Z.mu);
        m->state = MEMX_PAGE_RESIDENT;
        grant_rw(pa);
        return 0;
    }
    old = MEMX_CAS8(&m->state, MEMX_PAGE_COMPRESSED, MEMX_PAGE_COMPRESSING);
    LeaveCriticalSection(&Z.mu);
    if (old != MEMX_PAGE_COMPRESSING) {
        if (m->state == MEMX_PAGE_RESIDENT || m->state == MEMX_PAGE_HOT) {
            m->pool_offset = 0;
            m->comp_size = 0;
            grant_rw(pa);
        }
        return 0;
    }
    grant_none(pa);
    return 1;
}

static DWORD WINAPI compressor_main(LPVOID arg) {
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
                } else {
                    uint8_t old = MEMX_CAS8(&m->state, MEMX_PAGE_RESIDENT, MEMX_PAGE_HOT);
                    if (old == MEMX_PAGE_HOT) m->stable_ticks = 0;
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
        if (did == 0) Sleep(2);
    }
    return 0;
}

/* ── public surface ─────────────────────────────────────────────── */

int memx_win_init(size_t region_bytes, uint16_t tensor_role, uint16_t tensor_dtype,
                  uint32_t tensor_flags) {
    memset(&Z, 0, sizeof(Z));
    if (region_bytes < PAGE_SZ || (region_bytes % PAGE_SZ) != 0) return -1;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    size_t hw = si.dwPageSize;
    if (hw == 0 || hw > PAGE_SZ || (PAGE_SZ % hw) != 0) return -2;
    Z.region_bytes = region_bytes;
    Z.npages = region_bytes / PAGE_SZ;
    Z.role = tensor_role;
    Z.dtype = tensor_dtype;
    Z.tflags = tensor_flags;

    Z.region = (uint8_t *)VirtualAlloc(NULL, region_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
    if (!Z.region) return -1;
    Z.meta = (PageMeta *)calloc(Z.npages, sizeof(PageMeta));
    Z.arena = (uint8_t *)calloc(Z.npages, PAGE_SZ);
    if (!Z.meta || !Z.arena) return -1;
    for (size_t p = 0; p < Z.npages; p++) {
        Z.meta[p].state = MEMX_PAGE_NONE;
        Z.meta[p].tensor_role = tensor_role;
        Z.meta[p].tensor_dtype = tensor_dtype;
        Z.meta[p].tensor_flags = tensor_flags;
    }
    InitializeCriticalSection(&Z.mu);
    Z.mu_ready = 1;

    Z.veh = AddVectoredExceptionHandler(1, win_fault_handler);
    if (!Z.veh) return -1;

    Z.running = 1;
    Z.thr = CreateThread(NULL, 0, compressor_main, NULL, 0, NULL);
    if (!Z.thr) {
        Z.running = 0;
        return -1;
    }
    return 0;
}

void memx_win_shutdown(void) {
    if (Z.running) {
        InterlockedExchange(&Z.running, 0);
        WaitForSingleObject(Z.thr, INFINITE);
        CloseHandle(Z.thr);
    }
    if (Z.veh) {
        RemoveVectoredExceptionHandler(Z.veh);
        Z.veh = NULL;
    }
    if (Z.mu_ready) {
        DeleteCriticalSection(&Z.mu);
        Z.mu_ready = 0;
    }
    if (Z.region) VirtualFree(Z.region, 0, MEM_RELEASE);
    free(Z.meta);
    free(Z.arena);
    memset(&Z, 0, sizeof(Z));
}

uint8_t *memx_win_region(void) {
    return Z.region;
}

size_t memx_win_region_bytes(void) {
    return Z.region_bytes;
}

uint64_t memx_win_compressed_pages(void) {
    uint64_t n = 0;
    for (size_t p = 0; p < Z.npages; p++) {
        if (Z.meta[p].state == MEMX_PAGE_COMPRESSED) n++;
    }
    return n;
}

uint64_t memx_win_faults(void) {
    return (uint64_t)Z.faults;
}

void memx_win_dump_states(void) {
    int cnt[6] = {0, 0, 0, 0, 0, 0};
    for (size_t p = 0; p < Z.npages; p++) {
        int st = Z.meta[p].state;
        if (st < 0 || st > 4) st = 5;
        cnt[st]++;
    }
    fprintf(stderr, "[win] none=%d res=%d comp=%d hot=%d compressing=%d other=%d arena=%zu\n",
            cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cnt[5], Z.arena_next);
}