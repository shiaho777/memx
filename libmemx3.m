// MemX explicit runtime:
// managed allocations, quota-aware contexts, and compressed virtual pages.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <pthread.h>
#include <time.h>
#include <mach/mach.h>
#include <dlfcn.h>
#include <malloc/malloc.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>

#include "memx_runtime.h"
#include <zlib.h>
#include <dispatch/dispatch.h>

#import <Metal/Metal.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MEMX_HAS_NEON 1
#else
#define MEMX_HAS_NEON 0
#endif

static const char *memx_mode_label(void) { return "explicit runtime"; }

#define PAGE_SZ 16384
#define MB (1024ULL*1024)
#define LARGE_THRESHOLD 65536

// ─── GPU Shaders: Adaptive compression for the explicit runtime ───
// Zero-heavy pages skip LZ77 (faster), structured pages use full RLE+LZ77
// Output format remains compatible with the paired decompressor below
static NSString *const shader_src = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"constant uint PS=16384;\n"
"uint h4(threadgroup const uchar* p){return ((uint)p[0]|((uint)p[1]<<8)|((uint)p[2]<<16)|((uint)p[3]<<24))*2654435761u;}\n"
"kernel void cp(device const uchar* s[[buffer(0)]],device uchar* d[[buffer(1)]],device uint* z[[buffer(2)]],uint t[[thread_position_in_threadgroup]],uint pg[[threadgroup_position_in_grid]],uint ts[[threads_per_threadgroup]]){"
"threadgroup uchar dp[16384];threadgroup uint hk[2048],hv[2048];uint po=pg*PS;"
"if(t<256){if(t==0){dp[0]=s[po];for(uint i=1;i<64;i++)dp[i]=s[po+i]-s[po+i-1];}else{dp[t*64]=s[po+t*64]-s[po+t*64-1];for(uint i=1;i<64;i++)dp[t*64+i]=s[po+t*64+i]-s[po+t*64+i-1];}}"
"threadgroup_barrier(mem_flags::mem_threadgroup);"
"for(uint i=t;i<2048;i+=ts){hk[i]=0xFFFFFFFFu;hv[i]=0;}"
"threadgroup_barrier(mem_flags::mem_threadgroup);"
"if(t==0){uint db=pg*PS,op=4;uint ip=0;"
"uint zc=0;for(uint i=0;i<PS;i+=64)if(dp[i]==0)zc++;uint use_lz=(zc<PS/128)?1u:0u;"
"while(ip<PS){uint rl=1;while(ip+rl<PS&&dp[ip+rl]==dp[ip]&&rl<65535)rl++;"
"if(rl>=4&&op+4<=PS){d[db+op++]=0xFD;d[db+op++]=dp[ip];d[db+op++]=(uchar)(rl&0xFF);d[db+op++]=(uchar)((rl>>8)&0xFF);ip+=rl;continue;}"
"if(use_lz&&ip+4<=PS){uint h=h4(dp+ip)&2047;uint pp=hv[h],pk=hk[h];uint ck=(uint)dp[ip]|((uint)dp[ip+1]<<8)|((uint)dp[ip+2]<<16)|((uint)dp[ip+3]<<24);hk[h]=ck;hv[h]=ip;"
"if(pk==ck&&pp<ip&&(ip-pp)<4096){uint ml=0;while(ml<65535&&ip+ml<PS&&dp[ip+ml]==dp[pp+ml])ml++;"
"if(ml>=4&&op+5<=PS){d[db+op++]=0xFF;d[db+op++]=(uchar)((ip-pp)&0xFF);d[db+op++]=(uchar)(((ip-pp)>>8)&0xFF);d[db+op++]=(uchar)(ml&0xFF);d[db+op++]=(uchar)((ml>>8)&0xFF);ip+=ml;continue;}}}"
"if(dp[ip]==0xFD){if(op+2<=PS){d[db+op++]=0xFE;d[db+op++]=0xFD;}else break;}"
"else if(dp[ip]==0xFE){if(op+2<=PS){d[db+op++]=0xFE;d[db+op++]=0xFE;}else break;}"
"else if(dp[ip]==0xFF){if(op+2<=PS){d[db+op++]=0xFE;d[db+op++]=0xFF;}else break;}"
"else{if(op+1<=PS)d[db+op++]=dp[ip];else break;}ip++;}"
"if(op>=PS){z[pg]=PS;for(uint i=t;i<PS;i+=ts)d[pg*PS+i]=s[po+i];}"
"else{d[db]=0x4D;d[db+1]=0x58;d[db+2]=3;d[db+3]=0;z[pg]=op;}}"
"threadgroup_barrier(mem_flags::mem_threadgroup);"
"if(z[pg]==PS){for(uint i=t;i<PS;i+=ts)d[pg*PS+i]=s[po+i];}}\n"
"kernel void dp(device const uchar* s[[buffer(0)]],device uchar* d[[buffer(1)]],device const uint* z[[buffer(2)]],uint t[[thread_position_in_threadgroup]],uint pg[[threadgroup_position_in_grid]],uint ts[[threads_per_threadgroup]]){uint po=pg*PS,sb=pg*PS,cs=z[pg];if(cs==PS){for(uint i=t;i<PS;i+=ts)d[po+i]=s[sb+i];return;}if(s[sb]!=0x4D||s[sb+1]!=0x58){for(uint i=t;i<PS;i+=ts)d[po+i]=s[sb+i];return;}uint ver=s[sb+2];threadgroup uchar db[16384];if(t==0){uint ip=4,op=0;while(ip<cs&&op<PS){uchar b=s[sb+ip];if(b==0xFD&&ver>=2&&ip+3<cs){uchar vb=s[sb+ip+1];uint rl=(uint)s[sb+ip+2]|((uint)s[sb+ip+3]<<8);ip+=4;for(uint i=0;i<rl&&op<PS;i++)db[op++]=vb;}else if(b==0xFF&&ip+4<cs){ip++;uint off=(uint)s[sb+ip]|(((uint)s[sb+ip+1])<<8);ip+=2;uint ml=(uint)s[sb+ip]|(((uint)s[sb+ip+1])<<8);ip+=2;uint ms=op-off;for(uint i=0;i<ml&&op<PS;i++)db[op++]=db[ms+i];}else if(b==0xFE&&ip+1<cs){ip++;db[op++]=s[sb+ip++];}else{db[op++]=b;ip++;}}if(op<PS)for(uint i=op;i<PS;i++)db[i]=0;}threadgroup_barrier(mem_flags::mem_threadgroup);if(t==0){d[po]=db[0];for(uint i=1;i<PS;i++)d[po+i]=d[po+i-1]+db[i];}}\n";

#define PAGE_NONE       0
#define PAGE_RESIDENT   1
#define PAGE_COMPRESSED 2
#define PAGE_HOT        3  // Recently decompressed, cooldown before compressible
#define PAGE_COMPRESSING 4
static malloc_zone_t *s_default_zone = NULL;
static malloc_zone_t memx_zone;  // Forward: defined below

static void *real_malloc(size_t size) {
    return malloc(size);
}
static void real_free(void *ptr) {
    free(ptr);
}
static void *real_calloc(size_t nmemb, size_t size) {
    return calloc(nmemb, size);
}
static void *real_realloc(void *ptr, size_t size) {
    return realloc(ptr, size);
}
static size_t real_malloc_size(const void *ptr) {
    malloc_zone_t *z = malloc_zone_from_ptr((void*)ptr);
    if (z && z->size) return z->size(z, ptr);
    return 0;
}

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
    uint16_t tensor_role;
    uint16_t tensor_dtype;
    uint16_t tensor_layout;
    uint32_t tensor_flags;
    uint32_t tensor_layer;
    uint32_t tensor_head;
    size_t   alloc_size;
    uintptr_t owner_tag;
} PageMeta;

typedef struct {
    void *ptr;
    size_t hot_off;
    size_t hot_end;
    size_t alloc_size;
    uint32_t gen;
    uint8_t active;
} memx_ws_track_t;

#define MEMX_WS_TRACK_MAX 32

struct memx_runtime_context {
    uint64_t magic;
    char name[64];
    volatile uint64_t bytes_in_use;
    volatile uint64_t peak_bytes_in_use;
    volatile uint64_t allocations_live;
    volatile uint64_t allocations_total;
    volatile uint64_t quota_bytes;
    volatile uint64_t allocation_failures_quota;
    volatile uint64_t pressure_events;
    volatile uint64_t tensor_bytes_in_use;
    volatile uint64_t tensor_allocations_live;
    volatile uint64_t weight_bytes_in_use;
    volatile uint64_t kv_cache_bytes_in_use;
    volatile uint64_t hot_bytes_in_use;
    volatile uint64_t no_compress_bytes_in_use;
    uint32_t epoch_phase;
    uint32_t epoch_gen;
    uint64_t hot_budget_bytes;
    volatile uint64_t ws_hot_bytes;
    memx_ws_track_t ws_tracks[MEMX_WS_TRACK_MAX];
    pthread_mutex_t ws_mutex;
    int ws_mutex_inited;
};

#define MEMX_CONTEXT_MAGIC 0x4D584354585431ULL

typedef struct {
    void *ptr;
    size_t offset;
    size_t length;
    uintptr_t owner_tag;
} async_seal_job_t;

typedef struct {
    void        *vmem;       uint64_t    vmem_size;   size_t      npages;
    uint64_t    vmem_next;   // next-fit hint for allocation (page index * PAGE_SZ)
    uint8_t    *pool;        uint64_t    pool_size;   uint64_t    pool_used;
    uint64_t    pool_next;
    pthread_mutex_t alloc_mutex;  // protects vmem_next, pool_next, meta state transitions
    PageMeta   *meta;
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLComputePipelineState> comp_pipe;
    id<MTLComputePipelineState> decomp_pipe;
    volatile uint64_t  faults;      volatile uint64_t  compressions;
    volatile uint64_t  bytes_saved; volatile int       running;
    volatile int       attached;    pthread_t          bg_thread;
    volatile uint64_t  tensor_codec_pages;
    volatile uint64_t  tensor_codec_bytes_saved;
    volatile uint64_t  tensor_split_pages;
    volatile uint64_t  tensor_split_bytes_saved;
    volatile uint64_t  tensor_bitplane_pages;
    volatile uint64_t  tensor_bitplane_bytes_saved;
    volatile uint64_t  tensor_sparse_pages;
    volatile uint64_t  tensor_sparse_bytes_saved;
    volatile uint64_t  tensor_delta_split_pages;
    volatile uint64_t  tensor_delta_split_bytes_saved;
    volatile uint64_t  tensor_exp_pack_pages;
    volatile uint64_t  tensor_exp_pack_bytes_saved;
    volatile uint64_t  weight_compressed_pages;
    volatile uint64_t  weight_bytes_saved;
    volatile uint64_t  kv_cache_compressed_pages;
    volatile uint64_t  kv_cache_bytes_saved;
    // Persistent GPU buffers (pre-allocated, reused every batch)
    id<MTLBuffer>       gpu_sb;      id<MTLBuffer>       gpu_db;      id<MTLBuffer>       gpu_zb;
    uint8_t            *tmp_src;     uint8_t            *tmp_dst;     uint32_t           *tmp_sz;
    size_t              batch_cap;  // max pages per batch
    // Page deduplication: same-content pages share one compressed copy
    #define DEDUP_HT_SIZE 16384
    #define DEDUP_HT_MASK (DEDUP_HT_SIZE-1)
    uint64_t           *dedup_hash;  // hash of compressed data [DEDUP_HT_SIZE]
    uint64_t           *dedup_off;   // pool_offset [DEDUP_HT_SIZE]
    uint32_t           *dedup_sz;    // comp_size [DEDUP_HT_SIZE]
    uint32_t           *dedup_ref;   // reference count [DEDUP_HT_SIZE]
    uint8_t            *dedup_pending_free; // slot has reached ref=0 and needs pool reclaim
    volatile uint32_t   dedup_pending_free_count;
    uint32_t           *dedup_rev;
    uint32_t            dedup_rev_size;
    uint32_t            dedup_rev_mask;
    volatile uint64_t   dedup_hits;
    volatile uint64_t   dedup_bytes_saved;
    #define POOL_FREE_EXTENTS_MAX 131072
    uint64_t           *pool_free_off; // sorted free extents by offset
    uint32_t           *pool_free_sz;
    uint32_t            pool_free_count;
    uint32_t            pool_free_cap;
    volatile uint64_t   pool_reclaim_bytes_total;
    volatile uint64_t   pool_reclaim_events;
    // Predictive prefetch: detect sequential access and prefetch ahead
    volatile uint64_t   prefetch_hits;   // pages that were prefetched before fault
    volatile uint64_t   prefetch_misses; // pages prefetched but never accessed
    volatile uint64_t   prefetch_count;  // total prefetch operations
    #define PREFETCH_AHEAD 6
    #define PREFETCH_AHEAD_KV 48
    #define PREFETCH_AHEAD_WEIGHT 12
    #define PREFETCH_STRIDE_MIN 1
    #define PREFETCH_STRIDE_MAX 64
    #define FAULT_STREAMS 3
    volatile size_t     last_fault_page;
    volatile int        last_fault_stride;
    volatile uint32_t   last_fault_role;
    volatile size_t     stream_fault_page[FAULT_STREAMS];
    volatile int        stream_fault_stride[FAULT_STREAMS];
    #define ASYNC_PF_Q_SIZE 8192
    #define ASYNC_PF_Q_MASK (ASYNC_PF_Q_SIZE - 1)
    #define ASYNC_PF_WORKERS 6
    #define COMP_CPU_WORKERS 8
    uint32_t           *async_pf_q;
    volatile uint32_t   async_pf_head;
    volatile uint32_t   async_pf_tail;
    pthread_mutex_t     async_pf_mutex;
    pthread_cond_t      async_pf_cond;
    pthread_t           async_pf_threads[ASYNC_PF_WORKERS];
    int                 async_pf_nworkers;
    volatile int        async_pf_running;
    volatile uint64_t   async_pf_enqueued;
    volatile uint64_t   async_pf_completed;
    #define ASYNC_SEAL_Q_SIZE 512
    #define ASYNC_SEAL_Q_MASK (ASYNC_SEAL_Q_SIZE - 1)
    #define ASYNC_SEAL_WORKERS 3
    async_seal_job_t   *async_seal_q;
    volatile uint32_t   async_seal_head;
    volatile uint32_t   async_seal_tail;
    pthread_mutex_t     async_seal_mutex;
    pthread_cond_t      async_seal_cond;
    pthread_cond_t      async_seal_idle_cond;
    pthread_t           async_seal_threads[ASYNC_SEAL_WORKERS];
    int                 async_seal_nworkers;
    volatile int        async_seal_running;
    volatile int        async_seal_active;
    volatile uint64_t   async_seal_enqueued;
    volatile uint64_t   async_seal_completed;
    pthread_t           encode_threads[COMP_CPU_WORKERS];
    int                 encode_nworkers;
    volatile int        encode_pool_running;
    pthread_mutex_t     encode_mutex;
    pthread_cond_t      encode_cond;
    pthread_cond_t      encode_done_cond;
    void               *encode_job;
    volatile int        encode_workers_active;
    volatile int        encode_epoch;
    volatile int        encode_seen_epoch[COMP_CPU_WORKERS];
    volatile uint64_t   live_compressed_pages;
    volatile uint64_t   live_resident_pages;
    volatile uint64_t   live_hot_flag_pages;
    volatile uint64_t   live_nocomp_flag_pages;
    // Free page bitmap for O(1) allocation instead of linear scan
    // Each bit = 1 page, 1=free, 0=used. 6M pages = 750KB bitmap.
    uint64_t           *free_bm;         // [npages/64] rounded up
    uint32_t            free_bm_size;     // number of uint64_t entries
    volatile uint64_t   free_pages_count;
    // Active page tracking: compact lists for compressor (avoid scanning all 6M pages)
    uint32_t           *hot_list;        // page indices that are PAGE_HOT
    volatile uint32_t   hot_count;        // number of entries in hot_list
    uint32_t            hot_cap;          // capacity of hot_list
    uint32_t           *res_list;        // page indices that are PAGE_RESIDENT
    volatile uint32_t   res_count;        // number of entries in res_list
    uint32_t            res_cap;          // capacity of res_list
    int                 idle_count;       // adaptive sleep counter for compressor
} MemXZone3;

static void note_page_compressed(MemXZone3 *s, size_t page_index, uint8_t codec, uint32_t comp_size);

// ─── Shared stats export for GUI dashboard ───
typedef struct {
    uint32_t magic;       // 0x4D585331 ('MXS1')
    uint32_t pid;         // process ID
    uint64_t compressions;
    uint64_t faults;
    uint64_t bytes_saved;
    uint64_t dedup_hits;
    uint64_t prefetch_count;
    uint64_t prefetch_hits;
    uint64_t vmem_size;   // total virtual MB
    uint64_t pool_used;   // compressed pool bytes used
    uint64_t npages;      // total pages managed
    uint64_t npages_compressed; // pages currently compressed
    uint64_t npages_resident;   // pages currently resident
    uint64_t _reserved[8];
} MemXSharedStats;
#define MEMX_SHARED_MAGIC 0x4D585331
static MemXSharedStats *g_shared_stats = NULL;
static char g_shared_stats_path[256];
static uint64_t pool_reclaim_pending_locked(MemXZone3 *s);

static void shared_stats_init(MemXZone3 *s) {
    snprintf(g_shared_stats_path, sizeof(g_shared_stats_path), "/tmp/memx_stats_%d", getpid());
    int fd = open(g_shared_stats_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    ftruncate(fd, sizeof(MemXSharedStats));
    g_shared_stats = mmap(NULL, sizeof(MemXSharedStats), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (g_shared_stats == MAP_FAILED) { g_shared_stats = NULL; return; }
    memset(g_shared_stats, 0, sizeof(MemXSharedStats));
    g_shared_stats->magic = MEMX_SHARED_MAGIC;
    g_shared_stats->pid = getpid();
}

static void shared_stats_update(MemXZone3 *s) {
    if (!g_shared_stats) return;
    g_shared_stats->compressions = s->compressions;
    g_shared_stats->faults = s->faults;
    g_shared_stats->bytes_saved = s->bytes_saved;
    g_shared_stats->dedup_hits = s->dedup_hits;
    g_shared_stats->prefetch_count = s->prefetch_count;
    g_shared_stats->prefetch_hits = s->prefetch_hits;
    uint64_t nc = s->live_compressed_pages;
    uint64_t nr = s->live_resident_pages;
    g_shared_stats->npages_compressed = nc;
    g_shared_stats->npages_resident = nr;
    g_shared_stats->vmem_size = (nc + nr) * PAGE_SZ / MB;
    g_shared_stats->pool_used = s->pool_used;               // compressed pool bytes used
    g_shared_stats->npages = s->npages;
}

static void shared_stats_cleanup(void) {
    if (g_shared_stats) { munmap(g_shared_stats, sizeof(MemXSharedStats)); g_shared_stats = NULL; }
    unlink(g_shared_stats_path);
}

static MemXZone3 *g_z = NULL;
static __thread int in_memx = 0;  // Per-thread recursion guard

static inline memx_runtime_context_t *context_from_tag(uintptr_t owner_tag) {
    memx_runtime_context_t *ctx = (memx_runtime_context_t *)owner_tag;
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC) return NULL;
    return ctx;
}

static void context_note_alloc(uintptr_t owner_tag, size_t size) {
    memx_runtime_context_t *ctx = context_from_tag(owner_tag);
    if (!ctx) return;
    uint64_t in_use = __sync_add_and_fetch(&ctx->bytes_in_use, size);
    __sync_add_and_fetch(&ctx->allocations_live, 1);
    __sync_add_and_fetch(&ctx->allocations_total, 1);
    while (1) {
        uint64_t peak = ctx->peak_bytes_in_use;
        if (in_use <= peak) break;
        if (__sync_bool_compare_and_swap(&ctx->peak_bytes_in_use, peak, in_use)) break;
    }
}

static void context_note_free(uintptr_t owner_tag, size_t size) {
    memx_runtime_context_t *ctx = context_from_tag(owner_tag);
    if (!ctx) return;
    __sync_fetch_and_sub(&ctx->bytes_in_use, size);
    __sync_fetch_and_sub(&ctx->allocations_live, 1);
}

static void context_note_tensor_alloc(uintptr_t owner_tag, size_t size, const memx_runtime_tensor_desc_t *desc) {
    memx_runtime_context_t *ctx = context_from_tag(owner_tag);
    if (!ctx || !desc) return;
    __sync_fetch_and_add(&ctx->tensor_bytes_in_use, size);
    __sync_fetch_and_add(&ctx->tensor_allocations_live, 1);
    if (desc->role == MEMX_TENSOR_ROLE_WEIGHT) {
        __sync_fetch_and_add(&ctx->weight_bytes_in_use, size);
    } else if (desc->role == MEMX_TENSOR_ROLE_KV_CACHE) {
        __sync_fetch_and_add(&ctx->kv_cache_bytes_in_use, size);
    }
    if (desc->flags & MEMX_TENSOR_FLAG_HOT) __sync_fetch_and_add(&ctx->hot_bytes_in_use, size);
    if (desc->flags & MEMX_TENSOR_FLAG_NO_COMPRESS) __sync_fetch_and_add(&ctx->no_compress_bytes_in_use, size);
}

static void context_note_tensor_free(uintptr_t owner_tag, size_t size, uint32_t tensor_role, uint64_t hot_bytes, uint64_t no_compress_bytes) {
    memx_runtime_context_t *ctx = context_from_tag(owner_tag);
    if (!ctx || tensor_role == MEMX_TENSOR_ROLE_UNKNOWN) return;
    __sync_fetch_and_sub(&ctx->tensor_bytes_in_use, size);
    __sync_fetch_and_sub(&ctx->tensor_allocations_live, 1);
    if (tensor_role == MEMX_TENSOR_ROLE_WEIGHT) {
        __sync_fetch_and_sub(&ctx->weight_bytes_in_use, size);
    } else if (tensor_role == MEMX_TENSOR_ROLE_KV_CACHE) {
        __sync_fetch_and_sub(&ctx->kv_cache_bytes_in_use, size);
    }
    if (hot_bytes) __sync_fetch_and_sub(&ctx->hot_bytes_in_use, hot_bytes);
    if (no_compress_bytes) __sync_fetch_and_sub(&ctx->no_compress_bytes_in_use, no_compress_bytes);
}

static void context_adjust_tensor_flag_bytes(uintptr_t owner_tag, uint64_t hot_delta_sub, uint64_t hot_delta_add,
                                             uint64_t no_compress_delta_sub, uint64_t no_compress_delta_add) {
    memx_runtime_context_t *ctx = context_from_tag(owner_tag);
    if (!ctx) return;
    if (hot_delta_sub) __sync_fetch_and_sub(&ctx->hot_bytes_in_use, hot_delta_sub);
    if (hot_delta_add) __sync_fetch_and_add(&ctx->hot_bytes_in_use, hot_delta_add);
    if (no_compress_delta_sub) __sync_fetch_and_sub(&ctx->no_compress_bytes_in_use, no_compress_delta_sub);
    if (no_compress_delta_add) __sync_fetch_and_add(&ctx->no_compress_bytes_in_use, no_compress_delta_add);
}

static size_t allocation_page_bytes(size_t allocation_size, size_t rel_page) {
    size_t start = rel_page * PAGE_SZ;
    if (start >= allocation_size) return 0;
    size_t remaining = allocation_size - start;
    return remaining < PAGE_SZ ? remaining : PAGE_SZ;
}

static uint64_t count_free_pages_locked(MemXZone3 *s) {
    return s ? s->free_pages_count : 0;
}

static void pool_release_physical_range(MemXZone3 *s, uint64_t off, uint64_t sz) {
    if (!s || !s->pool || sz == 0) return;
    if (off >= s->pool_size) return;
    if (off + sz > s->pool_size) sz = s->pool_size - off;
    uint64_t start = (off + (PAGE_SZ - 1)) & ~((uint64_t)PAGE_SZ - 1);
    uint64_t end = (off + sz) & ~((uint64_t)PAGE_SZ - 1);
    if (end <= start) return;
    uint8_t *pa = s->pool + start;
    size_t bytes = (size_t)(end - start);
#if defined(MADV_FREE_REUSABLE)
    madvise(pa, bytes, MADV_FREE_REUSABLE);
#endif
#if defined(MADV_DONTNEED)
    madvise(pa, bytes, MADV_DONTNEED);
#elif defined(MADV_FREE)
    madvise(pa, bytes, MADV_FREE);
#endif
}

static void pool_prepare_write_range(MemXZone3 *s, uint64_t off, uint64_t sz) {
    if (!s || !s->pool || sz == 0) return;
    if (off >= s->pool_size) return;
    if (off + sz > s->pool_size) sz = s->pool_size - off;
    uint64_t start = off & ~((uint64_t)PAGE_SZ - 1);
    uint64_t end = (off + sz + PAGE_SZ - 1) & ~((uint64_t)PAGE_SZ - 1);
    if (end <= start) return;
    if (end > s->pool_size) end = s->pool_size;
    uint8_t *pa = s->pool + start;
    size_t bytes = (size_t)(end - start);
    mprotect(pa, bytes, PROT_READ | PROT_WRITE);
#if defined(MADV_FREE_REUSE)
    madvise(pa, bytes, MADV_FREE_REUSE);
#endif
}

static void pool_release_all_free_extents_locked(MemXZone3 *s) {
    if (!s) return;
    for (uint32_t i = 0; i < s->pool_free_count; i++) {
        pool_release_physical_range(s, s->pool_free_off[i], s->pool_free_sz[i]);
    }
    if (s->pool_next < s->pool_size) {
        pool_release_physical_range(s, s->pool_next, s->pool_size - s->pool_next);
    }
}

static void pool_trim_tail_locked(MemXZone3 *s) {
    while (s && s->pool_free_count > 0) {
        uint32_t idx = s->pool_free_count - 1;
        uint64_t off = s->pool_free_off[idx];
        uint32_t sz = s->pool_free_sz[idx];
        if (off + sz != s->pool_next) break;
        uint64_t old_next = s->pool_next;
        s->pool_next = off;
        s->pool_free_count--;
        if (old_next > off) pool_release_physical_range(s, off, old_next - off);
    }
}

static int pool_free_insert_locked(MemXZone3 *s, uint64_t off, uint32_t sz) {
    if (!s || sz == 0) return -1;
    if (s->pool_free_count >= s->pool_free_cap) return -1;
    uint32_t pos = 0;
    while (pos < s->pool_free_count && s->pool_free_off[pos] < off) pos++;
    if (pos > 0) {
        uint64_t prev_off = s->pool_free_off[pos - 1];
        uint32_t prev_sz = s->pool_free_sz[pos - 1];
        if (prev_off + prev_sz == off) {
            off = prev_off;
            sz += prev_sz;
            pos--;
            memmove(&s->pool_free_off[pos], &s->pool_free_off[pos + 1],
                    (s->pool_free_count - (pos + 1)) * sizeof(*s->pool_free_off));
            memmove(&s->pool_free_sz[pos], &s->pool_free_sz[pos + 1],
                    (s->pool_free_count - (pos + 1)) * sizeof(*s->pool_free_sz));
            s->pool_free_count--;
        }
    }
    if (pos < s->pool_free_count) {
        uint64_t next_off = s->pool_free_off[pos];
        uint32_t next_sz = s->pool_free_sz[pos];
        if (off + sz == next_off) {
            sz += next_sz;
            memmove(&s->pool_free_off[pos], &s->pool_free_off[pos + 1],
                    (s->pool_free_count - (pos + 1)) * sizeof(*s->pool_free_off));
            memmove(&s->pool_free_sz[pos], &s->pool_free_sz[pos + 1],
                    (s->pool_free_count - (pos + 1)) * sizeof(*s->pool_free_sz));
            s->pool_free_count--;
        }
    }
    if (pos < s->pool_free_count) {
        memmove(&s->pool_free_off[pos + 1], &s->pool_free_off[pos],
                (s->pool_free_count - pos) * sizeof(*s->pool_free_off));
        memmove(&s->pool_free_sz[pos + 1], &s->pool_free_sz[pos],
                (s->pool_free_count - pos) * sizeof(*s->pool_free_sz));
    }
    s->pool_free_off[pos] = off;
    s->pool_free_sz[pos] = sz;
    s->pool_free_count++;
    pool_trim_tail_locked(s);
    return 0;
}

static int pool_alloc_extent_locked(MemXZone3 *s, uint32_t sz, uint64_t *out_off) {
    if (!s || !out_off || sz == 0) return -1;
    for (uint32_t i = 0; i < s->pool_free_count; i++) {
        if (s->pool_free_sz[i] < sz) continue;
        uint64_t off = s->pool_free_off[i];
        if (s->pool_free_sz[i] == sz) {
            memmove(&s->pool_free_off[i], &s->pool_free_off[i + 1],
                    (s->pool_free_count - (i + 1)) * sizeof(*s->pool_free_off));
            memmove(&s->pool_free_sz[i], &s->pool_free_sz[i + 1],
                    (s->pool_free_count - (i + 1)) * sizeof(*s->pool_free_sz));
            s->pool_free_count--;
        } else {
            s->pool_free_off[i] += sz;
            s->pool_free_sz[i] -= sz;
        }
        *out_off = off;
        return 0;
    }
    if (s->pool_next + sz > s->pool_size) return -1;
    *out_off = s->pool_next;
    s->pool_next += sz;
    return 0;
}

static uint64_t pool_free_extent_bytes_locked(MemXZone3 *s) {
    if (!s) return 0;
    uint64_t total = 0;
    for (uint32_t i = 0; i < s->pool_free_count; i++) total += s->pool_free_sz[i];
    return total;
}

static uint64_t pool_largest_free_extent_locked(MemXZone3 *s) {
    if (!s) return 0;
    uint64_t largest = 0;
    for (uint32_t i = 0; i < s->pool_free_count; i++) {
        if (s->pool_free_sz[i] > largest) largest = s->pool_free_sz[i];
    }
    return largest;
}

static uint32_t memx_pool_pressure_percent_locked(MemXZone3 *s) {
    if (!s || s->pool_size == 0) return 0;
    uint64_t free_extent_bytes = pool_free_extent_bytes_locked(s);
    uint64_t packed = s->pool_used;
    uint64_t live = s->pool_next > free_extent_bytes ? s->pool_next - free_extent_bytes : 0;
    uint64_t used = packed > live ? packed : live;
    if (used > s->pool_size) used = s->pool_size;
    return (uint32_t)((used * 100ULL) / s->pool_size);
}

static uint32_t memx_pool_near_full_locked(MemXZone3 *s) {
    if (!s || s->pool_size == 0) return 0;
    uint32_t occupancy = memx_pool_pressure_percent_locked(s);
    if (occupancy >= 95) return 1;
    uint64_t headroom = s->pool_size > s->pool_next ? (s->pool_size - s->pool_next) : 0;
    if (headroom * 20ULL <= s->pool_size) return 1;
    return 0;
}

static uint64_t pool_compact_locked(MemXZone3 *s) {
    if (!s || !s->pool || s->pool_next == 0) return 0;

    typedef struct { uint64_t off; uint32_t sz; } pool_live_t;
    size_t cap = 4096;
    size_t nlive = 0;
    pool_live_t *live = (pool_live_t *)malloc(sizeof(pool_live_t) * cap);
    if (!live) return 0;

    for (size_t p = 0; p < s->npages; p++) {
        PageMeta *m = &s->meta[p];
        if (m->comp_size == 0) continue;
        if (m->state != PAGE_COMPRESSED && m->state != PAGE_HOT && m->state != PAGE_COMPRESSING) continue;
        uint64_t off = m->pool_offset;
        uint32_t sz = m->comp_size;
        if (sz == 0 || off + (uint64_t)sz > s->pool_size) continue;
        int exists = 0;
        for (size_t i = 0; i < nlive; i++) {
            if (live[i].off == off && live[i].sz == sz) { exists = 1; break; }
        }
        if (exists) continue;
        if (nlive >= cap) {
            size_t ncap = cap * 2;
            pool_live_t *nl = (pool_live_t *)realloc(live, sizeof(pool_live_t) * ncap);
            if (!nl) { free(live); return 0; }
            live = nl;
            cap = ncap;
        }
        live[nlive].off = off;
        live[nlive].sz = sz;
        nlive++;
    }

    if (nlive == 0) {
        free(live);
        uint64_t old_next = s->pool_next;
        s->pool_next = 0;
        s->pool_free_count = 0;
        s->pool_used = 0;
        if (old_next) pool_release_physical_range(s, 0, old_next < s->pool_size ? old_next : s->pool_size);
        return old_next;
    }

    for (size_t i = 1; i < nlive; i++) {
        pool_live_t key = live[i];
        size_t j = i;
        while (j > 0 && live[j - 1].off > key.off) {
            live[j] = live[j - 1];
            j--;
        }
        live[j] = key;
    }

    uint64_t *old_off = (uint64_t *)malloc(sizeof(uint64_t) * nlive);
    uint64_t *new_off = (uint64_t *)malloc(sizeof(uint64_t) * nlive);
    if (!old_off || !new_off) {
        free(live);
        free(old_off);
        free(new_off);
        return 0;
    }

    uint64_t cursor = 0;
    uint64_t moved = 0;
    for (size_t i = 0; i < nlive; i++) {
        uint64_t src = live[i].off;
        uint32_t sz = live[i].sz;
        old_off[i] = src;
        new_off[i] = cursor;
        if (src != cursor) {
            if (cursor > src) { free(live); free(old_off); free(new_off); return 0; }
            pool_prepare_write_range(s, cursor, sz);
            pool_prepare_write_range(s, src, sz);
            memmove(s->pool + cursor, s->pool + src, sz);
            {
                uint64_t st = cursor & ~((uint64_t)PAGE_SZ - 1);
                uint64_t en = (cursor + sz + PAGE_SZ - 1) & ~((uint64_t)PAGE_SZ - 1);
                if (en > s->pool_size) en = s->pool_size;
                if (en > st) mprotect(s->pool + st, (size_t)(en - st), PROT_READ);
            }
            moved += sz;
        }
        cursor += sz;
    }

    for (uint32_t i = 0; i < DEDUP_HT_SIZE; i++) {
        if (s->dedup_ref[i] == 0 || s->dedup_sz[i] == 0) continue;
        uint64_t off = s->dedup_off[i];
        uint32_t sz = s->dedup_sz[i];
        for (size_t j = 0; j < nlive; j++) {
            if (old_off[j] == off && live[j].sz == sz) {
                if (s->dedup_rev && s->dedup_rev_size) {
                    uint32_t old_pp = (uint32_t)(off / PAGE_SZ) & s->dedup_rev_mask;
                    if (s->dedup_rev[old_pp] == i) s->dedup_rev[old_pp] = 0xFFFFFFFFU;
                    uint32_t new_pp = (uint32_t)(new_off[j] / PAGE_SZ) & s->dedup_rev_mask;
                    s->dedup_rev[new_pp] = i;
                }
                s->dedup_off[i] = new_off[j];
                break;
            }
        }
    }

    for (size_t p = 0; p < s->npages; p++) {
        PageMeta *m = &s->meta[p];
        if (m->comp_size == 0) continue;
        if (m->state != PAGE_COMPRESSED && m->state != PAGE_HOT && m->state != PAGE_COMPRESSING) continue;
        uint64_t off = m->pool_offset;
        uint32_t sz = m->comp_size;
        for (size_t j = 0; j < nlive; j++) {
            if (old_off[j] == off && live[j].sz == sz) {
                m->pool_offset = new_off[j];
                break;
            }
        }
    }

    uint64_t old_next = s->pool_next;
    s->pool_next = cursor;
    s->pool_free_count = 0;
    s->pool_used = cursor;
    if (old_next > cursor) pool_release_physical_range(s, cursor, old_next - cursor);
    if (s->pool_next < s->pool_size) pool_release_physical_range(s, s->pool_next, s->pool_size - s->pool_next);

    free(live);
    free(old_off);
    free(new_off);
    return moved ? moved : (old_next > cursor ? (old_next - cursor) : 0);
}

static uint64_t memx_runtime_reclaim_locked(MemXZone3 *s) {
    if (!s) return 0;
    uint64_t reclaimed = pool_reclaim_pending_locked(s);
    reclaimed += pool_reclaim_pending_locked(s);
    pool_trim_tail_locked(s);
    pool_release_all_free_extents_locked(s);
    return reclaimed;
}

static uint64_t memx_runtime_reclaim_and_compact_locked(MemXZone3 *s) {
    if (!s) return 0;
    uint64_t reclaimed = memx_runtime_reclaim_locked(s);
    uint64_t free_bytes = pool_free_extent_bytes_locked(s);
    if (free_bytes >= (PAGE_SZ * 16) || s->pool_free_count >= 4) {
        uint64_t moved = pool_compact_locked(s);
        if (moved) reclaimed += moved;
        reclaimed += memx_runtime_reclaim_locked(s);
    }
    return reclaimed;
}



static int context_preflight_locked(uintptr_t owner_tag, size_t size, size_t npages, int enforce_pressure,
                                    size_t quota_credit) {
    memx_runtime_context_t *ctx = context_from_tag(owner_tag);
    if (ctx && ctx->quota_bytes > 0) {
        uint64_t in_use = ctx->bytes_in_use;
        if (quota_credit > in_use) quota_credit = (size_t)in_use;
        uint64_t effective_in_use = in_use - quota_credit;
        uint64_t quota = ctx->quota_bytes;
        if (size > quota || effective_in_use > quota - size) {
            __sync_fetch_and_add(&ctx->allocation_failures_quota, 1);
            errno = ENOMEM;
            return -1;
        }
    }
    if (!g_z) {
        errno = ENOMEM;
        return -1;
    }
    uint64_t free_pages = count_free_pages_locked(g_z);
    if (npages > free_pages) {
        if (ctx) __sync_fetch_and_add(&ctx->pressure_events, 1);
        errno = ENOMEM;
        return -1;
    }
    if (enforce_pressure) {
        uint32_t pressure = memx_pool_pressure_percent_locked(g_z);
        if (pressure >= 95) {
            memx_runtime_reclaim_locked(g_z);
            pressure = memx_pool_pressure_percent_locked(g_z);
        }
        if (pressure >= 95) {
            if (ctx) __sync_fetch_and_add(&ctx->pressure_events, 1);
            errno = ENOMEM;
            return -1;
        }
    }
    return 0;
}

// ─── Active page list helpers ───
static inline void hot_list_add(MemXZone3 *s, uint32_t page) {
    for (;;) {
        uint32_t n = s->hot_count;
        if (n >= s->hot_cap) return;
        if (__sync_bool_compare_and_swap(&s->hot_count, n, n + 1)) {
            s->hot_list[n] = page;
            return;
        }
    }
}
static inline void res_list_add(MemXZone3 *s, uint32_t page) {
    for (;;) {
        uint32_t n = s->res_count;
        if (n >= s->res_cap) return;
        if (__sync_bool_compare_and_swap(&s->res_count, n, n + 1)) {
            s->res_list[n] = page;
            return;
        }
    }
}

static inline int page_wants_write_protect(const PageMeta *m) {
    if (!m) return 0;
    if (m->tensor_flags & MEMX_TENSOR_FLAG_NO_COMPRESS) return 0;
    if (m->tensor_flags & (MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_SEQUENTIAL))
        return 1;
    if (m->tensor_role == MEMX_TENSOR_ROLE_WEIGHT ||
        m->tensor_role == MEMX_TENSOR_ROLE_EMBEDDING ||
        m->tensor_role == MEMX_TENSOR_ROLE_KV_CACHE)
        return 1;
    return 0;
}

static inline int fault_stream_for_role(uint16_t role) {
    if (role == MEMX_TENSOR_ROLE_KV_CACHE) return 0;
    if (role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING) return 1;
    return 2;
}

static inline int page_stable_need(const PageMeta *m) {
    if (!m) return 1;
    if (m->tensor_flags & MEMX_TENSOR_FLAG_HOT) return 0;
    if (m->tensor_flags & MEMX_TENSOR_FLAG_NO_COMPRESS) return 0;
    if (m->tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL) return 4;
    if (m->tensor_role == MEMX_TENSOR_ROLE_KV_CACHE) {
        if (m->tensor_flags & MEMX_TENSOR_FLAG_COLD) return 1;
        return 1;
    }
    if (m->tensor_role == MEMX_TENSOR_ROLE_WEIGHT || m->tensor_role == MEMX_TENSOR_ROLE_EMBEDDING) {
        if (m->tensor_flags & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY)) return 0;
        return 1;
    }
    if (m->tensor_role == MEMX_TENSOR_ROLE_ACTIVATION) return 1;
    return 2;
}


static inline void page_release_physical(MemXZone3 *s, size_t page) {
    if (!s) return;
    uint8_t *pa = (uint8_t*)s->vmem + page * PAGE_SZ;
#if defined(MADV_FREE_REUSABLE)
    madvise(pa, PAGE_SZ, MADV_FREE_REUSABLE);
#endif
#if defined(MADV_DONTNEED)
    madvise(pa, PAGE_SZ, MADV_DONTNEED);
#elif defined(MADV_FREE)
    madvise(pa, PAGE_SZ, MADV_FREE);
#endif
}

static inline void page_release_physical_range(MemXZone3 *s, size_t first, size_t last_inclusive) {
    if (!s || last_inclusive < first || first >= s->npages) return;
    if (last_inclusive >= s->npages) last_inclusive = s->npages - 1;
    size_t n = last_inclusive - first + 1;
    uint8_t *pa = (uint8_t*)s->vmem + first * PAGE_SZ;
    size_t bytes = n * PAGE_SZ;
#if defined(MADV_FREE_REUSABLE)
    madvise(pa, bytes, MADV_FREE_REUSABLE);
#endif
#if defined(MADV_DONTNEED)
    madvise(pa, bytes, MADV_DONTNEED);
#elif defined(MADV_FREE)
    madvise(pa, bytes, MADV_FREE);
#endif
}

static inline void restore_compressing_page(MemXZone3 *s, size_t page) {
    uint8_t old = __sync_val_compare_and_swap(&s->meta[page].state, PAGE_COMPRESSING, PAGE_RESIDENT);
    if (old == PAGE_COMPRESSING) {
        int prot = PROT_READ | PROT_WRITE;
        if (page_wants_write_protect(&s->meta[page])) prot = PROT_READ;
        mprotect((uint8_t*)s->vmem + page * PAGE_SZ, PAGE_SZ, prot);
        res_list_add(s, (uint32_t)page);
    }
}

// ─── Free bitmap helpers ───
static inline void bm_set_free(MemXZone3 *s, size_t page) {
    uint64_t mask = (1ULL << (page % 64));
    uint64_t *word = &s->free_bm[page / 64];
    if (((*word) & mask) == 0) {
        *word |= mask;
        __sync_fetch_and_add(&s->free_pages_count, 1);
    }
}
static inline void bm_set_used(MemXZone3 *s, size_t page) {
    uint64_t mask = (1ULL << (page % 64));
    uint64_t *word = &s->free_bm[page / 64];
    if (((*word) & mask) != 0) {
        *word &= ~mask;
        __sync_fetch_and_sub(&s->free_pages_count, 1);
    }
}
static inline int bm_is_free(MemXZone3 *s, size_t page) {
    return (s->free_bm[page / 64] >> (page % 64)) & 1;
}
// Find N consecutive free pages starting from hint. Returns page index or -1.
// Uses __builtin_ctzll for fast first-free-bit finding within each word.
static inline ssize_t bm_find_free_run(MemXZone3 *s, size_t npages, size_t hint) {
    size_t total = s->npages;
    for (size_t start = hint; start < total; ) {
        uint64_t word = s->free_bm[start / 64];
        if (word == 0) { start = (start / 64 + 1) * 64; continue; }
        // Use ctzll to find first set bit from (start % 64) onwards
        size_t base = start & ~63ULL;
        size_t bit_off = start - base;
        uint64_t masked = word >> bit_off;  // Shift out bits before start
        if (masked == 0) { start = base + 64; continue; }
        int bit = __builtin_ctzll(masked) + (int)bit_off;
        size_t found = base + bit;
        if (found >= total) return -1;
        // Check consecutive free pages (same as original)
        size_t cont = 0;
        for (size_t j = found; j < total && cont < npages; j++) {
            if (bm_is_free(s, j)) cont++;
            else break;
        }
        if (cont >= npages) return (ssize_t)found;
        start = found + cont + 1;
    }
    return -1;
}


// ─── FNV-1a hash (word-at-a-time for compressed data) ───
static uint64_t fnv1a_word(const uint8_t *data, uint32_t len) {
    uint64_t h = 14695981039346656037ULL;
    uint32_t i = 0;
    // Process 8 bytes at a time
    for (; i + 8 <= len; i += 8) {
        uint64_t w;
        memcpy(&w, data + i, 8);  // safe unaligned read
        h ^= w;
        h *= 1099511628211ULL;
    }
    // Remaining bytes
    for (; i < len; i++) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

#define MEMX_CODEC_TENSOR_FP16_SPLIT MEMX_RUNTIME_CODEC_TENSOR_FP16_SPLIT
#define MEMX_CODEC_TENSOR_BITPLANE16 MEMX_RUNTIME_CODEC_TENSOR_BITPLANE16
#define MEMX_CODEC_TENSOR_SPARSE_BYTE MEMX_RUNTIME_CODEC_TENSOR_SPARSE_BYTE
#define MEMX_CODEC_TENSOR_FP16_DELTA_SPLIT MEMX_RUNTIME_CODEC_TENSOR_FP16_DELTA_SPLIT
#define MEMX_CODEC_ZLIB MEMX_RUNTIME_CODEC_ZLIB
#define MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT MEMX_RUNTIME_CODEC_TENSOR_FP16_ZLIB_SPLIT
#define MEMX_CODEC_TENSOR_EXP_PACK MEMX_RUNTIME_CODEC_TENSOR_EXP_PACK

static uint32_t rle8_encode(const uint8_t *src, uint32_t len, uint8_t *dst, uint32_t cap) {
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

static int tensor_fp16_split_eligible(MemXZone3 *s, size_t page_index) {
    if (!s) return 0;
    uint16_t dtype = s->meta[page_index].tensor_dtype;
    uint16_t role = s->meta[page_index].tensor_role;
    if (dtype != MEMX_TENSOR_DTYPE_FP16 && dtype != MEMX_TENSOR_DTYPE_BF16) return 0;
    if (role != MEMX_TENSOR_ROLE_WEIGHT &&
        role != MEMX_TENSOR_ROLE_KV_CACHE &&
        role != MEMX_TENSOR_ROLE_ACTIVATION &&
        role != MEMX_TENSOR_ROLE_EMBEDDING) return 0;
    return 1;
}

static int tensor_sparse_byte_eligible(MemXZone3 *s, size_t page_index) {
    if (!s) return 0;
    uint16_t dtype = s->meta[page_index].tensor_dtype;
    uint16_t role = s->meta[page_index].tensor_role;
    if (role != MEMX_TENSOR_ROLE_WEIGHT &&
        role != MEMX_TENSOR_ROLE_KV_CACHE &&
        role != MEMX_TENSOR_ROLE_ACTIVATION &&
        role != MEMX_TENSOR_ROLE_EMBEDDING &&
        role != MEMX_TENSOR_ROLE_TEMPORARY) return 0;
    if (dtype != MEMX_TENSOR_DTYPE_FP16 &&
        dtype != MEMX_TENSOR_DTYPE_BF16 &&
        dtype != MEMX_TENSOR_DTYPE_FP32 &&
        dtype != MEMX_TENSOR_DTYPE_INT8 &&
        dtype != MEMX_TENSOR_DTYPE_UINT8 &&
        dtype != MEMX_TENSOR_DTYPE_INT32) return 0;
    return 1;
}

static int page_is_all_zero(const uint8_t *src) {
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

static int page_bytes_equal(const uint8_t *a, const uint8_t *b) {
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

static inline int page_compress_content_ok(MemXZone3 *s, size_t pidx, uint32_t seq0, const uint8_t *snap) {
    PageMeta *m = &s->meta[pidx];
    if (m->state != PAGE_COMPRESSING || m->dirty || m->write_seq != seq0) return 0;
    return page_bytes_equal(snap, (const uint8_t *)s->vmem + pidx * PAGE_SZ);
}

static inline int commit_compressed_page(MemXZone3 *s, size_t pidx, uint32_t seq0, const uint8_t *snap) {
    PageMeta *m = &s->meta[pidx];
    uint8_t *pa = (uint8_t *)s->vmem + pidx * PAGE_SZ;
    if (!snap) return 0;
    if (m->state != PAGE_COMPRESSING || m->dirty || m->write_seq != seq0) return 0;
    mprotect(pa, PAGE_SZ, PROT_READ);
    __sync_synchronize();
    if (m->state != PAGE_COMPRESSING || m->dirty || m->write_seq != seq0) {
        mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
        return 0;
    }
    if (!page_bytes_equal(snap, pa)) {
        mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
        return 0;
    }
    mprotect(pa, PAGE_SZ, PROT_NONE);
    __sync_synchronize();
    if (m->dirty || m->write_seq != seq0 || m->state != PAGE_COMPRESSING) {
        mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
        return 0;
    }
    uint8_t cas = __sync_val_compare_and_swap(&m->state, PAGE_COMPRESSING, PAGE_COMPRESSED);
    if (cas != PAGE_COMPRESSING) {
        if (cas == PAGE_HOT || cas == PAGE_RESIDENT)
            mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
        return 0;
    }
    return 1;
}


static uint32_t tensor_fp16_split_compress(const uint8_t *src, uint8_t *dst, uint32_t cap) {
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

static uint32_t tensor_fp16_delta_split_compress(const uint8_t *src, uint8_t *dst, uint32_t cap) {
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

static uint32_t tensor_bitplane16_compress(const uint8_t *src, uint8_t *dst, uint32_t cap) {
    if (!src || !dst || cap < 34) return 0;
    uint8_t planes[16][PAGE_SZ / 16];
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

static uint32_t tensor_sparse_byte_compress(const uint8_t *src, uint8_t *dst, uint32_t cap) {
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

static __thread z_stream g_deflate_zs;
static __thread int g_deflate_ready = 0;
static __thread z_stream g_inflate_zs;
static __thread int g_inflate_ready = 0;

static int memx_deflate_once(const uint8_t *src, uLong src_len, uint8_t *dst, uLongf *out_len) {
    if (!g_deflate_ready) {
        memset(&g_deflate_zs, 0, sizeof(g_deflate_zs));
        if (deflateInit(&g_deflate_zs, 1) != Z_OK) return Z_MEM_ERROR;
        g_deflate_ready = 1;
    }
    if (deflateReset(&g_deflate_zs) != Z_OK) return Z_STREAM_ERROR;
    g_deflate_zs.next_in = (Bytef *)src;
    g_deflate_zs.avail_in = (uInt)src_len;
    g_deflate_zs.next_out = (Bytef *)dst;
    g_deflate_zs.avail_out = (uInt)(*out_len);
    int rc = deflate(&g_deflate_zs, Z_FINISH);
    if (rc != Z_STREAM_END) return rc == Z_OK ? Z_BUF_ERROR : rc;
    *out_len = g_deflate_zs.total_out;
    return Z_OK;
}

static int memx_inflate_once(const uint8_t *src, uLong src_len, uint8_t *dst, uLongf *out_len) {
    if (!g_inflate_ready) {
        memset(&g_inflate_zs, 0, sizeof(g_inflate_zs));
        if (inflateInit(&g_inflate_zs) != Z_OK) return Z_MEM_ERROR;
        g_inflate_ready = 1;
    }
    if (inflateReset(&g_inflate_zs) != Z_OK) return Z_STREAM_ERROR;
    g_inflate_zs.next_in = (Bytef *)src;
    g_inflate_zs.avail_in = (uInt)src_len;
    g_inflate_zs.next_out = (Bytef *)dst;
    g_inflate_zs.avail_out = (uInt)(*out_len);
    int rc = inflate(&g_inflate_zs, Z_FINISH);
    if (rc != Z_STREAM_END) return rc == Z_OK ? Z_BUF_ERROR : rc;
    *out_len = g_inflate_zs.total_out;
    return Z_OK;
}

static uint32_t zlib_page_compress(const uint8_t *src, uint8_t *dst, uint32_t cap) {
    if (!src || !dst || cap < 16) return 0;
    uLongf bound = compressBound(PAGE_SZ);
    if (bound + 8 > cap) {
        if (cap <= 8) return 0;
        bound = cap - 8;
    }
    uLongf out_len = bound;
    int rc = memx_deflate_once(src, PAGE_SZ, dst + 8, &out_len);
    if (rc != Z_OK || out_len == 0) return 0;
    uint32_t total = (uint32_t)(out_len + 8);
    if (total >= PAGE_SZ || total >= (PAGE_SZ * 15) / 16) return 0;
    dst[0] = 0x4D;
    dst[1] = 0x58;
    dst[2] = MEMX_CODEC_ZLIB;
    dst[3] = 1;
    dst[4] = (uint8_t)(out_len & 0xFF);
    dst[5] = (uint8_t)((out_len >> 8) & 0xFF);
    dst[6] = (uint8_t)((out_len >> 16) & 0xFF);
    dst[7] = (uint8_t)((out_len >> 24) & 0xFF);
    return total;
}

static uint32_t tensor_fp16_zlib_split_compress(const uint8_t *src, uint8_t *dst, uint32_t cap) {
    if (!src || !dst || cap < 24) return 0;
    const uint32_t half_count = PAGE_SZ / 2;
    uint8_t lo_tmp[PAGE_SZ / 2];
    uint8_t hi_tmp[PAGE_SZ / 2];
#if MEMX_HAS_NEON
    {
        uint32_t i = 0;
        for (; i + 16 <= half_count; i += 16) {
            uint8x16x2_t z = vld2q_u8(src + i * 2);
            vst1q_u8(lo_tmp + i, z.val[0]);
            vst1q_u8(hi_tmp + i, z.val[1]);
        }
        for (; i < half_count; i++) {
            lo_tmp[i] = src[i * 2];
            hi_tmp[i] = src[i * 2 + 1];
        }
    }
#else
    for (uint32_t i = 0; i < half_count; i++) {
        lo_tmp[i] = src[i * 2];
        hi_tmp[i] = src[i * 2 + 1];
    }
#endif
    if (cap < 16 + half_count + 32) return 0;
    uLongf bound = compressBound(half_count);
    if (bound + 16 + half_count > cap) {
        if (cap <= 16 + half_count) return 0;
        bound = cap - 16 - half_count;
    }
    uLongf hi_len = bound;
    int rc = memx_deflate_once(hi_tmp, half_count, dst + 16 + half_count, &hi_len);
    if (rc != Z_OK || hi_len == 0) return 0;
    uint32_t total = (uint32_t)(16 + half_count + hi_len);
    if (total >= PAGE_SZ || total >= (PAGE_SZ * 15) / 16 || total > cap) return 0;
    dst[0] = 0x4D;
    dst[1] = 0x58;
    dst[2] = MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT;
    dst[3] = 1;
    dst[4] = (uint8_t)(half_count & 0xFF);
    dst[5] = (uint8_t)((half_count >> 8) & 0xFF);
    dst[6] = (uint8_t)((half_count >> 16) & 0xFF);
    dst[7] = (uint8_t)((half_count >> 24) & 0xFF);
    dst[8] = (uint8_t)(hi_len & 0xFF);
    dst[9] = (uint8_t)((hi_len >> 8) & 0xFF);
    dst[10] = (uint8_t)((hi_len >> 16) & 0xFF);
    dst[11] = (uint8_t)((hi_len >> 24) & 0xFF);
    dst[12] = 0;
    dst[13] = 0;
    dst[14] = 0;
    dst[15] = 0;
    memcpy(dst + 16, lo_tmp, half_count);
    return total;
}



static uint32_t tensor_exp_pack_compress(const uint8_t *src, uint8_t *dst, uint32_t cap, int is_bf16) {
    if (!src || !dst || cap < 40) return 0;
    const uint32_t half_count = PAGE_SZ / 2;
    const uint32_t sign_bytes = (half_count + 7u) / 8u;
    uint8_t sign_raw[PAGE_SZ / 16];
    uint8_t exp_raw[PAGE_SZ / 2];
    uint8_t mant_raw[PAGE_SZ];
    uint32_t mant_len;
    if (is_bf16) {
        mant_len = (half_count * 7u + 7u) / 8u;
    } else {
        mant_len = (half_count * 10u + 7u) / 8u;
    }
    if (sign_bytes > sizeof(sign_raw) || mant_len > sizeof(mant_raw)) return 0;
    memset(sign_raw, 0, sign_bytes);
    memset(mant_raw, 0, mant_len);
    if (is_bf16) {
        uint32_t bit_pos = 0;
        for (uint32_t i = 0; i < half_count; i++) {
            uint16_t h = (uint16_t)src[i * 2] | ((uint16_t)src[i * 2 + 1] << 8);
            if (h & 0x8000u) sign_raw[i >> 3] |= (uint8_t)(1u << (i & 7u));
            exp_raw[i] = (uint8_t)((h >> 7) & 0xFFu);
            uint32_t m = (uint32_t)(h & 0x7Fu);
            for (int b = 0; b < 7; b++) {
                if (m & (1u << b))
                    mant_raw[bit_pos >> 3] |= (uint8_t)(1u << (bit_pos & 7u));
                bit_pos++;
            }
        }
    } else {
        uint32_t bit_pos = 0;
        for (uint32_t i = 0; i < half_count; i++) {
            uint16_t h = (uint16_t)src[i * 2] | ((uint16_t)src[i * 2 + 1] << 8);
            if (h & 0x8000u) sign_raw[i >> 3] |= (uint8_t)(1u << (i & 7u));
            exp_raw[i] = (uint8_t)((h >> 10) & 0x1Fu);
            uint32_t m = (uint32_t)(h & 0x3FFu);
            for (int b = 0; b < 10; b++) {
                if (m & (1u << b))
                    mant_raw[bit_pos >> 3] |= (uint8_t)(1u << (bit_pos & 7u));
                bit_pos++;
            }
        }
    }
    const uint32_t hdr = 28;
    if (cap <= hdr + sign_bytes + mant_len + 32) return 0;
    uint8_t *sign_out = dst + hdr;
    uLongf sign_bound = compressBound(sign_bytes);
    uint32_t rem_for_sign = cap - hdr - mant_len - 16;
    if (rem_for_sign < 8) return 0;
    if (sign_bound > rem_for_sign) sign_bound = rem_for_sign;
    uLongf sign_len = sign_bound;
    int rc = compress2(sign_out, &sign_len, sign_raw, sign_bytes, 1);
    uint8_t sign_raw_store = 0;
    if (rc != Z_OK || sign_len == 0 || sign_len >= sign_bytes) {
        memcpy(sign_out, sign_raw, sign_bytes);
        sign_len = sign_bytes;
        sign_raw_store = 1;
    }
    uint8_t *exp_out = sign_out + sign_len;
    uint32_t rem = cap - (uint32_t)(exp_out - dst) - mant_len;
    if (rem < 16) return 0;
    uLongf exp_bound = compressBound(half_count);
    if (exp_bound > rem) exp_bound = rem;
    uLongf exp_len = exp_bound;
    rc = compress2(exp_out, &exp_len, exp_raw, half_count, 1);
    if (rc != Z_OK || exp_len == 0 || exp_len >= half_count) return 0;
    uint8_t *mant_out = exp_out + exp_len;
    uint8_t mant_raw_store = 1;
    uLongf mant_len_z = mant_len;
    uLongf mant_bound = compressBound(mant_len);
    uint32_t rem_m = cap - (uint32_t)(mant_out - dst);
    if (mant_bound > rem_m) mant_bound = rem_m;
    if (mant_bound >= 16) {
        uLongf zlen = mant_bound;
        rc = compress2(mant_out, &zlen, mant_raw, mant_len, 1);
        if (rc == Z_OK && zlen > 0 && zlen + 32 < mant_len) {
            mant_len_z = zlen;
            mant_raw_store = 0;
        }
    }
    if (mant_raw_store) {
        if (mant_len > rem_m) return 0;
        memcpy(mant_out, mant_raw, mant_len);
        mant_len_z = mant_len;
    }
    uint32_t total = (uint32_t)(hdr + sign_len + exp_len + mant_len_z);
    if (total >= PAGE_SZ || total >= (PAGE_SZ * 15) / 16 || total > cap) return 0;
    dst[0] = 0x4D;
    dst[1] = 0x58;
    dst[2] = MEMX_CODEC_TENSOR_EXP_PACK;
    dst[3] = 1;
    dst[4] = (uint8_t)(half_count & 0xFF);
    dst[5] = (uint8_t)((half_count >> 8) & 0xFF);
    dst[6] = (uint8_t)((half_count >> 16) & 0xFF);
    dst[7] = (uint8_t)((half_count >> 24) & 0xFF);
    dst[8] = (uint8_t)(sign_len & 0xFF);
    dst[9] = (uint8_t)((sign_len >> 8) & 0xFF);
    dst[10] = (uint8_t)((sign_len >> 16) & 0xFF);
    dst[11] = (uint8_t)((sign_len >> 24) & 0xFF);
    dst[12] = (uint8_t)(exp_len & 0xFF);
    dst[13] = (uint8_t)((exp_len >> 8) & 0xFF);
    dst[14] = (uint8_t)((exp_len >> 16) & 0xFF);
    dst[15] = (uint8_t)((exp_len >> 24) & 0xFF);
    dst[16] = (uint8_t)(mant_len_z & 0xFF);
    dst[17] = (uint8_t)((mant_len_z >> 8) & 0xFF);
    dst[18] = (uint8_t)((mant_len_z >> 16) & 0xFF);
    dst[19] = (uint8_t)((mant_len_z >> 24) & 0xFF);
    dst[20] = is_bf16 ? 2 : 1;
    dst[21] = (uint8_t)((sign_raw_store ? 1u : 0u) | (mant_raw_store ? 2u : 0u) | 4u);
    dst[22] = 0;
    dst[23] = 0;
    dst[24] = (uint8_t)(mant_len & 0xFF);
    dst[25] = (uint8_t)((mant_len >> 8) & 0xFF);
    dst[26] = (uint8_t)((mant_len >> 16) & 0xFF);
    dst[27] = (uint8_t)((mant_len >> 24) & 0xFF);
    return total;
}

static inline void memx_interleave_lo_hi(const uint8_t *lo, const uint8_t *hi, uint8_t *dst, uint32_t half_count) {
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

static void cpu_decompress(const uint8_t *src, uint32_t cs, uint8_t *dst) {
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
    // Decode RLE/LZ77 directly into dst (no intermediate buffer — saves 16KB stack)
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
        if(flags&1u){
            if(sign_zlen!=sign_bytes){memset(dst,0,PAGE_SZ);return;}
            memcpy(sign_raw,p,sign_bytes);
        }else{
            uLongf destLen=sign_bytes;
            if(uncompress(sign_raw,&destLen,p,sign_zlen)!=Z_OK||destLen!=sign_bytes){memset(dst,0,PAGE_SZ);return;}
        }
        p+=sign_zlen;
        {
            uLongf destLen=half_count;
            if(uncompress(exp_raw,&destLen,p,exp_zlen)!=Z_OK||destLen!=half_count){memset(dst,0,PAGE_SZ);return;}
        }
        p+=exp_zlen;
        if(flags&2u){
            if(mant_zlen!=mant_len){memset(dst,0,PAGE_SZ);return;}
            memcpy(mant_raw,p,mant_len);
        }else{
            uLongf destLen=mant_len;
            if(uncompress(mant_raw,&destLen,p,mant_zlen)!=Z_OK||destLen!=mant_len){memset(dst,0,PAGE_SZ);return;}
        }
        if(flags & 4u){
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
        uint32_t hi_zlen=(uint32_t)src[8]|((uint32_t)src[9]<<8)|((uint32_t)src[10]<<16)|((uint32_t)src[11]<<24);
        if(half_count!=PAGE_SZ/2||hi_zlen==0||16+half_count+hi_zlen>cs){memset(dst,0,PAGE_SZ);return;}
        uint8_t hi[PAGE_SZ/2];
        uLongf destLen=half_count;
        if(memx_inflate_once(src+16+half_count,hi_zlen,hi,&destLen)!=Z_OK||destLen!=half_count){memset(dst,0,PAGE_SZ);return;}
        memx_interleave_lo_hi(src+16, hi, dst, half_count);
        return;
    }
    if(ver==MEMX_CODEC_ZLIB){
        if(cs<9){memset(dst,0,PAGE_SZ);return;}
        uint32_t zlen=(uint32_t)src[4]|((uint32_t)src[5]<<8)|((uint32_t)src[6]<<16)|((uint32_t)src[7]<<24);
        if(zlen==0||8+zlen>cs){memset(dst,0,PAGE_SZ);return;}
        uLongf destLen=PAGE_SZ;
        if(memx_inflate_once(src+8,zlen,dst,&destLen)!=Z_OK||destLen!=PAGE_SZ){memset(dst,0,PAGE_SZ);}
        return;
    }
    uint32_t ip=4,op=0;
    while(ip<cs&&op<PAGE_SZ){uint8_t b=src[ip];
    if(b==0xFD&&ver>=2&&ip+3<cs){uint8_t vb=src[ip+1];uint32_t rl=(uint32_t)src[ip+2]|((uint32_t)src[ip+3]<<8);ip+=4;memset(dst+op,vb,rl<PAGE_SZ-op?rl:PAGE_SZ-op);op+=rl<PAGE_SZ-op?rl:PAGE_SZ-op;}
    else if(b==0xFF&&ip+4<cs){ip++;uint32_t o=(uint32_t)src[ip]|((uint32_t)src[ip+1]<<8);ip+=2;uint32_t m=(uint32_t)src[ip]|((uint32_t)src[ip+1]<<8);ip+=2;uint32_t s=op-o;for(uint32_t i=0;i<m&&op<PAGE_SZ;i++)dst[op++]=dst[s+i];}
    else if(b==0xFE&&ip+1<cs){ip++;dst[op++]=src[ip++];}
    else{dst[op++]=b;ip++;}}
    if(op<PAGE_SZ)memset(dst+op,0,PAGE_SZ-op);
    // Delta decode (prefix sum) — vectorized with 8-byte accumulator
    if(op>0){uint8_t acc=dst[0];for(uint32_t i=1;i<op;i++){acc+=dst[i];dst[i]=acc;}}
}

// Self-test: verify cpu_decompress works correctly for zero-page compressed data
__attribute__((constructor)) static void decomp_selftest(void) {
    uint8_t src[8] = {0x4D, 0x58, 0x03, 0x00, 0xFD, 0x00, 0x00, 0x40};
    uint8_t dst[16384];
    memset(dst, 0xCC, sizeof(dst));
    cpu_decompress(src, 8, dst);
    int ok = 1;
    for (int i = 0; i < 16384; i++) { if (dst[i] != 0) { ok = 0; break; } }
    if (!ok) write(2, "[SELFTEST] ZERO PAGE DECOMP FAILED!\n", 37);
    else write(2, "[SELFTEST] ZERO PAGE DECOMP OK\n", 31);
}

// ─── GPU compress (uses pre-allocated persistent buffers) ───
static int gpu_compress(MemXZone3 *s, size_t count) {
    size_t bytes=count*PAGE_SZ;
    memcpy([s->gpu_sb contents],s->tmp_src,bytes);
    id<MTLCommandBuffer> cb=[s->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:s->comp_pipe];
    [enc setBuffer:s->gpu_sb offset:0 atIndex:0];[enc setBuffer:s->gpu_db offset:0 atIndex:1];[enc setBuffer:s->gpu_zb offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
    memcpy(s->tmp_dst,[s->gpu_db contents],bytes);
    memcpy(s->tmp_sz,[s->gpu_zb contents],count*4);
    return 0;
}

// ─── Dedup reference management ───
static void dedup_decref(MemXZone3 *s, uint64_t pool_offset, uint32_t comp_size) {
    if (!s->dedup_rev || s->dedup_rev_size == 0) return;
    uint32_t pp = (uint32_t)(pool_offset / PAGE_SZ) & s->dedup_rev_mask;
    uint32_t slot = s->dedup_rev[pp];
    if (slot < DEDUP_HT_SIZE && s->dedup_ref[slot] > 0 && s->dedup_off[slot] == pool_offset && s->dedup_sz[slot] == comp_size) {
        uint32_t old = __sync_fetch_and_sub(&s->dedup_ref[slot], 1);
        if (old == 1 && s->dedup_pending_free && !s->dedup_pending_free[slot]) {
            s->dedup_pending_free[slot] = 1;
            __sync_fetch_and_add(&s->dedup_pending_free_count, 1);
        }
        return;
    }
    // Fallback: linear scan (for hash collisions in rev table)
    for (uint32_t i = 0; i < DEDUP_HT_SIZE; i++) {
        if (s->dedup_ref[i] > 0 && s->dedup_off[i] == pool_offset && s->dedup_sz[i] == comp_size) {
            uint32_t old = __sync_fetch_and_sub(&s->dedup_ref[i], 1);
            if (old == 1 && s->dedup_pending_free && !s->dedup_pending_free[i]) {
                s->dedup_pending_free[i] = 1;
                __sync_fetch_and_add(&s->dedup_pending_free_count, 1);
            }
            return;
        }
    }
}

static uint64_t pool_reclaim_pending_locked(MemXZone3 *s) {
    if (!s || !s->dedup_pending_free) return 0;
    if (s->dedup_pending_free_count == 0) return 0;
    uint64_t reclaimed = 0;
    uint64_t reclaimed_events = 0;
    for (uint32_t i = 0; i < DEDUP_HT_SIZE; i++) {
        if (!s->dedup_pending_free[i]) continue;
        if (s->dedup_ref[i] != 0) {
            s->dedup_pending_free[i] = 0;
            if (s->dedup_pending_free_count > 0) s->dedup_pending_free_count--;
            continue;
        }
        uint64_t off = s->dedup_off[i];
        uint32_t sz = s->dedup_sz[i];
        if (sz > 0) {
            pool_free_insert_locked(s, off, sz);
            if (s->pool_used >= sz) s->pool_used -= sz;
            else s->pool_used = 0;
            reclaimed += sz;
            reclaimed_events++;
            uint32_t pp = (uint32_t)(off / PAGE_SZ) & s->dedup_rev_mask;
            if (s->dedup_rev[pp] == i) s->dedup_rev[pp] = 0xFFFFFFFFU;
        }
        s->dedup_hash[i] = 0;
        s->dedup_off[i] = 0;
        s->dedup_sz[i] = 0;
        s->dedup_ref[i] = 0;
        s->dedup_pending_free[i] = 0;
        if (s->dedup_pending_free_count > 0) s->dedup_pending_free_count--;
    }
    if (reclaimed > 0) {
        __sync_fetch_and_add(&s->pool_reclaim_bytes_total, reclaimed);
        __sync_fetch_and_add(&s->pool_reclaim_events, reclaimed_events);
    }
    return reclaimed;
}

static void wait_decompress_complete(PageMeta *m) {
    if (!m) return;
    for (int i = 0; i < 1000000; i++) {
        uint8_t st = m->state;
        uint32_t cs = m->comp_size;
        if (st != PAGE_HOT && st != PAGE_COMPRESSED) return;
        if (st == PAGE_HOT && cs == 0) return;
        if (st == PAGE_COMPRESSED) return;
#if defined(__aarch64__)
        __asm__ __volatile__("yield");
#else
        ;
#endif
    }
}

static __thread uint8_t g_decomp_scratch[PAGE_SZ];
static __thread uint8_t g_codec_scratch[PAGE_SZ];
static __thread uint8_t g_comp_payload[PAGE_SZ];

static int decompress_compressed_page(MemXZone3 *s, size_t page_index, uint8_t prefetched, uint8_t cooldown) {
    if (!s || page_index >= s->npages) return 0;
    PageMeta *m = &s->meta[page_index];
    for (int attempt = 0; attempt < 64; attempt++) {
        uint8_t st = m->state;
        if (st == PAGE_HOT) {
            if (m->comp_size == 0) return 0;
            wait_decompress_complete(m);
            return 0;
        }
        if (st != PAGE_COMPRESSED) return 0;
        uint64_t d_off = 0;
        uint32_t d_sz = 0;
        pthread_mutex_lock(&s->alloc_mutex);
        if (m->state != PAGE_COMPRESSED) {
            pthread_mutex_unlock(&s->alloc_mutex);
            wait_decompress_complete(m);
            return 0;
        }
        uint8_t old = __sync_val_compare_and_swap(&m->state, PAGE_COMPRESSED, PAGE_HOT);
        if (old != PAGE_COMPRESSED) {
            pthread_mutex_unlock(&s->alloc_mutex);
            wait_decompress_complete(m);
            return 0;
        }
        __sync_fetch_and_sub(&s->live_compressed_pages, 1);
        __sync_fetch_and_add(&s->live_resident_pages, 1);
        d_off = m->pool_offset;
        d_sz = m->comp_size;
        if (d_sz > 0 && d_sz <= PAGE_SZ) {
            memcpy(g_comp_payload, s->pool + d_off, d_sz);
        }
        if (d_sz > 0 && d_sz <= PAGE_SZ) dedup_decref(s, d_off, d_sz);
        pthread_mutex_unlock(&s->alloc_mutex);

        uint8_t *pa = (uint8_t *)s->vmem + page_index * PAGE_SZ;
        uint8_t *tmp = g_decomp_scratch;
        if (d_sz == 0 || d_sz > PAGE_SZ) {
            memset(tmp, 0, PAGE_SZ);
        } else {
            cpu_decompress(g_comp_payload, d_sz, tmp);
        }
        mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
#if defined(MADV_FREE_REUSE)
        madvise(pa, PAGE_SZ, MADV_FREE_REUSE);
#endif
        memcpy(pa, tmp, PAGE_SZ);
        m->prefetched = prefetched;
        m->cooldown = cooldown;
        m->pool_offset = 0;
        m->codec = 0;
        __sync_synchronize();
        m->comp_size = 0;
        if ((m->tensor_flags & MEMX_TENSOR_FLAG_READ_MOSTLY) != 0)
            mprotect(pa, PAGE_SZ, PROT_READ);
        hot_list_add(s, (uint32_t)page_index);
        return 1;
    }
    return 0;
}

static int prefetch_page(MemXZone3 *s, size_t page_index, uint8_t cooldown) {
    if (!s || page_index >= s->npages) return 0;
    if (decompress_compressed_page(s, page_index, 1, cooldown)) return 1;
    PageMeta *m = &s->meta[page_index];
    uint8_t old = __sync_val_compare_and_swap(&m->state, PAGE_RESIDENT, PAGE_HOT);
    if (old == PAGE_RESIDENT) {
        m->prefetched = 1;
        m->cooldown = cooldown;
        hot_list_add(s, (uint32_t)page_index);
        return 1;
    }
    if (old == PAGE_HOT && m->cooldown < cooldown) {
        m->prefetched = 1;
        m->cooldown = cooldown;
    }
    return 0;
}

static int async_pf_enqueue_n(MemXZone3 *s, const uint32_t *pages, uint8_t *cooldowns, int n, int wake) {
    if (!s || !s->async_pf_q || !s->async_pf_running || !pages || n <= 0) return 0;
    int enq = 0;
    for (int i = 0; i < n; i++) {
        uint32_t page_index = pages[i];
        if (page_index >= s->npages) continue;
        if (s->meta[page_index].state != PAGE_COMPRESSED) continue;
        uint8_t cooldown = cooldowns ? cooldowns[i] : 5;
        uint32_t packed = 0x80000000u | ((uint32_t)cooldown << 23) | (page_index & 0x007FFFFFu);
        int ok = 0;
        for (int spin = 0; spin < 16; spin++) {
            uint32_t head = s->async_pf_head;
            uint32_t next = (head + 1) & ASYNC_PF_Q_MASK;
            if (next == s->async_pf_tail) break;
            if (!__sync_bool_compare_and_swap(&s->async_pf_head, head, next)) continue;
            s->async_pf_q[head] = packed;
            __sync_synchronize();
            __sync_fetch_and_add(&s->async_pf_enqueued, 1);
            enq++;
            ok = 1;
            break;
        }
        if (!ok) break;
    }
    if (enq > 0 && wake) {
        if (pthread_mutex_trylock(&s->async_pf_mutex) == 0) {
            if (enq >= 4) pthread_cond_broadcast(&s->async_pf_cond);
            else pthread_cond_signal(&s->async_pf_cond);
            pthread_mutex_unlock(&s->async_pf_mutex);
        }
    }
    return enq;
}

static int async_pf_enqueue(MemXZone3 *s, uint32_t page_index, uint8_t cooldown) {
    uint8_t cd = cooldown;
    return async_pf_enqueue_n(s, &page_index, &cd, 1, 1);
}

static int force_compress_page_now(MemXZone3 *s, size_t pidx);

static int async_seal_enqueue(MemXZone3 *s, void *ptr, size_t offset, size_t length, uintptr_t owner_tag) {
    if (!s || !s->async_seal_q || !s->async_seal_running || !ptr || length == 0) return 0;
    pthread_mutex_lock(&s->async_seal_mutex);
    uint32_t head = s->async_seal_head;
    uint32_t next = (head + 1) & ASYNC_SEAL_Q_MASK;
    if (next == s->async_seal_tail) {
        pthread_mutex_unlock(&s->async_seal_mutex);
        return 0;
    }
    int prioritize = (length >= (size_t)(256 * 1024));
    if (prioritize) {
        uint32_t tail = s->async_seal_tail;
        uint32_t new_tail = (tail + ASYNC_SEAL_Q_SIZE - 1) & ASYNC_SEAL_Q_MASK;
        if (new_tail != head) {
            async_seal_job_t *job = &s->async_seal_q[new_tail];
            job->ptr = ptr;
            job->offset = offset;
            job->length = length;
            job->owner_tag = owner_tag;
            __sync_synchronize();
            s->async_seal_tail = new_tail;
            __sync_fetch_and_add(&s->async_seal_enqueued, 1);
            pthread_cond_broadcast(&s->async_seal_cond);
            pthread_mutex_unlock(&s->async_seal_mutex);
            return 1;
        }
    }
    async_seal_job_t *job = &s->async_seal_q[head];
    job->ptr = ptr;
    job->offset = offset;
    job->length = length;
    job->owner_tag = owner_tag;
    __sync_synchronize();
    s->async_seal_head = next;
    __sync_fetch_and_add(&s->async_seal_enqueued, 1);
    if (prioritize || ((s->async_seal_head - s->async_seal_tail) & ASYNC_SEAL_Q_MASK) >= 4)
        pthread_cond_broadcast(&s->async_seal_cond);
    else
        pthread_cond_signal(&s->async_seal_cond);
    pthread_mutex_unlock(&s->async_seal_mutex);
    return 1;
}

static void *async_seal_worker(void *arg) {
    MemXZone3 *s = (MemXZone3 *)arg;
    in_memx = 1;
    while (1) {
        async_seal_job_t job;
        int got = 0;
        pthread_mutex_lock(&s->async_seal_mutex);
        while (s->async_seal_running && s->async_seal_tail == s->async_seal_head)
            pthread_cond_wait(&s->async_seal_cond, &s->async_seal_mutex);
        if (s->async_seal_tail != s->async_seal_head) {
            uint32_t tail = s->async_seal_tail;
            job = s->async_seal_q[tail];
            s->async_seal_q[tail].ptr = NULL;
            s->async_seal_q[tail].offset = 0;
            s->async_seal_q[tail].length = 0;
            s->async_seal_q[tail].owner_tag = 0;
            s->async_seal_tail = (tail + 1) & ASYNC_SEAL_Q_MASK;
            s->async_seal_active++;
            got = 1;
        } else if (!s->async_seal_running) {
            pthread_mutex_unlock(&s->async_seal_mutex);
            break;
        }
        pthread_mutex_unlock(&s->async_seal_mutex);
        if (!got) continue;
        if (job.ptr && job.length && s->vmem) {
            uintptr_t base = (uintptr_t)s->vmem;
            uintptr_t p = (uintptr_t)job.ptr;
            if (p >= base && p < base + s->vmem_size) {
            size_t sp = (p - base) / PAGE_SZ;
            if (sp < s->npages && s->meta[sp].owner_tag == job.owner_tag && s->meta[sp].alloc_size > 0) {
                size_t size = s->meta[sp].alloc_size;
                if (job.offset < size && job.length <= size - job.offset) {
                    size_t first = sp + job.offset / PAGE_SZ;
                    size_t last = sp + (job.offset + job.length - 1) / PAGE_SZ;
                    for (size_t i = first; i <= last; i++)
                        force_compress_page_now(s, i);
                    size_t run_start = (size_t)-1;
                    for (size_t i = first; i <= last; i++) {
                        if (s->meta[i].state == PAGE_COMPRESSED) {
                            if (run_start == (size_t)-1) run_start = i;
                        } else if (run_start != (size_t)-1) {
                            page_release_physical_range(s, run_start, i - 1);
                            run_start = (size_t)-1;
                        }
                    }
                    if (run_start != (size_t)-1)
                        page_release_physical_range(s, run_start, last);
                }
            }
            }
        }
        uint64_t done = __sync_add_and_fetch(&s->async_seal_completed, 1);
        if ((done & 7ULL) == 0) {
            if (pthread_mutex_trylock(&s->alloc_mutex) == 0) {
                memx_runtime_reclaim_locked(s);
                pthread_mutex_unlock(&s->alloc_mutex);
            }
        }
        pthread_mutex_lock(&s->async_seal_mutex);
        if (s->async_seal_active > 0) s->async_seal_active--;
        if (s->async_seal_tail == s->async_seal_head && s->async_seal_active == 0)
            pthread_cond_broadcast(&s->async_seal_idle_cond);
        pthread_mutex_unlock(&s->async_seal_mutex);
    }
    return NULL;
}

static void *async_pf_worker(void *arg) {
    MemXZone3 *s = (MemXZone3 *)arg;
    in_memx = 1;
    while (s->async_pf_running || s->async_pf_tail != s->async_pf_head) {
        int drained = 0;
        for (int item = 0; item < 48; item++) {
            uint32_t packed = 0;
            int got = 0;
            for (int spin = 0; spin < 32; spin++) {
                uint32_t tail = s->async_pf_tail;
                uint32_t head = s->async_pf_head;
                if (tail == head) break;
                packed = s->async_pf_q[tail];
                if ((packed & 0x80000000u) == 0) {
#if defined(__aarch64__)
                    __asm__ __volatile__("yield");
#endif
                    continue;
                }
                if (!__sync_bool_compare_and_swap(&s->async_pf_tail, tail, (tail + 1) & ASYNC_PF_Q_MASK))
                    continue;
                s->async_pf_q[tail] = 0;
                got = 1;
                break;
            }
            if (!got) break;
            uint32_t page = packed & 0x007FFFFFu;
            uint8_t cooldown = (uint8_t)((packed >> 23) & 0xFF);
            if (page < s->npages && prefetch_page(s, page, cooldown)) {
                __sync_fetch_and_add(&s->async_pf_completed, 1);
            }
            drained++;
        }
        if (drained == 0) {
            if (!s->async_pf_running) break;
            pthread_mutex_lock(&s->async_pf_mutex);
            if (s->async_pf_running && s->async_pf_tail == s->async_pf_head) {
                struct timespec ts;
                clock_gettime(CLOCK_REALTIME, &ts);
                ts.tv_nsec += 1500000L;
                if (ts.tv_nsec >= 1000000000L) {
                    ts.tv_sec += 1;
                    ts.tv_nsec -= 1000000000L;
                }
                pthread_cond_timedwait(&s->async_pf_cond, &s->async_pf_mutex, &ts);
            }
            pthread_mutex_unlock(&s->async_pf_mutex);
        }
    }
    return NULL;
}

// ─── Signal handler ───
static struct sigaction old_segv, old_bus;
static void fault_handler(int sig, siginfo_t *info, void *ctx) {
    if (!g_z||!g_z->running) goto chain;
    uintptr_t fa=(uintptr_t)info->si_addr, vs=(uintptr_t)g_z->vmem, ve=vs+g_z->vmem_size;
    if (fa<vs||fa>=ve) {
        // Not in vmem range - check if it's pool/meta access from compressor
        if (g_z->pool && fa>=(uintptr_t)g_z->pool && fa<(uintptr_t)g_z->pool+g_z->pool_size) {
            // Pool page fault - make it writable
            size_t pp = (fa - (uintptr_t)g_z->pool) / PAGE_SZ;
            mprotect(g_z->pool + pp*PAGE_SZ, PAGE_SZ, PROT_READ|PROT_WRITE);
            return;
        }
        goto chain;
    }
    size_t pi=(fa-vs)/PAGE_SZ; uint8_t *pa=(uint8_t*)g_z->vmem+pi*PAGE_SZ;
    PageMeta *m=&g_z->meta[pi];
    for (int resolve = 0; resolve < 8; resolve++) {
        uint8_t st = m->state;
        if (st == PAGE_NONE) {
            mprotect(pa,PAGE_SZ,PROT_READ|PROT_WRITE);
            uint8_t old = __sync_val_compare_and_swap(&m->state, PAGE_NONE, PAGE_RESIDENT);
            if (old == PAGE_NONE) {
                memset(pa,0,PAGE_SZ);
                res_list_add(g_z, pi);
                __sync_fetch_and_add(&g_z->live_resident_pages, 1);
                m->dirty = 1;
            }
            break;
        } else if (st == PAGE_RESIDENT || st == PAGE_HOT) {
            if (st == PAGE_HOT && m->comp_size != 0) {
                wait_decompress_complete(m);
            }
            if (!info || info->si_code == SEGV_ACCERR || info->si_code == BUS_ADRERR) {
                m->dirty = 1;
                m->stable_ticks = 0;
                __sync_fetch_and_add(&m->write_seq, 1);
                __sync_synchronize();
            }
            mprotect(pa,PAGE_SZ,PROT_READ|PROT_WRITE);
            if (!info || info->si_code == SEGV_ACCERR || info->si_code == BUS_ADRERR) {
                if (st == PAGE_RESIDENT) {
                    uint8_t old = __sync_val_compare_and_swap(&m->state, PAGE_RESIDENT, PAGE_HOT);
                    if (old == PAGE_RESIDENT) {
                        uint8_t cd = 8;
                        if (m->tensor_role == MEMX_TENSOR_ROLE_KV_CACHE) cd = 6;
                        else if (m->tensor_role == MEMX_TENSOR_ROLE_WEIGHT ||
                                 m->tensor_role == MEMX_TENSOR_ROLE_EMBEDDING) {
                            if (m->tensor_flags & (MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD))
                                cd = 3;
                            else
                                cd = 6;
                        }
                        if (m->tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL) {
                            if (cd < 12) cd = 12;
                        }
                        m->cooldown = cd;
                        m->prefetched = 0;
                        hot_list_add(g_z, (uint32_t)pi);
                    }
                } else if (m->cooldown < 3) {
                    m->cooldown = 3;
                }
            }
            break;
        } else if (st == PAGE_COMPRESSED) {
            {
                uint8_t cd = 4;
                if ((m->tensor_flags & MEMX_TENSOR_FLAG_COLD) != 0 &&
                    (m->tensor_role == MEMX_TENSOR_ROLE_WEIGHT ||
                     m->tensor_role == MEMX_TENSOR_ROLE_EMBEDDING))
                    cd = 1;
                else if ((m->tensor_flags & MEMX_TENSOR_FLAG_READ_MOSTLY) != 0 &&
                    (m->tensor_role == MEMX_TENSOR_ROLE_WEIGHT ||
                     m->tensor_role == MEMX_TENSOR_ROLE_EMBEDDING))
                    cd = 2;
                else if (m->tensor_role == MEMX_TENSOR_ROLE_KV_CACHE)
                    cd = 3;
                decompress_compressed_page(g_z, pi, 0, cd);
            }
            if (m->state == PAGE_HOT && m->comp_size != 0) {
                wait_decompress_complete(m);
            }
            if (m->state == PAGE_COMPRESSED) continue;
            if (!info || info->si_code == SEGV_ACCERR || info->si_code == BUS_ADRERR) {
                mprotect(pa,PAGE_SZ,PROT_READ|PROT_WRITE);
                m->dirty = 1;
                m->stable_ticks = 0;
                __sync_fetch_and_add(&m->write_seq, 1);
                if ((m->tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL) != 0) {
                    if (m->cooldown < 12) m->cooldown = 12;
                }
            }
            break;
        } else if (st == PAGE_COMPRESSING) {
            m->dirty = 1;
            m->stable_ticks = 0;
            __sync_fetch_and_add(&m->write_seq, 1);
            __sync_synchronize();
            mprotect(pa,PAGE_SZ,PROT_READ|PROT_WRITE);
            uint8_t old = __sync_val_compare_and_swap(&m->state, PAGE_COMPRESSING, PAGE_HOT);
            if (old == PAGE_COMPRESSING) {
                m->prefetched = 0;
                m->cooldown = 12;
                m->comp_size = 0;
                m->codec = 0;
                m->pool_offset = 0;
                hot_list_add(g_z, (uint32_t)pi);
                break;
            }
            continue;
        } else {
            mprotect(pa,PAGE_SZ,PROT_READ|PROT_WRITE);
            break;
        }
    }
    __sync_fetch_and_add(&g_z->faults,1);
    if (m->prefetched) {
        m->prefetched = 0;
        __sync_fetch_and_add(&g_z->prefetch_hits, 1);
    }
    {
        int stream = fault_stream_for_role(m->tensor_role);
        int stride = 1;
        size_t prev = g_z->stream_fault_page[stream];
        int prev_stride = g_z->stream_fault_stride[stream];
        if (prev == (size_t)-1) {
            prev = g_z->last_fault_page;
            prev_stride = g_z->last_fault_stride;
        }
        if (prev != (size_t)-1 && pi > prev && (pi - prev) <= PREFETCH_STRIDE_MAX) {
            int observed = (int)(pi - prev);
            if (prev_stride > 0 && observed == prev_stride) stride = observed;
            else if (m->tensor_role == MEMX_TENSOR_ROLE_KV_CACHE ||
                     (m->tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL)) stride = observed > 0 ? observed : 1;
            else if (m->tensor_role == MEMX_TENSOR_ROLE_WEIGHT ||
                     m->tensor_role == MEMX_TENSOR_ROLE_EMBEDDING) stride = observed > 0 ? observed : 1;
            else if (observed == 1) stride = 1;
            g_z->stream_fault_stride[stream] = observed;
            g_z->last_fault_stride = observed;
        } else {
            g_z->stream_fault_stride[stream] = 1;
            g_z->last_fault_stride = 1;
        }
        g_z->stream_fault_page[stream] = pi;
        g_z->last_fault_page = pi;
        g_z->last_fault_role = m->tensor_role;

        int ahead = PREFETCH_AHEAD;
        uint8_t pf_cooldown = 5;
        if (m->tensor_role == MEMX_TENSOR_ROLE_KV_CACHE) {
            ahead = (m->tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL) ? PREFETCH_AHEAD_KV : 12;
            pf_cooldown = 10;
        } else if (m->tensor_role == MEMX_TENSOR_ROLE_WEIGHT ||
                   m->tensor_role == MEMX_TENSOR_ROLE_EMBEDDING) {
            ahead = 8;
            if (m->tensor_flags & MEMX_TENSOR_FLAG_HOT)
                ahead = 10;
            if (m->tensor_flags & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY))
                pf_cooldown = 1;
            else
                pf_cooldown = 2;
        } else if (m->tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL) {
            ahead = 10;
        }
        if (stride > 1) {
            int scaled = ahead / stride;
            if (scaled < 2) scaled = 2;
            ahead = scaled;
        }

        int seq_count = 0;
        for (int k = 1; k <= ahead * 2 && pi + (size_t)k * (size_t)stride < g_z->npages; k++) {
            size_t pj = pi + (size_t)k * (size_t)stride;
            if (g_z->meta[pj].owner_tag != m->owner_tag && m->owner_tag != 0) break;
            if (m->tensor_role != MEMX_TENSOR_ROLE_UNKNOWN &&
                g_z->meta[pj].tensor_role != m->tensor_role) break;
            uint8_t st = g_z->meta[pj].state;
            if (st == PAGE_COMPRESSED) seq_count++;
            else if (st == PAGE_HOT || st == PAGE_RESIDENT) {
                if (k == 1) continue;
                break;
            } else break;
        }
        int min_seq = (m->tensor_role == MEMX_TENSOR_ROLE_KV_CACHE ||
                       m->tensor_role == MEMX_TENSOR_ROLE_WEIGHT ||
                       (m->tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL)) ? 1 : 2;
        if (seq_count >= min_seq) {
            int pf = 0;
            int sync_limit = (m->tensor_role == MEMX_TENSOR_ROLE_KV_CACHE) ? 4 :
                             (m->tensor_role == MEMX_TENSOR_ROLE_WEIGHT ? 3 : 1);
            uint32_t async_pages[64];
            uint8_t async_cds[64];
            int async_n = 0;
            for (int k = 1; k <= ahead && pi + (size_t)k * (size_t)stride < g_z->npages && pf + async_n < ahead; k++) {
                size_t pj = pi + (size_t)k * (size_t)stride;
                if (g_z->meta[pj].owner_tag != m->owner_tag && m->owner_tag != 0) break;
                if (m->tensor_role != MEMX_TENSOR_ROLE_UNKNOWN &&
                    g_z->meta[pj].tensor_role != m->tensor_role) break;
                if (g_z->meta[pj].state != PAGE_COMPRESSED) continue;
                if (k <= sync_limit) {
                    if (decompress_compressed_page(g_z, pj, 1, pf_cooldown)) pf++;
                } else if (async_n < 64) {
                    async_pages[async_n] = (uint32_t)pj;
                    async_cds[async_n] = pf_cooldown;
                    async_n++;
                }
            }
            if (async_n > 0) pf += async_pf_enqueue_n(g_z, async_pages, async_cds, async_n, 1);
            if (pf > 0) __sync_fetch_and_add(&g_z->prefetch_count, 1);
        }
    }
    return;
chain:
    if(sig==SIGSEGV&&old_segv.sa_handler!=SIG_DFL&&old_segv.sa_handler!=SIG_IGN){if(old_segv.sa_flags&SA_SIGINFO)old_segv.sa_sigaction(sig,info,ctx);else if(old_segv.sa_handler!=SIG_ERR)old_segv.sa_handler(sig);}
    else if(sig==SIGBUS&&old_bus.sa_handler!=SIG_DFL&&old_bus.sa_handler!=SIG_IGN){if(old_bus.sa_flags&SA_SIGINFO)old_bus.sa_sigaction(sig,info,ctx);else if(old_bus.sa_handler!=SIG_ERR)old_bus.sa_handler(sig);}
    else{signal(sig,SIG_DFL);raise(sig);}
}

// ─── Background compressor ───

typedef struct {
    MemXZone3 *s;
    size_t *tc;
    uint8_t *page_valid;
    uint32_t *page_seq;
    uint8_t **page_cdata;
    uint32_t *page_csz;
    uint8_t *page_codec;
    size_t *work;
    size_t nwork;
    volatile size_t next;
} tensor_encode_job_t;

static void encode_tensor_page_one(MemXZone3 *s, size_t pidx, const uint8_t *src, uint8_t *tensor_dst,
                                   uint8_t *codec_tmp, uint32_t *out_csz, uint8_t *out_codec) {
    uint32_t tensor_csz = 0;
    uint8_t tensor_codec = 0;
    uint16_t role = s->meta[pidx].tensor_role;
    uint32_t tflags = s->meta[pidx].tensor_flags;
    uint8_t preferred = s->meta[pidx].preferred_codec;
    int coldish = (tflags & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY)) != 0;
    int prefer_ratio = coldish ||
        role == MEMX_TENSOR_ROLE_WEIGHT ||
        role == MEMX_TENSOR_ROLE_KV_CACHE ||
        role == MEMX_TENSOR_ROLE_EMBEDDING;
    int try_sparse = tensor_sparse_byte_eligible(s, pidx);
    int try_fp16 = tensor_fp16_split_eligible(s, pidx);
    int try_bitplane = try_fp16 && prefer_ratio;
    int try_delta = try_fp16 && (role == MEMX_TENSOR_ROLE_KV_CACHE || coldish || role == MEMX_TENSOR_ROLE_ACTIVATION);
    int is_bf16 = (s->meta[pidx].tensor_dtype == MEMX_TENSOR_DTYPE_BF16);
    if (role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING) {
        preferred = try_fp16 ? MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT : MEMX_CODEC_ZLIB;
        s->meta[pidx].preferred_codec = preferred;
    } else if (role == MEMX_TENSOR_ROLE_KV_CACHE && try_delta) {
        preferred = MEMX_CODEC_TENSOR_FP16_DELTA_SPLIT;
        s->meta[pidx].preferred_codec = preferred;
    } else if (role == MEMX_TENSOR_ROLE_ACTIVATION && try_fp16) {
        preferred = MEMX_CODEC_TENSOR_FP16_SPLIT;
        s->meta[pidx].preferred_codec = preferred;
    }
    int sticky_ok = preferred != 0 && s->meta[pidx].codec_fail_streak < 4;
    uint32_t sticky_good = (role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING || coldish)
        ? (PAGE_SZ * 15 / 16)
        : (role == MEMX_TENSOR_ROLE_KV_CACHE ? (PAGE_SZ * 15 / 16) : (PAGE_SZ - 32));
    {
        uint32_t pressure = memx_pool_pressure_percent_locked(s);
        if (pressure >= 80) sticky_good = PAGE_SZ - 32;
        else if (pressure >= 60 && sticky_good < (PAGE_SZ * 7 / 8))
            sticky_good = PAGE_SZ * 7 / 8;
    }
    if (sticky_ok) {
        uint32_t sticky_csz = 0;
        if (preferred == MEMX_CODEC_TENSOR_SPARSE_BYTE && try_sparse)
            sticky_csz = tensor_sparse_byte_compress(src, tensor_dst, PAGE_SZ);
        else if (preferred == MEMX_CODEC_TENSOR_EXP_PACK && try_fp16)
            sticky_csz = tensor_exp_pack_compress(src, tensor_dst, PAGE_SZ, is_bf16);
        else if (preferred == MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT && try_fp16)
            sticky_csz = tensor_fp16_zlib_split_compress(src, tensor_dst, PAGE_SZ);
        else if (preferred == MEMX_CODEC_ZLIB)
            sticky_csz = zlib_page_compress(src, tensor_dst, PAGE_SZ);
        else if (preferred == MEMX_CODEC_TENSOR_BITPLANE16 && try_bitplane)
            sticky_csz = tensor_bitplane16_compress(src, tensor_dst, PAGE_SZ);
        else if (preferred == MEMX_CODEC_TENSOR_FP16_SPLIT && try_fp16)
            sticky_csz = tensor_fp16_split_compress(src, tensor_dst, PAGE_SZ);
        else if (preferred == MEMX_CODEC_TENSOR_FP16_DELTA_SPLIT && try_delta)
            sticky_csz = tensor_fp16_delta_split_compress(src, tensor_dst, PAGE_SZ);
        if (sticky_csz > 0 && sticky_csz < sticky_good) {
            tensor_csz = sticky_csz;
            tensor_codec = preferred;
        } else if (sticky_csz > 0 && sticky_csz < PAGE_SZ - 32) {
            tensor_csz = sticky_csz;
            tensor_codec = preferred;
            s->meta[pidx].codec_fail_streak++;
        } else {
            s->meta[pidx].codec_fail_streak++;
        }
    }
    int need_compete = (tensor_csz == 0);
    if (!(role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING)) {
        if (prefer_ratio && tensor_csz >= sticky_good) need_compete = 1;
    }
    if (need_compete) {
        uint32_t best_csz = tensor_csz;
        uint8_t best_codec = tensor_codec;
        if (try_sparse) {
            uint32_t sparse_csz = tensor_sparse_byte_compress(src, codec_tmp, PAGE_SZ);
            if (sparse_csz > 0 && (best_csz == 0 || sparse_csz < best_csz)) {
                best_csz = sparse_csz;
                best_codec = MEMX_CODEC_TENSOR_SPARSE_BYTE;
                memcpy(tensor_dst, codec_tmp, sparse_csz);
            }
        }
        if (try_fp16 && !is_bf16 &&
            (role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING || coldish)) {
            uint32_t expz = tensor_exp_pack_compress(src, codec_tmp, PAGE_SZ, is_bf16);
            if (expz > 0 && (best_csz == 0 || expz < best_csz)) {
                best_csz = expz;
                best_codec = MEMX_CODEC_TENSOR_EXP_PACK;
                memcpy(tensor_dst, codec_tmp, expz);
            }
        }
        if (try_fp16 &&
            (role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING || coldish ||
             role == MEMX_TENSOR_ROLE_KV_CACHE)) {
            uint32_t zsplit = tensor_fp16_zlib_split_compress(src, codec_tmp, PAGE_SZ);
            if (zsplit > 0 && (best_csz == 0 || zsplit < best_csz)) {
                best_csz = zsplit;
                best_codec = MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT;
                memcpy(tensor_dst, codec_tmp, zsplit);
            }
        }
        if (role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING || coldish ||
            role == MEMX_TENSOR_ROLE_KV_CACHE) {
            uint32_t zcsz = zlib_page_compress(src, codec_tmp, PAGE_SZ);
            if (zcsz > 0 && (best_csz == 0 || zcsz < best_csz)) {
                best_csz = zcsz;
                best_codec = MEMX_CODEC_ZLIB;
                memcpy(tensor_dst, codec_tmp, zcsz);
            }
        }
        if (try_delta) {
            uint32_t delta_split_csz = tensor_fp16_delta_split_compress(src, codec_tmp, PAGE_SZ);
            if (delta_split_csz > 0 && (best_csz == 0 || delta_split_csz < best_csz)) {
                best_csz = delta_split_csz;
                best_codec = MEMX_CODEC_TENSOR_FP16_DELTA_SPLIT;
                memcpy(tensor_dst, codec_tmp, delta_split_csz);
            }
        }
        if (try_fp16) {
            uint32_t split_csz = tensor_fp16_split_compress(src, codec_tmp, PAGE_SZ);
            if (split_csz > 0 && (best_csz == 0 || split_csz < best_csz)) {
                best_csz = split_csz;
                best_codec = MEMX_CODEC_TENSOR_FP16_SPLIT;
                memcpy(tensor_dst, codec_tmp, split_csz);
            }
        }
        if (try_bitplane &&
            (best_csz == 0 || best_csz > (PAGE_SZ * 5 / 8)) &&
            (role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING || coldish ||
             role == MEMX_TENSOR_ROLE_KV_CACHE)) {
            uint32_t bitplane_csz = tensor_bitplane16_compress(src, codec_tmp, PAGE_SZ);
            if (bitplane_csz > 0 && (best_csz == 0 || bitplane_csz < best_csz)) {
                best_csz = bitplane_csz;
                best_codec = MEMX_CODEC_TENSOR_BITPLANE16;
                memcpy(tensor_dst, codec_tmp, bitplane_csz);
            }
        }
        if (best_csz > 0) {
            tensor_csz = best_csz;
            tensor_codec = best_codec;
        }
    }
    *out_csz = tensor_csz;
    *out_codec = tensor_codec;
}

static void tensor_encode_process_range(MemXZone3 *s, tensor_encode_job_t *job, uint8_t *local_scratch) {
    for (;;) {
        size_t wi = __sync_fetch_and_add(&job->next, 1);
        if (wi >= job->nwork) break;
        size_t i = job->work[wi];
        if (!job->page_valid[i]) continue;
        size_t pidx = job->tc[i];
        if (s->meta[pidx].state != PAGE_COMPRESSING ||
            s->meta[pidx].dirty ||
            s->meta[pidx].write_seq != job->page_seq[i]) {
            job->page_valid[i] = 0;
            job->page_csz[i] = 0;
            job->page_cdata[i] = NULL;
            continue;
        }
        const uint8_t *src = s->tmp_src + i * PAGE_SZ;
        uint8_t *tensor_dst = s->tmp_dst + i * PAGE_SZ;
        uint32_t tensor_csz = 0;
        uint8_t tensor_codec = 0;
        encode_tensor_page_one(s, pidx, src, tensor_dst, local_scratch, &tensor_csz, &tensor_codec);
        if (tensor_csz > 0) {
            job->page_cdata[i] = tensor_dst;
            job->page_csz[i] = tensor_csz;
            job->page_codec[i] = tensor_codec;
        } else {
            job->page_csz[i] = 0;
            job->page_codec[i] = 0;
            job->page_cdata[i] = NULL;
        }
    }
}

static void *tensor_encode_pool_worker(void *arg) {
    MemXZone3 *s = (MemXZone3 *)arg;
    in_memx = 1;
    uint8_t local_scratch[PAGE_SZ];
    while (s->encode_pool_running) {
        pthread_mutex_lock(&s->encode_mutex);
        while (s->encode_pool_running && s->encode_job == NULL)
            pthread_cond_wait(&s->encode_cond, &s->encode_mutex);
        tensor_encode_job_t *job = (tensor_encode_job_t *)s->encode_job;
        pthread_mutex_unlock(&s->encode_mutex);
        if (!s->encode_pool_running) break;
        if (!job) continue;
        tensor_encode_process_range(s, job, local_scratch);
        if (__sync_sub_and_fetch(&s->encode_workers_active, 1) == 0) {
            pthread_mutex_lock(&s->encode_mutex);
            if (s->encode_job == job) s->encode_job = NULL;
            pthread_cond_broadcast(&s->encode_done_cond);
            pthread_mutex_unlock(&s->encode_mutex);
        }
    }
    return NULL;
}

static void tensor_encode_run_parallel(MemXZone3 *s, tensor_encode_job_t *job, int nworkers) {
    if (!s || !job || job->nwork == 0) return;
    uint8_t local_scratch[PAGE_SZ];
    if (nworkers < 1) nworkers = 1;
    if (s->encode_nworkers <= 0 || !s->encode_pool_running || nworkers == 1 || job->nwork < 3) {
        job->next = 0;
        tensor_encode_process_range(s, job, local_scratch);
        return;
    }
    int pool_n = s->encode_nworkers;
    if (pool_n < 1) {
        job->next = 0;
        tensor_encode_process_range(s, job, local_scratch);
        return;
    }
    job->next = 0;
    pthread_mutex_lock(&s->encode_mutex);
    while (s->encode_job != NULL)
        pthread_cond_wait(&s->encode_done_cond, &s->encode_mutex);
    s->encode_job = job;
    __sync_lock_test_and_set(&s->encode_workers_active, pool_n);
    pthread_cond_broadcast(&s->encode_cond);
    pthread_mutex_unlock(&s->encode_mutex);
    tensor_encode_process_range(s, job, local_scratch);
    pthread_mutex_lock(&s->encode_mutex);
    while (s->encode_job != NULL)
        pthread_cond_wait(&s->encode_done_cond, &s->encode_mutex);
    pthread_mutex_unlock(&s->encode_mutex);
}

static void *bg_compressor(void *arg) {
    MemXZone3 *s=(MemXZone3*)arg;
    in_memx = 1;  // CRITICAL: prevent Metal internal mallocs from going to our pool
    const size_t BATCH=s->batch_cap;
    while(s->running){
        // Cooldown: only scan HOT pages from compact list (not all 6M pages)
        uint32_t hc = s->hot_count;
        uint32_t new_hot = 0;
        for(uint32_t i=0; i<hc; i++) {
            uint32_t pi = s->hot_list[i];
            if(s->meta[pi].state==PAGE_HOT) {
                if (s->meta[pi].comp_size != 0) {
                    if (new_hot != i) s->hot_list[new_hot] = pi;
                    new_hot++;
                } else if (s->meta[pi].dirty) {
                    if (page_wants_write_protect(&s->meta[pi])) {
                        mprotect((uint8_t*)s->vmem + (size_t)pi * PAGE_SZ, PAGE_SZ, PROT_READ);
                    }
                    __sync_synchronize();
                    if ((s->meta[pi].tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL) != 0) {
                        if (s->meta[pi].cooldown == 0) s->meta[pi].cooldown = 12;
                        if (s->meta[pi].cooldown > 0) s->meta[pi].cooldown--;
                        if (s->meta[pi].cooldown == 0) {
                            uint32_t seq0 = s->meta[pi].write_seq;
                            __sync_synchronize();
                            if (!s->meta[pi].dirty || s->meta[pi].write_seq != seq0) {
                                s->meta[pi].cooldown = 8;
                            } else {
                                s->meta[pi].dirty = 0;
                                s->meta[pi].stable_ticks = 0;
                                s->meta[pi].cooldown = 6;
                            }
                        }
                        if (new_hot != i) s->hot_list[new_hot] = pi;
                        new_hot++;
                    } else {
                        if (s->meta[pi].dirty) {
                            s->meta[pi].dirty = 0;
                            if (s->meta[pi].cooldown < 8) s->meta[pi].cooldown = 8;
                        }
                        if (new_hot != i) s->hot_list[new_hot] = pi;
                        new_hot++;
                    }
                } else if(s->meta[pi].cooldown > 0) {
                    uint8_t dec = 1;
                    if ((s->meta[pi].tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL) == 0 &&
                        (s->meta[pi].tensor_flags & (MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD)) != 0 &&
                        (s->meta[pi].tensor_role == MEMX_TENSOR_ROLE_WEIGHT ||
                         s->meta[pi].tensor_role == MEMX_TENSOR_ROLE_EMBEDDING) &&
                        !s->meta[pi].dirty) {
                        dec = 2;
                    }
                    if (s->meta[pi].cooldown > dec) s->meta[pi].cooldown = (uint8_t)(s->meta[pi].cooldown - dec);
                    else s->meta[pi].cooldown = 0;
                    if (page_wants_write_protect(&s->meta[pi])) {
                        mprotect((uint8_t*)s->vmem + (size_t)pi * PAGE_SZ, PAGE_SZ, PROT_READ);
                    }
                    if (new_hot != i) s->hot_list[new_hot] = pi;
                    new_hot++;
                } else {
                    if (s->meta[pi].dirty) {
                        s->meta[pi].dirty = 0;
                        s->meta[pi].cooldown = 8;
                        if (page_wants_write_protect(&s->meta[pi])) {
                            mprotect((uint8_t*)s->vmem + (size_t)pi * PAGE_SZ, PAGE_SZ, PROT_READ);
                        }
                        if (new_hot != i) s->hot_list[new_hot] = pi;
                        new_hot++;
                    } else {
                        uint8_t old = __sync_val_compare_and_swap(&s->meta[pi].state, PAGE_HOT, PAGE_RESIDENT);
                        if(old == PAGE_HOT) {
                            if (s->meta[pi].dirty) {
                                s->meta[pi].state = PAGE_HOT;
                                s->meta[pi].dirty = 0;
                                s->meta[pi].cooldown = 8;
                                if (page_wants_write_protect(&s->meta[pi])) {
                                    mprotect((uint8_t*)s->vmem + (size_t)pi * PAGE_SZ, PAGE_SZ, PROT_READ);
                                }
                                if (new_hot != i) s->hot_list[new_hot] = pi;
                                new_hot++;
                            } else {
                                s->meta[pi].prefetched=0;
                                if ((s->meta[pi].tensor_flags & (MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD)) != 0)
                                    s->meta[pi].stable_ticks = 255;
                                else
                                    s->meta[pi].stable_ticks = 0;
                                if (page_wants_write_protect(&s->meta[pi])) {
                                    mprotect((uint8_t*)s->vmem + (size_t)pi * PAGE_SZ, PAGE_SZ, PROT_READ);
                                }
                                res_list_add(s, pi);
                            }
                        } else {
                            if (new_hot != i) s->hot_list[new_hot] = pi;
                            new_hot++;
                        }
                    }
                }
            }
            // else: page was freed or transitioned by fault handler, drop from list
        }
        s->hot_count = new_hot;
        
        size_t tc[BATCH]; size_t nc=0;
        size_t pri[BATCH]; size_t npc=0;
        uint32_t rc = s->res_count;
        uint32_t new_res = 0;
        for(uint32_t i=0; i<rc && s->running; i++) {
            uint32_t pi = s->res_list[i];
            if(s->meta[pi].state==PAGE_RESIDENT) {
                if ((s->meta[pi].tensor_flags & (MEMX_TENSOR_FLAG_HOT | MEMX_TENSOR_FLAG_NO_COMPRESS)) == 0) {
                    int cold_pri = (s->meta[pi].tensor_flags & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY)) != 0;
                    int llm_pri =
                        s->meta[pi].tensor_role == MEMX_TENSOR_ROLE_KV_CACHE ||
                        s->meta[pi].tensor_role == MEMX_TENSOR_ROLE_WEIGHT ||
                        s->meta[pi].tensor_role == MEMX_TENSOR_ROLE_EMBEDDING;
                    if (cold_pri && npc < BATCH) pri[npc++] = pi;
                    else if (llm_pri && npc < BATCH) pri[npc++] = pi;
                    else if (nc < BATCH) tc[nc++] = pi;
                }
                if (new_res != i) s->res_list[new_res] = pi;
                new_res++;
            }
        }
        if (npc > 0) {
            size_t merged[BATCH];
            size_t nm = 0;
            for (size_t i = 0; i < npc && nm < BATCH; i++) merged[nm++] = pri[i];
            for (size_t i = 0; i < nc && nm < BATCH; i++) merged[nm++] = tc[i];
            memcpy(tc, merged, nm * sizeof(size_t));
            nc = nm;
        }
        // Compact remaining entries
        for(uint32_t i=new_res; rc > BATCH && i<rc; i++) {
            uint32_t pi = s->res_list[i];
            if(s->meta[pi].state==PAGE_RESIDENT) {
                s->res_list[new_res++] = pi;
            }
        }
        s->res_count = new_res;
        
        if(nc==0){
            // Fallback: if res_list is empty but there may be untracked RESIDENT pages
            // (e.g. pages allocated before list was populated, or list overflow)
            // Do a bitmap-guided scan of used pages
            {
                static size_t scan_cursor = 0;
                size_t scanned = 0;
                size_t limit = s->npages < 65536 ? s->npages : 65536;
                while (nc < BATCH && scanned < limit && s->running) {
                    size_t i = scan_cursor;
                    if (++scan_cursor >= s->npages) scan_cursor = 0;
                    scanned++;
                    if (!bm_is_free(s, i) && s->meta[i].state == PAGE_RESIDENT &&
                        (s->meta[i].tensor_flags & (MEMX_TENSOR_FLAG_HOT | MEMX_TENSOR_FLAG_NO_COMPRESS)) == 0) {
                        tc[nc++] = i;
                        res_list_add(s, (uint32_t)i);
                    }
                }
            }
            if(nc==0){
                shared_stats_update(s);
                if (s->res_count > 512) {
                    s->idle_count = 0;
                    struct timespec ts={0,1000000};
                    nanosleep(&ts,NULL);
                } else if(++s->idle_count > 15) {
                    struct timespec ts={0,100000000};
                    nanosleep(&ts,NULL);
                    s->idle_count = 15;
                } else {
                    struct timespec ts={0,5000000};
                    nanosleep(&ts,NULL);
                }
                continue;
            }
        }
        uint8_t page_valid[BATCH];
        uint32_t page_seq[BATCH];
        memset(page_valid, 0, nc);
        memset(page_seq, 0, sizeof(uint32_t) * nc);
        for(size_t i=0;i<nc;) {
            if(s->meta[tc[i]].state!=PAGE_RESIDENT) { i++; continue; }
            size_t j=i+1;
            while(j<nc && tc[j]==tc[j-1]+1) j++;
            pthread_mutex_lock(&s->alloc_mutex);
            if (j > i) {
                size_t run_pages = j - i;
                mprotect((uint8_t*)s->vmem + tc[i]*PAGE_SZ, run_pages * PAGE_SZ, PROT_READ);
            }
            for(size_t k=i; k<j; k++) {
                page_valid[k] = 0;
                size_t pidx = tc[k];
                if (s->meta[pidx].state != PAGE_RESIDENT) continue;
                if (s->meta[pidx].dirty) {
                    if (page_wants_write_protect(&s->meta[pidx]))
                        mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ, PAGE_SZ, PROT_READ);
                    if ((s->meta[pidx].tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL) != 0) {
                        uint32_t seq0 = s->meta[pidx].write_seq;
                        __sync_synchronize();
                        if (s->meta[pidx].write_seq != seq0 || s->meta[pidx].dirty == 0) {
                            s->meta[pidx].stable_ticks = 0;
                            continue;
                        }
                        int need = page_stable_need(&s->meta[pidx]);
                        if (need < 3) need = 3;
                        if (s->meta[pidx].stable_ticks < (uint8_t)need) {
                            s->meta[pidx].stable_ticks++;
                            continue;
                        }
                    }
                    s->meta[pidx].dirty = 0;
                    s->meta[pidx].stable_ticks = 0;
                    continue;
                }
                {
                    int need = page_stable_need(&s->meta[pidx]);
                    uint32_t pressure = memx_pool_pressure_percent_locked(s);
                    if ((s->meta[pidx].tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL) == 0) {
                        if (pressure >= 85 && need > 1) need = 1;
                        if (pressure >= 95) need = 0;
                    } else if (need < 2) {
                        need = 2;
                    }
                    if (page_wants_write_protect(&s->meta[pidx]) && s->meta[pidx].stable_ticks < (uint8_t)need) {
                        if (s->meta[pidx].stable_ticks < 255) s->meta[pidx].stable_ticks++;
                        continue;
                    }
                }
                uint32_t seq0 = s->meta[pidx].write_seq;
                __sync_synchronize();
                if (s->meta[pidx].dirty || s->meta[pidx].state != PAGE_RESIDENT ||
                    s->meta[pidx].write_seq != seq0) {
                    s->meta[pidx].stable_ticks = 0;
                    if (page_wants_write_protect(&s->meta[pidx]))
                        mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ, PAGE_SZ, PROT_READ);
                    else
                        mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ, PAGE_SZ, PROT_READ|PROT_WRITE);
                    continue;
                }
                uint8_t old = __sync_val_compare_and_swap(&s->meta[pidx].state, PAGE_RESIDENT, PAGE_COMPRESSING);
                if (old != PAGE_RESIDENT) {
                    s->meta[pidx].stable_ticks = 0;
                    if (page_wants_write_protect(&s->meta[pidx]))
                        mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ, PAGE_SZ, PROT_READ);
                    else
                        mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ, PAGE_SZ, PROT_READ|PROT_WRITE);
                    continue;
                }
                if (s->meta[pidx].dirty || s->meta[pidx].write_seq != seq0) {
                    s->meta[pidx].state = PAGE_HOT;
                    s->meta[pidx].cooldown = 4;
                    s->meta[pidx].stable_ticks = 0;
                    mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ, PAGE_SZ, PROT_READ|PROT_WRITE);
                    hot_list_add(s, (uint32_t)pidx);
                    continue;
                }
                uint8_t *srcp = (uint8_t*)s->vmem+pidx*PAGE_SZ;
                uint8_t *dstp = s->tmp_src+k*PAGE_SZ;
                if ((s->meta[pidx].tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL) != 0) {
                    mprotect(srcp, PAGE_SZ, PROT_NONE);
                    __sync_synchronize();
                    mprotect(srcp, PAGE_SZ, PROT_READ);
                    __sync_synchronize();
                    if (s->meta[pidx].dirty || s->meta[pidx].write_seq != seq0) {
                        s->meta[pidx].state = PAGE_HOT;
                        s->meta[pidx].cooldown = 8;
                        s->meta[pidx].stable_ticks = 0;
                        mprotect(srcp, PAGE_SZ, PROT_READ|PROT_WRITE);
                        hot_list_add(s, (uint32_t)pidx);
                        continue;
                    }
                }
                memcpy(dstp, srcp, PAGE_SZ);
                __sync_synchronize();
                if (!page_compress_content_ok(s, pidx, seq0, dstp)) {
                    if (s->meta[pidx].state == PAGE_COMPRESSING) {
                        s->meta[pidx].state = PAGE_HOT;
                        s->meta[pidx].cooldown = 6;
                        hot_list_add(s, (uint32_t)pidx);
                    }
                    s->meta[pidx].stable_ticks = 0;
                    mprotect(srcp, PAGE_SZ, PROT_READ|PROT_WRITE);
                    continue;
                }
                page_seq[k] = seq0;
                page_valid[k] = 1;
            }
            pthread_mutex_unlock(&s->alloc_mutex);
            i = j;
        }
        // Zero-page fast path: detect all-zero pages before GPU compress
        // Zero pages compress to 8 bytes (MX header + RLE), skip GPU entirely
        static const uint8_t ZERO_COMPRESSED[] = {0x4D,0x58,0x03,0x00, 0xFD,0x00,0x00,0x40};  // ver3, RLE(zero, 16384)
        size_t gpu_nc = 0;  // pages that actually need GPU compression
        size_t gpu_map[BATCH];  // gpu_map[gi] = original index i
        uint8_t *page_cdata[BATCH];
        uint32_t page_csz[BATCH];
        uint8_t page_codec[BATCH];
        size_t tensor_nc = 0;
        size_t tensor_work[BATCH];
        size_t n_tensor_work = 0;
        for(size_t i=0;i<nc;i++){
            if(!page_valid[i]) continue;
            uint8_t *src = s->tmp_src + i*PAGE_SZ;
            page_codec[i] = 0;
            page_csz[i] = 0;
            page_cdata[i] = NULL;
            if (s->meta[tc[i]].state != PAGE_COMPRESSING ||
                s->meta[tc[i]].dirty ||
                s->meta[tc[i]].write_seq != page_seq[i]) {
                restore_compressing_page(s, tc[i]);
                page_valid[i] = 0;
                continue;
            }
            int is_zero = page_is_all_zero(src);
            if(is_zero) {
                page_cdata[i] = (uint8_t*)ZERO_COMPRESSED;
                page_csz[i] = 8;
            } else if (tensor_sparse_byte_eligible(s, tc[i]) || tensor_fp16_split_eligible(s, tc[i])) {
                tensor_work[n_tensor_work++] = i;
            } else {
                if (n_tensor_work > 0) {
                    restore_compressing_page(s, tc[i]);
                    page_valid[i] = 0;
                } else {
                    if(gpu_nc != i) memcpy(s->tmp_src + gpu_nc*PAGE_SZ, src, PAGE_SZ);
                    gpu_map[gpu_nc] = i;
                    gpu_nc++;
                }
            }
        }
        if (n_tensor_work > 0) {
            tensor_encode_job_t job;
            job.s = s;
            job.tc = tc;
            job.page_valid = page_valid;
            job.page_seq = page_seq;
            job.page_cdata = page_cdata;
            job.page_csz = page_csz;
            job.page_codec = page_codec;
            job.work = tensor_work;
            job.nwork = n_tensor_work;
            job.next = 0;
            int nworkers = COMP_CPU_WORKERS;
            if ((size_t)nworkers > n_tensor_work) nworkers = (int)n_tensor_work;
            if (nworkers < 1) nworkers = 1;
            if (n_tensor_work < 3) nworkers = 1;
            {
                int has_seq = 0;
                for (size_t wi = 0; wi < n_tensor_work; wi++) {
                    size_t pi = tc[tensor_work[wi]];
                    if (s->meta[pi].tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL) { has_seq = 1; break; }
                }
                if (has_seq) nworkers = 1;
            }
            tensor_encode_run_parallel(s, &job, nworkers);
            for (size_t wi = 0; wi < n_tensor_work; wi++) {
                size_t i = tensor_work[wi];
                size_t pidx = tc[i];
                if (!page_valid[i] ||
                    s->meta[pidx].state != PAGE_COMPRESSING ||
                    s->meta[pidx].dirty ||
                    s->meta[pidx].write_seq != page_seq[i]) {
                    if (s->meta[pidx].state == PAGE_COMPRESSING)
                        restore_compressing_page(s, pidx);
                    page_valid[i] = 0;
                    continue;
                }
                if (page_csz[i] > 0) {
                    tensor_nc++;
                } else {
                    uint16_t role = s->meta[pidx].tensor_role;
                    uint32_t tflags = s->meta[pidx].tensor_flags;
                    int coldish = (tflags & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY)) != 0;
                    if (role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING || coldish) {
                        s->meta[pidx].stable_ticks = 8;
                        restore_compressing_page(s, pidx);
                        page_valid[i] = 0;
                    } else {
                        uint8_t *src = s->tmp_src + i*PAGE_SZ;
                        if(gpu_nc != i) memcpy(s->tmp_src + gpu_nc*PAGE_SZ, src, PAGE_SZ);
                        gpu_map[gpu_nc] = i;
                        gpu_nc++;
                    }
                }
            }
        }
        // GPU compress only non-zero pages
        if(gpu_nc > 0) {
            int gr=gpu_compress(s,gpu_nc);
            if(gr!=0){
                pthread_mutex_lock(&s->alloc_mutex);
                for(size_t i=0;i<nc;i++) if(page_valid[i]) restore_compressing_page(s, tc[i]);
                pthread_mutex_unlock(&s->alloc_mutex);
                struct timespec ts={0,100000000}; nanosleep(&ts,NULL); continue;
            }
            // Map GPU output back to original page indices
            for(size_t gi=0; gi<gpu_nc; gi++) {
                size_t orig_i = gpu_map[gi];
                page_cdata[orig_i] = s->tmp_dst + gi*PAGE_SZ;
                page_csz[orig_i] = s->tmp_sz[gi];
                page_codec[orig_i] = 0;
            }
        }
        // Reset idle counter - we had work to do
        s->idle_count = 0;
        pthread_mutex_lock(&s->alloc_mutex);
        if (s->dedup_pending_free_count > 0) memx_runtime_reclaim_locked(s);
        for(size_t i=0;i<nc;i++){
            if(!page_valid[i]) continue;
            uint32_t cs=page_csz[i]; size_t pidx=tc[i];
            {
                uint32_t max_cs = PAGE_SZ - 32;
                uint16_t role = s->meta[pidx].tensor_role;
                uint32_t tflags = s->meta[pidx].tensor_flags;
                if (role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING ||
                    role == MEMX_TENSOR_ROLE_KV_CACHE ||
                    (tflags & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY))) {
                    max_cs = (PAGE_SZ * 15) / 16;
                }
                if (cs == 0 || cs >= PAGE_SZ || cs >= max_cs) {
                    restore_compressing_page(s, pidx);
                    continue;
                }
            }
            // ─── Dedup: check if this compressed page already exists in pool ───
            uint8_t *cdata = page_cdata[i];
            uint64_t h = fnv1a_word(cdata, cs);
            uint32_t slot = (uint32_t)(h & DEDUP_HT_MASK);
            int dedup_found = 0;
            // Open-addressing probe (up to 8 probes)
            for(int probe=0; probe<8; probe++) {
                uint32_t s2 = (slot + probe) & DEDUP_HT_MASK;
                if (s->dedup_hash[s2] == 0) break;  // truly empty slot
                if (s->dedup_hash[s2] == h && s->dedup_sz[s2] == cs && s->dedup_ref[s2] > 0) {
                    // Potential match - verify actual data
                    uint64_t existing_off = s->dedup_off[s2];
                    if (memcmp(cdata, s->pool + existing_off, cs) == 0) {
                        if (s->meta[pidx].state != PAGE_COMPRESSING || s->meta[pidx].dirty ||
                            s->meta[pidx].write_seq != page_seq[i]) {
                            if (s->meta[pidx].state == PAGE_COMPRESSING) {
                                s->meta[pidx].state = PAGE_HOT;
                                s->meta[pidx].cooldown = 1;
                                hot_list_add(s, (uint32_t)pidx);
                            }
                            mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ,PAGE_SZ,PROT_READ|PROT_WRITE);
                            continue;
                        }
                        if (!page_compress_content_ok(s, pidx, page_seq[i], s->tmp_src + i * PAGE_SZ)) {
                            if (s->meta[pidx].state == PAGE_COMPRESSING) {
                                s->meta[pidx].state = PAGE_HOT;
                                s->meta[pidx].cooldown = 6;
                                hot_list_add(s, (uint32_t)pidx);
                            }
                            mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ,PAGE_SZ,PROT_READ|PROT_WRITE);
                            continue;
                        }
                        __sync_fetch_and_add(&s->dedup_ref[s2], 1);
                        s->meta[pidx].pool_offset=existing_off;
                        s->meta[pidx].codec=page_codec[i];
                        s->meta[pidx].comp_size=cs;
                        __sync_synchronize();
                        if (!page_compress_content_ok(s, pidx, page_seq[i], s->tmp_src + i * PAGE_SZ)) {
                            s->meta[pidx].pool_offset=0;
                            s->meta[pidx].codec=0;
                            s->meta[pidx].comp_size=0;
                            __sync_fetch_and_sub(&s->dedup_ref[s2], 1);
                            if (s->meta[pidx].state == PAGE_COMPRESSING) {
                                s->meta[pidx].state = PAGE_HOT;
                                s->meta[pidx].cooldown = 6;
                                hot_list_add(s, (uint32_t)pidx);
                            }
                            mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ,PAGE_SZ,PROT_READ|PROT_WRITE);
                            continue;
                        }
                        if (!commit_compressed_page(s, pidx, page_seq[i], s->tmp_src + i * PAGE_SZ)) {
                            s->meta[pidx].pool_offset=0;
                            s->meta[pidx].codec=0;
                            s->meta[pidx].comp_size=0;
                            __sync_fetch_and_sub(&s->dedup_ref[s2], 1);
                            if (s->meta[pidx].state == PAGE_COMPRESSING) {
                                s->meta[pidx].state = PAGE_HOT;
                                s->meta[pidx].cooldown = 6;
                                hot_list_add(s, (uint32_t)pidx);
                            }
                            mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ,PAGE_SZ,PROT_READ|PROT_WRITE);
                            continue;
                        }
                        if (s->live_resident_pages) __sync_fetch_and_sub(&s->live_resident_pages, 1);
                        __sync_fetch_and_add(&s->live_compressed_pages, 1);
                        s->meta[pidx].preferred_codec = page_codec[i] ? page_codec[i] : s->meta[pidx].preferred_codec;
                        s->meta[pidx].codec_fail_streak = 0;
                        if (s->meta[pidx].state == PAGE_COMPRESSED) s->meta[pidx].dirty = 0;
                        note_page_compressed(s, pidx, page_codec[i], cs);
                        if (s->meta[pidx].state == PAGE_COMPRESSED) page_release_physical(s, pidx);
                        __sync_fetch_and_add(&s->dedup_hits,1);
                        __sync_fetch_and_add(&s->dedup_bytes_saved,cs);
                        dedup_found = 1;
                        break;
                    }
                }
            }
            if (dedup_found) continue;
            // ─── No dedup hit: store new compressed data ───
            uint64_t off = 0;
            if (pool_alloc_extent_locked(s, cs, &off) != 0) { restore_compressing_page(s, pidx); continue; }
            pool_prepare_write_range(s, off, cs);
            if (!page_compress_content_ok(s, pidx, page_seq[i], s->tmp_src + i * PAGE_SZ)) {
                pool_free_insert_locked(s, off, cs);
                if (s->meta[pidx].state == PAGE_COMPRESSING) {
                    s->meta[pidx].state = PAGE_HOT;
                    s->meta[pidx].cooldown = 6;
                    hot_list_add(s, (uint32_t)pidx);
                }
                mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ,PAGE_SZ,PROT_READ|PROT_WRITE);
                continue;
            }
            memcpy(s->pool+off,cdata,cs);
            __sync_fetch_and_add(&s->pool_used,cs);
            if (!page_compress_content_ok(s, pidx, page_seq[i], s->tmp_src + i * PAGE_SZ)) {
                pool_free_insert_locked(s, off, cs);
                if(s->pool_used >= cs) __sync_fetch_and_sub(&s->pool_used, cs);
                if (s->meta[pidx].state == PAGE_COMPRESSING) {
                    s->meta[pidx].state = PAGE_HOT;
                    s->meta[pidx].cooldown = 6;
                    hot_list_add(s, (uint32_t)pidx);
                }
                mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ,PAGE_SZ,PROT_READ|PROT_WRITE);
                continue;
            }
            s->meta[pidx].pool_offset=off;
            s->meta[pidx].codec=page_codec[i];
            s->meta[pidx].comp_size=cs;
            __sync_synchronize();
            if (s->meta[pidx].dirty || s->meta[pidx].write_seq != page_seq[i] ||
                s->meta[pidx].state != PAGE_COMPRESSING) {
                s->meta[pidx].pool_offset=0;
                s->meta[pidx].codec=0;
                s->meta[pidx].comp_size=0;
                pool_free_insert_locked(s, off, cs);
                if(s->pool_used >= cs) __sync_fetch_and_sub(&s->pool_used, cs);
                if (s->meta[pidx].state == PAGE_COMPRESSING) {
                    s->meta[pidx].state = PAGE_HOT;
                    s->meta[pidx].cooldown = 1;
                    hot_list_add(s, (uint32_t)pidx);
                }
                mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ,PAGE_SZ,PROT_READ|PROT_WRITE);
                continue;
            }
            if (!commit_compressed_page(s, pidx, page_seq[i], s->tmp_src + i * PAGE_SZ)) {
                s->meta[pidx].pool_offset=0;
                s->meta[pidx].codec=0;
                s->meta[pidx].comp_size=0;
                pool_free_insert_locked(s, off, cs);
                if(s->pool_used >= cs) __sync_fetch_and_sub(&s->pool_used, cs);
                if (s->meta[pidx].state == PAGE_COMPRESSING) {
                    s->meta[pidx].state = PAGE_HOT;
                    s->meta[pidx].cooldown = 6;
                    hot_list_add(s, (uint32_t)pidx);
                }
                mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ,PAGE_SZ,PROT_READ|PROT_WRITE);
                continue;
            }
            if (s->live_resident_pages) __sync_fetch_and_sub(&s->live_resident_pages, 1);
            __sync_fetch_and_add(&s->live_compressed_pages, 1);
            s->meta[pidx].preferred_codec = page_codec[i] ? page_codec[i] : s->meta[pidx].preferred_codec;
            s->meta[pidx].codec_fail_streak = 0;
            if (s->meta[pidx].state == PAGE_COMPRESSED) s->meta[pidx].dirty = 0;
            note_page_compressed(s, pidx, page_codec[i], cs);
            if (s->meta[pidx].state == PAGE_COMPRESSED) page_release_physical(s, pidx);
            for(int probe=0; probe<8; probe++) {
                uint32_t s2 = (slot + probe) & DEDUP_HT_MASK;
                if (s->dedup_hash[s2] == 0 || s->dedup_ref[s2] == 0) {
                    s->dedup_hash[s2] = h;
                    s->dedup_off[s2] = off;
                    s->dedup_sz[s2] = cs;
                    __sync_lock_test_and_set(&s->dedup_ref[s2], 1);  // atomic write
                    if (s->dedup_rev && s->dedup_rev_size) {
                        s->dedup_rev[(uint32_t)(off / PAGE_SZ) & s->dedup_rev_mask] = s2;
                    }
                    break;
                }
            }
        }
        pthread_mutex_unlock(&s->alloc_mutex);
        {
            uint32_t qdepth = (s->async_pf_head >= s->async_pf_tail)
                ? (s->async_pf_head - s->async_pf_tail)
                : ((s->async_pf_head + ASYNC_PF_Q_SIZE) - s->async_pf_tail);
            long ns = (nc >= (BATCH/2) || qdepth > 8) ? 50000L : (nc > 0 ? 150000L : 400000L);
            struct timespec ts={0,ns}; nanosleep(&ts,NULL);
        }
    }
    return NULL;
}

// ─── Address check ───
static inline int is_ours(void *ptr) {
    if(__builtin_expect(!g_z||!g_z->vmem,0)) return 0;
    uintptr_t a=(uintptr_t)ptr;
    return a>=(uintptr_t)g_z->vmem && a<(uintptr_t)g_z->vmem+g_z->vmem_size;
}

static int tensor_desc_is_valid(const memx_runtime_tensor_desc_t *desc) {
    if (!desc) return 1;
    if (desc->struct_size != 0 && desc->struct_size < offsetof(memx_runtime_tensor_desc_t, reserved)) return 0;
    if (desc->role > MEMX_TENSOR_ROLE_TEMPORARY) return 0;
    if (desc->dtype > MEMX_TENSOR_DTYPE_INT32) return 0;
    if (desc->layout > MEMX_TENSOR_LAYOUT_INTERLEAVED) return 0;
    if (desc->rank > 4) return 0;
    return 1;
}

static void note_page_compressed(MemXZone3 *s, size_t page_index, uint8_t codec, uint32_t comp_size) {
    uint64_t saved = PAGE_SZ - comp_size;
    __sync_fetch_and_add(&s->compressions, 1);
    __sync_fetch_and_add(&s->bytes_saved, saved);
    if (codec == MEMX_CODEC_TENSOR_FP16_SPLIT ||
        codec == MEMX_CODEC_TENSOR_FP16_DELTA_SPLIT ||
        codec == MEMX_CODEC_TENSOR_BITPLANE16 ||
        codec == MEMX_CODEC_TENSOR_SPARSE_BYTE ||
        codec == MEMX_CODEC_ZLIB ||
        codec == MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT ||
        codec == MEMX_CODEC_TENSOR_EXP_PACK) {
        __sync_fetch_and_add(&s->tensor_codec_pages, 1);
        __sync_fetch_and_add(&s->tensor_codec_bytes_saved, saved);
    }
    if (codec == MEMX_CODEC_TENSOR_FP16_SPLIT ||
        codec == MEMX_CODEC_TENSOR_FP16_DELTA_SPLIT ||
        codec == MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT) {
        __sync_fetch_and_add(&s->tensor_split_pages, 1);
        __sync_fetch_and_add(&s->tensor_split_bytes_saved, saved);
        if (codec == MEMX_CODEC_TENSOR_FP16_DELTA_SPLIT) {
            __sync_fetch_and_add(&s->tensor_delta_split_pages, 1);
            __sync_fetch_and_add(&s->tensor_delta_split_bytes_saved, saved);
        }
    } else if (codec == MEMX_CODEC_TENSOR_EXP_PACK) {
        __sync_fetch_and_add(&s->tensor_exp_pack_pages, 1);
        __sync_fetch_and_add(&s->tensor_exp_pack_bytes_saved, saved);
    } else if (codec == MEMX_CODEC_TENSOR_BITPLANE16) {
        __sync_fetch_and_add(&s->tensor_bitplane_pages, 1);
        __sync_fetch_and_add(&s->tensor_bitplane_bytes_saved, saved);
    } else if (codec == MEMX_CODEC_TENSOR_SPARSE_BYTE ||
        codec == MEMX_CODEC_ZLIB) {
        __sync_fetch_and_add(&s->tensor_sparse_pages, 1);
        __sync_fetch_and_add(&s->tensor_sparse_bytes_saved, saved);
    }
    if (s->meta[page_index].tensor_role == MEMX_TENSOR_ROLE_WEIGHT) {
        __sync_fetch_and_add(&s->weight_compressed_pages, 1);
        __sync_fetch_and_add(&s->weight_bytes_saved, saved);
    } else if (s->meta[page_index].tensor_role == MEMX_TENSOR_ROLE_KV_CACHE) {
        __sync_fetch_and_add(&s->kv_cache_compressed_pages, 1);
        __sync_fetch_and_add(&s->kv_cache_bytes_saved, saved);
    }
}

// ─── Explicit runtime allocation helpers ───
static void init_memx(void);
static void *memx_alloc_internal(size_t size, uintptr_t owner_tag, int force_managed, int allow_fallback,
                                 size_t quota_credit, const memx_runtime_tensor_desc_t *desc) {
    if (size == 0) size = 1;
    if (!tensor_desc_is_valid(desc)) {
        errno = EINVAL;
        return NULL;
    }
    if (in_memx) return allow_fallback ? real_malloc(size) : NULL;
    if (!g_z && (force_managed || size >= LARGE_THRESHOLD)) init_memx();
    if (!g_z || !g_z->running) {
        if (allow_fallback) return real_malloc(size);
        errno = ENOMEM;
        return NULL;
    }
    if (!force_managed && size < LARGE_THRESHOLD) {
        return allow_fallback ? real_malloc(size) : NULL;
    }
    
    in_memx = 1;
    
    // Allocate from our compressed pool (no size header needed — stored in meta)
    pthread_mutex_lock(&g_z->alloc_mutex);
    size_t alloc_size = ((size + PAGE_SZ - 1) / PAGE_SZ) * PAGE_SZ;
    size_t npages = alloc_size / PAGE_SZ;
    if (context_preflight_locked(owner_tag, size, npages, force_managed, quota_credit) != 0) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        in_memx = 0;
        if (allow_fallback) return real_malloc(size);
        return NULL;
    }
    size_t hint = g_z->vmem_next / PAGE_SZ;
    ssize_t found_s = bm_find_free_run(g_z, npages, hint);
    if (found_s < 0) {
        memx_runtime_context_t *ctx = context_from_tag(owner_tag);
        if (ctx) __sync_fetch_and_add(&ctx->pressure_events, 1);
        pthread_mutex_unlock(&g_z->alloc_mutex);
        in_memx = 0;
        if (allow_fallback) return real_malloc(size);
        errno = ENOMEM;
        return NULL;
    }
    size_t found = (size_t)found_s;
    
    void *result = (uint8_t*)g_z->vmem + found * PAGE_SZ;
    g_z->vmem_next = (found + npages) * PAGE_SZ;
    uint16_t tensor_role = desc ? (uint16_t)desc->role : MEMX_TENSOR_ROLE_UNKNOWN;
    uint16_t tensor_dtype = desc ? (uint16_t)desc->dtype : MEMX_TENSOR_DTYPE_UNKNOWN;
    uint16_t tensor_layout = desc ? (uint16_t)desc->layout : MEMX_TENSOR_LAYOUT_UNKNOWN;
    uint32_t tensor_flags = desc ? desc->flags : 0;
    uint32_t tensor_layer = desc ? desc->layer_index : 0;
    uint32_t tensor_head = desc ? desc->head_index : 0;
    uint8_t default_pref = 0;
    if (desc) {
        if (tensor_role == MEMX_TENSOR_ROLE_KV_CACHE) {
            if ((tensor_flags & (MEMX_TENSOR_FLAG_HOT | MEMX_TENSOR_FLAG_NO_COMPRESS | MEMX_TENSOR_FLAG_COLD)) == 0)
                tensor_flags |= MEMX_TENSOR_FLAG_SEQUENTIAL;
            if (tensor_dtype == MEMX_TENSOR_DTYPE_FP16 || tensor_dtype == MEMX_TENSOR_DTYPE_BF16)
                default_pref = MEMX_CODEC_TENSOR_FP16_DELTA_SPLIT;
        } else if (tensor_role == MEMX_TENSOR_ROLE_WEIGHT || tensor_role == MEMX_TENSOR_ROLE_EMBEDDING) {
            if ((tensor_flags & MEMX_TENSOR_FLAG_HOT) == 0)
                tensor_flags |= MEMX_TENSOR_FLAG_READ_MOSTLY;
            if (tensor_dtype == MEMX_TENSOR_DTYPE_FP16 || tensor_dtype == MEMX_TENSOR_DTYPE_BF16)
                default_pref = MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT;
        } else if (tensor_role == MEMX_TENSOR_ROLE_ACTIVATION) {
            if ((tensor_flags & (MEMX_TENSOR_FLAG_HOT | MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_NO_COMPRESS)) == 0)
                tensor_flags |= MEMX_TENSOR_FLAG_HOT;
            if (tensor_dtype == MEMX_TENSOR_DTYPE_FP16 || tensor_dtype == MEMX_TENSOR_DTYPE_BF16)
                default_pref = MEMX_CODEC_TENSOR_FP16_SPLIT;
        }
    }
    for (size_t i=found; i<found+npages; i++) {
        g_z->meta[i].state = PAGE_RESIDENT;
        g_z->meta[i].codec = 0;
        g_z->meta[i].comp_size = 0;
        g_z->meta[i].pool_offset = 0;
        g_z->meta[i].tensor_role = tensor_role;
        g_z->meta[i].tensor_dtype = tensor_dtype;
        g_z->meta[i].tensor_layout = tensor_layout;
        g_z->meta[i].tensor_flags = tensor_flags;
        g_z->meta[i].tensor_layer = tensor_layer;
        g_z->meta[i].tensor_head = tensor_head;
        g_z->meta[i].owner_tag = owner_tag;
        g_z->meta[i].alloc_size = 0;
        g_z->meta[i].prefetched = 0;
        g_z->meta[i].cooldown = 0;
        g_z->meta[i].preferred_codec = default_pref;
        g_z->meta[i].codec_fail_streak = 0;
        g_z->meta[i].stable_ticks = 0;
        g_z->meta[i].dirty = 1;
        bm_set_used(g_z, i);
        res_list_add(g_z, i);
        if (tensor_flags & MEMX_TENSOR_FLAG_HOT) __sync_fetch_and_add(&g_z->live_hot_flag_pages, 1);
        if (tensor_flags & MEMX_TENSOR_FLAG_NO_COMPRESS) __sync_fetch_and_add(&g_z->live_nocomp_flag_pages, 1);
    }
    __sync_fetch_and_add(&g_z->live_resident_pages, npages);
    mprotect(result, npages * PAGE_SZ, PROT_READ|PROT_WRITE);
    g_z->meta[found].alloc_size = size;
    pthread_mutex_unlock(&g_z->alloc_mutex);
    
    context_note_alloc(owner_tag, size);
    if (desc) {
        memx_runtime_tensor_desc_t effective = *desc;
        effective.flags = tensor_flags;
        context_note_tensor_alloc(owner_tag, size, &effective);
    }
    in_memx = 0;
    return result;
}

static void runtime_managed_free_internal(void *ptr) {
    if (!ptr) return;
    if (!g_z || !is_ours(ptr)) {
        real_free(ptr);
        return;
    }
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    size_t size = g_z->meta[sp].alloc_size;
    uintptr_t owner_tag = g_z->meta[sp].owner_tag;
    uint32_t tensor_role = g_z->meta[sp].tensor_role;
    size_t alloc_size = ((size + PAGE_SZ - 1) / PAGE_SZ) * PAGE_SZ;
    size_t np = alloc_size / PAGE_SZ;
    mprotect((uint8_t*)g_z->vmem + sp * PAGE_SZ, np * PAGE_SZ, PROT_READ | PROT_WRITE);
    pthread_mutex_lock(&g_z->alloc_mutex);
    uint64_t hot_bytes = 0;
    uint64_t no_compress_bytes = 0;
    for (size_t i = sp; i < sp + np && i < g_z->npages; i++) {
        size_t rel_page = i - sp;
        size_t page_bytes = allocation_page_bytes(size, rel_page);
        if (g_z->meta[i].tensor_flags & MEMX_TENSOR_FLAG_HOT) {
            hot_bytes += page_bytes;
            if (g_z->live_hot_flag_pages) __sync_fetch_and_sub(&g_z->live_hot_flag_pages, 1);
        }
        if (g_z->meta[i].tensor_flags & MEMX_TENSOR_FLAG_NO_COMPRESS) {
            no_compress_bytes += page_bytes;
            if (g_z->live_nocomp_flag_pages) __sync_fetch_and_sub(&g_z->live_nocomp_flag_pages, 1);
        }
        if (g_z->meta[i].state == PAGE_COMPRESSED) {
            if (g_z->live_compressed_pages) __sync_fetch_and_sub(&g_z->live_compressed_pages, 1);
            if (g_z->meta[i].comp_size > 0) dedup_decref(g_z, g_z->meta[i].pool_offset, g_z->meta[i].comp_size);
        } else if (g_z->meta[i].state == PAGE_RESIDENT || g_z->meta[i].state == PAGE_HOT || g_z->meta[i].state == PAGE_COMPRESSING) {
            if (g_z->live_resident_pages) __sync_fetch_and_sub(&g_z->live_resident_pages, 1);
        }
        g_z->meta[i].state = PAGE_NONE;
        g_z->meta[i].codec = 0;
        g_z->meta[i].comp_size = 0;
        g_z->meta[i].pool_offset = 0;
        g_z->meta[i].owner_tag = 0;
        g_z->meta[i].tensor_role = MEMX_TENSOR_ROLE_UNKNOWN;
        g_z->meta[i].tensor_dtype = MEMX_TENSOR_DTYPE_UNKNOWN;
        g_z->meta[i].tensor_layout = MEMX_TENSOR_LAYOUT_UNKNOWN;
        g_z->meta[i].tensor_flags = 0;
        g_z->meta[i].tensor_layer = 0;
        g_z->meta[i].tensor_head = 0;
        bm_set_free(g_z, i);
    }
    memx_runtime_reclaim_locked(g_z);
    pthread_mutex_unlock(&g_z->alloc_mutex);
    mprotect((uint8_t*)g_z->vmem + sp * PAGE_SZ, np * PAGE_SZ, PROT_NONE);
    context_note_tensor_free(owner_tag, size, tensor_role, hot_bytes, no_compress_bytes);
    context_note_free(owner_tag, size);
}

static void runtime_free_internal(void *ptr) {
    runtime_managed_free_internal(ptr);
}

static void *memx_calloc_internal(size_t nmemb, size_t size, uintptr_t owner_tag, int force_managed, int allow_fallback) {
    if (nmemb != 0 && size > SIZE_MAX / nmemb) {
        errno = ENOMEM;
        return NULL;
    }
    size_t total = nmemb * size;
    // Recursion guard
    if (in_memx) return allow_fallback ? real_calloc(nmemb, size) : NULL;
    void *ptr = memx_alloc_internal(total, owner_tag, force_managed, allow_fallback, 0, NULL);
    if (ptr && is_ours(ptr)) { /* pages zero-filled on fault */ }
    else if (ptr) memset(ptr, 0, total);
    return ptr;
}

static void *memx_realloc_internal(void *ptr, size_t size, uintptr_t requested_owner_tag, int force_managed, int allow_fallback) {
    if (!ptr) return memx_alloc_internal(size, requested_owner_tag, force_managed, allow_fallback, 0, NULL);
    if (size == 0) { memx_runtime_free(ptr); return NULL; }
    uintptr_t owner_tag = requested_owner_tag;
    // Fast path: if new size is small, always use system realloc
    if (!force_managed && __builtin_expect(size < LARGE_THRESHOLD, 1)) {
        if (!g_z || !is_ours(ptr)) return real_realloc(ptr, size);
        // Our allocation shrinking to small: copy out and free
        size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
        size_t old_size = g_z->meta[sp].alloc_size;
        owner_tag = g_z->meta[sp].owner_tag;
        void *new_ptr = real_malloc(size);
        if (!new_ptr) return NULL;
        memcpy(new_ptr, ptr, old_size < size ? old_size : size);
        real_free(ptr);
        context_note_alloc(owner_tag, size);
        context_note_free(owner_tag, old_size);
        return new_ptr;
    }
    if (!g_z || !is_ours(ptr)) {
        if (!allow_fallback) errno = EINVAL;
        return allow_fallback ? real_realloc(ptr, size) : NULL;
    }
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    size_t old_size = g_z->meta[sp].alloc_size;
    if (owner_tag == 0) owner_tag = g_z->meta[sp].owner_tag;
    if (requested_owner_tag && requested_owner_tag != g_z->meta[sp].owner_tag) {
        errno = EINVAL;
        return NULL;
    }
    void *new_ptr = memx_alloc_internal(size, owner_tag, force_managed, allow_fallback, old_size, NULL);
    if (!new_ptr) return NULL;
    size_t copy = old_size < size ? old_size : size;
    memcpy(new_ptr, ptr, copy);
    real_free(ptr);
    return new_ptr;
}

// ─── Init ───

static void init_memx(void) {
    static pthread_mutex_t init_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&init_mutex);
    if (g_z) { pthread_mutex_unlock(&init_mutex); return; }
    in_memx = 1;  // Prevent recursion during Metal init
    
    g_z = (MemXZone3*)mmap(NULL, sizeof(MemXZone3), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (g_z == MAP_FAILED) { g_z = NULL; pthread_mutex_unlock(&init_mutex); return; }
    memset(g_z, 0, sizeof(MemXZone3));
    
    // GPU
    g_z->device = MTLCreateSystemDefaultDevice();
    if (!g_z->device) { munmap(g_z, sizeof(MemXZone3)); g_z = NULL; pthread_mutex_unlock(&init_mutex); return; }
    NSError *err = nil;
    id<MTLLibrary> lib = [g_z->device newLibraryWithSource:shader_src options:nil error:&err];
    if (!lib) { munmap(g_z, sizeof(MemXZone3)); g_z = NULL; pthread_mutex_unlock(&init_mutex); return; }
    g_z->queue = [g_z->device newCommandQueue];
    g_z->comp_pipe = [g_z->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"cp"] error:&err];
    g_z->decomp_pipe = [g_z->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dp"] error:&err];
    if (!g_z->comp_pipe || !g_z->decomp_pipe) { munmap(g_z, sizeof(MemXZone3)); g_z = NULL; pthread_mutex_unlock(&init_mutex); return; }
    
    // Pre-allocate persistent GPU + temp buffers (512 pages = 8MB each)
    // Larger batch = better GPU utilization, fewer dispatches
    g_z->batch_cap = 512;
    size_t batch_bytes = g_z->batch_cap * PAGE_SZ;
    g_z->gpu_sb = [g_z->device newBufferWithLength:batch_bytes options:MTLResourceStorageModeShared];
    g_z->gpu_db = [g_z->device newBufferWithLength:batch_bytes options:MTLResourceStorageModeShared];
    g_z->gpu_zb = [g_z->device newBufferWithLength:g_z->batch_cap*4 options:MTLResourceStorageModeShared];
    g_z->tmp_src = (uint8_t*)mmap(NULL, batch_bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    g_z->tmp_dst = (uint8_t*)mmap(NULL, batch_bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    g_z->tmp_sz  = (uint32_t*)mmap(NULL, g_z->batch_cap*4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    // Dedup hash tables
    g_z->dedup_hash = (uint64_t*)mmap(NULL, DEDUP_HT_SIZE*8, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    g_z->dedup_off  = (uint64_t*)mmap(NULL, DEDUP_HT_SIZE*8, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    g_z->dedup_sz   = (uint32_t*)mmap(NULL, DEDUP_HT_SIZE*4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    g_z->dedup_ref  = (uint32_t*)mmap(NULL, DEDUP_HT_SIZE*4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    g_z->dedup_pending_free = (uint8_t*)mmap(NULL, DEDUP_HT_SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    g_z->dedup_rev = NULL;
    g_z->dedup_rev_size = 0;
    g_z->dedup_rev_mask = 0;
    g_z->pool_free_cap = POOL_FREE_EXTENTS_MAX;
    g_z->pool_free_off = (uint64_t*)mmap(NULL, g_z->pool_free_cap * sizeof(uint64_t), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    g_z->pool_free_sz = (uint32_t*)mmap(NULL, g_z->pool_free_cap * sizeof(uint32_t), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(g_z->dedup_hash, 0, DEDUP_HT_SIZE*8);
    memset(g_z->dedup_ref, 0, DEDUP_HT_SIZE*4);
    memset(g_z->dedup_pending_free, 0, DEDUP_HT_SIZE);
    // Active page tracking lists (compact: only track active pages, not all 6M)
    g_z->hot_cap = 131072;   // max HOT pages tracked (512KB)
    g_z->res_cap = 4194304;  // max RESIDENT pages tracked (16MB, covers 64GB of pages)
    g_z->hot_list = (uint32_t*)mmap(NULL, g_z->hot_cap * 4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    g_z->res_list = (uint32_t*)mmap(NULL, g_z->res_cap * 4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    g_z->hot_count = 0;
    g_z->res_count = 0;
    if (!g_z->gpu_sb||!g_z->gpu_db||!g_z->gpu_zb||!g_z->tmp_src||!g_z->tmp_dst||!g_z->tmp_sz||
        !g_z->dedup_hash||!g_z->dedup_pending_free||!g_z->pool_free_off||!g_z->pool_free_sz||
        !g_z->hot_list||!g_z->res_list) {
        munmap(g_z, sizeof(MemXZone3)); g_z = NULL; pthread_mutex_unlock(&init_mutex); return;
    }
    
    // Virtual memory
    int64_t ms=0; size_t len=sizeof(ms);
    sysctlbyname("hw.memsize", &ms, &len, NULL, 0);
    g_z->vmem_size = ms * 4;
    g_z->npages = g_z->vmem_size / PAGE_SZ;
    g_z->vmem = mmap(NULL, g_z->vmem_size, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (g_z->vmem == MAP_FAILED) { munmap(g_z, sizeof(MemXZone3)); g_z = NULL; pthread_mutex_unlock(&init_mutex); return; }
    g_z->pool_size = g_z->vmem_size / 2;
    g_z->pool = mmap(NULL, g_z->pool_size, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (g_z->pool == MAP_FAILED) { munmap(g_z->vmem, g_z->vmem_size); munmap(g_z, sizeof(MemXZone3)); g_z = NULL; pthread_mutex_unlock(&init_mutex); return; }
    {
        uint64_t pool_pages = (g_z->pool_size + PAGE_SZ - 1) / PAGE_SZ;
        uint64_t rev = 1;
        while (rev < pool_pages) rev <<= 1;
        if (rev < 8192) rev = 8192;
        if (rev > (1ull << 26)) rev = (1ull << 26);
        g_z->dedup_rev_size = (uint32_t)rev;
        g_z->dedup_rev_mask = g_z->dedup_rev_size - 1;
        g_z->dedup_rev = (uint32_t*)mmap(NULL, (size_t)g_z->dedup_rev_size * 4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (g_z->dedup_rev == MAP_FAILED) {
            munmap(g_z->pool, g_z->pool_size);
            munmap(g_z->vmem, g_z->vmem_size);
            munmap(g_z, sizeof(MemXZone3));
            g_z = NULL;
            pthread_mutex_unlock(&init_mutex);
            return;
        }
        memset(g_z->dedup_rev, 0xFF, (size_t)g_z->dedup_rev_size * 4);
    }
    g_z->last_fault_page = (size_t)-1;
    g_z->last_fault_stride = 1;
    g_z->last_fault_role = MEMX_TENSOR_ROLE_UNKNOWN;
    for (int si = 0; si < FAULT_STREAMS; si++) {
        g_z->stream_fault_page[si] = (size_t)-1;
        g_z->stream_fault_stride[si] = 1;
    }
    size_t meta_sz = g_z->npages * sizeof(PageMeta);
    g_z->meta = (PageMeta*)mmap(NULL, meta_sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (g_z->meta == MAP_FAILED) { munmap(g_z->pool, g_z->pool_size); munmap(g_z->vmem, g_z->vmem_size); munmap(g_z, sizeof(MemXZone3)); g_z = NULL; pthread_mutex_unlock(&init_mutex); return; }
    
    // Free page bitmap: all pages start as free (1=free, 0=used)
    g_z->free_bm_size = (g_z->npages + 63) / 64;
    size_t bm_sz = g_z->free_bm_size * sizeof(uint64_t);
    g_z->free_bm = (uint64_t*)mmap(NULL, bm_sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (g_z->free_bm == MAP_FAILED) { munmap(g_z->meta, meta_sz); munmap(g_z->pool, g_z->pool_size); munmap(g_z->vmem, g_z->vmem_size); munmap(g_z, sizeof(MemXZone3)); g_z = NULL; pthread_mutex_unlock(&init_mutex); return; }
    memset(g_z->free_bm, 0xFF, bm_sz);  // All pages free initially
    g_z->free_pages_count = g_z->npages;
    
    // Signal handler with alternate stack (decompressor uses 16KB on stack)
    stack_t ss;
    ss.ss_sp = mmap(NULL, SIGSTKSZ*16, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    ss.ss_size = SIGSTKSZ*16;
    ss.ss_flags = 0;
    sigaltstack(&ss, NULL);
    struct sigaction sa; memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = fault_handler; sa.sa_flags = SA_SIGINFO|SA_NODEFER|SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old_segv); sigaction(SIGBUS, &sa, &old_bus);
    g_z->attached = 1;

    pthread_mutex_init(&g_z->alloc_mutex, NULL);
    
    // Background compressor
    g_z->running = 1;
    g_z->async_pf_q = (uint32_t*)mmap(NULL, ASYNC_PF_Q_SIZE * 4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (g_z->async_pf_q == MAP_FAILED) g_z->async_pf_q = NULL;
    g_z->async_pf_head = 0;
    g_z->async_pf_tail = 0;
    g_z->async_pf_enqueued = 0;
    g_z->async_pf_completed = 0;
    g_z->live_compressed_pages = 0;
    g_z->live_resident_pages = 0;
    g_z->live_hot_flag_pages = 0;
    g_z->live_nocomp_flag_pages = 0;
    pthread_mutex_init(&g_z->async_pf_mutex, NULL);
    pthread_cond_init(&g_z->async_pf_cond, NULL);
    g_z->async_pf_running = 1;
    g_z->async_pf_nworkers = 0;
    g_z->encode_nworkers = 0;
    g_z->encode_pool_running = 0;
    g_z->encode_job = NULL;
    g_z->encode_workers_active = 0;
    pthread_mutex_init(&g_z->encode_mutex, NULL);
    pthread_cond_init(&g_z->encode_cond, NULL);
    pthread_cond_init(&g_z->encode_done_cond, NULL);
    {
        int ncpu = 0;
        size_t len = sizeof(ncpu);
        if (sysctlbyname("hw.activecpu", &ncpu, &len, NULL, 0) != 0 || ncpu < 2)
            ncpu = 4;
        int want = ncpu - 1;
        if (want < 2) want = 2;
        if (want > COMP_CPU_WORKERS) want = COMP_CPU_WORKERS;
        g_z->encode_pool_running = 1;
        for (int wi = 0; wi < want; wi++) {
            if (pthread_create(&g_z->encode_threads[wi], NULL, tensor_encode_pool_worker, g_z) == 0)
                g_z->encode_nworkers++;
            else break;
        }
        if (g_z->encode_nworkers == 0) g_z->encode_pool_running = 0;
    }
    pthread_create(&g_z->bg_thread, NULL, bg_compressor, g_z);
    if (g_z->async_pf_q) {
        for (int wi = 0; wi < ASYNC_PF_WORKERS; wi++) {
            if (pthread_create(&g_z->async_pf_threads[wi], NULL, async_pf_worker, g_z) == 0)
                g_z->async_pf_nworkers++;
            else break;
        }
        if (g_z->async_pf_nworkers == 0) g_z->async_pf_running = 0;
    } else {
        g_z->async_pf_running = 0;
    }
    g_z->async_seal_q = (async_seal_job_t *)mmap(NULL, ASYNC_SEAL_Q_SIZE * sizeof(async_seal_job_t), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (g_z->async_seal_q == MAP_FAILED) g_z->async_seal_q = NULL;
    g_z->async_seal_head = 0;
    g_z->async_seal_tail = 0;
    g_z->async_seal_enqueued = 0;
    g_z->async_seal_completed = 0;
    g_z->async_seal_active = 0;
    g_z->async_seal_nworkers = 0;
    pthread_mutex_init(&g_z->async_seal_mutex, NULL);
    pthread_cond_init(&g_z->async_seal_cond, NULL);
    pthread_cond_init(&g_z->async_seal_idle_cond, NULL);
    g_z->async_seal_running = 1;
    if (g_z->async_seal_q) {
        for (int wi = 0; wi < ASYNC_SEAL_WORKERS; wi++) {
            if (pthread_create(&g_z->async_seal_threads[wi], NULL, async_seal_worker, g_z) == 0)
                g_z->async_seal_nworkers++;
            else break;
        }
        if (g_z->async_seal_nworkers == 0) g_z->async_seal_running = 0;
    } else {
        g_z->async_seal_running = 0;
    }
    shared_stats_init(g_z);
    
    fprintf(stderr, "[memx] ✅ GPU memory expansion active (%llu MB virtual, %s)\n",
            (unsigned long long)(g_z->vmem_size / MB), memx_mode_label());
    in_memx = 0;  // Allow this thread to use our pool
    pthread_mutex_unlock(&init_mutex);
}

static void fini_memx(void) {
    if (!g_z) return;
    shared_stats_update(g_z);
    g_z->running = 0;
    g_z->async_pf_running = 0;
    g_z->async_seal_running = 0;
    if (g_z->async_seal_q) {
        pthread_mutex_lock(&g_z->async_seal_mutex);
        pthread_cond_broadcast(&g_z->async_seal_cond);
        pthread_cond_broadcast(&g_z->async_seal_idle_cond);
        pthread_mutex_unlock(&g_z->async_seal_mutex);
        for (int wi = 0; wi < g_z->async_seal_nworkers; wi++)
            pthread_join(g_z->async_seal_threads[wi], NULL);
        g_z->async_seal_nworkers = 0;
        pthread_mutex_destroy(&g_z->async_seal_mutex);
        pthread_cond_destroy(&g_z->async_seal_cond);
        pthread_cond_destroy(&g_z->async_seal_idle_cond);
        munmap(g_z->async_seal_q, ASYNC_SEAL_Q_SIZE * sizeof(async_seal_job_t));
        g_z->async_seal_q = NULL;
    }
    if (g_z->encode_pool_running) {
        pthread_mutex_lock(&g_z->encode_mutex);
        g_z->encode_pool_running = 0;
        g_z->encode_job = NULL;
        pthread_cond_broadcast(&g_z->encode_cond);
        pthread_cond_broadcast(&g_z->encode_done_cond);
        pthread_mutex_unlock(&g_z->encode_mutex);
        for (int wi = 0; wi < g_z->encode_nworkers; wi++)
            pthread_join(g_z->encode_threads[wi], NULL);
        g_z->encode_nworkers = 0;
        pthread_mutex_destroy(&g_z->encode_mutex);
        pthread_cond_destroy(&g_z->encode_cond);
        pthread_cond_destroy(&g_z->encode_done_cond);
    }
    if (g_z->async_pf_q) {
        pthread_mutex_lock(&g_z->async_pf_mutex);
        pthread_cond_broadcast(&g_z->async_pf_cond);
        pthread_mutex_unlock(&g_z->async_pf_mutex);
        for (int wi = 0; wi < g_z->async_pf_nworkers; wi++)
            pthread_join(g_z->async_pf_threads[wi], NULL);
    }
    pthread_join(g_z->bg_thread, NULL);
    pthread_mutex_destroy(&g_z->async_pf_mutex);
    pthread_cond_destroy(&g_z->async_pf_cond);
    if (g_z->async_pf_q) munmap(g_z->async_pf_q, ASYNC_PF_Q_SIZE * 4);
    // Restore signal handlers FIRST to prevent faults during cleanup
    if (g_z->attached) { sigaction(SIGSEGV, &old_segv, NULL); sigaction(SIGBUS, &old_bus, NULL); g_z->attached = 0; }
    for (size_t i=0; i<g_z->npages; i++)
        if (g_z->meta[i].state==PAGE_COMPRESSED||g_z->meta[i].state==PAGE_NONE)
            mprotect((uint8_t*)g_z->vmem+i*PAGE_SZ, PAGE_SZ, PROT_READ|PROT_WRITE);
    // Release persistent GPU buffers (ObjC release may trigger malloc/free)
    g_z->gpu_sb = nil; g_z->gpu_db = nil; g_z->gpu_zb = nil;
    size_t batch_bytes = g_z->batch_cap * PAGE_SZ;
    munmap(g_z->tmp_src, batch_bytes);
    munmap(g_z->tmp_dst, batch_bytes);
    munmap(g_z->tmp_sz, g_z->batch_cap * 4);
    munmap(g_z->vmem, g_z->vmem_size);
    munmap(g_z->pool, g_z->pool_size);
    munmap(g_z->meta, g_z->npages * sizeof(PageMeta));
    munmap(g_z->free_bm, g_z->free_bm_size * sizeof(uint64_t));
    munmap(g_z->hot_list, g_z->hot_cap * 4);
    munmap(g_z->res_list, g_z->res_cap * 4);
    munmap(g_z->dedup_hash, DEDUP_HT_SIZE*8);
    munmap(g_z->dedup_off, DEDUP_HT_SIZE*8);
    munmap(g_z->dedup_sz, DEDUP_HT_SIZE*4);
    munmap(g_z->dedup_ref, DEDUP_HT_SIZE*4);
    munmap(g_z->dedup_pending_free, DEDUP_HT_SIZE);
    if (g_z->dedup_rev) munmap(g_z->dedup_rev, (size_t)g_z->dedup_rev_size * 4);
    munmap(g_z->pool_free_off, g_z->pool_free_cap * sizeof(uint64_t));
    munmap(g_z->pool_free_sz, g_z->pool_free_cap * sizeof(uint32_t));
    shared_stats_cleanup();
    MemXZone3 *old_z = g_z;
    g_z = NULL;  // Prevent concurrent access before munmap
    fprintf(stderr, "[memx] ✅ Shutdown. Compressed %llu pages, saved %llu MB, resolved %llu faults, dedup %llu hits, prefetch %llu (hits %llu)\n",
            (unsigned long long)old_z->compressions, (unsigned long long)(old_z->bytes_saved/MB), (unsigned long long)old_z->faults, (unsigned long long)old_z->dedup_hits, (unsigned long long)old_z->prefetch_count, (unsigned long long)old_z->prefetch_hits);
    munmap(old_z, sizeof(MemXZone3));
}

// ─── mmap interposition ───
// macOS malloc uses mmap directly for large allocations, bypassing zone->malloc.
// We intercept mmap for large anonymous private mappings and route to our pool.
// CRITICAL: use volatile static flag (not TLS) for early-safety check,
// since TLS may not be initialized when dyld calls mmap very early.

// Hash-based allocation table for mmap-routed allocations (O(1) lookup)
#define MMAP_TABLE_MAX 8192
#define MMAP_HT_MASK 8191
static struct { void *base; size_t npages; size_t requested_size; uintptr_t owner_tag; } mmap_table[MMAP_TABLE_MAX];
static int mmap_table_count = 0;

static inline uint32_t mmap_ht_hash(void *ptr) {
    uintptr_t k = (uintptr_t)ptr;
    k = ((k >> 16) ^ k) * 0x45d9f3b;
    k = ((k >> 16) ^ k) * 0x45d9f3b;
    k = (k >> 16) ^ k;
    return (uint32_t)(k & MMAP_HT_MASK);
}

static inline void mmap_table_insert(void *base, size_t npages, size_t requested_size, uintptr_t owner_tag) {
    uint32_t slot = mmap_ht_hash(base);
    for (int probe = 0; probe < 8; probe++) {
        uint32_t idx = (slot + probe) & MMAP_HT_MASK;
        if (mmap_table[idx].base == NULL) {
            mmap_table[idx].base = base;
            mmap_table[idx].npages = npages;
            mmap_table[idx].requested_size = requested_size;
            mmap_table[idx].owner_tag = owner_tag;
            mmap_table_count++;
            return;
        }
    }
    // Fallback: find any empty slot
    for (uint32_t idx = 0; idx < MMAP_TABLE_MAX; idx++) {
        if (mmap_table[idx].base == NULL) {
            mmap_table[idx].base = base;
            mmap_table[idx].npages = npages;
            mmap_table[idx].requested_size = requested_size;
            mmap_table[idx].owner_tag = owner_tag;
            mmap_table_count++;
            return;
        }
    }
}

static inline int mmap_table_find_and_remove(void *base, size_t *out_npages, size_t *out_requested_size, uintptr_t *out_owner_tag) {
    uint32_t slot = mmap_ht_hash(base);
    for (int probe = 0; probe < 8; probe++) {
        uint32_t idx = (slot + probe) & MMAP_HT_MASK;
        if (mmap_table[idx].base == base) {
            *out_npages = mmap_table[idx].npages;
            if (out_requested_size) *out_requested_size = mmap_table[idx].requested_size;
            if (out_owner_tag) *out_owner_tag = mmap_table[idx].owner_tag;
            mmap_table[idx].base = NULL;
            mmap_table[idx].npages = 0;
            mmap_table[idx].requested_size = 0;
            mmap_table[idx].owner_tag = 0;
            mmap_table_count--;
            return 1;
        }
        if (mmap_table[idx].base == NULL) break;
    }
    return 0;
}

static void *memx_mmap_internal(void *addr, size_t length, int prot, int flags, int fd, off_t offset,
                                uintptr_t owner_tag, int require_managed) {
    if (!g_z || !g_z->running || addr != NULL || fd != -1 || (!require_managed && length < LARGE_THRESHOLD)) goto passthrough;
    if (!(flags & MAP_ANON) || !(flags & MAP_PRIVATE) || (flags & MAP_FIXED)) goto passthrough;
    if (prot & PROT_EXEC) goto passthrough;
    if (in_memx) goto passthrough;

    {
        in_memx = 1;
        pthread_mutex_lock(&g_z->alloc_mutex);
        size_t npages = (length + PAGE_SZ - 1) / PAGE_SZ;
        if (context_preflight_locked(owner_tag, length, npages, require_managed, 0) != 0) {
            pthread_mutex_unlock(&g_z->alloc_mutex);
            in_memx = 0;
            goto passthrough;
        }
        size_t hint = g_z->vmem_next / PAGE_SZ;
        ssize_t found_s = bm_find_free_run(g_z, npages, hint);
        if (found_s >= 0) {
            size_t found = (size_t)found_s;
            void *result = (uint8_t*)g_z->vmem + found * PAGE_SZ;
            g_z->vmem_next = (found + npages) * PAGE_SZ;
            for (size_t i=found; i<found+npages; i++) {
                g_z->meta[i].state = PAGE_RESIDENT;
                g_z->meta[i].codec = 0;
                g_z->meta[i].comp_size = 0;
                g_z->meta[i].pool_offset = 0;
                g_z->meta[i].owner_tag = owner_tag;
                g_z->meta[i].alloc_size = 0;
                g_z->meta[i].prefetched = 0;
                g_z->meta[i].cooldown = 0;
                g_z->meta[i].preferred_codec = 0;
                g_z->meta[i].codec_fail_streak = 0;
                bm_set_used(g_z, i);
                res_list_add(g_z, i);
            }
            __sync_fetch_and_add(&g_z->live_resident_pages, npages);
            mprotect(result, npages * PAGE_SZ, PROT_READ|PROT_WRITE);
            g_z->meta[found].alloc_size = length;
            mmap_table_insert(result, npages, length, owner_tag);
            pthread_mutex_unlock(&g_z->alloc_mutex);
            context_note_alloc(owner_tag, length);
            in_memx = 0;
            return result;
        }
        pthread_mutex_unlock(&g_z->alloc_mutex);
        in_memx = 0;
    }
passthrough:
    if (require_managed) {
        errno = ENOMEM;
        return MAP_FAILED;
    }
    return mmap(addr, length, prot, flags, fd, offset);
}

static void *memx_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    return memx_mmap_internal(addr, length, prot, flags, fd, offset, 0, 0);
}

static int memx_munmap(void *addr, size_t length) {
    if (g_z && g_z->running && is_ours(addr)) {
        // Find in mmap allocation table (hash-based O(1) lookup)
        size_t npages = 0;
        size_t requested_size = 0;
        uintptr_t owner_tag = 0;
        size_t sp = 0;
        if (mmap_table_find_and_remove(addr, &npages, &requested_size, &owner_tag)) {
            sp = ((uintptr_t)addr - (uintptr_t)g_z->vmem) / PAGE_SZ;
        }
        // Not in table: may be a direct managed allocation
        if (npages == 0) {
            if (is_ours(addr)) {
                size_t sp2 = ((uintptr_t)addr - (uintptr_t)g_z->vmem) / PAGE_SZ;
                size_t stored_size = g_z->meta[sp2].alloc_size;
                owner_tag = g_z->meta[sp2].owner_tag;
                requested_size = stored_size;
                size_t alloc_size = ((stored_size + PAGE_SZ - 1) / PAGE_SZ) * PAGE_SZ;
                npages = alloc_size / PAGE_SZ;
                sp = sp2;
            }
        }
        if (npages > 0 && sp < g_z->npages) {
            // Phase 1: Decompress any compressed pages
            for (size_t i=sp; i<sp+npages && i<g_z->npages; i++) {
                if (g_z->meta[i].state==PAGE_COMPRESSED) mprotect((uint8_t*)g_z->vmem+i*PAGE_SZ, PAGE_SZ, PROT_READ|PROT_WRITE);
            }
            // Phase 2: Under mutex, set state=NONE and free bitmap
            pthread_mutex_lock(&g_z->alloc_mutex);
            for (size_t i=sp; i<sp+npages && i<g_z->npages; i++) {
                g_z->meta[i].state=PAGE_NONE; g_z->meta[i].comp_size=0;
                g_z->meta[i].owner_tag = 0;
                bm_set_free(g_z, i);
            }
            pthread_mutex_unlock(&g_z->alloc_mutex);
            // Phase 3: Batch protect entire range
            mprotect((uint8_t*)g_z->vmem+sp*PAGE_SZ, npages*PAGE_SZ, PROT_NONE);
            if (requested_size > 0) context_note_free(owner_tag, requested_size);
            return 0;
        }
    }
    return munmap(addr, length);
}

int memx_runtime_init(void) {
    if (g_z && g_z->running) return 0;
    init_memx();
    return (g_z && g_z->running) ? 0 : -1;
}

void memx_runtime_shutdown(void) {
    fini_memx();
}

int memx_runtime_context_create(const char *name, memx_runtime_context_t **out_ctx) {
    if (!out_ctx) return EINVAL;
    *out_ctx = NULL;
    if (memx_runtime_init() != 0) return ENOMEM;
    memx_runtime_context_t *ctx = (memx_runtime_context_t *)real_calloc(1, sizeof(*ctx));
    if (!ctx) return ENOMEM;
    ctx->magic = MEMX_CONTEXT_MAGIC;
    if (name && *name) {
        strncpy(ctx->name, name, sizeof(ctx->name) - 1);
        ctx->name[sizeof(ctx->name) - 1] = '\0';
    }
    if (pthread_mutex_init(&ctx->ws_mutex, NULL) != 0) {
        real_free(ctx);
        return ENOMEM;
    }
    ctx->ws_mutex_inited = 1;
    ctx->epoch_phase = 0;
    ctx->epoch_gen = 0;
    ctx->hot_budget_bytes = 0;
    ctx->ws_hot_bytes = 0;
    *out_ctx = ctx;
    return 0;
}

int memx_runtime_context_destroy(memx_runtime_context_t *ctx) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC) return EINVAL;
    if (ctx->allocations_live != 0) return EBUSY;
    if (ctx->ws_mutex_inited) {
        pthread_mutex_destroy(&ctx->ws_mutex);
        ctx->ws_mutex_inited = 0;
    }
    ctx->magic = 0;
    real_free(ctx);
    return 0;
}

int memx_runtime_context_get_stats(const memx_runtime_context_t *ctx, memx_runtime_context_stats_t *out_stats) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !out_stats) return EINVAL;
    memset(out_stats, 0, sizeof(*out_stats));
    out_stats->bytes_in_use = ctx->bytes_in_use;
    out_stats->peak_bytes_in_use = ctx->peak_bytes_in_use;
    out_stats->allocations_live = ctx->allocations_live;
    out_stats->allocations_total = ctx->allocations_total;
    out_stats->quota_bytes = ctx->quota_bytes;
    out_stats->allocation_failures_quota = ctx->allocation_failures_quota;
    out_stats->pressure_events = ctx->pressure_events;
    out_stats->tensor_bytes_in_use = ctx->tensor_bytes_in_use;
    out_stats->tensor_allocations_live = ctx->tensor_allocations_live;
    out_stats->weight_bytes_in_use = ctx->weight_bytes_in_use;
    out_stats->kv_cache_bytes_in_use = ctx->kv_cache_bytes_in_use;
    out_stats->hot_bytes_in_use = ctx->hot_bytes_in_use;
    out_stats->no_compress_bytes_in_use = ctx->no_compress_bytes_in_use;
    return 0;
}

int memx_runtime_context_set_quota(memx_runtime_context_t *ctx, uint64_t quota_bytes) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC) return EINVAL;
    ctx->quota_bytes = quota_bytes;
    return 0;
}

int memx_runtime_context_get_quota(const memx_runtime_context_t *ctx, uint64_t *out_quota_bytes) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !out_quota_bytes) return EINVAL;
    *out_quota_bytes = ctx->quota_bytes;
    return 0;
}

int memx_runtime_is_active(void) {
    return g_z && g_z->running;
}

int memx_runtime_owns_pointer(const void *ptr) {
    return ptr ? is_ours((void *)ptr) : 0;
}

static int allocation_info_range_locked(size_t sp, size_t offset, size_t length, memx_runtime_allocation_info_t *out_info) {
    if (sp >= g_z->npages || g_z->meta[sp].alloc_size == 0 || g_z->meta[sp].state == PAGE_NONE) return ENOENT;
    size_t alloc_size = g_z->meta[sp].alloc_size;
    if (length == 0 || offset >= alloc_size || length > alloc_size - offset) return EINVAL;
    size_t alloc_pages = ((alloc_size + PAGE_SZ - 1) / PAGE_SZ);
    size_t first_rel_page = offset / PAGE_SZ;
    size_t last_rel_page = (offset + length - 1) / PAGE_SZ;
    if (first_rel_page >= alloc_pages || last_rel_page >= alloc_pages || sp + last_rel_page >= g_z->npages) return EINVAL;
    size_t first_page = sp + first_rel_page;
    size_t last_page = sp + last_rel_page;
    uint64_t compressed_pages = 0;
    uint64_t compressed_bytes = 0;
    uint64_t tensor_codec_pages = 0;
    uint32_t primary_codec = 0;
    for (size_t i = first_page; i <= last_page; i++) {
        if (g_z->meta[i].state == PAGE_COMPRESSED) {
            compressed_pages++;
            compressed_bytes += g_z->meta[i].comp_size;
            if (primary_codec == 0 && g_z->meta[i].codec != 0) primary_codec = g_z->meta[i].codec;
            if (g_z->meta[i].codec == MEMX_CODEC_TENSOR_FP16_SPLIT ||
                g_z->meta[i].codec == MEMX_CODEC_TENSOR_FP16_DELTA_SPLIT ||
                g_z->meta[i].codec == MEMX_CODEC_TENSOR_BITPLANE16 ||
                g_z->meta[i].codec == MEMX_CODEC_TENSOR_SPARSE_BYTE ||
                g_z->meta[i].codec == MEMX_CODEC_ZLIB ||
                g_z->meta[i].codec == MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT ||
                g_z->meta[i].codec == MEMX_CODEC_TENSOR_EXP_PACK) tensor_codec_pages++;
        }
    }
    out_info->size = length;
    out_info->page_count = last_rel_page - first_rel_page + 1;
    out_info->compressed_pages = compressed_pages;
    out_info->compressed_bytes = compressed_bytes;
    out_info->tensor_role = g_z->meta[first_page].tensor_role;
    out_info->tensor_dtype = g_z->meta[first_page].tensor_dtype;
    out_info->tensor_layout = g_z->meta[first_page].tensor_layout;
    out_info->tensor_flags = g_z->meta[first_page].tensor_flags;
    out_info->primary_codec = primary_codec;
    out_info->tensor_codec_pages = tensor_codec_pages;
    out_info->managed = 1;
    return 0;
}

int memx_runtime_get_allocation_info(const void *ptr, memx_runtime_allocation_info_t *out_info) {
    if (!ptr || !out_info) return EINVAL;
    memset(out_info, 0, sizeof(*out_info));
    if (!g_z || !is_ours((void *)ptr)) return ENOENT;
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    pthread_mutex_lock(&g_z->alloc_mutex);
    if (sp >= g_z->npages || g_z->meta[sp].alloc_size == 0 || g_z->meta[sp].state == PAGE_NONE) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return ENOENT;
    }
    size_t size = g_z->meta[sp].alloc_size;
    int rc = allocation_info_range_locked(sp, 0, size, out_info);
    pthread_mutex_unlock(&g_z->alloc_mutex);
    return rc;
}

int memx_runtime_get_allocation_info_range(const void *ptr, size_t offset, size_t length, memx_runtime_allocation_info_t *out_info) {
    if (!ptr || !out_info) return EINVAL;
    memset(out_info, 0, sizeof(*out_info));
    if (!g_z || !is_ours((void *)ptr)) return ENOENT;
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    pthread_mutex_lock(&g_z->alloc_mutex);
    int rc = allocation_info_range_locked(sp, offset, length, out_info);
    pthread_mutex_unlock(&g_z->alloc_mutex);
    return rc;
}

static int prefetch_range_internal(const void *ptr, size_t offset, size_t length, uintptr_t owner_tag, int check_owner) {
    if (!ptr) return EINVAL;
    if (length == 0) return 0;
    if (!g_z || !g_z->running || !is_ours((void *)ptr)) return ENOENT;
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    size_t first_page = 0;
    size_t last_page = 0;
    pthread_mutex_lock(&g_z->alloc_mutex);
    if (sp >= g_z->npages ||
        g_z->meta[sp].alloc_size == 0 ||
        g_z->meta[sp].state == PAGE_NONE ||
        (check_owner && g_z->meta[sp].owner_tag != owner_tag)) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return EINVAL;
    }
    size_t size = g_z->meta[sp].alloc_size;
    if (offset >= size || length > size - offset) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return EINVAL;
    }
    size_t first_rel_page = offset / PAGE_SZ;
    size_t last_rel_page = (offset + length - 1) / PAGE_SZ;
    size_t allocation_pages = (size + PAGE_SZ - 1) / PAGE_SZ;
    if (first_rel_page >= allocation_pages || last_rel_page >= allocation_pages || sp + last_rel_page >= g_z->npages) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return EINVAL;
    }
    first_page = sp + first_rel_page;
    last_page = sp + last_rel_page;
    pthread_mutex_unlock(&g_z->alloc_mutex);
    uint64_t prefetched = 0;
    size_t total_pages = last_page - first_page + 1;
    uint16_t role0 = g_z->meta[first_page].tensor_role;
    uint32_t flags0 = g_z->meta[first_page].tensor_flags;
    uint8_t cd = 8;
    size_t sync_budget = total_pages <= 8 ? total_pages : 6;
    size_t async_cap = total_pages;
    if (role0 == MEMX_TENSOR_ROLE_WEIGHT || role0 == MEMX_TENSOR_ROLE_EMBEDDING) {
        cd = (flags0 & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY)) ? 1 : 2;
        if (flags0 & MEMX_TENSOR_FLAG_HOT) {
            cd = 3;
            if (total_pages <= 8) sync_budget = total_pages;
            else if (total_pages <= 24) sync_budget = total_pages;
            else if (total_pages <= 48) sync_budget = 24;
            else if (total_pages <= 96) sync_budget = 16;
            else if (total_pages <= 192) sync_budget = 12;
            else sync_budget = 8;
            async_cap = total_pages;
            if (async_cap > 256) async_cap = 256;
        } else {
            if (total_pages > 128) sync_budget = 3;
            else if (total_pages > 48) sync_budget = 4;
            else if (total_pages > 16) sync_budget = 5;
            else if (total_pages > 8) sync_budget = 6;
            if (total_pages > 96) async_cap = 80;
            else if (total_pages > 64) async_cap = 64;
            else if (total_pages > 32) async_cap = 40;
        }
    } else if (role0 == MEMX_TENSOR_ROLE_KV_CACHE) {
        cd = 10;
        if (total_pages > 32) sync_budget = 4;
        if (total_pages > 16) async_cap = total_pages <= 64 ? total_pages : 64;
    }
    size_t idx = 0;
    uint32_t async_pages[192];
    uint8_t async_cds[192];
    int async_n = 0;
    size_t async_budget = (async_cap > sync_budget) ? (async_cap - sync_budget) : 0;
    size_t async_seen = 0;
    for (size_t i = first_page; i <= last_page; i++, idx++) {
        int ok = 0;
        if (idx < sync_budget) {
            ok = prefetch_page(g_z, i, cd);
            if (ok) prefetched++;
        } else if (async_seen < async_budget) {
            if (g_z->meta[i].state == PAGE_COMPRESSED) {
                async_pages[async_n] = (uint32_t)i;
                async_cds[async_n] = cd;
                async_n++;
                async_seen++;
                if (async_n >= 192) {
                    int enq = async_pf_enqueue_n(g_z, async_pages, async_cds, async_n, 1);
                    if (enq > 0) prefetched += (uint64_t)enq;
                    async_n = 0;
                }
            }
        }
    }
    if (async_n > 0) {
        int enq = async_pf_enqueue_n(g_z, async_pages, async_cds, async_n, 1);
        if (enq > 0) prefetched += (uint64_t)enq;
    }
    if (prefetched > 0) __sync_fetch_and_add(&g_z->prefetch_count, 1);
    return 0;
}

int memx_runtime_prefetch_range(const void *ptr, size_t offset, size_t length) {
    return prefetch_range_internal(ptr, offset, length, 0, 0);
}

int memx_runtime_context_prefetch_range(memx_runtime_context_t *ctx, const void *ptr, size_t offset, size_t length) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC) return EINVAL;
    return prefetch_range_internal(ptr, offset, length, (uintptr_t)ctx, 1);
}

static int mark_access_range_internal(const void *ptr, size_t offset, size_t length, uintptr_t owner_tag, int check_owner) {
    if (!ptr) return EINVAL;
    if (length == 0) return 0;
    if (!g_z || !g_z->running || !is_ours((void *)ptr)) return ENOENT;
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    pthread_mutex_lock(&g_z->alloc_mutex);
    if (sp >= g_z->npages ||
        g_z->meta[sp].alloc_size == 0 ||
        g_z->meta[sp].state == PAGE_NONE ||
        (check_owner && g_z->meta[sp].owner_tag != owner_tag)) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return EINVAL;
    }
    size_t size = g_z->meta[sp].alloc_size;
    if (offset >= size || length > size - offset) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return EINVAL;
    }
    size_t first_rel_page = offset / PAGE_SZ;
    size_t last_rel_page = (offset + length - 1) / PAGE_SZ;
    size_t allocation_pages = (size + PAGE_SZ - 1) / PAGE_SZ;
    if (first_rel_page >= allocation_pages || last_rel_page >= allocation_pages || sp + last_rel_page >= g_z->npages) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return EINVAL;
    }
    uint64_t hits = 0;
    for (size_t i = sp + first_rel_page; i <= sp + last_rel_page; i++) {
        if (g_z->meta[i].prefetched) {
            g_z->meta[i].prefetched = 0;
            hits++;
        }
    }
    pthread_mutex_unlock(&g_z->alloc_mutex);
    if (hits > 0) __sync_fetch_and_add(&g_z->prefetch_hits, hits);
    return 0;
}

int memx_runtime_mark_access_range(const void *ptr, size_t offset, size_t length) {
    return mark_access_range_internal(ptr, offset, length, 0, 0);
}

int memx_runtime_context_mark_access_range(memx_runtime_context_t *ctx, const void *ptr, size_t offset, size_t length) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC) return EINVAL;
    return mark_access_range_internal(ptr, offset, length, (uintptr_t)ctx, 1);
}

static int range_fits_allocation(size_t offset, size_t length, size_t allocation_size) {
    if (length == 0) return 1;
    if (offset >= allocation_size) return 0;
    return length <= allocation_size - offset;
}

static int update_tensor_window(memx_runtime_context_t *ctx, void *ptr, size_t managed_offset, size_t managed_length,
                                size_t hot_offset, size_t hot_length, size_t prefetch_offset, size_t prefetch_length,
                                uint32_t required_role) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !ptr) return EINVAL;
    if (!g_z || !is_ours(ptr)) return EINVAL;
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    size_t allocation_size = 0;
    pthread_mutex_lock(&g_z->alloc_mutex);
    if (sp >= g_z->npages ||
        g_z->meta[sp].owner_tag != (uintptr_t)ctx ||
        g_z->meta[sp].alloc_size == 0 ||
        g_z->meta[sp].tensor_role != required_role) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return EINVAL;
    }
    allocation_size = g_z->meta[sp].alloc_size;
    pthread_mutex_unlock(&g_z->alloc_mutex);
    if (!range_fits_allocation(managed_offset, managed_length, allocation_size) ||
        !range_fits_allocation(hot_offset, hot_length, allocation_size) ||
        !range_fits_allocation(prefetch_offset, prefetch_length, allocation_size)) {
        return EINVAL;
    }
    if (managed_length > 0) {
        uint32_t cold_flags = MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY;
        if (required_role == MEMX_TENSOR_ROLE_KV_CACHE)
            cold_flags |= MEMX_TENSOR_FLAG_SEQUENTIAL;
        int rc = memx_runtime_context_update_tensor_flags_range(ctx, ptr, managed_offset, managed_length, cold_flags);
        if (rc != 0) return rc;
    }
    if (hot_length > 0) {
        uint32_t hot_flags = MEMX_TENSOR_FLAG_HOT | MEMX_TENSOR_FLAG_NO_COMPRESS;
        if (required_role == MEMX_TENSOR_ROLE_KV_CACHE)
            hot_flags |= MEMX_TENSOR_FLAG_SEQUENTIAL;
        int rc = memx_runtime_context_update_tensor_flags_range(ctx, ptr, hot_offset, hot_length, hot_flags);
        if (rc != 0) return rc;
        rc = memx_runtime_context_prefetch_range(ctx, ptr, hot_offset, hot_length);
        if (rc != 0) return rc;
    }
    if (prefetch_length > 0) {
        int rc = memx_runtime_context_prefetch_range(ctx, ptr, prefetch_offset, prefetch_length);
        if (rc != 0) return rc;
        if (required_role == MEMX_TENSOR_ROLE_KV_CACHE || required_role == MEMX_TENSOR_ROLE_WEIGHT) {
            size_t sp2 = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
            size_t first_rel = prefetch_offset / PAGE_SZ;
            size_t last_rel = (prefetch_offset + prefetch_length - 1) / PAGE_SZ;
            pthread_mutex_lock(&g_z->alloc_mutex);
            for (size_t i = sp2 + first_rel; i <= sp2 + last_rel && i < g_z->npages; i++) {
                if (g_z->meta[i].state == PAGE_HOT && g_z->meta[i].cooldown < 32)
                    g_z->meta[i].cooldown = 32;
            }
            pthread_mutex_unlock(&g_z->alloc_mutex);
        }
    }
    return 0;
}

int memx_runtime_context_update_kv_cache_window(memx_runtime_context_t *ctx, void *ptr, const memx_runtime_kv_cache_window_t *window) {
    if (!window) return EINVAL;
    if (window->struct_size != 0 && window->struct_size < offsetof(memx_runtime_kv_cache_window_t, reserved)) return EINVAL;
    return update_tensor_window(ctx, ptr,
                                window->managed_offset, window->managed_length,
                                window->hot_offset, window->hot_length,
                                window->prefetch_offset, window->prefetch_length,
                                MEMX_TENSOR_ROLE_KV_CACHE);
}

int memx_runtime_context_update_weight_window(memx_runtime_context_t *ctx, void *ptr, const memx_runtime_weight_window_t *window) {
    if (!window) return EINVAL;
    if (window->struct_size != 0 && window->struct_size < offsetof(memx_runtime_weight_window_t, reserved)) return EINVAL;
    return update_tensor_window(ctx, ptr,
                                window->managed_offset, window->managed_length,
                                window->hot_offset, window->hot_length,
                                window->prefetch_offset, window->prefetch_length,
                                MEMX_TENSOR_ROLE_WEIGHT);
}


static __thread uint8_t g_force_src[PAGE_SZ];
static __thread uint8_t g_force_dst[PAGE_SZ];
static __thread uint8_t g_force_tmp[PAGE_SZ];

static int force_compress_page_now(MemXZone3 *s, size_t pidx) {
    if (!s || pidx >= s->npages) return 0;
    PageMeta *m = &s->meta[pidx];
    if (m->tensor_flags & MEMX_TENSOR_FLAG_NO_COMPRESS) {
        m->tensor_flags &= ~MEMX_TENSOR_FLAG_NO_COMPRESS;
        m->tensor_flags &= ~MEMX_TENSOR_FLAG_HOT;
        m->tensor_flags |= (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY);
    }
    uint8_t *pa = (uint8_t *)s->vmem + pidx * PAGE_SZ;
    uint8_t st = m->state;
    if (st == PAGE_COMPRESSED) return 1;
    if (st == PAGE_HOT) {
        if (m->comp_size != 0) return 0;
        m->dirty = 0;
        m->cooldown = 0;
        m->stable_ticks = 255;
        m->tensor_flags &= ~MEMX_TENSOR_FLAG_HOT;
        uint8_t old = __sync_val_compare_and_swap(&m->state, PAGE_HOT, PAGE_RESIDENT);
        if (old != PAGE_HOT && m->state != PAGE_RESIDENT) return 0;
        res_list_add(s, (uint32_t)pidx);
    } else if (st != PAGE_RESIDENT) {
        return 0;
    }
    m->dirty = 0;
    m->cooldown = 0;
    m->stable_ticks = 255;
    m->tensor_flags &= ~MEMX_TENSOR_FLAG_HOT;
    mprotect(pa, PAGE_SZ, PROT_READ);
    uint32_t seq0 = m->write_seq;
    uint8_t old = __sync_val_compare_and_swap(&m->state, PAGE_RESIDENT, PAGE_COMPRESSING);
    if (old != PAGE_RESIDENT) return 0;
    uint8_t *src = g_force_src;
    uint8_t *dst = g_force_dst;
    uint8_t *tmp = g_force_tmp;
    if ((m->tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL) != 0) {
        mprotect(pa, PAGE_SZ, PROT_NONE);
        __sync_synchronize();
        mprotect(pa, PAGE_SZ, PROT_READ);
        __sync_synchronize();
        if (m->dirty || m->write_seq != seq0) {
            restore_compressing_page(s, pidx);
            return 0;
        }
    }
    memcpy(src, pa, PAGE_SZ);
    if (m->dirty || m->write_seq != seq0) {
        restore_compressing_page(s, pidx);
        return 0;
    }
    uint32_t csz = 0;
    uint8_t codec = 0;
    encode_tensor_page_one(s, pidx, src, dst, tmp, &csz, &codec);
    if (csz == 0 || csz >= PAGE_SZ || csz >= (PAGE_SZ * 15) / 16) {
        restore_compressing_page(s, pidx);
        return 0;
    }
    if (!page_compress_content_ok(s, pidx, seq0, src)) {
        restore_compressing_page(s, pidx);
        return 0;
    }
    pthread_mutex_lock(&s->alloc_mutex);
    if (s->dedup_pending_free_count > 0) memx_runtime_reclaim_locked(s);
    uint64_t h = fnv1a_word(dst, csz);
    uint32_t slot = (uint32_t)(h & DEDUP_HT_MASK);
    int committed = 0;
    for (int probe = 0; probe < 8; probe++) {
        uint32_t s2 = (slot + probe) & DEDUP_HT_MASK;
        if (s->dedup_hash[s2] == 0) break;
        if (s->dedup_hash[s2] == h && s->dedup_sz[s2] == csz && s->dedup_ref[s2] > 0) {
            uint64_t existing_off = s->dedup_off[s2];
            if (memcmp(dst, s->pool + existing_off, csz) == 0) {
                if (!page_compress_content_ok(s, pidx, seq0, src)) break;
                __sync_fetch_and_add(&s->dedup_ref[s2], 1);
                m->pool_offset = existing_off;
                m->codec = codec;
                m->comp_size = csz;
                __sync_synchronize();
                if (!page_compress_content_ok(s, pidx, seq0, src) || m->dirty || m->write_seq != seq0) {
                    m->pool_offset = 0;
                    m->codec = 0;
                    m->comp_size = 0;
                    __sync_fetch_and_sub(&s->dedup_ref[s2], 1);
                    break;
                }
                if (!commit_compressed_page(s, pidx, seq0, src)) {
                    m->pool_offset = 0;
                    m->codec = 0;
                    m->comp_size = 0;
                    __sync_fetch_and_sub(&s->dedup_ref[s2], 1);
                    break;
                }
                if (s->live_resident_pages) __sync_fetch_and_sub(&s->live_resident_pages, 1);
                __sync_fetch_and_add(&s->live_compressed_pages, 1);
                m->preferred_codec = codec ? codec : m->preferred_codec;
                m->codec_fail_streak = 0;
                if (m->state == PAGE_COMPRESSED) m->dirty = 0;
                note_page_compressed(s, pidx, codec, csz);
                if (m->state == PAGE_COMPRESSED) page_release_physical(s, pidx);
                __sync_fetch_and_add(&s->dedup_hits, 1);
                __sync_fetch_and_add(&s->dedup_bytes_saved, csz);
                committed = 1;
                break;
            }
        }
    }
    if (!committed) {
        uint64_t off = 0;
        if (pool_alloc_extent_locked(s, csz, &off) != 0) {
            restore_compressing_page(s, pidx);
            pthread_mutex_unlock(&s->alloc_mutex);
            return 0;
        }
        pool_prepare_write_range(s, off, csz);
        if (!page_compress_content_ok(s, pidx, seq0, src)) {
            pool_free_insert_locked(s, off, csz);
            restore_compressing_page(s, pidx);
            pthread_mutex_unlock(&s->alloc_mutex);
            return 0;
        }
        memcpy(s->pool + off, dst, csz);
        {
            uint64_t start = off & ~((uint64_t)PAGE_SZ - 1);
            uint64_t end = (off + csz + PAGE_SZ - 1) & ~((uint64_t)PAGE_SZ - 1);
            if (end > s->pool_size) end = s->pool_size;
            if (end > start) mprotect(s->pool + start, (size_t)(end - start), PROT_READ);
        }
        if (!page_compress_content_ok(s, pidx, seq0, src) || m->dirty || m->write_seq != seq0) {
            pool_free_insert_locked(s, off, csz);
            restore_compressing_page(s, pidx);
            pthread_mutex_unlock(&s->alloc_mutex);
            return 0;
        }
        m->pool_offset = off;
        m->codec = codec;
        m->comp_size = csz;
        __sync_synchronize();
        if (!commit_compressed_page(s, pidx, seq0, src)) {
            m->pool_offset = 0;
            m->codec = 0;
            m->comp_size = 0;
            pool_free_insert_locked(s, off, csz);
            restore_compressing_page(s, pidx);
            pthread_mutex_unlock(&s->alloc_mutex);
            return 0;
        }
        if (s->live_resident_pages) __sync_fetch_and_sub(&s->live_resident_pages, 1);
        __sync_fetch_and_add(&s->live_compressed_pages, 1);
        m->preferred_codec = codec ? codec : m->preferred_codec;
        m->codec_fail_streak = 0;
        if (m->state == PAGE_COMPRESSED) m->dirty = 0;
        note_page_compressed(s, pidx, codec, csz);
        if (m->state == PAGE_COMPRESSED) page_release_physical(s, pidx);
        for (int probe = 0; probe < 8; probe++) {
            uint32_t s2 = (slot + probe) & DEDUP_HT_MASK;
            if (s->dedup_hash[s2] == 0 || s->dedup_ref[s2] == 0) {
                s->dedup_hash[s2] = h;
                s->dedup_off[s2] = off;
                s->dedup_sz[s2] = csz;
                s->dedup_ref[s2] = 1;
                if (s->dedup_rev && s->dedup_rev_size) s->dedup_rev[(uint32_t)(off / PAGE_SZ) & s->dedup_rev_mask] = s2;
                break;
            }
        }
        committed = 1;
    }
    pthread_mutex_unlock(&s->alloc_mutex);
    return committed;
}

int memx_runtime_context_force_compress_range(memx_runtime_context_t *ctx, void *ptr, size_t offset, size_t length, uint64_t *out_compressed_pages) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !ptr) return EINVAL;
    if (!g_z || !is_ours(ptr)) return EINVAL;
    if (length == 0) {
        if (out_compressed_pages) *out_compressed_pages = 0;
        return 0;
    }
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    if (sp >= g_z->npages || g_z->meta[sp].owner_tag != (uintptr_t)ctx || g_z->meta[sp].alloc_size == 0)
        return EINVAL;
    size_t size = g_z->meta[sp].alloc_size;
    if (offset >= size || length > size - offset) return EINVAL;
    size_t first = sp + offset / PAGE_SZ;
    size_t last = sp + (offset + length - 1) / PAGE_SZ;
    uint64_t done = 0;
    for (size_t i = first; i <= last; i++) {
        if (force_compress_page_now(g_z, i)) done++;
    }
    size_t run_start = (size_t)-1;
    for (size_t i = first; i <= last; i++) {
        if (g_z->meta[i].state == PAGE_COMPRESSED) {
            if (run_start == (size_t)-1) run_start = i;
        } else if (run_start != (size_t)-1) {
            page_release_physical_range(g_z, run_start, i - 1);
            run_start = (size_t)-1;
        }
    }
    if (run_start != (size_t)-1)
        page_release_physical_range(g_z, run_start, last);
    if (out_compressed_pages) *out_compressed_pages = done;
    return 0;
}

int memx_runtime_context_seal_range(memx_runtime_context_t *ctx, void *ptr, size_t offset, size_t length, uint64_t *out_compressed_pages) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !ptr) return EINVAL;
    if (!g_z || !is_ours(ptr)) return EINVAL;
    if (length == 0) {
        if (out_compressed_pages) *out_compressed_pages = 0;
        return 0;
    }
    int rc = memx_runtime_context_update_tensor_flags_range(
        ctx, ptr, offset, length,
        MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY);
    if (rc != 0) return rc;
    uint64_t done = 0;
    rc = memx_runtime_context_force_compress_range(ctx, ptr, offset, length, &done);
    if (rc != 0) return rc;
    if (done > 0) {
        pthread_mutex_lock(&g_z->alloc_mutex);
        memx_runtime_reclaim_locked(g_z);
        pthread_mutex_unlock(&g_z->alloc_mutex);
    }
    if (out_compressed_pages) *out_compressed_pages = done;
    return 0;
}


int memx_runtime_context_seal_range_async(memx_runtime_context_t *ctx, void *ptr, size_t offset, size_t length) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !ptr) return EINVAL;
    if (!g_z || !g_z->running || !is_ours(ptr)) return ENOENT;
    if (length == 0) return 0;
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    if (sp >= g_z->npages || g_z->meta[sp].owner_tag != (uintptr_t)ctx || g_z->meta[sp].alloc_size == 0)
        return EINVAL;
    size_t size = g_z->meta[sp].alloc_size;
    if (offset >= size || length > size - offset) return EINVAL;
    /* mark cold immediately so compressor won't keep hot residency */
    memx_runtime_context_update_tensor_flags_range(
        ctx, ptr, offset, length,
        MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY);
    if (!async_seal_enqueue(g_z, ptr, offset, length, (uintptr_t)ctx)) {
        /* fallback sync if queue full */
        uint64_t done = 0;
        return memx_runtime_context_force_compress_range(ctx, ptr, offset, length, &done);
    }
    return 0;
}

int memx_runtime_seal_flush(uint64_t *out_pending) {
    if (!g_z) return ENOENT;
    if (!g_z->async_seal_q) {
        if (out_pending) *out_pending = 0;
        return 0;
    }
    for (int i = 0; i < 5000; i++) {
        uint32_t head = g_z->async_seal_head;
        uint32_t tail = g_z->async_seal_tail;
        int active = g_z->async_seal_active;
        if (head == tail && active == 0) break;
        pthread_mutex_lock(&g_z->async_seal_mutex);
        if (g_z->async_seal_head != g_z->async_seal_tail || g_z->async_seal_active != 0) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 5000000L;
            if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
            pthread_cond_timedwait(&g_z->async_seal_idle_cond, &g_z->async_seal_mutex, &ts);
        }
        pthread_mutex_unlock(&g_z->async_seal_mutex);
    }
    uint32_t head = g_z->async_seal_head;
    uint32_t tail = g_z->async_seal_tail;
    uint64_t pending = 0;
    if (head >= tail) pending = head - tail;
    else pending = (ASYNC_SEAL_Q_SIZE - tail) + head;
    pending += (uint64_t)g_z->async_seal_active;
    if (out_pending) *out_pending = pending;
    pthread_mutex_lock(&g_z->alloc_mutex);
    memx_runtime_reclaim_locked(g_z);
    pthread_mutex_unlock(&g_z->alloc_mutex);
    return 0;
}


int memx_runtime_context_update_tensor_flags_range(memx_runtime_context_t *ctx, void *ptr, size_t offset, size_t length, uint32_t flags);

static int ws_resolve_alloc(memx_runtime_context_t *ctx, void *ptr, size_t *out_sp, size_t *out_size) {
    if (!ctx || !ptr || !g_z || !is_ours(ptr)) return EINVAL;
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    if (sp >= g_z->npages || g_z->meta[sp].owner_tag != (uintptr_t)ctx || g_z->meta[sp].alloc_size == 0)
        return EINVAL;
    *out_sp = sp;
    *out_size = g_z->meta[sp].alloc_size;
    return 0;
}

static size_t ws_align_down(size_t v) {
    return v & ~((size_t)PAGE_SZ - 1);
}

static size_t ws_align_up(size_t v, size_t limit) {
    size_t a = (v + PAGE_SZ - 1) & ~((size_t)PAGE_SZ - 1);
    if (a > limit) a = limit;
    return a;
}

static int ws_find_track(memx_runtime_context_t *ctx, void *ptr) {
    for (int i = 0; i < MEMX_WS_TRACK_MAX; i++) {
        if (ctx->ws_tracks[i].active && ctx->ws_tracks[i].ptr == ptr)
            return i;
    }
    return -1;
}

static int ws_alloc_track(memx_runtime_context_t *ctx) {
    for (int i = 0; i < MEMX_WS_TRACK_MAX; i++) {
        if (!ctx->ws_tracks[i].active)
            return i;
    }
    size_t victim = 0;
    size_t best = 0;
    for (int i = 0; i < MEMX_WS_TRACK_MAX; i++) {
        size_t span = 0;
        if (ctx->ws_tracks[i].hot_end > ctx->ws_tracks[i].hot_off)
            span = ctx->ws_tracks[i].hot_end - ctx->ws_tracks[i].hot_off;
        if (i == 0 || span < best) {
            best = span;
            victim = (size_t)i;
        }
    }
    return (int)victim;
}

static void ws_track_set(memx_runtime_context_t *ctx, int idx, void *ptr, size_t hot_off, size_t hot_end, size_t alloc_size) {
    size_t old = 0;
    if (ctx->ws_tracks[idx].active && ctx->ws_tracks[idx].hot_end > ctx->ws_tracks[idx].hot_off)
        old = ctx->ws_tracks[idx].hot_end - ctx->ws_tracks[idx].hot_off;
    size_t neu = (hot_end > hot_off) ? (hot_end - hot_off) : 0;
    if (neu >= old)
        __sync_fetch_and_add(&ctx->ws_hot_bytes, neu - old);
    else
        __sync_fetch_and_sub(&ctx->ws_hot_bytes, old - neu);
    ctx->ws_tracks[idx].ptr = ptr;
    ctx->ws_tracks[idx].hot_off = hot_off;
    ctx->ws_tracks[idx].hot_end = hot_end;
    ctx->ws_tracks[idx].alloc_size = alloc_size;
    ctx->ws_tracks[idx].gen = ctx->epoch_gen;
    ctx->ws_tracks[idx].active = 1;
}

static void ws_track_clear(memx_runtime_context_t *ctx, int idx) {
    if (!ctx->ws_tracks[idx].active) return;
    size_t old = 0;
    if (ctx->ws_tracks[idx].hot_end > ctx->ws_tracks[idx].hot_off)
        old = ctx->ws_tracks[idx].hot_end - ctx->ws_tracks[idx].hot_off;
    if (old) __sync_fetch_and_sub(&ctx->ws_hot_bytes, old);
    ctx->ws_tracks[idx].ptr = NULL;
    ctx->ws_tracks[idx].hot_off = 0;
    ctx->ws_tracks[idx].hot_end = 0;
    ctx->ws_tracks[idx].alloc_size = 0;
    ctx->ws_tracks[idx].gen = 0;
    ctx->ws_tracks[idx].active = 0;
}

static int ws_cold_range(memx_runtime_context_t *ctx, void *ptr, size_t off, size_t ln) {
    if (ln == 0) return 0;
    return memx_runtime_context_update_tensor_flags_range(
        ctx, ptr, off, ln,
        MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD);
}

static int ws_retire_range(memx_runtime_context_t *ctx, void *ptr, size_t off, size_t ln, int sync) {
    if (ln == 0) return 0;
    if (sync) {
        uint64_t done = 0;
        return memx_runtime_context_seal_range(ctx, ptr, off, ln, &done);
    }
    int rc = memx_runtime_context_seal_range_async(ctx, ptr, off, ln);
    if (rc != 0) {
        uint64_t done = 0;
        return memx_runtime_context_seal_range(ctx, ptr, off, ln, &done);
    }
    return 0;
}

static int ws_release_range(memx_runtime_context_t *ctx, void *ptr, size_t off, size_t ln, int seal, int sync) {
    if (ln == 0) return 0;
    if (seal) return ws_retire_range(ctx, ptr, off, ln, sync);
    return ws_cold_range(ctx, ptr, off, ln);
}

static int ws_hot_range(memx_runtime_context_t *ctx, void *ptr, size_t off, size_t ln, int do_prefetch, size_t pref_cap) {
    if (ln == 0) return 0;
    int rc = memx_runtime_context_update_tensor_flags_range(
        ctx, ptr, off, ln,
        MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_HOT);
    if (rc != 0) return rc;
    if (do_prefetch) {
        size_t cap = pref_cap ? pref_cap : ln;
        if (cap > ln) cap = ln;
        if (cap > 0) {
            rc = memx_runtime_context_prefetch_range(ctx, ptr, off, cap);
            if (rc != 0) return rc;
        }
    }
    return 0;
}

static void ws_prefetch_ahead(memx_runtime_context_t *ctx, void *ptr, size_t a0, size_t a1, size_t cap_pages) {
    if (!ptr || a1 <= a0) return;
    size_t ln = a1 - a0;
    size_t cap = cap_pages * PAGE_SZ;
    if (cap == 0) cap = 8 * PAGE_SZ;
    if (ln > cap) ln = cap;
    memx_runtime_context_prefetch_range(ctx, ptr, a0, ln);
}

static size_t ws_pressure_prefetch_pages(void) {
    if (!g_z) return 24;
    uint32_t p = 0;
    pthread_mutex_lock(&g_z->alloc_mutex);
    p = memx_pool_pressure_percent_locked(g_z);
    pthread_mutex_unlock(&g_z->alloc_mutex);
    if (p >= 70) return 8;
    if (p >= 45) return 16;
    if (p >= 25) return 24;
    return 32;
}

static int ws_apply_one(memx_runtime_context_t *ctx, const memx_runtime_ws_intent_t *it) {
    if (!it || !it->ptr) return EINVAL;
    size_t sp = 0, alloc_size = 0;
    int rc = ws_resolve_alloc(ctx, it->ptr, &sp, &alloc_size);
    if (rc != 0) return rc;
    size_t off = it->offset;
    size_t ln = it->length;
    if (ln == 0) return 0;
    if (off >= alloc_size) return EINVAL;
    if (off + ln > alloc_size) ln = alloc_size - off;
    size_t a0 = ws_align_down(off);
    size_t a1 = ws_align_up(off + ln, alloc_size);
    if (a1 <= a0) return 0;
    uint32_t flags = it->flags;
    int want_hot = (flags & MEMX_WS_FLAG_HOT) != 0;
    int want_pref = (flags & MEMX_WS_FLAG_PREFETCH) != 0;
    int want_retire = (flags & MEMX_WS_FLAG_RETIRE) != 0;
    int retire_sync = (flags & MEMX_WS_FLAG_RETIRE_SYNC) != 0 || (flags & MEMX_WS_FLAG_NO_ASYNC) != 0;
    int mark = (flags & MEMX_WS_FLAG_MARK_ACCESS) != 0;
    int trail_seal = want_retire;

    if (want_retire && !want_hot) {
        return ws_retire_range(ctx, it->ptr, a0, a1 - a0, retire_sync);
    }

    if (want_pref && !want_hot) {
        size_t pref_ln = it->prefetch_length ? it->prefetch_length : (a1 - a0);
        size_t cap_pages = ws_pressure_prefetch_pages();
        if (pref_ln > cap_pages * PAGE_SZ) pref_ln = cap_pages * PAGE_SZ;
        if (pref_ln > a1 - a0) pref_ln = a1 - a0;
        return memx_runtime_context_prefetch_range(ctx, it->ptr, a0, pref_ln);
    }

    if (!want_hot) return 0;

    size_t pref_pages = ws_pressure_prefetch_pages();
    size_t pref_cap = pref_pages * PAGE_SZ;
    size_t pref_extra = it->prefetch_length;
    size_t hot_end = a1;
    size_t ahead0 = 0, ahead1 = 0;
    if (pref_extra > 0) {
        ahead0 = a1;
        ahead1 = ws_align_up(a1 + pref_extra, alloc_size);
        if (ahead1 <= ahead0) { ahead0 = 0; ahead1 = 0; }
    }

    pthread_mutex_lock(&ctx->ws_mutex);
    int tidx = ws_find_track(ctx, it->ptr);
    size_t prev0 = 0, prev1 = 0;
    int had = 0;
    if (tidx >= 0) {
        prev0 = ctx->ws_tracks[tidx].hot_off;
        prev1 = ctx->ws_tracks[tidx].hot_end;
        had = 1;
    } else {
        tidx = ws_alloc_track(ctx);
        if (ctx->ws_tracks[tidx].active && ctx->ws_tracks[tidx].ptr != it->ptr) {
            prev0 = ctx->ws_tracks[tidx].hot_off;
            prev1 = ctx->ws_tracks[tidx].hot_end;
            void *oldp = ctx->ws_tracks[tidx].ptr;
            ws_track_clear(ctx, tidx);
            pthread_mutex_unlock(&ctx->ws_mutex);
            if (oldp && prev1 > prev0)
                ws_retire_range(ctx, oldp, prev0, prev1 - prev0, 0);
            pthread_mutex_lock(&ctx->ws_mutex);
            had = 0;
            tidx = ws_alloc_track(ctx);
        }
    }

    if (had) {
        if (a0 >= prev0 && hot_end <= prev1) {
            ws_track_set(ctx, tidx, it->ptr, prev0, prev1, alloc_size);
            pthread_mutex_unlock(&ctx->ws_mutex);
            if (ahead1 > ahead0) ws_prefetch_ahead(ctx, it->ptr, ahead0, ahead1, pref_pages);
            return 0;
        }
        if (a0 == prev0 && hot_end > prev1) {
            size_t hoff = prev1, hln = hot_end - prev1;
            ws_track_set(ctx, tidx, it->ptr, a0, hot_end, alloc_size);
            pthread_mutex_unlock(&ctx->ws_mutex);
            int rc2 = ws_hot_range(ctx, it->ptr, hoff, hln, 1, pref_cap);
            if (ahead1 > ahead0) ws_prefetch_ahead(ctx, it->ptr, ahead0, ahead1, pref_pages);
            return rc2;
        }
        if (a0 > prev0 && a0 < prev1 && hot_end >= prev1) {
            size_t roff = prev0, rln = a0 - prev0;
            size_t hoff = prev1, hln = (hot_end > prev1) ? (hot_end - prev1) : 0;
            int do_trail = trail_seal || (rln >= (PAGE_SZ * 4));
            size_t track0 = do_trail ? a0 : prev0;
            ws_track_set(ctx, tidx, it->ptr, track0, hot_end, alloc_size);
            pthread_mutex_unlock(&ctx->ws_mutex);
            if (do_trail) ws_release_range(ctx, it->ptr, roff, rln, trail_seal, retire_sync);
            int rc2 = 0;
            if (hln) rc2 = ws_hot_range(ctx, it->ptr, hoff, hln, 1, pref_cap);
            if (ahead1 > ahead0) ws_prefetch_ahead(ctx, it->ptr, ahead0, ahead1, pref_pages);
            return rc2;
        }
        if (a0 < prev1 && hot_end > prev0) {
            size_t trail0 = 0, trail1 = 0, lead0 = 0, lead1 = 0, tail0 = 0, tail1 = 0;
            if (a0 > prev0) { trail0 = prev0; trail1 = a0; }
            if (hot_end < prev1) { lead0 = hot_end; lead1 = prev1; }
            if (a0 < prev0) { tail0 = a0; tail1 = prev0; }
            size_t ext0 = 0, ext1 = 0;
            if (hot_end > prev1) { ext0 = prev1; ext1 = hot_end; }
            ws_track_set(ctx, tidx, it->ptr, a0, hot_end, alloc_size);
            pthread_mutex_unlock(&ctx->ws_mutex);
            if (trail1 > trail0) ws_release_range(ctx, it->ptr, trail0, trail1 - trail0, trail_seal, retire_sync);
            if (lead1 > lead0) ws_release_range(ctx, it->ptr, lead0, lead1 - lead0, trail_seal, retire_sync);
            if (tail1 > tail0) {
                rc = ws_hot_range(ctx, it->ptr, tail0, tail1 - tail0, 1, pref_cap);
                if (rc != 0) return rc;
            }
            if (ext1 > ext0) {
                rc = ws_hot_range(ctx, it->ptr, ext0, ext1 - ext0, 1, pref_cap);
                if (ahead1 > ahead0) ws_prefetch_ahead(ctx, it->ptr, ahead0, ahead1, pref_pages);
                return rc;
            }
            if (ahead1 > ahead0) ws_prefetch_ahead(ctx, it->ptr, ahead0, ahead1, pref_pages);
            return 0;
        }
        size_t roff = prev0, rln = prev1 - prev0;
        ws_track_set(ctx, tidx, it->ptr, a0, hot_end, alloc_size);
        pthread_mutex_unlock(&ctx->ws_mutex);
        if (rln) ws_release_range(ctx, it->ptr, roff, rln, trail_seal, retire_sync);
        rc = ws_hot_range(ctx, it->ptr, a0, hot_end - a0, 1, pref_cap);
        if (ahead1 > ahead0) ws_prefetch_ahead(ctx, it->ptr, ahead0, ahead1, pref_pages);
        return rc;
    }

    ws_track_set(ctx, tidx, it->ptr, a0, hot_end, alloc_size);
    pthread_mutex_unlock(&ctx->ws_mutex);
    rc = ws_hot_range(ctx, it->ptr, a0, hot_end - a0, 1, pref_cap);
    if (ahead1 > ahead0) ws_prefetch_ahead(ctx, it->ptr, ahead0, ahead1, pref_pages);
    return rc;
}

int memx_runtime_context_begin_epoch(memx_runtime_context_t *ctx, uint32_t phase, uint64_t hot_budget_bytes) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC) return EINVAL;
    if (!ctx->ws_mutex_inited) {
        if (pthread_mutex_init(&ctx->ws_mutex, NULL) != 0) return ENOMEM;
        ctx->ws_mutex_inited = 1;
    }
    pthread_mutex_lock(&ctx->ws_mutex);
    ctx->epoch_phase = phase;
    ctx->epoch_gen++;
    if (ctx->epoch_gen == 0) ctx->epoch_gen = 1;
    ctx->hot_budget_bytes = hot_budget_bytes;
    for (int i = 0; i < MEMX_WS_TRACK_MAX; i++) {
        if (ctx->ws_tracks[i].active && ctx->ws_tracks[i].gen != ctx->epoch_gen) {
            /* keep tracks across phase switch; budget only */
        }
    }
    pthread_mutex_unlock(&ctx->ws_mutex);
    return 0;
}

int memx_runtime_context_apply_ws(memx_runtime_context_t *ctx, const memx_runtime_ws_intent_t *intents, size_t nintents) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC) return EINVAL;
    if (!intents || nintents == 0) return 0;
    for (size_t i = 0; i < nintents; i++) {
        const memx_runtime_ws_intent_t *it = &intents[i];
        if (it->struct_size != 0 && it->struct_size < sizeof(memx_runtime_ws_intent_t))
            continue;
        int rc = ws_apply_one(ctx, it);
        if (rc != 0) return rc;
    }
    return 0;
}

int memx_runtime_context_ws_advance(memx_runtime_context_t *ctx, void *ptr, size_t hot_offset, size_t hot_length, size_t prefetch_length, uint32_t flags) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !ptr) return EINVAL;
    memx_runtime_ws_intent_t it;
    memset(&it, 0, sizeof(it));
    it.struct_size = (uint32_t)sizeof(it);
    it.flags = flags ? flags : (MEMX_WS_FLAG_HOT | MEMX_WS_FLAG_PREFETCH | MEMX_WS_FLAG_MARK_ACCESS);
    it.ptr = ptr;
    it.offset = hot_offset;
    it.length = hot_length;
    it.prefetch_length = prefetch_length;
    it.priority = 0;
    return ws_apply_one(ctx, &it);
}

int memx_runtime_context_ws_close(memx_runtime_context_t *ctx, void *ptr, uint32_t flags) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !ptr) return EINVAL;
    int sync = (flags & MEMX_WS_FLAG_RETIRE_SYNC) != 0 || (flags & MEMX_WS_FLAG_NO_ASYNC) != 0;
    size_t off = 0, end = 0;
    int had = 0;
    pthread_mutex_lock(&ctx->ws_mutex);
    int tidx = ws_find_track(ctx, ptr);
    if (tidx >= 0) {
        off = ctx->ws_tracks[tidx].hot_off;
        end = ctx->ws_tracks[tidx].hot_end;
        had = 1;
        ws_track_clear(ctx, tidx);
    }
    pthread_mutex_unlock(&ctx->ws_mutex);
    if (!had) {
        size_t sp = 0, sz = 0;
        if (ws_resolve_alloc(ctx, ptr, &sp, &sz) != 0) return EINVAL;
        if (flags & MEMX_WS_FLAG_RETIRE)
            return ws_retire_range(ctx, ptr, 0, sz, sync);
        return memx_runtime_context_update_tensor_flags_range(
            ctx, ptr, 0, sz,
            MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD);
    }
    if (end > off) {
        if (flags & MEMX_WS_FLAG_RETIRE)
            return ws_retire_range(ctx, ptr, off, end - off, sync);
        return memx_runtime_context_update_tensor_flags_range(
            ctx, ptr, off, end - off,
            MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD);
    }
    return 0;
}

int memx_runtime_context_end_epoch(memx_runtime_context_t *ctx, int seal_tracked) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC) return EINVAL;
    typedef struct { void *ptr; size_t off; size_t ln; } seal_item_t;
    seal_item_t items[MEMX_WS_TRACK_MAX];
    int n = 0;
    pthread_mutex_lock(&ctx->ws_mutex);
    for (int i = 0; i < MEMX_WS_TRACK_MAX; i++) {
        if (!ctx->ws_tracks[i].active) continue;
        if (seal_tracked && ctx->ws_tracks[i].hot_end > ctx->ws_tracks[i].hot_off) {
            items[n].ptr = ctx->ws_tracks[i].ptr;
            items[n].off = ctx->ws_tracks[i].hot_off;
            items[n].ln = ctx->ws_tracks[i].hot_end - ctx->ws_tracks[i].hot_off;
            n++;
        }
        ws_track_clear(ctx, i);
    }
    ctx->epoch_phase = 0;
    pthread_mutex_unlock(&ctx->ws_mutex);
    for (int i = 0; i < n; i++) {
        ws_retire_range(ctx, items[i].ptr, items[i].off, items[i].ln, 0);
    }
    return 0;
}

int memx_runtime_context_purge(memx_runtime_context_t *ctx, void *ptr) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !ptr) return EINVAL;
    if (!g_z || !is_ours(ptr)) return EINVAL;
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    if (sp >= g_z->npages || g_z->meta[sp].owner_tag != (uintptr_t)ctx || g_z->meta[sp].alloc_size == 0)
        return EINVAL;
    size_t size = g_z->meta[sp].alloc_size;
    uint64_t dummy = 0;
    int rc = memx_runtime_context_force_compress_range(ctx, ptr, 0, size, &dummy);
    if (rc != 0) return rc;
    size_t npages = (size + PAGE_SZ - 1) / PAGE_SZ;
    size_t last = sp + npages - 1;
    if (last >= g_z->npages) last = g_z->npages - 1;
    page_release_physical_range(g_z, sp, last);
    pthread_mutex_lock(&g_z->alloc_mutex);
    memx_runtime_reclaim_and_compact_locked(g_z);
    pthread_mutex_unlock(&g_z->alloc_mutex);
    return 0;
}

int memx_runtime_get_stats(memx_runtime_stats_t *out_stats) {
    if (!out_stats) return EINVAL;
    memset(out_stats, 0, sizeof(*out_stats));
    if (!g_z || !g_z->running) return ENOENT;

    uint64_t compressed = g_z->live_compressed_pages;
    uint64_t resident = g_z->live_resident_pages;
    uint64_t hot_resident = g_z->live_hot_flag_pages;
    uint64_t no_compress_resident = g_z->live_nocomp_flag_pages;

    out_stats->compressions = g_z->compressions;
    out_stats->faults = g_z->faults;
    out_stats->bytes_saved = g_z->bytes_saved;
    out_stats->dedup_hits = g_z->dedup_hits;
    out_stats->dedup_bytes_saved = g_z->dedup_bytes_saved;
    out_stats->prefetch_count = g_z->prefetch_count;
    out_stats->prefetch_hits = g_z->prefetch_hits;
    out_stats->virtual_bytes = (compressed + resident) * PAGE_SZ;
    out_stats->pool_used_bytes = g_z->pool_used;
    out_stats->total_pages = g_z->npages;
    out_stats->compressed_pages = compressed;
    out_stats->resident_pages = resident;
    out_stats->pool_capacity_bytes = g_z->pool_size;
    out_stats->pool_cursor_bytes = g_z->pool_next;
    out_stats->pool_headroom_bytes = (g_z->pool_size > g_z->pool_next) ? (g_z->pool_size - g_z->pool_next) : 0;
    out_stats->free_pages = g_z->free_pages_count;
    out_stats->pool_reclaim_bytes = g_z->pool_reclaim_bytes_total;
    out_stats->pool_reclaim_events = g_z->pool_reclaim_events;
    out_stats->tensor_codec_pages = g_z->tensor_codec_pages;
    out_stats->tensor_codec_bytes_saved = g_z->tensor_codec_bytes_saved;
    out_stats->tensor_split_pages = g_z->tensor_split_pages;
    out_stats->tensor_split_bytes_saved = g_z->tensor_split_bytes_saved;
    out_stats->tensor_bitplane_pages = g_z->tensor_bitplane_pages;
    out_stats->tensor_bitplane_bytes_saved = g_z->tensor_bitplane_bytes_saved;
    out_stats->tensor_sparse_pages = g_z->tensor_sparse_pages;
    out_stats->tensor_sparse_bytes_saved = g_z->tensor_sparse_bytes_saved;
    out_stats->weight_compressed_pages = g_z->weight_compressed_pages;
    out_stats->weight_bytes_saved = g_z->weight_bytes_saved;
    out_stats->kv_cache_compressed_pages = g_z->kv_cache_compressed_pages;
    out_stats->kv_cache_bytes_saved = g_z->kv_cache_bytes_saved;
    out_stats->hot_resident_pages = hot_resident;
    out_stats->hot_resident_bytes = hot_resident * PAGE_SZ;
    out_stats->no_compress_resident_pages = no_compress_resident;
    out_stats->no_compress_resident_bytes = no_compress_resident * PAGE_SZ;
    out_stats->pool_pressure_percent = memx_pool_pressure_percent_locked(g_z);
    out_stats->tensor_delta_split_pages = g_z->tensor_delta_split_pages;
    out_stats->tensor_delta_split_bytes_saved = g_z->tensor_delta_split_bytes_saved;
    out_stats->tensor_exp_pack_pages = g_z->tensor_exp_pack_pages;
    out_stats->tensor_exp_pack_bytes_saved = g_z->tensor_exp_pack_bytes_saved;
    out_stats->running = 1;
    return 0;
}

int memx_runtime_get_pressure(memx_runtime_pressure_t *out_pressure) {
    if (!out_pressure) return EINVAL;
    memset(out_pressure, 0, sizeof(*out_pressure));
    if (!g_z || !g_z->running) return ENOENT;

    uint64_t free_pages = g_z->free_pages_count;
    uint64_t free_bytes = free_pages * PAGE_SZ;
    uint64_t used_bytes = g_z->vmem_size > free_bytes ? (g_z->vmem_size - free_bytes) : 0;
    uint32_t pressure = memx_pool_pressure_percent_locked(g_z);
    uint64_t free_extent_bytes = 0;
    uint64_t largest_free_extent = 0;
    uint32_t free_extent_count = 0;
    uint32_t fragmentation = 0;

    pthread_mutex_lock(&g_z->alloc_mutex);
    free_extent_bytes = pool_free_extent_bytes_locked(g_z);
    largest_free_extent = pool_largest_free_extent_locked(g_z);
    free_extent_count = g_z->pool_free_count;
    if (free_extent_bytes > 0 && largest_free_extent < free_extent_bytes) {
        fragmentation = (uint32_t)(((free_extent_bytes - largest_free_extent) * 100ULL) / free_extent_bytes);
    }
    pthread_mutex_unlock(&g_z->alloc_mutex);

    out_pressure->virtual_capacity_bytes = g_z->vmem_size;
    out_pressure->virtual_used_bytes = used_bytes;
    out_pressure->virtual_free_bytes = free_bytes;
    out_pressure->pool_capacity_bytes = g_z->pool_size;
    out_pressure->pool_cursor_bytes = g_z->pool_next;
    out_pressure->pool_used_bytes = g_z->pool_used;
    out_pressure->pool_headroom_bytes = (g_z->pool_size > g_z->pool_next) ? (g_z->pool_size - g_z->pool_next) : 0;
    out_pressure->pool_free_extent_bytes = free_extent_bytes;
    out_pressure->pool_largest_free_extent_bytes = largest_free_extent;
    out_pressure->pool_free_extent_count = free_extent_count;
    out_pressure->pool_fragmentation_percent = fragmentation;
    out_pressure->free_pages = free_pages;
    out_pressure->pool_pressure_percent = pressure;
    if (pressure < 95) {
        uint64_t headroom = g_z->pool_size > g_z->pool_next ? (g_z->pool_size - g_z->pool_next) : 0;
        uint32_t cursor_fill = g_z->pool_size ? (uint32_t)((g_z->pool_next * 100ULL) / g_z->pool_size) : 0;
        if (headroom * 20ULL <= g_z->pool_size && cursor_fill >= 95) pressure = cursor_fill;
    }
    out_pressure->pool_pressure_percent = pressure;
    out_pressure->pool_near_full = (pressure >= 95 || memx_pool_near_full_locked(g_z)) ? 1U : 0U;
    return 0;
}

int memx_runtime_reclaim(uint64_t *out_reclaimed_bytes) {
    if (!g_z || !g_z->running) return ENOENT;
    uint64_t reclaimed = 0;
    pthread_mutex_lock(&g_z->alloc_mutex);
    reclaimed = memx_runtime_reclaim_locked(g_z);
    pthread_mutex_unlock(&g_z->alloc_mutex);
    if (out_reclaimed_bytes) *out_reclaimed_bytes = reclaimed;
    return 0;
}

int memx_runtime_compact(uint64_t *out_reclaimed_bytes) {
    if (!g_z || !g_z->running) return ENOENT;
    uint64_t reclaimed = 0;
    pthread_mutex_lock(&g_z->alloc_mutex);
    reclaimed = memx_runtime_reclaim_and_compact_locked(g_z);
    pthread_mutex_unlock(&g_z->alloc_mutex);
    if (out_reclaimed_bytes) *out_reclaimed_bytes = reclaimed;
    return 0;
}

int memx_runtime_test_set_pool_cursor(size_t cursor_bytes) {
    if (!g_z || !g_z->running) return ENOENT;
    pthread_mutex_lock(&g_z->alloc_mutex);
    if (cursor_bytes > g_z->pool_size || cursor_bytes < g_z->pool_next) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return EINVAL;
    }
    g_z->pool_next = cursor_bytes;
    pthread_mutex_unlock(&g_z->alloc_mutex);
    return 0;
}

void *memx_runtime_malloc(size_t size) {
    if (memx_runtime_init() != 0) return NULL;
    return memx_alloc_internal(size, 0, 1, 0, 0, NULL);
}

void memx_runtime_free(void *ptr) {
    runtime_free_internal(ptr);
}

void *memx_runtime_calloc(size_t nmemb, size_t size) {
    if (memx_runtime_init() != 0) return NULL;
    return memx_calloc_internal(nmemb, size, 0, 1, 0);
}

void *memx_runtime_realloc(void *ptr, size_t size) {
    if (!ptr && memx_runtime_init() != 0) return NULL;
    return memx_realloc_internal(ptr, size, 0, 1, 0);
}

int memx_runtime_posix_memalign(void **memptr, size_t alignment, size_t size) {
    if (memx_runtime_init() != 0) return ENOMEM;
    if (!memptr) return EINVAL;
    if ((alignment & (alignment - 1)) != 0 || alignment < sizeof(void*)) return EINVAL;
    void *ptr = memx_alloc_internal(size, 0, 1, 0, 0, NULL);
    if (!ptr) return ENOMEM;
    if (((uintptr_t)ptr % alignment) != 0) {
        memx_runtime_free(ptr);
        return ENOMEM;
    }
    *memptr = ptr;
    return 0;
}

void *memx_runtime_aligned_alloc(size_t alignment, size_t size) {
    void *ptr = NULL;
    if (memx_runtime_posix_memalign(&ptr, alignment, size) != 0) return NULL;
    return ptr;
}

void *memx_runtime_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    if (memx_runtime_init() != 0) return MAP_FAILED;
    return memx_mmap_internal(addr, length, prot, flags, fd, offset, 0, 1);
}

int memx_runtime_munmap(void *addr, size_t length) {
    return memx_munmap(addr, length);
}

void *memx_runtime_context_malloc(memx_runtime_context_t *ctx, size_t size) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC) { errno = EINVAL; return NULL; }
    return memx_alloc_internal(size, (uintptr_t)ctx, 1, 0, 0, NULL);
}

void memx_runtime_context_free(memx_runtime_context_t *ctx, void *ptr) {
    if (!ptr) return;
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC) return;
    if (g_z && is_ours(ptr)) {
        size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
        if (g_z->meta[sp].owner_tag != (uintptr_t)ctx) return;
    } else {
        return;
    }
    memx_runtime_free(ptr);
}

void *memx_runtime_context_calloc(memx_runtime_context_t *ctx, size_t nmemb, size_t size) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC) { errno = EINVAL; return NULL; }
    return memx_calloc_internal(nmemb, size, (uintptr_t)ctx, 1, 0);
}

void *memx_runtime_context_realloc(memx_runtime_context_t *ctx, void *ptr, size_t size) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC) { errno = EINVAL; return NULL; }
    if (ptr && (!g_z || !is_ours(ptr))) { errno = EINVAL; return NULL; }
    return memx_realloc_internal(ptr, size, (uintptr_t)ctx, 1, 0);
}

void *memx_runtime_context_malloc_tensor(memx_runtime_context_t *ctx, size_t size, const memx_runtime_tensor_desc_t *desc) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !desc || !tensor_desc_is_valid(desc)) {
        errno = EINVAL;
        return NULL;
    }
    return memx_alloc_internal(size, (uintptr_t)ctx, 1, 0, 0, desc);
}

int memx_runtime_context_update_tensor_flags_range(memx_runtime_context_t *ctx, void *ptr, size_t offset, size_t length, uint32_t flags);

int memx_runtime_context_update_tensor_flags(memx_runtime_context_t *ctx, void *ptr, uint32_t flags) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !ptr) return EINVAL;
    if (!g_z || !is_ours(ptr)) return EINVAL;
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    pthread_mutex_lock(&g_z->alloc_mutex);
    if (sp >= g_z->npages ||
        g_z->meta[sp].owner_tag != (uintptr_t)ctx ||
        g_z->meta[sp].alloc_size == 0 ||
        g_z->meta[sp].tensor_role == MEMX_TENSOR_ROLE_UNKNOWN) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return EINVAL;
    }
    size_t size = g_z->meta[sp].alloc_size;
    pthread_mutex_unlock(&g_z->alloc_mutex);
    return memx_runtime_context_update_tensor_flags_range(ctx, ptr, 0, size, flags);
}

int memx_runtime_context_update_tensor_flags_range(memx_runtime_context_t *ctx, void *ptr, size_t offset, size_t length, uint32_t flags) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !ptr) return EINVAL;
    if (!g_z || !is_ours(ptr)) return EINVAL;
    if (length == 0) return 0;
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    pthread_mutex_lock(&g_z->alloc_mutex);
    if (sp >= g_z->npages ||
        g_z->meta[sp].owner_tag != (uintptr_t)ctx ||
        g_z->meta[sp].alloc_size == 0 ||
        g_z->meta[sp].tensor_role == MEMX_TENSOR_ROLE_UNKNOWN) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return EINVAL;
    }
    size_t size = g_z->meta[sp].alloc_size;
    if (offset >= size || length > size - offset) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return EINVAL;
    }
    size_t first_rel_page = offset / PAGE_SZ;
    size_t last_rel_page = (offset + length - 1) / PAGE_SZ;
    size_t first_page = sp + first_rel_page;
    size_t last_page = sp + last_rel_page;
    size_t allocation_pages = (size + PAGE_SZ - 1) / PAGE_SZ;
    if (first_rel_page >= allocation_pages || last_rel_page >= allocation_pages || last_page >= g_z->npages) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return EINVAL;
    }
    uint64_t hot_sub = 0, hot_add = 0, no_comp_sub = 0, no_comp_add = 0;
    size_t promote_pages[512];
    size_t promote_count = 0;
    int promote_overflow = 0;
    int want_hot = (flags & (MEMX_TENSOR_FLAG_HOT | MEMX_TENSOR_FLAG_NO_COMPRESS)) != 0;
    for (size_t i = first_page; i <= last_page; i++) {
        size_t rel_page = i - sp;
        size_t page_bytes = allocation_page_bytes(size, rel_page);
        uint32_t old_flags = g_z->meta[i].tensor_flags;
        if (old_flags == flags) {
            if (want_hot) {
                if (g_z->meta[i].state == PAGE_COMPRESSED) {
                    if (promote_count < 512) promote_pages[promote_count++] = i;
                    else promote_overflow = 1;
                } else if (g_z->meta[i].state == PAGE_HOT) {
                    if (g_z->meta[i].cooldown < 100) g_z->meta[i].cooldown = 100;
                } else if (g_z->meta[i].state == PAGE_RESIDENT) {
                    uint8_t old = __sync_val_compare_and_swap(&g_z->meta[i].state, PAGE_RESIDENT, PAGE_HOT);
                    if (old == PAGE_RESIDENT) {
                        g_z->meta[i].cooldown = 100;
                        g_z->meta[i].prefetched = 1;
                        hot_list_add(g_z, (uint32_t)i);
                    }
                }
            }
            continue;
        }
        if ((old_flags & MEMX_TENSOR_FLAG_HOT) && !(flags & MEMX_TENSOR_FLAG_HOT)) {
            hot_sub += page_bytes;
            if (g_z->live_hot_flag_pages) __sync_fetch_and_sub(&g_z->live_hot_flag_pages, 1);
        } else if (!(old_flags & MEMX_TENSOR_FLAG_HOT) && (flags & MEMX_TENSOR_FLAG_HOT)) {
            hot_add += page_bytes;
            __sync_fetch_and_add(&g_z->live_hot_flag_pages, 1);
        }
        if ((old_flags & MEMX_TENSOR_FLAG_NO_COMPRESS) && !(flags & MEMX_TENSOR_FLAG_NO_COMPRESS)) {
            no_comp_sub += page_bytes;
            if (g_z->live_nocomp_flag_pages) __sync_fetch_and_sub(&g_z->live_nocomp_flag_pages, 1);
        } else if (!(old_flags & MEMX_TENSOR_FLAG_NO_COMPRESS) && (flags & MEMX_TENSOR_FLAG_NO_COMPRESS)) {
            no_comp_add += page_bytes;
            __sync_fetch_and_add(&g_z->live_nocomp_flag_pages, 1);
        }
        g_z->meta[i].tensor_flags = flags;
        if (g_z->meta[i].tensor_dtype == MEMX_TENSOR_DTYPE_FP16 ||
            g_z->meta[i].tensor_dtype == MEMX_TENSOR_DTYPE_BF16) {
            if ((g_z->meta[i].tensor_role == MEMX_TENSOR_ROLE_WEIGHT ||
                 g_z->meta[i].tensor_role == MEMX_TENSOR_ROLE_EMBEDDING) &&
                (flags & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY))) {
                g_z->meta[i].preferred_codec = MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT;
                g_z->meta[i].codec_fail_streak = 0;
            } else if (g_z->meta[i].tensor_role == MEMX_TENSOR_ROLE_KV_CACHE) {
                g_z->meta[i].preferred_codec = MEMX_CODEC_TENSOR_FP16_DELTA_SPLIT;
                g_z->meta[i].codec_fail_streak = 0;
            }
        }
        if (!want_hot) {
            int seal = ((flags & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_SEQUENTIAL)) != 0) ||
                       g_z->meta[i].tensor_role == MEMX_TENSOR_ROLE_WEIGHT ||
                       g_z->meta[i].tensor_role == MEMX_TENSOR_ROLE_EMBEDDING ||
                       g_z->meta[i].tensor_role == MEMX_TENSOR_ROLE_KV_CACHE;
            if (seal) {
                g_z->meta[i].dirty = 0;
                g_z->meta[i].stable_ticks = 255;
                g_z->meta[i].cooldown = 0;
            }
            if (g_z->meta[i].state == PAGE_HOT) {
                uint8_t old = __sync_val_compare_and_swap(&g_z->meta[i].state, PAGE_HOT, PAGE_RESIDENT);
                if (old == PAGE_HOT) {
                    g_z->meta[i].prefetched = 0;
                    g_z->meta[i].cooldown = 0;
                    if (seal) {
                        mprotect((uint8_t*)g_z->vmem + i * PAGE_SZ, PAGE_SZ, PROT_READ);
                    }
                    res_list_add(g_z, (uint32_t)i);
                }
            } else if (g_z->meta[i].state == PAGE_RESIDENT) {
                g_z->meta[i].cooldown = 0;
                if (seal) {
                    mprotect((uint8_t*)g_z->vmem + i * PAGE_SZ, PAGE_SZ, PROT_READ);
                }
                res_list_add(g_z, (uint32_t)i);
            }
        } else {
            if (g_z->meta[i].state == PAGE_COMPRESSED) {
                if (promote_count < 512) promote_pages[promote_count++] = i;
                else promote_overflow = 1;
            } else if (g_z->meta[i].state == PAGE_RESIDENT) {
                uint8_t old = __sync_val_compare_and_swap(&g_z->meta[i].state, PAGE_RESIDENT, PAGE_HOT);
                if (old == PAGE_RESIDENT) {
                    g_z->meta[i].cooldown = 100;
                    g_z->meta[i].prefetched = 1;
                    hot_list_add(g_z, (uint32_t)i);
                }
            } else if (g_z->meta[i].state == PAGE_HOT) {
                if (g_z->meta[i].cooldown < 100) g_z->meta[i].cooldown = 100;
            }
        }
    }
    size_t overflow_first = first_page;
    size_t overflow_last = last_page;
    pthread_mutex_unlock(&g_z->alloc_mutex);
    for (size_t i = 0; i < promote_count; i++) {
        prefetch_page(g_z, promote_pages[i], 100);
    }
    if (promote_overflow) {
        for (size_t i = overflow_first; i <= overflow_last; i++) {
            if (g_z->meta[i].state == PAGE_COMPRESSED) prefetch_page(g_z, i, 100);
        }
    }
    context_adjust_tensor_flag_bytes((uintptr_t)ctx, hot_sub, hot_add, no_comp_sub, no_comp_add);
    return 0;
}

int memx_runtime_context_posix_memalign(memx_runtime_context_t *ctx, void **memptr, size_t alignment, size_t size) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !memptr) return EINVAL;
    if ((alignment & (alignment - 1)) != 0 || alignment < sizeof(void*)) return EINVAL;
    void *ptr = memx_alloc_internal(size, (uintptr_t)ctx, 1, 0, 0, NULL);
    if (!ptr) return ENOMEM;
    if (((uintptr_t)ptr % alignment) != 0) {
        memx_runtime_free(ptr);
        return ENOMEM;
    }
    *memptr = ptr;
    return 0;
}

void *memx_runtime_context_aligned_alloc(memx_runtime_context_t *ctx, size_t alignment, size_t size) {
    void *ptr = NULL;
    if (memx_runtime_context_posix_memalign(ctx, &ptr, alignment, size) != 0) return NULL;
    return ptr;
}

void *memx_runtime_context_mmap(memx_runtime_context_t *ctx, void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC) { errno = EINVAL; return MAP_FAILED; }
    if (memx_runtime_init() != 0) return MAP_FAILED;
    return memx_mmap_internal(addr, length, prot, flags, fd, offset, (uintptr_t)ctx, 1);
}

int memx_runtime_context_munmap(memx_runtime_context_t *ctx, void *addr, size_t length) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC) return EINVAL;
    if (!g_z || !is_ours(addr)) return EINVAL;
    if (g_z && is_ours(addr)) {
        size_t sp = ((uintptr_t)addr - (uintptr_t)g_z->vmem) / PAGE_SZ;
        if (g_z->meta[sp].owner_tag != (uintptr_t)ctx) return EINVAL;
    }
    return memx_munmap(addr, length);
}
