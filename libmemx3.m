// MemX explicit runtime:
// managed allocations, quota-aware contexts, and compressed virtual pages.

#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#if defined(__APPLE__)
#include <sys/clonefile.h>
#include <copyfile.h>
#endif
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
    int         pool_spill_fd;
    int         pool_detached;
    int         pool_ghost;
    uint64_t    pool_spill_bytes;
    uint64_t    pool_spill_events;
    uint64_t    pool_ghost_flushed;
    uint64_t    pool_ghost_stores;
    int         pool_vault_native;
    uint64_t    pool_vault_stores;
    uint64_t    pool_vault_reads;
    uint64_t    pool_vault_cache_hits;
    uint64_t    pool_vault_window_hits;
    uint64_t    pool_vault_wbuf_flushes;
    uint8_t    *vault_cache;
    uint64_t    vault_cache_bytes;
    uint64_t    vault_cache_next;
    uint32_t    vault_cache_slots;
    uint64_t   *vault_cache_off;
    uint32_t   *vault_cache_sz;
    uint32_t   *vault_cache_pos;
    uint8_t    *vault_cache_live;
    uint64_t   *vault_cache_gen;
    uint64_t    vault_cache_prefer_bytes;
    uint64_t    vault_avcs_events;
    uint64_t    vault_ring_reclaims;
    uint64_t    sov_tca_pages;
    uint64_t    sov_tca_bytes;
    uint64_t    vault_epoch;
    uint8_t    *vault_wbuf;
    uint64_t    vault_wbuf_cap;
    uint64_t    vault_wbuf_base;
    uint64_t    vault_wbuf_len;
    uint8_t    *vault_win[4];
    uint64_t    vault_win_base[4];
    uint64_t    vault_win_len[4];
    uint32_t    vault_win_clock;
    int         sovereign;
    int         sovereign_frozen;
    int         phoenix_sealed;
    uint32_t    sov_count;
    uint32_t    sov_cap;
    void       *sov_ents;
    uint64_t    sov_bytes;
    uint64_t    sov_hits;
    void       *sov_off_idx;
    uint64_t    sov_off_idx_bytes;
    uint32_t   *sov_pidx_map;
    uint64_t    sov_pidx_map_bytes;
    uint64_t    sov_warm_bytes;
    uint64_t    sov_stream_injects;
    uint64_t    sov_crw_spans;
    uint64_t    sov_crw_pages;
    uint64_t    sov_crw_bytes;
    uint64_t    sov_chronos_injects;
    uint8_t    *sov_warp_buf;
    uint64_t    sov_warp_buf_bytes;
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
    #define DEDUP_HT_SIZE 65536
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
    #define ASYNC_PF_WORKERS 4
    #define COMP_CPU_WORKERS 6
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
    #define ASYNC_SEAL_WORKERS 2
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

#ifndef MADV_PAGEOUT
#define MADV_PAGEOUT 10
#endif

static volatile int g_hard_quiesce = 0;

static void pool_pageout_range(MemXZone3 *s, uint64_t off, uint64_t sz) {
    if (!s || !s->pool || sz == 0) return;
    if (off >= s->pool_size) return;
    if (off + sz > s->pool_size) sz = s->pool_size - off;
    uint64_t start = off & ~((uint64_t)PAGE_SZ - 1);
    uint64_t end = (off + sz + PAGE_SZ - 1) & ~((uint64_t)PAGE_SZ - 1);
    if (end > s->pool_size) end = s->pool_size;
    if (end <= start) return;
    const uint64_t max_chunk = (uint64_t)PAGE_SZ * 64;
    for (uint64_t cur = start; cur < end; ) {
        uint64_t nend = cur + max_chunk;
        if (nend > end) nend = end;
        uint8_t *pa = s->pool + cur;
        size_t bytes = (size_t)(nend - cur);
        if (madvise(pa, bytes, MADV_PAGEOUT) != 0) {
#if defined(MADV_DONTNEED)
            /* pageout unavailable: do not DONTNEED live compressed blobs */
#endif
        }
        cur = nend;
    }
}

static void pool_pageout_live_locked(MemXZone3 *s) {
    if (!s || !s->pool) return;
    if (s->pool_next > 0) {
        pool_pageout_range(s, 0, s->pool_next);
        pool_pageout_range(s, 0, s->pool_next);
    }
}

static int g_pool_spill_force = 0;
static int pool_copy_blob_locked(MemXZone3 *s, uint64_t d_off, uint32_t d_sz, uint8_t *payload);
static void pool_hard_decommit_range(MemXZone3 *s, uint64_t off, uint64_t sz);
static void pool_prepare_write_range(MemXZone3 *s, uint64_t off, uint64_t sz);
static int pool_ensure_spill_fd_locked(MemXZone3 *s);
static int pool_vault_window_enabled(void);
static int pool_vault_wbuf_flush_locked(MemXZone3 *s);

static int pool_vault_native_enabled(void) {
    const char *e = getenv("MEMX_POOL_VAULT_NATIVE");
    if (e && e[0] == '0') return 0;
    if (e && e[0] == '1') return 1;
    e = getenv("MEMX_POOL_SPILL");
    if (e && e[0] == '0') return 0;
    return 1;
}

static int pool_is_vault_native(MemXZone3 *s) {
    return s && s->pool_vault_native;
}

typedef struct {
    uint32_t pidx;
    uint32_t csz;
    uint64_t off;
    uint32_t seq;
    uint8_t  codec;
    uint8_t  _pad[3];
} sov_ent_t;

typedef struct {
    uint64_t off;
    uint32_t idx;
} sov_off_pair_t;

static int sov_off_pair_cmp(const void *a, const void *b) {
    const sov_off_pair_t *x = (const sov_off_pair_t *)a;
    const sov_off_pair_t *y = (const sov_off_pair_t *)b;
    if (x->off < y->off) return -1;
    if (x->off > y->off) return 1;
    if (x->idx < y->idx) return -1;
    if (x->idx > y->idx) return 1;
    return 0;
}

static int sov_ent_cmp(const void *a, const void *b) {
    const sov_ent_t *x = (const sov_ent_t *)a;
    const sov_ent_t *y = (const sov_ent_t *)b;
    if (x->pidx < y->pidx) return -1;
    if (x->pidx > y->pidx) return 1;
    return 0;
}

static sov_ent_t *sov_find_locked(MemXZone3 *s, uint32_t pidx) {
    if (!s || !s->sov_ents || s->sov_count == 0) return NULL;
    sov_ent_t *ents = (sov_ent_t *)s->sov_ents;
    if (s->sov_pidx_map && s->sov_pidx_map_bytes) {
        size_t nmap = s->sov_pidx_map_bytes / sizeof(uint32_t);
        if ((size_t)pidx < nmap) {
            uint32_t ei1 = s->sov_pidx_map[pidx];
            if (ei1 != 0) {
                uint32_t ei = ei1 - 1u;
                if (ei < s->sov_count && ents[ei].pidx == pidx) return &ents[ei];
            }
            return NULL;
        }
    }
    uint32_t lo = 0, hi = s->sov_count;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (ents[mid].pidx < pidx) lo = mid + 1;
        else hi = mid;
    }
    if (lo < s->sov_count && ents[lo].pidx == pidx) return &ents[lo];
    return NULL;
}

static void sov_drop_locked(MemXZone3 *s) {
    if (!s) return;
    if (s->sov_ents && s->sov_ents != MAP_FAILED && s->sov_bytes) {
        munmap(s->sov_ents, (size_t)s->sov_bytes);
    }
    if (s->sov_off_idx && s->sov_off_idx != MAP_FAILED && s->sov_off_idx_bytes) {
        munmap(s->sov_off_idx, (size_t)s->sov_off_idx_bytes);
    }
    if (s->sov_pidx_map && s->sov_pidx_map != MAP_FAILED && s->sov_pidx_map_bytes) {
        munmap(s->sov_pidx_map, (size_t)s->sov_pidx_map_bytes);
    }
    s->sov_ents = NULL;
    s->sov_bytes = 0;
    s->sov_count = 0;
    s->sov_cap = 0;
    s->sov_off_idx = NULL;
    s->sov_off_idx_bytes = 0;
    s->sov_pidx_map = NULL;
    s->sov_pidx_map_bytes = 0;
    s->sov_warm_bytes = 0;
    s->sov_stream_injects = 0;
    s->sov_crw_spans = 0;
    s->sov_crw_pages = 0;
    s->sov_crw_bytes = 0;
    s->sov_chronos_injects = 0;
    s->sovereign = 0;
    s->sovereign_frozen = 0;
}

static int sov_enabled_env(void) {
    const char *e = getenv("MEMX_SOVEREIGN");
    if (e && e[0] == '0') return 0;
    if (e && e[0] == '1') return 1;
    return 1;
}

static void meta_release_physical_range(MemXZone3 *s, size_t first, size_t last_inclusive);

static int sov_build_locked(MemXZone3 *s) {
    if (!s || !s->meta) return -1;
    if (s->vault_wbuf_len) (void)pool_vault_wbuf_flush_locked(s);
    size_t last = s->vmem_next / PAGE_SZ;
    if (last > s->npages) last = s->npages;
    if (last == 0) last = s->npages;
    uint32_t n = 0;
    for (size_t i = 0; i < last; i++) {
        PageMeta *m = &s->meta[i];
        if (m->state != PAGE_COMPRESSED) continue;
        if (m->comp_size == 0 || m->comp_size > PAGE_SZ) continue;
        if (m->pool_offset + (uint64_t)m->comp_size > s->pool_size) continue;
        n++;
    }
    if (n == 0) {
        sov_drop_locked(s);
        s->sovereign = 1;
        s->sovereign_frozen = 1;
        return 0;
    }
    size_t bytes = (size_t)n * sizeof(sov_ent_t);
    bytes = (bytes + PAGE_SZ - 1) & ~((size_t)PAGE_SZ - 1);
    void *mem = mmap(NULL, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) return -1;
    sov_ent_t *ents = (sov_ent_t *)mem;
    uint32_t w = 0;
    for (size_t i = 0; i < last && w < n; i++) {
        PageMeta *m = &s->meta[i];
        if (m->state != PAGE_COMPRESSED) continue;
        if (m->comp_size == 0 || m->comp_size > PAGE_SZ) continue;
        if (m->pool_offset + (uint64_t)m->comp_size > s->pool_size) continue;
        ents[w].pidx = (uint32_t)i;
        ents[w].csz = m->comp_size;
        ents[w].off = m->pool_offset;
        ents[w].seq = m->write_seq;
        ents[w].codec = m->codec;
        ents[w]._pad[0] = ents[w]._pad[1] = ents[w]._pad[2] = 0;
        w++;
    }
    if (w > 1) qsort(ents, w, sizeof(sov_ent_t), sov_ent_cmp);
    if (s->sov_ents && s->sov_ents != MAP_FAILED && s->sov_bytes)
        munmap(s->sov_ents, (size_t)s->sov_bytes);
    s->sov_ents = ents;
    s->sov_bytes = bytes;
    s->sov_count = w;
    s->sov_cap = w;
    s->sovereign = 1;
    s->sovereign_frozen = 1;
    s->sov_hits = 0;
    s->sov_warm_bytes = 0;
    s->sov_stream_injects = 0;
    s->sov_crw_spans = 0;
    s->sov_crw_pages = 0;
    s->sov_crw_bytes = 0;
    s->sov_chronos_injects = 0;
    if (s->sov_off_idx && s->sov_off_idx != MAP_FAILED && s->sov_off_idx_bytes) {
        munmap(s->sov_off_idx, (size_t)s->sov_off_idx_bytes);
        s->sov_off_idx = NULL;
        s->sov_off_idx_bytes = 0;
    }
    if (w > 0) {
        size_t ib = (size_t)w * sizeof(uint32_t);
        ib = (ib + PAGE_SZ - 1) & ~((size_t)PAGE_SZ - 1);
        void *im = mmap(NULL, ib, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        size_t pb = (size_t)w * sizeof(sov_off_pair_t);
        sov_off_pair_t *pairs = (sov_off_pair_t *)mmap(NULL, pb, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (im != MAP_FAILED && pairs != MAP_FAILED) {
            uint32_t *idx = (uint32_t *)im;
            for (uint32_t i = 0; i < w; i++) {
                pairs[i].off = ents[i].off;
                pairs[i].idx = i;
            }
            qsort(pairs, w, sizeof(sov_off_pair_t), sov_off_pair_cmp);
            for (uint32_t i = 0; i < w; i++) idx[i] = pairs[i].idx;
            s->sov_off_idx = im;
            s->sov_off_idx_bytes = ib;
        } else {
            if (im != MAP_FAILED) munmap(im, ib);
        }
        if (pairs != MAP_FAILED) munmap(pairs, pb);
    }
    if (s->sov_pidx_map && s->sov_pidx_map != MAP_FAILED && s->sov_pidx_map_bytes) {
        munmap(s->sov_pidx_map, (size_t)s->sov_pidx_map_bytes);
        s->sov_pidx_map = NULL;
        s->sov_pidx_map_bytes = 0;
    }
    if (w > 0) {
        uint32_t hi = 0;
        for (uint32_t i = 0; i < w; i++) if (ents[i].pidx > hi) hi = ents[i].pidx;
        size_t nmap = (size_t)hi + 1ull;
        size_t mb = nmap * sizeof(uint32_t);
        mb = (mb + PAGE_SZ - 1) & ~((size_t)PAGE_SZ - 1);
        void *mm = mmap(NULL, mb, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (mm != MAP_FAILED) {
            uint32_t *pm = (uint32_t *)mm;
            for (uint32_t i = 0; i < w; i++) {
                uint32_t p = ents[i].pidx;
                if ((size_t)p < nmap) pm[p] = i + 1u;
            }
            s->sov_pidx_map = pm;
            s->sov_pidx_map_bytes = mb;
        }
    }
    return 0;
}

static void sov_release_structural_locked(MemXZone3 *s) {
    if (!s) return;
    if (s->dedup_hash && s->dedup_hash != MAP_FAILED) {
        madvise(s->dedup_hash, DEDUP_HT_SIZE * 8, MADV_DONTNEED);
        madvise(s->dedup_off, DEDUP_HT_SIZE * 8, MADV_DONTNEED);
        madvise(s->dedup_sz, DEDUP_HT_SIZE * 4, MADV_DONTNEED);
        madvise(s->dedup_ref, DEDUP_HT_SIZE * 4, MADV_DONTNEED);
        madvise(s->dedup_pending_free, DEDUP_HT_SIZE, MADV_DONTNEED);
    }
    if (s->dedup_rev && s->dedup_rev != MAP_FAILED && s->dedup_rev_size)
        madvise(s->dedup_rev, (size_t)s->dedup_rev_size * 4, MADV_DONTNEED);
    if (s->hot_list && s->hot_cap)
        madvise(s->hot_list, (size_t)s->hot_cap * 4, MADV_DONTNEED);
    if (s->res_list && s->res_cap)
        madvise(s->res_list, (size_t)s->res_cap * 4, MADV_DONTNEED);
    s->hot_count = 0;
    s->res_count = 0;
    if (s->meta && s->npages) {
        size_t used = s->vmem_next / PAGE_SZ;
        if (used > s->npages) used = s->npages;
        if (used + 8 < s->npages)
            meta_release_physical_range(s, used + 8, s->npages - 1);
        if (used > 0) {
            uintptr_t a0 = ((uintptr_t)&s->meta[0]) & ~((uintptr_t)PAGE_SZ - 1);
            uintptr_t a1 = ((uintptr_t)&s->meta[used - 1] + sizeof(PageMeta) + PAGE_SZ - 1) & ~((uintptr_t)PAGE_SZ - 1);
            if (a1 > a0) {
#if defined(MADV_FREE_REUSABLE)
                madvise((void*)a0, a1 - a0, MADV_FREE_REUSABLE);
#endif
                madvise((void*)a0, a1 - a0, MADV_PAGEOUT);
            }
        }
    }
    if (s->free_bm && s->free_bm_size) {
        size_t used = s->vmem_next / PAGE_SZ;
        size_t word0 = (used + 63) / 64;
        size_t start = word0 * sizeof(uint64_t);
        start = (start + PAGE_SZ - 1) & ~((size_t)PAGE_SZ - 1);
        size_t oldb = s->free_bm_size * sizeof(uint64_t);
        if (start < oldb)
            madvise((uint8_t*)s->free_bm + start, oldb - start, MADV_DONTNEED);
    }
}


static int pool_vault_cache_enabled(void) {
    const char *e = getenv("MEMX_VAULT_CACHE");
    if (e && e[0] == '0') return 0;
    return 1;
}

static void pool_vault_cache_reset_locked(MemXZone3 *s) {
    if (!s || !s->vault_cache_live || s->vault_cache_slots == 0) return;
    memset(s->vault_cache_live, 0, s->vault_cache_slots);
    s->vault_cache_next = 0;
}

static void pool_vault_cache_invalidate_range_locked(MemXZone3 *s, uint64_t off, uint32_t sz) {
    if (!s || !s->vault_cache_live || s->vault_cache_slots == 0 || sz == 0) return;
    uint64_t end = off + (uint64_t)sz;
    for (uint32_t i = 0; i < s->vault_cache_slots; i++) {
        if (!s->vault_cache_live[i]) continue;
        uint64_t o = s->vault_cache_off[i];
        uint32_t z = s->vault_cache_sz[i];
        if (z == 0) { s->vault_cache_live[i] = 0; continue; }
        uint64_t e = o + (uint64_t)z;
        if (e <= off || end <= o) continue;
        s->vault_cache_live[i] = 0;
    }
}

static int pool_vault_cache_init_locked(MemXZone3 *s) {
    if (!s || s->vault_cache) return 0;
    if (!pool_vault_cache_enabled()) return 0;
    size_t bytes = 64ull * 1024ull * 1024ull;
    if (s->vault_cache_prefer_bytes >= (4ull * 1024ull * 1024ull) &&
        s->vault_cache_prefer_bytes <= (512ull * 1024ull * 1024ull)) {
        bytes = s->vault_cache_prefer_bytes;
    } else {
        const char *e = getenv("MEMX_VAULT_CACHE_MB");
        if (e && e[0]) {
            long mb = strtol(e, NULL, 10);
            if (mb >= 4 && mb <= 512) bytes = (size_t)mb * 1024ull * 1024ull;
        }
    }
    uint32_t slots = (uint32_t)(bytes / PAGE_SZ);
    if (slots < 64) slots = 64;
    uint8_t *buf = (uint8_t*)mmap(NULL, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (buf == MAP_FAILED) return -1;
    uint64_t *off = (uint64_t*)mmap(NULL, slots * sizeof(uint64_t), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    uint32_t *sz = (uint32_t*)mmap(NULL, slots * sizeof(uint32_t), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    uint32_t *pos = (uint32_t*)mmap(NULL, slots * sizeof(uint32_t), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    uint8_t *live = (uint8_t*)mmap(NULL, slots, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    uint64_t *gen = (uint64_t*)mmap(NULL, slots * sizeof(uint64_t), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (off == MAP_FAILED || sz == MAP_FAILED || pos == MAP_FAILED || live == MAP_FAILED || gen == MAP_FAILED) {
        if (buf != MAP_FAILED) munmap(buf, bytes);
        if (off != MAP_FAILED) munmap(off, slots * sizeof(uint64_t));
        if (sz != MAP_FAILED) munmap(sz, slots * sizeof(uint32_t));
        if (pos != MAP_FAILED) munmap(pos, slots * sizeof(uint32_t));
        if (live != MAP_FAILED) munmap(live, slots);
        if (gen != MAP_FAILED) munmap(gen, slots * sizeof(uint64_t));
        return -1;
    }
    memset(live, 0, slots);
    memset(gen, 0, slots * sizeof(uint64_t));
    s->vault_cache = buf;
    s->vault_cache_bytes = bytes;
    s->vault_cache_next = 0;
    s->vault_cache_slots = slots;
    s->vault_cache_off = off;
    s->vault_cache_sz = sz;
    s->vault_cache_pos = pos;
    s->vault_cache_live = live;
    s->vault_cache_gen = gen;
    return 0;
}

static int pool_vault_cache_get_locked(MemXZone3 *s, uint64_t d_off, uint32_t d_sz, uint8_t *payload) {
    if (!s || !s->vault_cache || !s->vault_cache_live || d_sz == 0 || d_sz > PAGE_SZ) return 0;
    uint32_t n = s->vault_cache_slots;
    uint32_t h = (uint32_t)((d_off ^ (d_off >> 17) ^ d_sz) % n);
    for (uint32_t probe = 0; probe < 8; probe++) {
        uint32_t i = (h + probe) % n;
        if (!s->vault_cache_live[i]) continue;
        if (s->vault_cache_off[i] == d_off && s->vault_cache_sz[i] == d_sz &&
            s->vault_cache_gen && s->vault_cache_gen[i] == s->vault_epoch) {
            uint32_t p = s->vault_cache_pos[i];
            if ((uint64_t)p + d_sz > s->vault_cache_bytes) continue;
            memcpy(payload, s->vault_cache + p, d_sz);
            s->pool_vault_cache_hits++;
            return 1;
        }
    }
    return 0;
}

static void pool_vault_cache_put_locked(MemXZone3 *s, uint64_t d_off, uint32_t d_sz, const uint8_t *payload) {
    if (!s || !payload || d_sz == 0 || d_sz > PAGE_SZ) return;
    if (pool_vault_cache_init_locked(s) != 0) return;
    if (!s->vault_cache) return;
    if ((uint64_t)d_sz > s->vault_cache_bytes) return;
    pool_vault_cache_invalidate_range_locked(s, d_off, d_sz);
    if (s->vault_cache_next + d_sz > s->vault_cache_bytes) {
        pool_vault_cache_reset_locked(s);
        if (s->vault_cache && s->vault_cache_bytes) {
            void *rm = mmap(s->vault_cache, (size_t)s->vault_cache_bytes,
                            PROT_READ | PROT_WRITE,
                            MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (rm == MAP_FAILED) {
#if defined(MADV_FREE_REUSABLE)
                madvise(s->vault_cache, (size_t)s->vault_cache_bytes, MADV_FREE_REUSABLE);
#endif
                madvise(s->vault_cache, (size_t)s->vault_cache_bytes, MADV_DONTNEED);
            }
            s->vault_ring_reclaims++;
        }
        s->vault_cache_next = 0;
    }
    uint32_t p = (uint32_t)s->vault_cache_next;
    memcpy(s->vault_cache + p, payload, d_sz);
    s->vault_cache_next += d_sz;
    uint32_t n = s->vault_cache_slots;
    uint32_t h = (uint32_t)((d_off ^ (d_off >> 17) ^ d_sz) % n);
    uint32_t slot = h;
    for (uint32_t probe = 0; probe < 8; probe++) {
        uint32_t i = (h + probe) % n;
        if (!s->vault_cache_live[i] || (s->vault_cache_off[i] == d_off && s->vault_cache_sz[i] == d_sz)) {
            slot = i;
            break;
        }
        slot = i;
    }
    s->vault_cache_off[slot] = d_off;
    s->vault_cache_sz[slot] = d_sz;
    s->vault_cache_pos[slot] = p;
    s->vault_cache_live[slot] = 1;
    if (s->vault_cache_gen) s->vault_cache_gen[slot] = s->vault_epoch;
}

static void pool_vault_cache_release_locked(MemXZone3 *s) {
    if (!s) return;
    if (s->vault_cache && s->vault_cache != MAP_FAILED && s->vault_cache_bytes) {
        munmap(s->vault_cache, (size_t)s->vault_cache_bytes);
    }
    if (s->vault_cache_off && s->vault_cache_off != MAP_FAILED && s->vault_cache_slots)
        munmap(s->vault_cache_off, (size_t)s->vault_cache_slots * sizeof(uint64_t));
    if (s->vault_cache_sz && s->vault_cache_sz != MAP_FAILED && s->vault_cache_slots)
        munmap(s->vault_cache_sz, (size_t)s->vault_cache_slots * sizeof(uint32_t));
    if (s->vault_cache_pos && s->vault_cache_pos != MAP_FAILED && s->vault_cache_slots)
        munmap(s->vault_cache_pos, (size_t)s->vault_cache_slots * sizeof(uint32_t));
    if (s->vault_cache_live && s->vault_cache_live != MAP_FAILED && s->vault_cache_slots)
        munmap(s->vault_cache_live, (size_t)s->vault_cache_slots);
    if (s->vault_cache_gen && s->vault_cache_gen != MAP_FAILED && s->vault_cache_slots)
        munmap(s->vault_cache_gen, (size_t)s->vault_cache_slots * sizeof(uint64_t));
    s->vault_cache = NULL;
    s->vault_cache_bytes = 0;
    s->vault_cache_next = 0;
    s->vault_cache_slots = 0;
    s->vault_cache_off = NULL;
    s->vault_cache_sz = NULL;
    s->vault_cache_pos = NULL;
    s->vault_cache_live = NULL;
    s->vault_cache_gen = NULL;
}

static int pool_vault_avcs_enabled(void) {
    const char *e = getenv("MEMX_VAULT_AVCS");
    if (e && e[0] == '0') return 0;
    return 1;
}

static size_t pool_vault_cache_infer_target_bytes(void) {
    long mb = 32;
    const char *e = getenv("MEMX_VAULT_CACHE_INFER_MB");
    if (e && e[0]) {
        long v = strtol(e, NULL, 10);
        if (v >= 4 && v <= 512) mb = v;
    }
    return (size_t)mb * 1024ull * 1024ull;
}

static void pool_vault_cache_avcs_locked(MemXZone3 *s) {
    if (!s || !pool_vault_avcs_enabled()) return;
    size_t target = pool_vault_cache_infer_target_bytes();
    if (!s->vault_cache || s->vault_cache_bytes <= target) return;
    enum { KEEP_MAX = 8192 };
    typedef struct { uint64_t off; uint32_t sz; uint32_t pos; } keep_t;
    keep_t *keep = (keep_t *)malloc(sizeof(keep_t) * KEEP_MAX);
    int nk = 0;
    size_t kept_bytes = 0;
    if (keep && s->vault_cache_live && s->vault_cache_slots) {
        for (uint32_t i = 0; i < s->vault_cache_slots && nk < KEEP_MAX; i++) {
            if (!s->vault_cache_live[i]) continue;
            if (s->vault_cache_gen && s->vault_cache_gen[i] != s->vault_epoch) continue;
            uint32_t z = s->vault_cache_sz[i];
            uint32_t p = s->vault_cache_pos[i];
            if (z == 0 || z > PAGE_SZ) continue;
            if ((uint64_t)p + z > s->vault_cache_bytes) continue;
            keep[nk].off = s->vault_cache_off[i];
            keep[nk].sz = z;
            keep[nk].pos = p;
            nk++;
        }
        for (int a = 1; a < nk; a++) {
            keep_t key = keep[a];
            int b = a - 1;
            while (b >= 0 && keep[b].pos > key.pos) {
                keep[b + 1] = keep[b];
                b--;
            }
            keep[b + 1] = key;
        }
        kept_bytes = 0;
        int nk2 = 0;
        for (int i = 0; i < nk; i++) {
            if (kept_bytes + keep[i].sz > target) break;
            keep[nk2++] = keep[i];
            kept_bytes += keep[i].sz;
        }
        nk = nk2;
    }
    uint8_t *snapshot = NULL;
    if (nk > 0 && kept_bytes > 0) {
        snapshot = (uint8_t *)mmap(NULL, kept_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (snapshot == MAP_FAILED) snapshot = NULL;
        else {
            size_t cursor = 0;
            for (int i = 0; i < nk; i++) {
                memcpy(snapshot + cursor, s->vault_cache + keep[i].pos, keep[i].sz);
                keep[i].pos = (uint32_t)cursor;
                cursor += keep[i].sz;
            }
        }
    }
    pool_vault_cache_release_locked(s);
    s->vault_cache_prefer_bytes = target;
    s->vault_epoch++;
    if (s->vault_epoch == 0) s->vault_epoch = 1;
    (void)pool_vault_cache_init_locked(s);
    if (s->vault_cache && snapshot && nk > 0) {
        for (int i = 0; i < nk; i++) {
            pool_vault_cache_put_locked(s, keep[i].off, keep[i].sz, snapshot + keep[i].pos);
        }
    }
    if (snapshot && snapshot != MAP_FAILED) munmap(snapshot, kept_bytes ? kept_bytes : 1);
    free(keep);
    s->vault_avcs_events++;
}

static pthread_mutex_t g_tca_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static uint8_t *g_tca_pool = NULL;
static size_t g_tca_pool_bytes = 0;
static int g_tca_pool_inuse = 0;
static uint64_t g_tca_pool_hits = 0;
static uint64_t g_tca_pool_grows = 0;
static uint8_t *g_tca_pipe[2] = {NULL, NULL};
static size_t g_tca_pipe_cap[2] = {0, 0};
enum { TCA_STICKY_MAX = 512, TCA_STICKY_SLOTS = 8 };
typedef struct {
    size_t pidx[TCA_STICKY_MAX];
    int n;
    uint8_t *arena;
    size_t bytes;
    uint64_t hash;
    int valid;
    uint64_t gen;
} tca_sticky_slot_t;
static __thread tca_sticky_slot_t g_tca_slots[TCA_STICKY_SLOTS];
static __thread tca_sticky_slot_t *g_tca_active = NULL;
static __thread uint64_t g_tca_slot_gen = 1;
static uint64_t g_tca_sticky_hits = 0;
static uint64_t g_tca_sticky_partial = 0;
static uint64_t g_tca_sticky_diff_pages = 0;

static void tca_sticky_clear(void);
static void tca_pipe_release_all(void) {
    for (int i = 0; i < 2; i++) {
        if (g_tca_pipe[i] && g_tca_pipe[i] != MAP_FAILED && g_tca_pipe_cap[i]) {
            munmap(g_tca_pipe[i], g_tca_pipe_cap[i]);
        }
        g_tca_pipe[i] = NULL;
        g_tca_pipe_cap[i] = 0;
    }
}

static void tca_pool_destroy(void) {
    pthread_mutex_lock(&g_tca_pool_mu);
    if (g_tca_pool && g_tca_pool != MAP_FAILED && g_tca_pool_bytes && !g_tca_pool_inuse) {
        munmap(g_tca_pool, g_tca_pool_bytes);
        g_tca_pool = NULL;
        g_tca_pool_bytes = 0;
    } else if (g_tca_pool && g_tca_pool_inuse) {
        g_tca_pool_inuse = 0;
        if (g_tca_pool && g_tca_pool != MAP_FAILED && g_tca_pool_bytes) {
            munmap(g_tca_pool, g_tca_pool_bytes);
        }
        g_tca_pool = NULL;
        g_tca_pool_bytes = 0;
    }
    tca_pipe_release_all();
    pthread_mutex_unlock(&g_tca_pool_mu);
    tca_sticky_clear();
}

static uint8_t *tca_arena_acquire(size_t need, size_t *out_cap) {
    if (need == 0) return NULL;
    size_t cap = need;
    if (cap < (size_t)(1u << 20)) cap = (size_t)(1u << 20);
    cap = (cap + ((1u << 20) - 1)) & ~((size_t)(1u << 20) - 1);
    pthread_mutex_lock(&g_tca_pool_mu);
    if (g_tca_pool && g_tca_pool_bytes >= need && !g_tca_pool_inuse) {
        g_tca_pool_inuse = 1;
        g_tca_pool_hits++;
        if (out_cap) *out_cap = g_tca_pool_bytes;
        uint8_t *p = g_tca_pool;
        pthread_mutex_unlock(&g_tca_pool_mu);
        return p;
    }
    if (g_tca_pool && !g_tca_pool_inuse && g_tca_pool != MAP_FAILED && g_tca_pool_bytes) {
        munmap(g_tca_pool, g_tca_pool_bytes);
        g_tca_pool = NULL;
        g_tca_pool_bytes = 0;
    }
    if (g_tca_pool_inuse) {
        pthread_mutex_unlock(&g_tca_pool_mu);
        void *m = mmap(NULL, cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) return NULL;
        if (out_cap) *out_cap = cap;
        return (uint8_t *)m;
    }
    void *m = mmap(NULL, cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) {
        pthread_mutex_unlock(&g_tca_pool_mu);
        return NULL;
    }
    g_tca_pool = (uint8_t *)m;
    g_tca_pool_bytes = cap;
    g_tca_pool_inuse = 1;
    g_tca_pool_grows++;
    if (out_cap) *out_cap = cap;
    pthread_mutex_unlock(&g_tca_pool_mu);
    return g_tca_pool;
}

static void tca_arena_release(uint8_t *p, size_t cap, int destroy) {
    if (!p) return;
    pthread_mutex_lock(&g_tca_pool_mu);
    if (p == g_tca_pool) {
        g_tca_pool_inuse = 0;
        if (destroy) {
            munmap(g_tca_pool, g_tca_pool_bytes);
            g_tca_pool = NULL;
            g_tca_pool_bytes = 0;
        }
        pthread_mutex_unlock(&g_tca_pool_mu);
        return;
    }
    pthread_mutex_unlock(&g_tca_pool_mu);
    if (cap) munmap(p, cap);
}

static uint8_t *tca_pipe_buf(int slot, size_t need) {
    if (slot < 0 || slot > 1) return NULL;
    if (need < PAGE_SZ) need = PAGE_SZ;
    if (need > (1u << 20)) need = (1u << 20);
    if (g_tca_pipe[slot] && g_tca_pipe_cap[slot] >= need) return g_tca_pipe[slot];
    if (g_tca_pipe[slot] && g_tca_pipe[slot] != MAP_FAILED && g_tca_pipe_cap[slot])
        munmap(g_tca_pipe[slot], g_tca_pipe_cap[slot]);
    void *m = mmap(NULL, need, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) {
        g_tca_pipe[slot] = NULL;
        g_tca_pipe_cap[slot] = 0;
        return NULL;
    }
    g_tca_pipe[slot] = (uint8_t *)m;
    g_tca_pipe_cap[slot] = need;
    return g_tca_pipe[slot];
}

static int tca_sticky_enabled(void) {
    const char *e = getenv("MEMX_TCA_STICKY");
    if (e && e[0] == '0') return 0;
    return 1;
}

static int tca_sticky_diff_enabled(void) {
    const char *e = getenv("MEMX_TCA_DIFF");
    if (e && e[0] == '0') return 0;
    return 1;
}

static int tca_sticky_slot_count(void) {
    const char *e = getenv("MEMX_TCA_SLOTS");
    if (e && e[0]) {
        int n = atoi(e);
        if (n < 1) n = 1;
        if (n > TCA_STICKY_SLOTS) n = TCA_STICKY_SLOTS;
        return n;
    }
    return TCA_STICKY_SLOTS;
}

static uint64_t tca_pidx_hash(const size_t *uniq, int n) {
    uint64_t h = 14695981039346656037ull;
    for (int i = 0; i < n; i++) {
        h ^= (uint64_t)uniq[i] + 0x9e3779b97f4a7c15ull;
        h *= 1099511628211ull;
    }
    h ^= (uint64_t)(uint32_t)n * 0x85ebca6bull;
    return h ? h : 1ull;
}

static void tca_sticky_slot_free(tca_sticky_slot_t *sl) {
    if (!sl) return;
    if (sl->arena && sl->arena != MAP_FAILED && sl->bytes) {
        munmap(sl->arena, sl->bytes);
    }
    sl->arena = NULL;
    sl->bytes = 0;
    sl->n = 0;
    sl->valid = 0;
    sl->hash = 0;
    sl->gen = 0;
}

static void tca_sticky_clear(void) {
    for (int i = 0; i < TCA_STICKY_SLOTS; i++) tca_sticky_slot_free(&g_tca_slots[i]);
    g_tca_active = NULL;
}

static int tca_sticky_owns(const uint8_t *p) {
    if (!p) return 0;
    for (int i = 0; i < TCA_STICKY_SLOTS; i++) {
        if (g_tca_slots[i].arena == p) return 1;
    }
    return 0;
}

static int tca_sticky_eq(const tca_sticky_slot_t *sl, const size_t *uniq, int n, uint64_t h) {
    if (!sl || !sl->valid || !sl->arena || sl->n != n || sl->hash != h) return 0;
    for (int i = 0; i < n; i++) {
        if (sl->pidx[i] != uniq[i]) return 0;
    }
    return 1;
}

static tca_sticky_slot_t *tca_sticky_find(const size_t *uniq, int n, uint64_t h) {
    if (!tca_sticky_enabled() || !uniq || n <= 0 || n > TCA_STICKY_MAX) return NULL;
    int slots = tca_sticky_slot_count();
    for (int i = 0; i < slots; i++) {
        if (tca_sticky_eq(&g_tca_slots[i], uniq, n, h)) {
            g_tca_slots[i].gen = ++g_tca_slot_gen;
            return &g_tca_slots[i];
        }
    }
    return NULL;
}

static uint8_t *tca_sticky_try_hit(const size_t *uniq, int n, size_t *out_bytes) {
    if (!tca_sticky_enabled() || !uniq || n <= 0) return NULL;
    uint64_t h = tca_pidx_hash(uniq, n);
    tca_sticky_slot_t *sl = tca_sticky_find(uniq, n, h);
    if (!sl) return NULL;
    g_tca_active = sl;
    if (out_bytes) *out_bytes = sl->bytes;
    return sl->arena;
}

static tca_sticky_slot_t *tca_sticky_best_partial(const size_t *uniq, int n, int *out_hits) {
    tca_sticky_slot_t *best = NULL;
    int best_hits = 0;
    int slots = tca_sticky_slot_count();
    for (int s = 0; s < slots; s++) {
        tca_sticky_slot_t *sl = &g_tca_slots[s];
        if (!sl->valid || !sl->arena || sl->n <= 0) continue;
        int hits = 0;
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < sl->n; b++) {
                if (uniq[a] == sl->pidx[b]) { hits++; break; }
            }
        }
        if (hits > best_hits) {
            best_hits = hits;
            best = sl;
        }
    }
    if (out_hits) *out_hits = best_hits;
    return best;
}

static int tca_sticky_seed_from(tca_sticky_slot_t *src, const size_t *uniq, int n, uint8_t *dst, uint8_t *mask) {
    if (!src || !src->arena || !uniq || !dst || !mask || n <= 0) return 0;
    int hits = 0;
    for (int a = 0; a < n; a++) {
        mask[a] = 0;
        for (int b = 0; b < src->n; b++) {
            if (uniq[a] == src->pidx[b]) {
                memcpy(dst + (size_t)a * (size_t)PAGE_SZ,
                       src->arena + (size_t)b * (size_t)PAGE_SZ,
                       (size_t)PAGE_SZ);
                mask[a] = 1;
                hits++;
                break;
            }
        }
    }
    return hits;
}

static tca_sticky_slot_t *tca_sticky_pick_victim_except(tca_sticky_slot_t *keep) {
    int slots = tca_sticky_slot_count();
    tca_sticky_slot_t *empty = NULL;
    tca_sticky_slot_t *best = NULL;
    uint64_t best_gen = UINT64_MAX;
    for (int i = 0; i < slots; i++) {
        tca_sticky_slot_t *sl = &g_tca_slots[i];
        if (sl == keep) continue;
        if (!sl->valid && !sl->arena) return sl;
        if (!sl->valid && !empty) empty = sl;
        if (sl->gen < best_gen) {
            best_gen = sl->gen;
            best = sl;
        }
    }
    if (empty) return empty;
    if (best) return best;
    for (int i = 0; i < slots; i++) {
        if (&g_tca_slots[i] != keep) return &g_tca_slots[i];
    }
    return &g_tca_slots[0];
}

static size_t tca_sticky_budget_bytes(void) {
    const char *e = getenv("MEMX_TCA_BUDGET_MB");
    if (e && e[0]) {
        long mb = atol(e);
        if (mb < 8) mb = 8;
        if (mb > 512) mb = 512;
        return (size_t)mb << 20;
    }
    return (size_t)96 << 20;
}

static void tca_sticky_enforce_budget(tca_sticky_slot_t *keep_a, tca_sticky_slot_t *keep_b) {
    size_t budget = tca_sticky_budget_bytes();
    for (;;) {
        size_t total = 0;
        tca_sticky_slot_t *victim = NULL;
        uint64_t vgen = UINT64_MAX;
        int slots = tca_sticky_slot_count();
        for (int i = 0; i < slots; i++) {
            tca_sticky_slot_t *sl = &g_tca_slots[i];
            if (!sl->arena || !sl->bytes) continue;
            total += sl->bytes;
            if (sl == keep_a || sl == keep_b) continue;
            if (sl->gen < vgen) {
                vgen = sl->gen;
                victim = sl;
            }
        }
        if (total <= budget || !victim) break;
        tca_sticky_slot_free(victim);
    }
}

static uint8_t *tca_sticky_prepare(const size_t *uniq, int n, size_t *out_bytes, tca_sticky_slot_t *keep) {
    if (!tca_sticky_enabled() || !uniq || n <= 0 || n > TCA_STICKY_MAX) return NULL;
    uint64_t h = tca_pidx_hash(uniq, n);
    size_t need = (size_t)n * (size_t)PAGE_SZ;
    tca_sticky_slot_t *sl = tca_sticky_pick_victim_except(keep);
    if (sl->bytes < need || !sl->arena) {
        tca_sticky_slot_free(sl);
        tca_sticky_enforce_budget(keep, NULL);
        void *m = mmap(NULL, need, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) return NULL;
        sl->arena = (uint8_t *)m;
        sl->bytes = need;
    }
    sl->n = n;
    for (int i = 0; i < n; i++) sl->pidx[i] = uniq[i];
    sl->hash = h;
    sl->valid = 0;
    sl->gen = ++g_tca_slot_gen;
    g_tca_active = sl;
    tca_sticky_enforce_budget(sl, keep);
    if (out_bytes) *out_bytes = sl->bytes;
    return sl->arena;
}

static void tca_sticky_commit(void) {
    if (g_tca_active && g_tca_active->arena && g_tca_active->n > 0) g_tca_active->valid = 1;
}

static void tca_sticky_invalidate_active(void) {
    if (g_tca_active) g_tca_active->valid = 0;
}

static int sov_tca_enabled(void) {
    const char *e = getenv("MEMX_SOV_TCA");
    if (e && e[0] == '0') return 0;
    return 1;
}

static int sov_tca_novault(void) {
    const char *e = getenv("MEMX_TCA_NO_VAULT");
    if (e && e[0] == '0') return 0;
    return 1;
}

static int sov_tca_warp_enabled(void) {
    const char *e = getenv("MEMX_TCA_WARP");
    if (e && e[0] == '0') return 0;
    return 1;
}

static void cpu_decompress(const uint8_t *src, uint32_t cs, uint8_t *dst);

typedef struct {
    uint64_t off;
    uint32_t csz;
    uint32_t seq;
    int arena_i;
} tca_item_t;

static int tca_item_cmp(const void *a, const void *b) {
    const tca_item_t *x = (const tca_item_t *)a;
    const tca_item_t *y = (const tca_item_t *)b;
    if (x->off < y->off) return -1;
    if (x->off > y->off) return 1;
    if (x->arena_i < y->arena_i) return -1;
    if (x->arena_i > y->arena_i) return 1;
    return 0;
}

static void tca_rdadvise(int fd, off_t off, size_t len) {
    if (fd <= 2 || len == 0) return;
#if defined(F_RDADVISE)
    struct radvisory ra;
    ra.ra_offset = off;
    ra.ra_count = (int)((len > (size_t)0x7fffffff) ? 0x7fffffff : len);
    (void)fcntl(fd, F_RDADVISE, &ra);
#else
    (void)off; (void)len;
#endif
}

static int sov_tca_fill_arena_ex(MemXZone3 *s, const size_t *pidxs, int n, uint8_t *arena, const uint8_t *seed);

static int sov_tca_fill_arena(MemXZone3 *s, const size_t *pidxs, int n, uint8_t *arena) {
    return sov_tca_fill_arena_ex(s, pidxs, n, arena, NULL);
}

static int sov_tca_fill_arena_ex(MemXZone3 *s, const size_t *pidxs, int n, uint8_t *arena, const uint8_t *seed) {
    if (!s || !pidxs || n <= 0 || !arena) return 0;
    if (!s->sovereign || !s->sov_ents || s->sov_count == 0) return 0;
    if (s->pool_spill_fd <= 2 || s->pool_spill_bytes == 0) return 0;
    enum { NMAX = 2048 };
    if (n > NMAX) n = NMAX;
    tca_item_t items[NMAX];
    int m = 0;
    int novault = sov_tca_novault();
    int use_warp = sov_tca_warp_enabled();
    int seed_hits = 0;
    if (seed) {
        for (int i = 0; i < n; i++) if (seed[i]) seed_hits++;
    }
    for (int i = 0; i < n; i++) {
        if (seed && seed[i]) continue;
        size_t pidx = pidxs[i];
        if (pidx >= s->npages) continue;
        sov_ent_t *e = sov_find_locked(s, (uint32_t)pidx);
        if (!e || e->csz == 0 || e->csz > PAGE_SZ) continue;
        if (e->off + (uint64_t)e->csz > s->pool_spill_bytes) continue;
        items[m].off = e->off;
        items[m].csz = e->csz;
        items[m].seq = e->seq;
        items[m].arena_i = i;
        m++;
    }
    if (m <= 0) {
        if (seed_hits > 0) {
            __sync_fetch_and_add(&s->sov_hits, (uint64_t)seed_hits);
            return seed_hits;
        }
        return 0;
    }
    if (m > 1) qsort(items, (size_t)m, sizeof(items[0]), tca_item_cmp);

    int filled_stack[2048];
    int *filled = filled_stack;
    int filled_heap = 0;
    if (n > 2048) {
        filled = (int *)calloc((size_t)n, sizeof(int));
        if (!filled) return 0;
        filled_heap = 1;
    } else {
        memset(filled_stack, 0, (size_t)n * sizeof(int));
    }
    if (seed) {
        for (int i = 0; i < n; i++) if (seed[i]) filled[i] = 1;
    }
    __block int done = seed_hits;

    if (use_warp && m >= 1) {
        enum { SMAX = 256 };
        typedef struct { int i0, i1; uint64_t base; size_t span; } span_t;
        span_t spans[SMAX];
        int ns = 0;
        int i = 0;
        while (i < m && ns < SMAX) {
            uint64_t base = items[i].off;
            uint64_t run_end = items[i].off + items[i].csz;
            int j = i + 1;
            while (j < m) {
                uint64_t o = items[j].off;
                uint32_t c = items[j].csz;
                uint64_t e2 = o + c;
                if (o < run_end) {
                    if (e2 > run_end) run_end = e2;
                    j++;
                    continue;
                }
                if (o > run_end + 4096ull) break;
                if (e2 - base > (1ull << 20)) break;
                if (j - i >= 96) break;
                run_end = e2;
                j++;
            }
            size_t span = (size_t)(run_end - base);
            if (span > 0 && span <= (1u << 20)) {
                spans[ns].i0 = i;
                spans[ns].i1 = j;
                spans[ns].base = base;
                spans[ns].span = span;
                ns++;
            } else {
                for (int k = i; k < j; k++) {
                    uint8_t payload[PAGE_SZ];
                    if (pread(s->pool_spill_fd, payload, items[k].csz, (off_t)items[k].off) != (ssize_t)items[k].csz)
                        continue;
                    int ai = items[k].arena_i;
                    if (ai < 0 || ai >= n || filled[ai]) continue;
                    cpu_decompress(payload, items[k].csz, arena + (size_t)ai * PAGE_SZ);
                    filled[ai] = 1;
                    done++;
                }
            }
            i = j;
        }
        while (i < m) {
            uint8_t payload[PAGE_SZ];
            if (pread(s->pool_spill_fd, payload, items[i].csz, (off_t)items[i].off) == (ssize_t)items[i].csz) {
                int ai = items[i].arena_i;
                if (ai >= 0 && ai < n && !filled[ai]) {
                    cpu_decompress(payload, items[i].csz, arena + (size_t)ai * PAGE_SZ);
                    filled[ai] = 1;
                    done++;
                }
            }
            i++;
        }
        uint8_t small_slots[2][PAGE_SZ];
        int have = 0;
        uint8_t *cur_buf = NULL;
        size_t cur_span = 0;
        uint64_t cur_base = 0;
        int cur_i0 = 0, cur_i1 = 0;
        int slot = 0;
        for (int si = 0; si <= ns; si++) {
            uint8_t *nb = NULL;
            size_t nspan = 0;
            uint64_t nbase = 0;
            int ni0 = 0, ni1 = 0;
            int nok = 0;
            if (si < ns) {
                nspan = spans[si].span;
                nbase = spans[si].base;
                ni0 = spans[si].i0;
                ni1 = spans[si].i1;
                if (nspan <= PAGE_SZ) nb = small_slots[slot];
                else nb = tca_pipe_buf(slot, nspan);
                if (nb) tca_rdadvise(s->pool_spill_fd, (off_t)nbase, nspan);
                if (nb && pread(s->pool_spill_fd, nb, nspan, (off_t)nbase) == (ssize_t)nspan) {
                    nok = 1;
                    s->sov_crw_spans++;
                    s->sov_crw_pages += (uint64_t)(ni1 - ni0);
                    s->sov_crw_bytes += nspan;
                    s->pool_vault_reads++;
                } else {
                    for (int k = ni0; k < ni1; k++) {
                        uint8_t payload[PAGE_SZ];
                        if (pread(s->pool_spill_fd, payload, items[k].csz, (off_t)items[k].off) != (ssize_t)items[k].csz)
                            continue;
                        int ai = items[k].arena_i;
                        if (ai < 0 || ai >= n || filled[ai]) continue;
                        cpu_decompress(payload, items[k].csz, arena + (size_t)ai * PAGE_SZ);
                        filled[ai] = 1;
                        done++;
                    }
                }
            }
            if (have && cur_buf) {
                int count = cur_i1 - cur_i0;
                tca_item_t *items_h = items;
                uint8_t *buf_h = cur_buf;
                uint8_t *arena_h = arena;
                int *filled_h = filled;
                int n_h = n;
                int i0_h = cur_i0;
                uint64_t base_h = cur_base;
                size_t span_h = cur_span;
                __block int local = 0;
                dispatch_apply((size_t)count, DISPATCH_APPLY_AUTO, ^(size_t t) {
                    int k = i0_h + (int)t;
                    uint64_t rel = items_h[k].off - base_h;
                    uint32_t c = items_h[k].csz;
                    if (rel + c > span_h) return;
                    int ai = items_h[k].arena_i;
                    if (ai < 0 || ai >= n_h) return;
                    if (__sync_lock_test_and_set(&filled_h[ai], 1)) return;
                    cpu_decompress(buf_h + rel, c, arena_h + (size_t)ai * PAGE_SZ);
                    __sync_fetch_and_add(&local, 1);
                });
                done += local;
            }
            if (nok) {
                cur_buf = nb;
                cur_span = nspan;
                cur_base = nbase;
                cur_i0 = ni0;
                cur_i1 = ni1;
                have = 1;
                slot ^= 1;
            } else {
                have = 0;
                cur_buf = NULL;
            }
        }
            } else {
        tca_item_t *items_p = items;
        int novault_p = novault;
        int n_p = n;
        int fd = s->pool_spill_fd;
        MemXZone3 *sp = s;
        uint8_t *arena_p = arena;
        int *filled_p = filled;
        dispatch_apply((size_t)m, DISPATCH_APPLY_AUTO, ^(size_t t) {
            tca_item_t it = items_p[t];
            uint8_t payload[PAGE_SZ];
            int got = 0;
            if (!novault_p) {
                pthread_mutex_lock(&sp->alloc_mutex);
                if (pool_vault_cache_get_locked(sp, it.off, it.csz, payload)) got = 1;
                pthread_mutex_unlock(&sp->alloc_mutex);
            }
            if (!got) {
                if (pread(fd, payload, it.csz, (off_t)it.off) != (ssize_t)it.csz) return;
                got = 1;
                if (!novault_p) {
                    pthread_mutex_lock(&sp->alloc_mutex);
                    pool_vault_cache_put_locked(sp, it.off, it.csz, payload);
                    pthread_mutex_unlock(&sp->alloc_mutex);
                }
            }
            if (it.arena_i < 0 || it.arena_i >= n_p) return;
            if (__sync_lock_test_and_set(&filled_p[it.arena_i], 1)) return;
            cpu_decompress(payload, it.csz, arena_p + (size_t)it.arena_i * PAGE_SZ);
            __sync_fetch_and_add(&done, 1);
        });
    }

    if (filled_heap) free(filled);
    if (done > 0) {
        __sync_fetch_and_add(&s->sov_tca_pages, (uint64_t)done);
        __sync_fetch_and_add(&s->sov_tca_bytes, (uint64_t)done * (uint64_t)PAGE_SZ);
        __sync_fetch_and_add(&s->sov_hits, (uint64_t)done);
    }
    return done;
}

typedef struct {
    uint64_t off;
    uint32_t csz;
    uint32_t eidx;
} sov_warp_span_t;

static int sov_warm_enabled(void) {
    const char *e = getenv("MEMX_SOV_WARM");
    if (e && e[0] == '0') return 0;
    return 1;
}

static int sov_stream_enabled(void) {
    const char *e = getenv("MEMX_SOV_STREAM");
    if (e && e[0] == '0') return 0;
    return 1;
}

static void sov_warm_vault_stream_locked(MemXZone3 *s) {
    if (!s || !s->sov_ents || s->sov_count == 0 || !s->sov_off_idx) return;
    if (!sov_warm_enabled()) return;
    if (!pool_is_vault_native(s)) return;
    if (s->pool_spill_fd <= 2 || s->pool_spill_bytes == 0) return;
    if (pool_vault_cache_init_locked(s) != 0 || !s->vault_cache) return;
    sov_ent_t *ents = (sov_ent_t *)s->sov_ents;
    uint32_t *oidx = (uint32_t *)s->sov_off_idx;
    uint32_t n = s->sov_count;
    size_t budget = s->vault_cache_bytes;
    if (budget < (size_t)PAGE_SZ * 64) return;
    size_t filled = 0;
    uint32_t i = 0;
    uint8_t stack_payload[PAGE_SZ];
    while (i < n && filled + 4096 < budget) {
        uint32_t e0 = oidx[i];
        if (e0 >= n) { i++; continue; }
        uint64_t base = ents[e0].off;
        uint32_t c0 = ents[e0].csz;
        if (c0 == 0 || c0 > PAGE_SZ || base + c0 > s->pool_spill_bytes) { i++; continue; }
        uint64_t lim = base + (1ull << 20);
        if (lim > s->pool_spill_bytes) lim = s->pool_spill_bytes;
        if (filled + (size_t)(lim - base) > budget) {
            lim = base + (budget - filled);
            if (lim <= base + c0) lim = base + c0;
        }
        uint32_t j = i;
        uint64_t end = base;
        while (j < n) {
            uint32_t ei = oidx[j];
            if (ei >= n) break;
            uint64_t o = ents[ei].off;
            uint32_t c = ents[ei].csz;
            if (c == 0 || c > PAGE_SZ) break;
            if (o < end) break;
            if (o > end + 4096) break;
            if (o + c > lim) break;
            end = o + c;
            j++;
            if (j - i >= 96) break;
        }
        if (j <= i) { i++; continue; }
        size_t span = (size_t)(end - base);
        if (span == 0 || span > (1u << 20)) { i++; continue; }
        uint8_t *buf = NULL;
        int stacked = 0;
        if (span <= PAGE_SZ) {
            buf = stack_payload;
            stacked = 1;
        } else {
            buf = (uint8_t *)mmap(NULL, span, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
            if (buf == MAP_FAILED) {
                for (uint32_t k = i; k < j && filled + PAGE_SZ < budget; k++) {
                    uint32_t ei = oidx[k];
                    uint32_t c = ents[ei].csz;
                    uint64_t o = ents[ei].off;
                    if (pool_vault_cache_get_locked(s, o, c, stack_payload)) continue;
                    if (pread(s->pool_spill_fd, stack_payload, c, (off_t)o) != (ssize_t)c) continue;
                    pool_vault_cache_put_locked(s, o, c, stack_payload);
                    filled += c;
                }
                i = j;
                continue;
            }
        }
        ssize_t r = pread(s->pool_spill_fd, buf, span, (off_t)base);
        if (r == (ssize_t)span) {
            for (uint32_t k = i; k < j; k++) {
                uint32_t ei = oidx[k];
                uint32_t c = ents[ei].csz;
                uint64_t o = ents[ei].off;
                uint64_t rel = o - base;
                if (rel + c > span) continue;
                if (pool_vault_cache_get_locked(s, o, c, stack_payload)) continue;
                pool_vault_cache_put_locked(s, o, c, buf + rel);
                filled += c;
            }
        }
        if (!stacked && buf && buf != MAP_FAILED) munmap(buf, span);
        i = j;
    }
    s->sov_warm_bytes = filled;
}

static void sov_stream_inject_neighbors_locked(MemXZone3 *s, const sov_ent_t *cur) {
    if (!s || !cur || !s->sov_ents || !s->sov_off_idx || s->sov_count == 0) return;
    if (!sov_stream_enabled()) return;
    if (s->pool_spill_fd <= 2) return;
    if (!pool_is_vault_native(s) || !s->vault_cache) return;
    sov_ent_t *ents = (sov_ent_t *)s->sov_ents;
    uint32_t *oidx = (uint32_t *)s->sov_off_idx;
    uint32_t n = s->sov_count;
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        uint64_t mo = ents[oidx[mid]].off;
        if (mo < cur->off) lo = mid + 1;
        else hi = mid;
    }
    uint32_t start = lo;
    while (start < n && ents[oidx[start]].off == cur->off && ents[oidx[start]].csz == cur->csz)
        start++;
    if (start >= n) return;
    uint64_t base = ents[oidx[start]].off;
    if (base < cur->off) base = cur->off;
    uint64_t lim = base + (256ull * 1024ull);
    if (lim > s->pool_spill_bytes) lim = s->pool_spill_bytes;
    uint32_t end = start;
    uint64_t run_end = base;
    while (end < n && end < start + 64) {
        sov_ent_t *e = &ents[oidx[end]];
        if (e->csz == 0 || e->csz > PAGE_SZ) break;
        if (e->off < run_end) break;
        if (e->off > run_end + 4096) break;
        if (e->off + e->csz > lim) break;
        run_end = e->off + e->csz;
        end++;
    }
    if (end <= start) return;
    size_t span = (size_t)(run_end - base);
    if (span == 0 || span > (1u << 20)) return;
    uint8_t stack_payload[PAGE_SZ];
    uint8_t *buf = NULL;
    int stacked = 0;
    if (span <= PAGE_SZ) {
        buf = stack_payload;
        stacked = 1;
    } else {
        buf = (uint8_t *)mmap(NULL, span, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (buf == MAP_FAILED) {
            for (uint32_t k = start; k < end && k < start + 12; k++) {
                sov_ent_t *e = &ents[oidx[k]];
                if (pool_vault_cache_get_locked(s, e->off, e->csz, stack_payload)) continue;
                if (pread(s->pool_spill_fd, stack_payload, e->csz, (off_t)e->off) != (ssize_t)e->csz) continue;
                pool_vault_cache_put_locked(s, e->off, e->csz, stack_payload);
                s->sov_stream_injects++;
            }
            return;
        }
    }
    ssize_t r = pread(s->pool_spill_fd, buf, span, (off_t)base);
    if (r == (ssize_t)span) {
        for (uint32_t k = start; k < end; k++) {
            sov_ent_t *e = &ents[oidx[k]];
            uint64_t rel = e->off - base;
            if (rel + e->csz > span) continue;
            if (pool_vault_cache_get_locked(s, e->off, e->csz, stack_payload)) continue;
            pool_vault_cache_put_locked(s, e->off, e->csz, buf + rel);
            s->sov_stream_injects++;
        }
    }
    if (!stacked && buf && buf != MAP_FAILED) munmap(buf, span);
}

static void cpu_decompress(const uint8_t *src, uint32_t cs, uint8_t *dst);
static int mat_cache_get(size_t pidx, uint32_t write_seq, uint8_t *out_page);
static void mat_cache_put(size_t pidx, uint32_t write_seq, const uint8_t *page);

static __thread int g_sov_batch_mode = 0;

static int sov_crw_enabled(void) {
    const char *e = getenv("MEMX_SOV_CRW");
    if (e && e[0] == '0') return 0;
    return 1;
}

static int sov_crw_direct_enabled(void) {
    const char *e = getenv("MEMX_SOV_CRW_DIRECT");
    if (e && e[0] == '0') return 0;
    return 1;
}

static int sov_chronos_enabled(void) {
    const char *e = getenv("MEMX_SOV_CHRONOS");
    if (e && e[0] == '0') return 0;
    return 1;
}

static int sov_chronos_horizon_pages(void) {
    const char *e = getenv("MEMX_SOV_CHRONOS_PAGES");
    if (!e || !e[0]) return 96;
    int v = atoi(e);
    if (v < 0) v = 0;
    if (v > 512) v = 512;
    return v;
}

static void sov_warp_buf_release_locked(MemXZone3 *s) {
    if (!s) return;
    if (s->sov_warp_buf && s->sov_warp_buf != MAP_FAILED && s->sov_warp_buf_bytes) {
        munmap(s->sov_warp_buf, (size_t)s->sov_warp_buf_bytes);
    }
    s->sov_warp_buf = NULL;
    s->sov_warp_buf_bytes = 0;
}

static int sov_warp_buf_ensure_locked(MemXZone3 *s, size_t need) {
    if (!s) return -1;
    if (need < (size_t)(1u << 20)) need = (size_t)(1u << 20);
    if (need > (size_t)(4u << 20)) need = (size_t)(4u << 20);
    if (s->sov_warp_buf && s->sov_warp_buf != MAP_FAILED && s->sov_warp_buf_bytes >= need)
        return 0;
    sov_warp_buf_release_locked(s);
    void *m = mmap(NULL, need, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) {
        s->sov_warp_buf = NULL;
        s->sov_warp_buf_bytes = 0;
        return -1;
    }
    s->sov_warp_buf = (uint8_t *)m;
    s->sov_warp_buf_bytes = need;
    return 0;
}

typedef struct {
    uint64_t off;
    uint32_t csz;
    uint32_t pidx;
    uint32_t seq;
} sov_crw_item_t;

static int sov_crw_item_cmp(const void *a, const void *b) {
    const sov_crw_item_t *x = (const sov_crw_item_t *)a;
    const sov_crw_item_t *y = (const sov_crw_item_t *)b;
    if (x->off < y->off) return -1;
    if (x->off > y->off) return 1;
    if (x->pidx < y->pidx) return -1;
    if (x->pidx > y->pidx) return 1;
    return 0;
}

static int sov_capsule_readv_warp_locked(MemXZone3 *s, const size_t *pidxs, int n) {
    if (!sov_crw_enabled()) return 0;
    if (!s || !pidxs || n <= 0) return 0;
    if (!s->sovereign || !s->sov_ents || s->sov_count == 0) return 0;
    if (s->pool_spill_fd <= 2 || s->pool_spill_bytes == 0) return 0;
    if (!pool_is_vault_native(s)) return 0;
    if (pool_vault_cache_init_locked(s) != 0 || !s->vault_cache) return 0;
    enum { CRW_MAX = 2048 };
    if (n > CRW_MAX) n = CRW_MAX;
    sov_crw_item_t items[CRW_MAX];
    int m = 0;
    uint8_t stack_payload[PAGE_SZ];
    for (int i = 0; i < n; i++) {
        size_t pidx = pidxs[i];
        if (pidx >= s->npages) continue;
        sov_ent_t *e = sov_find_locked(s, (uint32_t)pidx);
        if (!e || e->csz == 0 || e->csz > PAGE_SZ) continue;
        if (e->off + (uint64_t)e->csz > s->pool_spill_bytes) continue;
        if (pool_vault_cache_get_locked(s, e->off, e->csz, stack_payload)) continue;
        items[m].off = e->off;
        items[m].csz = e->csz;
        items[m].pidx = (uint32_t)pidx;
        items[m].seq = e->seq;
        m++;
    }
    if (m <= 0) return 0;
    if (m > 1) qsort(items, (size_t)m, sizeof(items[0]), sov_crw_item_cmp);
    int i = 0;
    while (i < m) {
        uint64_t base = items[i].off;
        uint64_t run_end = items[i].off + items[i].csz;
        int j = i + 1;
        while (j < m) {
            uint64_t o = items[j].off;
            uint32_t c = items[j].csz;
            uint64_t e2 = o + c;
            if (o < run_end) {
                if (e2 > run_end) run_end = e2;
                j++;
                continue;
            }
            if (o > run_end + 4096ull) break;
            if (e2 - base > (1ull << 20)) break;
            if (j - i >= 96) break;
            run_end = e2;
            j++;
        }
        size_t span = (size_t)(run_end - base);
        if (span == 0 || span > (1u << 20)) {
            for (int k = i; k < j; k++) {
                if (pool_vault_cache_get_locked(s, items[k].off, items[k].csz, stack_payload)) continue;
                if (pread(s->pool_spill_fd, stack_payload, items[k].csz, (off_t)items[k].off) != (ssize_t)items[k].csz)
                    continue;
                pool_vault_cache_put_locked(s, items[k].off, items[k].csz, stack_payload);
                s->sov_crw_pages++;
                s->sov_crw_bytes += items[k].csz;
                s->pool_vault_reads++;
            }
            if (j > i) s->sov_crw_spans++;
            i = j;
            continue;
        }
        uint8_t *buf = NULL;
        int owned = 0;
        if (span <= PAGE_SZ) {
            buf = stack_payload;
        } else if (sov_warp_buf_ensure_locked(s, span) == 0) {
            buf = s->sov_warp_buf;
        } else {
            buf = (uint8_t *)mmap(NULL, span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (buf == MAP_FAILED) {
                for (int k = i; k < j; k++) {
                    if (pool_vault_cache_get_locked(s, items[k].off, items[k].csz, stack_payload)) continue;
                    if (pread(s->pool_spill_fd, stack_payload, items[k].csz, (off_t)items[k].off) != (ssize_t)items[k].csz)
                        continue;
                    pool_vault_cache_put_locked(s, items[k].off, items[k].csz, stack_payload);
                    s->sov_crw_pages++;
                    s->sov_crw_bytes += items[k].csz;
                    s->pool_vault_reads++;
                }
                if (j > i) s->sov_crw_spans++;
                i = j;
                continue;
            }
            owned = 1;
        }
        ssize_t r = pread(s->pool_spill_fd, buf, span, (off_t)base);
        if (r == (ssize_t)span) {
            s->sov_crw_spans++;
            s->sov_crw_bytes += span;
            s->pool_vault_reads++;
            for (int k = i; k < j; k++) {
                uint64_t rel = items[k].off - base;
                if (rel + items[k].csz > span) continue;
                if (pool_vault_cache_get_locked(s, items[k].off, items[k].csz, stack_payload)) continue;
                pool_vault_cache_put_locked(s, items[k].off, items[k].csz, buf + rel);
                s->sov_crw_pages++;
            }
        }
        if (owned && buf && buf != MAP_FAILED) munmap(buf, span);
        i = j;
    }
    return m;
}

static int sov_crw_direct_decomp_batch(MemXZone3 *s, const size_t *pidxs, int n) {
    if (!sov_crw_direct_enabled()) return 0;
    if (!s || !pidxs || n <= 0) return 0;
    if (!s->sovereign || !s->sov_ents || s->sov_count == 0) return 0;
    enum { DMAX = 2048 };
    if (n > DMAX) n = DMAX;
    size_t *work = (size_t *)malloc((size_t)n * sizeof(size_t));
    uint32_t *seqs = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    if (!work || !seqs) {
        free(work);
        free(seqs);
        return 0;
    }
    int w = 0;
    for (int i = 0; i < n; i++) {
        size_t pidx = pidxs[i];
        if (pidx >= s->npages) continue;
        sov_ent_t *e = sov_find_locked(s, (uint32_t)pidx);
        if (!e || e->csz == 0 || e->csz > PAGE_SZ) continue;
        uint8_t tmp[PAGE_SZ];
        if (mat_cache_get(pidx, e->seq, tmp)) continue;
        work[w] = pidx;
        seqs[w] = e->seq;
        w++;
    }
    if (w <= 0) {
        free(work);
        free(seqs);
        return 0;
    }
    size_t *work_h = work;
    uint32_t *seq_h = seqs;
    int ww = w;
    __block int done = 0;
    dispatch_apply((size_t)ww, DISPATCH_APPLY_AUTO, ^(size_t i) {
        size_t pidx = work_h[i];
        uint32_t seq = seq_h[i];
        uint8_t page[PAGE_SZ];
        uint8_t payload[PAGE_SZ];
        if (mat_cache_get(pidx, seq, page)) {
            __sync_fetch_and_add(&done, 1);
            return;
        }
        sov_ent_t *e = sov_find_locked(s, (uint32_t)pidx);
        if (!e || e->csz == 0 || e->csz > PAGE_SZ || e->seq != seq) return;
        int got = 0;
        pthread_mutex_lock(&s->alloc_mutex);
        if (pool_vault_cache_get_locked(s, e->off, e->csz, payload)) got = 1;
        pthread_mutex_unlock(&s->alloc_mutex);
        if (!got && s->pool_spill_fd > 2 && s->pool_spill_bytes >= e->off + (uint64_t)e->csz) {
            if (pread(s->pool_spill_fd, payload, e->csz, (off_t)e->off) == (ssize_t)e->csz) {
                got = 1;
                pthread_mutex_lock(&s->alloc_mutex);
                pool_vault_cache_put_locked(s, e->off, e->csz, payload);
                pthread_mutex_unlock(&s->alloc_mutex);
            }
        }
        if (!got) return;
        cpu_decompress(payload, e->csz, page);
        mat_cache_put(pidx, seq, page);
        __sync_fetch_and_add(&s->sov_hits, 1);
        __sync_fetch_and_add(&done, 1);
    });
    free(work);
    free(seqs);
    return done;
}

static void sov_chronos_horizon_locked(MemXZone3 *s, size_t pidx_hi, int horizon) {
    if (!sov_chronos_enabled()) return;
    if (!s || !s->sov_ents || s->sov_count == 0 || horizon <= 0) return;
    if (s->pool_spill_fd <= 2 || !pool_is_vault_native(s)) return;
    if (!s->vault_cache) return;
    sov_ent_t *ents = (sov_ent_t *)s->sov_ents;
    uint32_t lo = 0, hi = s->sov_count;
    uint32_t key = (uint32_t)pidx_hi;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (ents[mid].pidx <= key) lo = mid + 1;
        else hi = mid;
    }
    size_t limit = pidx_hi + (size_t)horizon;
    enum { HMAX = 192 };
    size_t plist[HMAX];
    int n = 0;
    for (uint32_t i = lo; i < s->sov_count && n < HMAX; i++) {
        if ((size_t)ents[i].pidx > limit) break;
        plist[n++] = (size_t)ents[i].pidx;
    }
    if (n < HMAX / 2 && s->sov_off_idx) {
        sov_ent_t *cur = sov_find_locked(s, (uint32_t)pidx_hi);
        if (cur) {
            uint32_t *oidx = (uint32_t *)s->sov_off_idx;
            uint32_t nent = s->sov_count;
            uint32_t olo = 0, ohi = nent;
            while (olo < ohi) {
                uint32_t mid = olo + ((ohi - olo) >> 1);
                uint64_t mo = ents[oidx[mid]].off;
                if (mo < cur->off) olo = mid + 1;
                else ohi = mid;
            }
            uint32_t start = olo;
            while (start < nent && ents[oidx[start]].off <= cur->off) start++;
            for (uint32_t k = start; k < nent && n < HMAX && k < start + (uint32_t)horizon; k++) {
                uint32_t pi = ents[oidx[k]].pidx;
                int seen = 0;
                for (int t = 0; t < n; t++) if (plist[t] == (size_t)pi) { seen = 1; break; }
                if (!seen) plist[n++] = (size_t)pi;
            }
        }
    }
    if (n <= 0) return;
    int r = sov_capsule_readv_warp_locked(s, plist, n);
    if (r > 0) s->sov_chronos_injects += (uint64_t)n;
}

static const uint8_t *pool_vault_cache_ptr_locked(MemXZone3 *s, uint64_t d_off, uint32_t d_sz) {
    if (!s || !s->vault_cache || !s->vault_cache_live || d_sz == 0 || d_sz > PAGE_SZ) return NULL;
    uint32_t n = s->vault_cache_slots;
    uint32_t h = (uint32_t)((d_off ^ (d_off >> 17) ^ d_sz) % n);
    for (uint32_t probe = 0; probe < 8; probe++) {
        uint32_t i = (h + probe) % n;
        if (!s->vault_cache_live[i]) continue;
        if (s->vault_cache_off[i] == d_off && s->vault_cache_sz[i] == d_sz &&
            s->vault_cache_gen && s->vault_cache_gen[i] == s->vault_epoch) {
            uint32_t p = s->vault_cache_pos[i];
            if ((uint64_t)p + d_sz > s->vault_cache_bytes) continue;
            s->pool_vault_cache_hits++;
            return s->vault_cache + p;
        }
    }
    return NULL;
}


static void pool_vault_windows_drop_locked(MemXZone3 *s) {
    if (!s) return;
    for (int i = 0; i < 4; i++) {
        if (s->vault_win[i] && s->vault_win[i] != MAP_FAILED && s->vault_win_len[i]) {
            munmap(s->vault_win[i], (size_t)s->vault_win_len[i]);
        }
        s->vault_win[i] = NULL;
        s->vault_win_base[i] = 0;
        s->vault_win_len[i] = 0;
    }
}

static int pool_vault_window_enabled(void) {
    const char *e = getenv("MEMX_VAULT_WINDOW");
    if (e && e[0] == '1') return 1;
    return 0;
}

static const uint8_t *pool_vault_window_ptr_locked(MemXZone3 *s, uint64_t d_off, uint32_t d_sz) {
    if (!s || d_sz == 0 || s->pool_spill_fd <= 2) return NULL;
    if (!pool_vault_window_enabled()) return NULL;
    if (d_off + (uint64_t)d_sz > s->pool_spill_bytes) return NULL;
    for (int i = 0; i < 4; i++) {
        if (!s->vault_win[i] || !s->vault_win_len[i]) continue;
        if (d_off >= s->vault_win_base[i] && d_off + d_sz <= s->vault_win_base[i] + s->vault_win_len[i]) {
            s->pool_vault_window_hits++;
            return s->vault_win[i] + (d_off - s->vault_win_base[i]);
        }
    }
    uint64_t win_sz = 4ull * 1024ull * 1024ull;
    const char *e = getenv("MEMX_VAULT_WINDOW_MB");
    if (e && e[0]) {
        long mb = strtol(e, NULL, 10);
        if (mb >= 1 && mb <= 64) win_sz = (uint64_t)mb * 1024ull * 1024ull;
    }
    uint64_t base = d_off & ~(win_sz - 1);
    uint64_t end = base + win_sz;
    if (end < d_off + d_sz) end = ((d_off + d_sz + win_sz - 1) / win_sz) * win_sz;
    if (end > s->pool_spill_bytes) end = s->pool_spill_bytes;
    if (end <= base) return NULL;
    size_t map_len = (size_t)(end - base);
    int slot = (int)(s->vault_win_clock++ & 3u);
    if (s->vault_win[slot] && s->vault_win[slot] != MAP_FAILED && s->vault_win_len[slot]) {
        munmap(s->vault_win[slot], (size_t)s->vault_win_len[slot]);
        s->vault_win[slot] = NULL;
        s->vault_win_len[slot] = 0;
    }
#if defined(__APPLE__)
    (void)fcntl(s->pool_spill_fd, F_NOCACHE, 0);
#endif
    void *m = mmap(NULL, map_len, PROT_READ, MAP_SHARED, s->pool_spill_fd, (off_t)base);
    if (m == MAP_FAILED) return NULL;
#if defined(MADV_SEQUENTIAL)
    madvise(m, map_len, MADV_SEQUENTIAL);
#endif
#if defined(MADV_WILLNEED)
    madvise(m, map_len, MADV_WILLNEED);
#endif
    s->vault_win[slot] = (uint8_t *)m;
    s->vault_win_base[slot] = base;
    s->vault_win_len[slot] = map_len;
    if (d_off >= base && d_off + d_sz <= base + map_len) {
        s->pool_vault_window_hits++;
        return s->vault_win[slot] + (d_off - base);
    }
    return NULL;
}

static int pool_vault_wbuf_enabled(void) {
    const char *e = getenv("MEMX_VAULT_WBUF");
    if (e && e[0] == '1') return 1;
    return 0;
}

static int pool_vault_wbuf_flush_locked(MemXZone3 *s) {
    if (!s || !s->vault_wbuf || s->vault_wbuf_len == 0) return 0;
    if (s->pool_spill_fd <= 2) return -1;
    uint64_t end = s->vault_wbuf_base + s->vault_wbuf_len;
    if (end > s->pool_spill_bytes) {
        if (ftruncate(s->pool_spill_fd, (off_t)end) != 0) return -1;
        s->pool_spill_bytes = end;
    }
    ssize_t w = pwrite(s->pool_spill_fd, s->vault_wbuf, (size_t)s->vault_wbuf_len, (off_t)s->vault_wbuf_base);
    if (w != (ssize_t)s->vault_wbuf_len) return -1;
    s->pool_vault_wbuf_flushes++;
    s->vault_wbuf_len = 0;
    return 0;
}

static int pool_vault_wbuf_store_locked(MemXZone3 *s, uint64_t off, const uint8_t *data, uint32_t sz) {
    if (!s || !data || sz == 0) return -1;
    if (!pool_vault_wbuf_enabled()) return 1;
    if (!s->vault_wbuf) {
        size_t cap = 1024ull * 1024ull;
        const char *e = getenv("MEMX_VAULT_WBUF_KB");
        if (e && e[0]) {
            long kb = strtol(e, NULL, 10);
            if (kb >= 64 && kb <= 16384) cap = (size_t)kb * 1024ull;
        }
        uint8_t *b = (uint8_t *)mmap(NULL, cap, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (b == MAP_FAILED) return 1;
        s->vault_wbuf = b;
        s->vault_wbuf_cap = cap;
        s->vault_wbuf_base = 0;
        s->vault_wbuf_len = 0;
    }
    if ((uint64_t)sz > s->vault_wbuf_cap) return 1;
    if (s->vault_wbuf_len > 0) {
        if (off != s->vault_wbuf_base + s->vault_wbuf_len ||
            s->vault_wbuf_len + sz > s->vault_wbuf_cap) {
            if (pool_vault_wbuf_flush_locked(s) != 0) return -1;
        }
    }
    if (s->vault_wbuf_len == 0) s->vault_wbuf_base = off;
    if (off != s->vault_wbuf_base + s->vault_wbuf_len) return 1;
    memcpy(s->vault_wbuf + s->vault_wbuf_len, data, sz);
    s->vault_wbuf_len += sz;
    if (s->vault_wbuf_len >= s->vault_wbuf_cap) {
        if (pool_vault_wbuf_flush_locked(s) != 0) return -1;
    }
    return 0;
}

static void pool_vault_wbuf_release_locked(MemXZone3 *s) {
    if (!s) return;
    (void)pool_vault_wbuf_flush_locked(s);
    if (s->vault_wbuf && s->vault_wbuf_cap) {
        munmap(s->vault_wbuf, (size_t)s->vault_wbuf_cap);
        s->vault_wbuf = NULL;
        s->vault_wbuf_cap = 0;
        s->vault_wbuf_base = 0;
        s->vault_wbuf_len = 0;
    }
}

static int pool_vault_bootstrap_locked(MemXZone3 *s) {
    if (!s) return -1;
    if (!s->pool_vault_native) return 0;
    if (pool_ensure_spill_fd_locked(s) != 0) return -1;
    s->pool_detached = 1;
    s->pool_ghost = 1;
    return 0;
}

static uint64_t pool_mincore_resident_bytes(const void *addr, uint64_t len) {
#if defined(__APPLE__) || defined(__linux__)
    if (!addr || len == 0) return 0;
    uint64_t start = ((uintptr_t)addr) & ~((uintptr_t)PAGE_SZ - 1);
    uint64_t end = ((uintptr_t)addr + len + PAGE_SZ - 1) & ~((uintptr_t)PAGE_SZ - 1);
    size_t np = (size_t)((end - start) / PAGE_SZ);
    if (np == 0) return 0;
    const size_t chunk = 4096;
    uint64_t res = 0;
    char vec[4096];
    for (size_t off = 0; off < np; ) {
        size_t n = np - off;
        if (n > chunk) n = chunk;
        if (mincore((void *)(start + off * PAGE_SZ), n * PAGE_SZ, vec) == 0) {
            for (size_t i = 0; i < n; i++) {
                if (vec[i] & 1) res += PAGE_SZ;
            }
        }
        off += n;
    }
    return res;
#else
    (void)addr; (void)len;
    return 0;
#endif
}

static void pool_vault_probe_locked(MemXZone3 *s) {
    if (!s) return;
    const char *e = getenv("MEMX_VAULT_PROBE");
    if (!e || e[0] != '1') return;
    static uint64_t last_sig = 0;
    uint64_t sig = ((uint64_t)s->pool_detached << 63) ^ s->pool_spill_bytes ^ (s->pool_vault_stores << 1) ^ s->pool_vault_reads;
    if (sig == last_sig) return;
    last_sig = sig;
    uint64_t used = s->pool_next;
    uint64_t pool_res = 0;
    uint64_t vmem_res = 0;
    if (s->pool && used) pool_res = pool_mincore_resident_bytes(s->pool, used);
    size_t last = s->vmem_next / PAGE_SZ;
    if (last > s->npages) last = s->npages;
    if (s->vmem && last) vmem_res = pool_mincore_resident_bytes(s->vmem, last * PAGE_SZ);
    fprintf(stderr,
            "[memx] vault_probe native=%d detached=%d spill_fd=%d spill_bytes=%llu pool_next=%llu pool_res=%llu vmem_res=%llu vault_stores=%llu vault_reads=%llu cache_hits=%llu win_hits=%llu wbuf_flush=%llu sov=%d ents=%u hits=%llu warm=%llu stream=%llu crw_sp=%llu crw_pg=%llu crw_b=%llu chrono=%llu avcs=%llu vbytes=%llu ring=%llu tca_pg=%llu tca_b=%llu stick=%llu stick_p=%llu stick_d=%llu\n",
            s->pool_vault_native, s->pool_detached, s->pool_spill_fd,
            (unsigned long long)s->pool_spill_bytes,
            (unsigned long long)s->pool_next,
            (unsigned long long)pool_res,
            (unsigned long long)vmem_res,
            (unsigned long long)s->pool_vault_stores,
            (unsigned long long)s->pool_vault_reads,
            (unsigned long long)s->pool_vault_cache_hits,
            (unsigned long long)s->pool_vault_window_hits,
            (unsigned long long)s->pool_vault_wbuf_flushes,
            s->sovereign, (unsigned)s->sov_count,
            (unsigned long long)s->sov_hits,
            (unsigned long long)s->sov_warm_bytes,
            (unsigned long long)s->sov_stream_injects,
            (unsigned long long)s->sov_crw_spans,
            (unsigned long long)s->sov_crw_pages,
            (unsigned long long)s->sov_crw_bytes,
            (unsigned long long)s->sov_chronos_injects,
            (unsigned long long)s->vault_avcs_events,
            (unsigned long long)s->vault_cache_bytes,
            (unsigned long long)s->vault_ring_reclaims,
            (unsigned long long)s->sov_tca_pages,
            (unsigned long long)s->sov_tca_bytes,
            (unsigned long long)g_tca_sticky_hits,
            (unsigned long long)g_tca_sticky_partial,
            (unsigned long long)g_tca_sticky_diff_pages);
}

static int pool_ghost_enabled(void) {
    const char *e = getenv("MEMX_POOL_GHOST");
    if (e && e[0] == '0') return 0;
    if (e && e[0] == '1') return 1;
    e = getenv("MEMX_POOL_WRITE_THROUGH");
    if (e && e[0] == '1') return 1;
    return 0;
}

static int pool_ghost_final_enabled(void) {
    const char *e = getenv("MEMX_POOL_GHOST_FINAL");
    if (e && e[0] == '0') return 0;
    if (e && e[0] == '1') return 1;
    e = getenv("MEMX_POOL_SPILL");
    if (e && e[0] == '1') return 1;
    return pool_ghost_enabled();
}

static int pool_ensure_spill_fd_locked(MemXZone3 *s) {
    if (!s) return -1;
    if (s->pool_spill_fd > 2) return 0;
    char path[320];
    const char *dir = getenv("MEMX_POOL_SPILL_DIR");
    if (!dir || !dir[0]) dir = "/tmp";
    snprintf(path, sizeof(path), "%s/memx_ghost_%d.spill", dir, (int)getpid());
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return -1;
#if defined(__APPLE__)
    {
        int nocache = 1;
        if (pool_vault_window_enabled()) nocache = 0;
        (void)fcntl(fd, F_NOCACHE, nocache);
#if defined(F_RDAHEAD)
        (void)fcntl(fd, F_RDAHEAD, 1);
#endif
#if defined(F_RDADVISE)
        {
            struct radvisory ra;
            ra.ra_offset = 0;
            ra.ra_count = (int)((1u << 20));
            (void)fcntl(fd, F_RDADVISE, &ra);
        }
#endif
    }
#endif
    s->pool_spill_fd = fd;
    {
        const char *keep = getenv("MEMX_POOL_SPILL_KEEP");
        int do_keep = 1;
        if (keep && keep[0] == '0') do_keep = 0;
        if (!do_keep) unlink(path);
    }
    s->pool_ghost = 1;
    return 0;
}

static int pool_ghost_pwrite_range_locked(MemXZone3 *s, uint64_t off, uint64_t len) {
    if (!s || !s->pool || len == 0) return 0;
    if (pool_ensure_spill_fd_locked(s) != 0) return -1;
    uint64_t end = off + len;
    if (end > s->pool_size) end = s->pool_size;
    if (end <= off) return 0;
    if (end > s->pool_spill_bytes) {
        if (ftruncate(s->pool_spill_fd, (off_t)end) != 0) return -1;
        s->pool_spill_bytes = end;
    }
    uint64_t cur = off;
    while (cur < end) {
        size_t n = (size_t)((end - cur) > (16ull * 1024ull * 1024ull) ? (16ull * 1024ull * 1024ull) : (end - cur));
        uint64_t p0 = cur & ~((uint64_t)PAGE_SZ - 1);
        uint64_t p1 = (cur + n + PAGE_SZ - 1) & ~((uint64_t)PAGE_SZ - 1);
        if (p1 > s->pool_size) p1 = s->pool_size;
        if (p1 > p0) {
            if (mprotect(s->pool + p0, (size_t)(p1 - p0), PROT_READ | PROT_WRITE) != 0) {
                void *m = mmap(s->pool + p0, (size_t)(p1 - p0), PROT_READ | PROT_WRITE,
                               MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (m == MAP_FAILED) return -1;
            }
#if defined(MADV_FREE_REUSE)
            (void)madvise(s->pool + p0, (size_t)(p1 - p0), MADV_FREE_REUSE);
#endif
        }
        ssize_t w = pwrite(s->pool_spill_fd, s->pool + cur, n, (off_t)cur);
        if (w != (ssize_t)n) return -1;
        cur += n;
    }
    return 0;
}

static void pool_ghost_detach_range_locked(MemXZone3 *s, uint64_t off, uint64_t len) {
    if (!s || !s->pool || len == 0) return;
    uint64_t start = (off + (PAGE_SZ - 1)) & ~((uint64_t)PAGE_SZ - 1);
    uint64_t end = (off + len) & ~((uint64_t)PAGE_SZ - 1);
    if (end <= start) {
        pool_pageout_range(s, off, len);
        return;
    }
    const uint64_t chunk = (uint64_t)PAGE_SZ * 64;
    for (uint64_t cur = start; cur < end; ) {
        uint64_t nend = cur + chunk;
        if (nend > end) nend = end;
        void *m = mmap(s->pool + cur, (size_t)(nend - cur), PROT_NONE,
                       MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m == MAP_FAILED) {
            pool_pageout_range(s, cur, nend - cur);
#if defined(MADV_DONTNEED)
            (void)madvise(s->pool + cur, (size_t)(nend - cur), MADV_DONTNEED);
#endif
        }
        cur = nend;
    }
}

static int pool_ghost_flush_locked(MemXZone3 *s) {
    if (!s || !s->pool) return 0;
    if (pool_is_vault_native(s)) {
        if (s->pool_next == 0) return 0;
        if (pool_ensure_spill_fd_locked(s) != 0) return -1;
        if (pool_vault_wbuf_flush_locked(s) != 0) return -1;
        if (s->pool_spill_bytes < s->pool_next) return -1;
        pool_ghost_detach_range_locked(s, 0, s->pool_next);
        if (s->pool_next < s->pool_size)
            pool_hard_decommit_range(s, s->pool_next, s->pool_size - s->pool_next);
        s->pool_detached = 1;
        s->pool_ghost = 1;
        s->pool_ghost_flushed = s->pool_next;
        return 1;
    }
    if (!pool_ghost_enabled() && !pool_ghost_final_enabled() && !s->pool_ghost && !g_pool_spill_force) return 0;
    if (s->pool_next == 0) return 0;
    if (pool_ensure_spill_fd_locked(s) != 0) return -1;
    uint64_t from = s->pool_ghost_flushed;
    uint64_t to = s->pool_next;
    if (to <= from && s->pool_detached) return 1;
    if (to > from) {
        if (pool_ghost_pwrite_range_locked(s, from, to - from) != 0) return -1;
    }
    if (s->pool_spill_bytes < to) {
        if (ftruncate(s->pool_spill_fd, (off_t)to) != 0) return -1;
        s->pool_spill_bytes = to;
    }
    pool_ghost_detach_range_locked(s, 0, to);
    if (to < s->pool_size)
        pool_hard_decommit_range(s, to, s->pool_size - to);
    s->pool_ghost_flushed = to;
    s->pool_detached = 1;
    s->pool_ghost = 1;
    s->pool_ghost_stores++;
    s->pool_spill_events++;
    return 1;
}

static int pool_store_blob_locked(MemXZone3 *s, uint64_t off, const uint8_t *data, uint32_t sz) {
    if (!s || !data || sz == 0) return -1;
    int vault = pool_is_vault_native(s);
    int durable = vault || pool_ghost_enabled() || pool_ghost_final_enabled() || s->pool_ghost;
    const char *em = getenv("MEMX_POOL_MIRROR");
    if (em && em[0] == '1') durable = 1;
    if (em && em[0] == '0' && !vault) durable = 0;
    if (durable || vault) {
        if (pool_ensure_spill_fd_locked(s) != 0) return -1;
        uint64_t end = off + (uint64_t)sz;
        int used_wbuf = 0;
        if (vault) {
            int wr = pool_vault_wbuf_store_locked(s, off, data, sz);
            if (wr < 0) return -1;
            if (wr == 0) used_wbuf = 1;
        }
        if (!used_wbuf) {
            if (end > s->pool_spill_bytes) {
                if (ftruncate(s->pool_spill_fd, (off_t)end) != 0) return -1;
                s->pool_spill_bytes = end;
            }
            ssize_t w = pwrite(s->pool_spill_fd, data, sz, (off_t)off);
            if (w != (ssize_t)sz) return -1;
        } else {
            if (end > s->pool_spill_bytes) s->pool_spill_bytes = end;
        }
        s->pool_ghost = 1;
        s->pool_ghost_stores++;
        if (vault) s->pool_vault_stores++;
        if (end > s->pool_ghost_flushed) s->pool_ghost_flushed = end;
    }
    if (vault) {
        s->pool_detached = 1;
        return 0;
    }
    if (!s->pool) return durable ? 0 : -1;
    if (s->pool_detached && durable) {
        return 0;
    }
    pool_prepare_write_range(s, off, sz);
    memcpy(s->pool + off, data, sz);
    return 0;
}



static int pool_blob_eq_locked(MemXZone3 *s, uint64_t off, const uint8_t *data, uint32_t sz) {
    if (!s || !data || sz == 0 || sz > PAGE_SZ) return 0;
    uint8_t tmp[PAGE_SZ];
    if (pool_copy_blob_locked(s, off, sz, tmp) != 0) return 0;
    return memcmp(tmp, data, sz) == 0;
}

static int pool_spill_to_file_locked(MemXZone3 *s) {
    if (!s || !s->pool || s->pool_next == 0) return 0;
    if (pool_is_vault_native(s)) {
        if (pool_ensure_spill_fd_locked(s) != 0) return -1;
        if (pool_vault_wbuf_flush_locked(s) != 0) return -1;
        if (s->pool_spill_bytes < s->pool_next) return -1;
        pool_ghost_detach_range_locked(s, 0, s->pool_next);
        if (s->pool_next < s->pool_size)
            pool_hard_decommit_range(s, s->pool_next, s->pool_size - s->pool_next);
        s->pool_detached = 1;
        s->pool_ghost = 1;
        s->pool_spill_events++;
        return 1;
    }
    if (s->pool_spill_fd > 2 && s->pool_detached) {
        pool_pageout_range(s, 0, s->pool_next);
        return 1;
    }
    const char *env = getenv("MEMX_POOL_SPILL");
    int allow = g_pool_spill_force;
    if (env && env[0] == '1') allow = 1;
    if (env && env[0] == '0') allow = 0;
    if (!allow) return 0;

    int want_detach = 1;
    const char *ed = getenv("MEMX_POOL_DETACH");
    if (ed && ed[0] == '0') want_detach = 0;

    char path[320];
    const char *dir = getenv("MEMX_POOL_SPILL_DIR");
    if (!dir || !dir[0]) dir = "/tmp";
    uint64_t used = s->pool_next;
    used = (used + (uint64_t)PAGE_SZ - 1) & ~((uint64_t)PAGE_SZ - 1);
    if (used == 0) used = (uint64_t)PAGE_SZ;
    if (used > s->pool_size) used = s->pool_size;

    int fd = s->pool_spill_fd;
    int reused = 0;
    if (fd > 2) {
        reused = 1;
    } else {
        snprintf(path, sizeof(path), "%s/memx_pool_%d_%llu.spill", dir, (int)getpid(),
                 (unsigned long long)used);
        fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) return -1;
#if defined(__APPLE__)
        (void)fcntl(fd, F_NOCACHE, 1);
#endif
    }

    if (!reused || !s->pool_detached) {
        uint64_t off = 0;
        while (off < used) {
            size_t n = (size_t)((used - off) > (32ull * 1024ull * 1024ull) ? (32ull * 1024ull * 1024ull) : (used - off));
            uint64_t p0 = off & ~((uint64_t)PAGE_SZ - 1);
            uint64_t p1 = (off + n + PAGE_SZ - 1) & ~((uint64_t)PAGE_SZ - 1);
            if (p1 > s->pool_size) p1 = s->pool_size;
            if (p1 > p0) {
                if (mprotect(s->pool + p0, (size_t)(p1 - p0), PROT_READ | PROT_WRITE) != 0) {
                    if (!reused) { close(fd); unlink(path); }
                    return -1;
                }
#if defined(MADV_FREE_REUSE)
                (void)madvise(s->pool + p0, (size_t)(p1 - p0), MADV_FREE_REUSE);
#endif
            }
            ssize_t w = pwrite(fd, s->pool + off, n, (off_t)off);
            if (w != (ssize_t)n) {
                if (!reused) { close(fd); unlink(path); }
                return -1;
            }
            off += n;
        }
        if (ftruncate(fd, (off_t)used) != 0) {
            if (!reused) { close(fd); unlink(path); }
            return -1;
        }
        if (fsync(fd) != 0) {
            if (!reused) { close(fd); unlink(path); }
            return -1;
        }
    }

    if (!want_detach) {
        void *m = mmap(s->pool, (size_t)used, PROT_READ | PROT_WRITE,
                       MAP_FIXED | MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) {
            if (!reused) { close(fd); unlink(path); }
            return -1;
        }
        if (used < s->pool_size) {
            void *t = mmap(s->pool + used, (size_t)(s->pool_size - used), PROT_NONE,
                           MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (t == MAP_FAILED)
                pool_hard_decommit_range(s, used, s->pool_size - used);
        }
        if (!reused) {
            if (s->pool_spill_fd > 2) close(s->pool_spill_fd);
            s->pool_spill_fd = fd;
            unlink(path);
        }
#if defined(__APPLE__)
        (void)fcntl(fd, F_NOCACHE, 1);
#endif
        s->pool_detached = 0;
        pool_pageout_range(s, 0, used);
        pool_pageout_range(s, 0, used);
        pool_pageout_range(s, 0, used);
        s->pool_spill_bytes = used;
        s->pool_spill_events++;
        return 1;
    }

    {
        int detached_ok = 1;
        const uint64_t chunk = (uint64_t)PAGE_SZ * 64;
        uint64_t cur = 0;
        while (cur < used) {
            uint64_t nend = cur + chunk;
            if (nend > used) nend = used;
            void *m = mmap(s->pool + cur, (size_t)(nend - cur), PROT_NONE,
                           MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (m == MAP_FAILED) {
                size_t off2 = (size_t)cur;
                while (off2 < (size_t)nend) {
                    size_t step = PAGE_SZ;
                    if (off2 + step > (size_t)nend) step = (size_t)nend - off2;
                    void *m1 = mmap(s->pool + off2, step, PROT_NONE,
                                    MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                    if (m1 == MAP_FAILED) {
                        (void)madvise(s->pool + off2, step, MADV_PAGEOUT);
#if defined(MADV_DONTNEED)
                        (void)madvise(s->pool + off2, step, MADV_DONTNEED);
#endif
                        detached_ok = 0;
                    }
                    off2 += step;
                }
            }
            cur = nend;
        }
        if (used < s->pool_size)
            pool_hard_decommit_range(s, used, s->pool_size - used);
        s->pool_detached = 1;
        if (!detached_ok) {
            pool_pageout_range(s, 0, used);
            pool_pageout_range(s, 0, used);
        }
    }
    if (!reused) {
        if (s->pool_spill_fd > 2) close(s->pool_spill_fd);
        s->pool_spill_fd = fd;
        unlink(path);
    }
#if defined(__APPLE__)
    (void)fcntl(fd, F_NOCACHE, 1);
#endif
    s->pool_spill_bytes = used;
    s->pool_spill_events++;
    return 1;
}

static void pool_hard_decommit_range(MemXZone3 *s, uint64_t off, uint64_t sz) {
    if (!s || !s->pool || sz == 0) return;
    if (off >= s->pool_size) return;
    if (off + sz > s->pool_size) sz = s->pool_size - off;
    uint64_t start = (off + (PAGE_SZ - 1)) & ~((uint64_t)PAGE_SZ - 1);
    uint64_t end = (off + sz) & ~((uint64_t)PAGE_SZ - 1);
    if (end <= start) return;
    const uint64_t max_chunk = (uint64_t)PAGE_SZ * 32;
    for (uint64_t cur = start; cur < end; ) {
        uint64_t nend = cur + max_chunk;
        if (nend > end) nend = end;
        uint8_t *pa = s->pool + cur;
        size_t bytes = (size_t)(nend - cur);
        void *m = mmap(pa, bytes, PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m == MAP_FAILED)
            pool_release_physical_range(s, cur, nend - cur);
        cur = nend;
    }
}

static void pool_prepare_write_range(MemXZone3 *s, uint64_t off, uint64_t sz) {
    if (!s || !s->pool || sz == 0) return;
    if (off >= s->pool_size) return;
    if (off + sz > s->pool_size) sz = s->pool_size - off;
    if (pool_is_vault_native(s)) {
        if (pool_ensure_spill_fd_locked(s) != 0) return;
        uint64_t end0 = off + (uint64_t)sz;
        if (end0 > s->pool_spill_bytes) {
            if (ftruncate(s->pool_spill_fd, (off_t)end0) == 0)
                s->pool_spill_bytes = end0;
        }
        return;
    }
    uint64_t start = off & ~((uint64_t)PAGE_SZ - 1);
    uint64_t end = (off + sz + PAGE_SZ - 1) & ~((uint64_t)PAGE_SZ - 1);
    if (end <= start) return;
    if (end > s->pool_size) end = s->pool_size;
    uint8_t *pa = s->pool + start;
    size_t bytes = (size_t)(end - start);
    if (s->pool_spill_fd > 2 && (s->pool_detached || s->pool_spill_bytes > 0)) {
        uint64_t file_end = s->pool_spill_bytes;
        if (end > file_end) {
            if (ftruncate(s->pool_spill_fd, (off_t)end) == 0)
                s->pool_spill_bytes = end;
        }
        void *m = mmap(pa, bytes, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, s->pool_spill_fd, (off_t)start);
        if (m != MAP_FAILED) {
#if defined(MADV_FREE_REUSE)
            madvise(pa, bytes, MADV_FREE_REUSE);
#endif
            return;
        }
    }
    if (s->pool_detached) {
        void *m = mmap(pa, bytes, PROT_READ | PROT_WRITE,
                       MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED)
            mprotect(pa, bytes, PROT_READ | PROT_WRITE);
#if defined(MADV_FREE_REUSE)
        madvise(pa, bytes, MADV_FREE_REUSE);
#endif
        return;
    }
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

static void pool_hard_decommit_all_free_extents_locked(MemXZone3 *s) {
    if (!s) return;
    for (uint32_t i = 0; i < s->pool_free_count; i++) {
        pool_hard_decommit_range(s, s->pool_free_off[i], s->pool_free_sz[i]);
    }
    if (s->pool_next < s->pool_size) {
        pool_hard_decommit_range(s, s->pool_next, s->pool_size - s->pool_next);
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
    if (s->vault_wbuf_len) (void)pool_vault_wbuf_flush_locked(s);
    s->vault_epoch++;
    if (s->vault_epoch == 0) s->vault_epoch = 1;
    pool_vault_cache_invalidate_range_locked(s, off, sz);
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
    if (pool_is_vault_native(s)) return 0;

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
                    if (s->dedup_rev[old_pp] == i + 1) s->dedup_rev[old_pp] = 0;
                    uint32_t new_pp = (uint32_t)(new_off[j] / PAGE_SZ) & s->dedup_rev_mask;
                    s->dedup_rev[new_pp] = (i) + 1;
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

static void mat_cache_invalidate(void);
static void release_all_compressed_physical_locked(MemXZone3 *s);

static uint64_t memx_runtime_reclaim_and_compact_locked(MemXZone3 *s) {
    if (!s) return 0;
    uint64_t reclaimed = memx_runtime_reclaim_locked(s);
    int allow = 1;
    const char *env_c = getenv("MEMX_SOFT_COMPACT");
    if (env_c && env_c[0] == '0') allow = 0;
    if (allow) {
        uint64_t free_bytes = pool_free_extent_bytes_locked(s);
        uint64_t live = s->pool_next > free_bytes ? (s->pool_next - free_bytes) : 0;
        if (free_bytes >= (PAGE_SZ * 1024) ||
            (live > 0 && free_bytes >= (live / 2)) ||
            s->pool_free_count >= 48) {
            uint64_t moved = pool_compact_locked(s);
            if (moved) reclaimed += moved;
            reclaimed += memx_runtime_reclaim_locked(s);
        }
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

static void page_hard_decommit_range(MemXZone3 *s, size_t first, size_t last_inclusive) {
    if (!s || !s->vmem || last_inclusive < first || first >= s->npages) return;
    if (last_inclusive >= s->npages) last_inclusive = s->npages - 1;
    const size_t max_chunk = 256;
    size_t cur = first;
    while (cur <= last_inclusive) {
        size_t end = cur + max_chunk - 1;
        if (end > last_inclusive) end = last_inclusive;
        size_t n = end - cur + 1;
        uint8_t *pa = (uint8_t*)s->vmem + cur * PAGE_SZ;
        size_t bytes = n * PAGE_SZ;
        void *m = mmap(pa, bytes, PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m == MAP_FAILED) {
            page_release_physical_range(s, cur, end);
        }
        cur = end + 1;
    }
}

static void release_all_compressed_physical_locked(MemXZone3 *s) {
    if (!s || !s->meta || !s->vmem) return;
    size_t run_start = (size_t)-1;
    size_t last = s->vmem_next / PAGE_SZ;
    if (last > s->npages) last = s->npages;
    if (last == 0) last = s->npages;
    for (size_t i = 0; i < last; i++) {
        if (s->meta[i].state == PAGE_COMPRESSED) {
            if (run_start == (size_t)-1) run_start = i;
        } else if (run_start != (size_t)-1) {
            page_release_physical_range(s, run_start, i - 1);
            run_start = (size_t)-1;
        }
    }
    if (run_start != (size_t)-1) page_release_physical_range(s, run_start, last - 1);
}

static void hard_release_all_compressed_physical_locked(MemXZone3 *s) {
    if (!s || !s->meta || !s->vmem) return;
    size_t last = s->vmem_next / PAGE_SZ;
    if (last > s->npages) last = s->npages;
    if (last == 0) last = s->npages;
    size_t run = (size_t)-1;
    const size_t max_chunk_pages = 256;
    for (size_t i = 0; i <= last; i++) {
        int is_comp = (i < last && s->meta[i].state == PAGE_COMPRESSED);
        if (is_comp) {
            if (run == (size_t)-1) run = i;
            continue;
        }
        if (run == (size_t)-1) continue;
        size_t end = i;
        size_t cur = run;
        while (cur < end) {
            size_t nend = cur + max_chunk_pages;
            if (nend > end) nend = end;
            size_t n = nend - cur;
            uint8_t *pa = (uint8_t*)s->vmem + cur * PAGE_SZ;
            size_t bytes = n * PAGE_SZ;
            void *m = mmap(pa, bytes, PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (m == MAP_FAILED) {
                for (size_t j = cur; j < nend; j++) {
                    uint8_t *pj = (uint8_t*)s->vmem + j * PAGE_SZ;
                    void *m1 = mmap(pj, PAGE_SZ, PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                    if (m1 == MAP_FAILED) page_release_physical(s, j);
                }
            }
            cur = nend;
        }
        run = (size_t)-1;
    }
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


static void meta_release_physical_range(MemXZone3 *s, size_t first, size_t last_inclusive) {
    if (!s || !s->meta || last_inclusive < first) return;
    if (last_inclusive >= s->npages) last_inclusive = s->npages - 1;
    uintptr_t a0 = ((uintptr_t)&s->meta[first]) & ~((uintptr_t)PAGE_SZ - 1);
    uintptr_t a1 = ((uintptr_t)&s->meta[last_inclusive] + sizeof(PageMeta) + PAGE_SZ - 1) & ~((uintptr_t)PAGE_SZ - 1);
    if (a1 <= a0) return;
#if defined(MADV_FREE_REUSABLE)
    madvise((void*)a0, a1 - a0, MADV_FREE_REUSABLE);
#endif
    madvise((void*)a0, a1 - a0, MADV_DONTNEED);
}

// ─── Free bitmap helpers ───
static inline void bm_set_free(MemXZone3 *s, size_t page) {
    uint64_t mask = (1ULL << (page % 64));
    uint64_t *word = &s->free_bm[page / 64];
    if (((*word) & mask) != 0) {
        *word &= ~mask;
        __sync_fetch_and_add(&s->free_pages_count, 1);
    }
}
static inline void bm_set_used(MemXZone3 *s, size_t page) {
    uint64_t mask = (1ULL << (page % 64));
    uint64_t *word = &s->free_bm[page / 64];
    if (((*word) & mask) == 0) {
        *word |= mask;
        __sync_fetch_and_sub(&s->free_pages_count, 1);
    }
}
static inline int bm_is_free(MemXZone3 *s, size_t page) {
    return (((s->free_bm[page / 64] >> (page % 64)) & 1) == 0);
}
static inline ssize_t bm_find_free_run(MemXZone3 *s, size_t npages, size_t hint) {
    size_t total = s->npages;
    for (size_t start = hint; start < total; ) {
        uint64_t used = s->free_bm[start / 64];
        uint64_t free_bits = ~used;
        if (free_bits == 0) { start = (start / 64 + 1) * 64; continue; }
        size_t base = start & ~63ULL;
        size_t bit_off = start - base;
        uint64_t masked = free_bits >> bit_off;
        if (masked == 0) { start = base + 64; continue; }
        int bit = __builtin_ctzll(masked) + (int)bit_off;
        size_t found = base + bit;
        if (found >= total) return -1;
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
static __thread int g_deflate_level_cur = -1;
static __thread int g_deflate_level_req = 1;
static __thread int g_recompress_mode = 0;
static __thread z_stream g_inflate_zs;
static __thread int g_inflate_ready = 0;

static int memx_deflate_level_get(void) {
    int lv = g_deflate_level_req;
    if (lv < 1) lv = 1;
    if (lv > 9) lv = 9;
    return lv;
}

static int memx_deflate_level_push(int level) {
    int prev = g_deflate_level_req;
    if (level < 1) level = 1;
    if (level > 9) level = 9;
    g_deflate_level_req = level;
    return prev;
}

static int memx_deflate_once(const uint8_t *src, uLong src_len, uint8_t *dst, uLongf *out_len) {
    int level = memx_deflate_level_get();
    if (!g_deflate_ready || g_deflate_level_cur != level) {
        if (g_deflate_ready) {
            deflateEnd(&g_deflate_zs);
            g_deflate_ready = 0;
            g_deflate_level_cur = -1;
        }
        memset(&g_deflate_zs, 0, sizeof(g_deflate_zs));
        if (deflateInit(&g_deflate_zs, level) != Z_OK) return Z_MEM_ERROR;
        g_deflate_ready = 1;
        g_deflate_level_cur = level;
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
    {
        uLongf lo_bound = compressBound(half_count);
        uLongf hi_bound = compressBound(half_count);
        if (24 + lo_bound + hi_bound > cap) {
            if (cap > 24) {
                uint32_t rem = cap - 24;
                lo_bound = rem / 2;
                hi_bound = rem - lo_bound;
            } else {
                lo_bound = 0;
                hi_bound = 0;
            }
        }
        if (lo_bound >= 16 && hi_bound >= 16) {
            uint8_t lo_z[PAGE_SZ];
            uint8_t hi_z[PAGE_SZ];
            uLongf lo_len = lo_bound;
            uLongf hi_len = hi_bound;
            int rc_lo = memx_deflate_once(lo_tmp, half_count, lo_z, &lo_len);
            int rc_hi = memx_deflate_once(hi_tmp, half_count, hi_z, &hi_len);
            if (rc_lo == Z_OK && rc_hi == Z_OK && lo_len > 0 && hi_len > 0 &&
                lo_len + 32 < half_count && hi_len + 8 < half_count) {
                uint32_t total = (uint32_t)(24 + lo_len + hi_len);
                if (total < PAGE_SZ && total < (PAGE_SZ * 15) / 16 && total <= cap) {
                    dst[0] = 0x4D;
                    dst[1] = 0x58;
                    dst[2] = MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT;
                    dst[3] = 2;
                    dst[4] = (uint8_t)(half_count & 0xFF);
                    dst[5] = (uint8_t)((half_count >> 8) & 0xFF);
                    dst[6] = (uint8_t)((half_count >> 16) & 0xFF);
                    dst[7] = (uint8_t)((half_count >> 24) & 0xFF);
                    dst[8] = (uint8_t)(lo_len & 0xFF);
                    dst[9] = (uint8_t)((lo_len >> 8) & 0xFF);
                    dst[10] = (uint8_t)((lo_len >> 16) & 0xFF);
                    dst[11] = (uint8_t)((lo_len >> 24) & 0xFF);
                    dst[12] = (uint8_t)(hi_len & 0xFF);
                    dst[13] = (uint8_t)((hi_len >> 8) & 0xFF);
                    dst[14] = (uint8_t)((hi_len >> 16) & 0xFF);
                    dst[15] = (uint8_t)((hi_len >> 24) & 0xFF);
                    memset(dst + 16, 0, 8);
                    memcpy(dst + 24, lo_z, lo_len);
                    memcpy(dst + 24 + lo_len, hi_z, hi_len);
                    return total;
                }
            }
        }
    }
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
        if(half_count!=PAGE_SZ/2){memset(dst,0,PAGE_SZ);return;}
        if(src[3]>=2){
            uint32_t lo_zlen=(uint32_t)src[8]|((uint32_t)src[9]<<8)|((uint32_t)src[10]<<16)|((uint32_t)src[11]<<24);
            uint32_t hi_zlen=(uint32_t)src[12]|((uint32_t)src[13]<<8)|((uint32_t)src[14]<<16)|((uint32_t)src[15]<<24);
            if(lo_zlen==0||hi_zlen==0||24+lo_zlen+hi_zlen>cs){memset(dst,0,PAGE_SZ);return;}
            uint8_t lo[PAGE_SZ/2];
            uint8_t hi[PAGE_SZ/2];
            uLongf loLen=half_count, hiLen=half_count;
            if(memx_inflate_once(src+24,lo_zlen,lo,&loLen)!=Z_OK||loLen!=half_count){memset(dst,0,PAGE_SZ);return;}
            if(memx_inflate_once(src+24+lo_zlen,hi_zlen,hi,&hiLen)!=Z_OK||hiLen!=half_count){memset(dst,0,PAGE_SZ);return;}
            memx_interleave_lo_hi(lo, hi, dst, half_count);
            return;
        }
        uint32_t hi_zlen=(uint32_t)src[8]|((uint32_t)src[9]<<8)|((uint32_t)src[10]<<16)|((uint32_t)src[11]<<24);
        if(hi_zlen==0||16+half_count+hi_zlen>cs){memset(dst,0,PAGE_SZ);return;}
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
    const char *nos = getenv("MEMX_NO_SELFTEST");
    if (nos && nos[0] == '1') return;
    uint8_t src[8] = {0x4D, 0x58, 0x03, 0x00, 0xFD, 0x00, 0x00, 0x40};
    uint8_t dst[16384];
    memset(dst, 0xCC, sizeof(dst));
    cpu_decompress(src, 8, dst);
    int ok = 1;
    for (int i = 0; i < 16384; i++) { if (dst[i] != 0) { ok = 0; break; } }
    if (!ok) write(2, "[SELFTEST] ZERO PAGE DECOMP FAILED!\n", 37);
    else write(2, "[SELFTEST] ZERO PAGE DECOMP OK\n", 31);
}

static int ensure_metal(MemXZone3 *s) {
    if (!s) return -1;
    if (s->device && s->comp_pipe && s->decomp_pipe && s->queue) return 0;
    const char *cpu_only = getenv("MEMX_CPU_ONLY");
    if (cpu_only && cpu_only[0] == '1') return -1;
    @autoreleasepool {
        if (!s->device) s->device = MTLCreateSystemDefaultDevice();
        if (!s->device) return -1;
        NSError *err = nil;
        id<MTLLibrary> lib = [s->device newLibraryWithSource:shader_src options:nil error:&err];
        if (!lib) return -1;
        if (!s->queue) s->queue = [s->device newCommandQueue];
        if (!s->comp_pipe)
            s->comp_pipe = [s->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"cp"] error:&err];
        if (!s->decomp_pipe)
            s->decomp_pipe = [s->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dp"] error:&err];
        if (!s->comp_pipe || !s->decomp_pipe || !s->queue) return -1;
    }
    return 0;
}

static int ensure_gpu_bufs(MemXZone3 *s) {
    if (!s) return -1;
    if (ensure_metal(s) != 0) return -1;
    size_t batch_bytes = s->batch_cap * PAGE_SZ;
    if (!s->gpu_sb)
        s->gpu_sb = [s->device newBufferWithLength:batch_bytes options:MTLResourceStorageModeShared];
    if (!s->gpu_db)
        s->gpu_db = [s->device newBufferWithLength:batch_bytes options:MTLResourceStorageModeShared];
    if (!s->gpu_zb)
        s->gpu_zb = [s->device newBufferWithLength:s->batch_cap * 4 options:MTLResourceStorageModeShared];
    return (s->gpu_sb && s->gpu_db && s->gpu_zb) ? 0 : -1;
}

static int gpu_compress(MemXZone3 *s, size_t count) {
    if (ensure_gpu_bufs(s) != 0) return -1;
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
    uint32_t slot1 = s->dedup_rev[pp];
    if (slot1 > 0 && slot1 <= DEDUP_HT_SIZE) {
        uint32_t slot = slot1 - 1;
        if (s->dedup_ref[slot] > 0 && s->dedup_off[slot] == pool_offset && s->dedup_sz[slot] == comp_size) {
            uint32_t old = __sync_fetch_and_sub(&s->dedup_ref[slot], 1);
            if (old == 1 && s->dedup_pending_free && !s->dedup_pending_free[slot]) {
                s->dedup_pending_free[slot] = 1;
                __sync_fetch_and_add(&s->dedup_pending_free_count, 1);
            }
            return;
        }
    }
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
            if (s->dedup_rev[pp] == i + 1) s->dedup_rev[pp] = 0;
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
            if ((pool_is_vault_native(s) || s->pool_detached) && s->pool_spill_fd > 2) {
                if (pool_copy_blob_locked(s, d_off, d_sz, g_comp_payload) != 0) {
                    m->state = PAGE_COMPRESSED;
                    __sync_fetch_and_add(&s->live_compressed_pages, 1);
                    if (s->live_resident_pages) __sync_fetch_and_sub(&s->live_resident_pages, 1);
                    pthread_mutex_unlock(&s->alloc_mutex);
                    return 0;
                }
            } else {
                uint64_t p0 = d_off & ~((uint64_t)PAGE_SZ - 1);
                uint64_t p1 = (d_off + d_sz + PAGE_SZ - 1) & ~((uint64_t)PAGE_SZ - 1);
                if (p1 > s->pool_size) p1 = s->pool_size;
                if (p1 > p0) {
                    size_t bytes = (size_t)(p1 - p0);
                    if (mprotect(s->pool + p0, bytes, PROT_READ) != 0)
                        (void)mprotect(s->pool + p0, bytes, PROT_READ | PROT_WRITE);
                }
                memcpy(g_comp_payload, s->pool + d_off, d_sz);
            }
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
    int try_delta = try_fp16 && (role == MEMX_TENSOR_ROLE_KV_CACHE || coldish || role == MEMX_TENSOR_ROLE_ACTIVATION ||
                                 role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING);
    int is_bf16 = (s->meta[pidx].tensor_dtype == MEMX_TENSOR_DTYPE_BF16);
    if (role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING) {
        if (try_fp16)
            preferred = MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT;
        else
            preferred = MEMX_CODEC_ZLIB;
        s->meta[pidx].preferred_codec = preferred;
    } else if (role == MEMX_TENSOR_ROLE_KV_CACHE && try_delta) {
        preferred = MEMX_CODEC_TENSOR_FP16_DELTA_SPLIT;
        s->meta[pidx].preferred_codec = preferred;
    } else if (role == MEMX_TENSOR_ROLE_ACTIVATION && try_fp16) {
        preferred = MEMX_CODEC_TENSOR_FP16_SPLIT;
        s->meta[pidx].preferred_codec = preferred;
    }
    int sticky_ok = preferred != 0 && s->meta[pidx].codec_fail_streak < 4;
    uint32_t sticky_good = (role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING)
        ? (PAGE_SZ * 11 / 16)
        : ((role == MEMX_TENSOR_ROLE_KV_CACHE || coldish) ? (PAGE_SZ * 13 / 16) : (PAGE_SZ - 32));
    {
        uint32_t pressure = memx_pool_pressure_percent_locked(s);
        if (pressure >= 80) sticky_good = PAGE_SZ * 7 / 16;
        else if (pressure >= 60 && sticky_good > (PAGE_SZ * 7 / 16))
            sticky_good = PAGE_SZ * 7 / 16;
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
    if (prefer_ratio && (tensor_csz == 0 || tensor_csz >= sticky_good)) need_compete = 1;
    if ((role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING) &&
        (tensor_csz == 0 || tensor_csz > (PAGE_SZ / 2))) need_compete = 1;
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
        if (try_fp16 &&
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
            if ((role == MEMX_TENSOR_ROLE_WEIGHT || role == MEMX_TENSOR_ROLE_EMBEDDING) &&
                best_codec == MEMX_CODEC_TENSOR_EXP_PACK) {
                uint32_t zsplit = tensor_fp16_zlib_split_compress(src, codec_tmp, PAGE_SZ);
                if (zsplit > 0 && zsplit <= (best_csz + (best_csz / 8) + 64)) {
                    best_csz = zsplit;
                    best_codec = MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT;
                    memcpy(tensor_dst, codec_tmp, zsplit);
                }
            }
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
        if (g_hard_quiesce || s->phoenix_sealed || !s->meta || !s->vmem) {
            struct timespec ts={0, 2000000L};
            nanosleep(&ts, NULL);
            continue;
        }
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
                    if (pool_blob_eq_locked(s, existing_off, cdata, cs)) {
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
            if (pool_store_blob_locked(s, off, cdata, cs) != 0) {
                pool_free_insert_locked(s, off, cs);
                restore_compressing_page(s, pidx);
                continue;
            }
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
                        s->dedup_rev[(uint32_t)(off / PAGE_SZ) & s->dedup_rev_mask] = (s2) + 1;
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
    if (np > 0) meta_release_physical_range(g_z, sp, sp + np - 1);
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

static int memx_pthread_spawn(pthread_t *th, void *(*fn)(void *), void *arg) {
    pthread_attr_t attr;
    int rc;
    if (pthread_attr_init(&attr) != 0)
        return pthread_create(th, NULL, fn, arg);
    (void)pthread_attr_setstacksize(&attr, 384 * 1024);
    rc = pthread_create(th, &attr, fn, arg);
    pthread_attr_destroy(&attr);
    if (rc != 0)
        rc = pthread_create(th, NULL, fn, arg);
    return rc;
}

static void init_memx(void) {
    static pthread_mutex_t init_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&init_mutex);
    if (g_z) { pthread_mutex_unlock(&init_mutex); return; }
    in_memx = 1;  // Prevent recursion during Metal init
    
    g_z = (MemXZone3*)mmap(NULL, sizeof(MemXZone3), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (g_z == MAP_FAILED) { g_z = NULL; pthread_mutex_unlock(&init_mutex); return; }
    memset(g_z, 0, sizeof(MemXZone3));
    g_z->pool_spill_fd = -1;
    g_z->pool_detached = 0;
    g_z->pool_ghost = 0;
    g_z->pool_spill_bytes = 0;
    g_z->pool_spill_events = 0;
    g_z->pool_ghost_flushed = 0;
    g_z->pool_ghost_stores = 0;
    g_z->pool_vault_native = 0;
    g_z->pool_vault_stores = 0;
    g_z->pool_vault_reads = 0;
    g_z->pool_vault_cache_hits = 0;
    g_z->pool_vault_window_hits = 0;
    g_z->pool_vault_wbuf_flushes = 0;
    g_z->vault_cache = NULL;
    g_z->vault_cache_bytes = 0;
    g_z->vault_cache_next = 0;
    g_z->vault_cache_slots = 0;
    g_z->vault_cache_off = NULL;
    g_z->vault_cache_sz = NULL;
    g_z->vault_cache_pos = NULL;
    g_z->vault_cache_live = NULL;
    g_z->vault_cache_gen = NULL;
    g_z->vault_epoch = 1;
    g_z->vault_cache_prefer_bytes = 0;
    g_z->vault_avcs_events = 0;
    g_z->vault_ring_reclaims = 0;
    g_z->sov_tca_pages = 0;
    g_z->sov_tca_bytes = 0;
    g_z->vault_wbuf = NULL;
    g_z->vault_wbuf_cap = 0;
    g_z->vault_wbuf_base = 0;
    g_z->vault_wbuf_len = 0;
    for (int wi = 0; wi < 4; wi++) {
        g_z->vault_win[wi] = NULL;
        g_z->vault_win_base[wi] = 0;
        g_z->vault_win_len[wi] = 0;
    }
    g_z->vault_win_clock = 0;
    g_z->sovereign = 0;
    g_z->sovereign_frozen = 0;
    g_z->phoenix_sealed = 0;
    g_z->sov_count = 0;
    g_z->sov_cap = 0;
    g_z->sov_ents = NULL;
    g_z->sov_bytes = 0;
    g_z->sov_hits = 0;
    g_z->sov_off_idx = NULL;
    g_z->sov_off_idx_bytes = 0;
    g_z->sov_warm_bytes = 0;
    g_z->sov_stream_injects = 0;
    g_z->sov_crw_spans = 0;
    g_z->sov_crw_pages = 0;
    g_z->sov_crw_bytes = 0;
    g_z->sov_chronos_injects = 0;
    g_z->sov_warp_buf = NULL;
    g_z->sov_warp_buf_bytes = 0;
    
    g_z->device = nil;
    g_z->queue = nil;
    g_z->comp_pipe = nil;
    g_z->decomp_pipe = nil;
    g_z->gpu_sb = nil;
    g_z->gpu_db = nil;
    g_z->gpu_zb = nil;
    g_z->batch_cap = 32;
    size_t batch_bytes = g_z->batch_cap * PAGE_SZ;
    g_z->tmp_src = (uint8_t*)mmap(NULL, batch_bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    g_z->tmp_dst = (uint8_t*)mmap(NULL, batch_bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    g_z->tmp_sz  = (uint32_t*)mmap(NULL, g_z->batch_cap*4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    // Dedup hash tables
    g_z->dedup_hash = (uint64_t*)mmap(NULL, DEDUP_HT_SIZE*8, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    g_z->dedup_off  = (uint64_t*)mmap(NULL, DEDUP_HT_SIZE*8, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    g_z->dedup_sz   = (uint32_t*)mmap(NULL, DEDUP_HT_SIZE*4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    g_z->dedup_ref  = (uint32_t*)mmap(NULL, DEDUP_HT_SIZE*4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    g_z->dedup_pending_free = (uint8_t*)mmap(NULL, DEDUP_HT_SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    memset(g_z->dedup_hash, 0, DEDUP_HT_SIZE*8);
    memset(g_z->dedup_ref, 0, DEDUP_HT_SIZE*4);
    memset(g_z->dedup_pending_free, 0, DEDUP_HT_SIZE);
    g_z->dedup_rev = NULL;
    g_z->dedup_rev_size = 0;
    g_z->dedup_rev_mask = 0;
    g_z->pool_free_cap = POOL_FREE_EXTENTS_MAX;
    g_z->pool_free_off = (uint64_t*)mmap(NULL, g_z->pool_free_cap * sizeof(uint64_t), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    g_z->pool_free_sz = (uint32_t*)mmap(NULL, g_z->pool_free_cap * sizeof(uint32_t), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    // Active page tracking lists (compact: only track active pages, not all 6M)
    g_z->hot_cap = 16384;
    g_z->res_cap = 131072;
    g_z->hot_list = (uint32_t*)mmap(NULL, g_z->hot_cap * 4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    g_z->res_list = (uint32_t*)mmap(NULL, g_z->res_cap * 4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    g_z->hot_count = 0;
    g_z->res_count = 0;
    if (!g_z->tmp_src||!g_z->tmp_dst||!g_z->tmp_sz||
        !g_z->dedup_hash||!g_z->dedup_pending_free||!g_z->pool_free_off||!g_z->pool_free_sz||
        !g_z->hot_list||!g_z->res_list) {
        munmap(g_z, sizeof(MemXZone3)); g_z = NULL; pthread_mutex_unlock(&init_mutex); return;
    }
    
    // Virtual memory
    int64_t ms=0; size_t len=sizeof(ms);
    sysctlbyname("hw.memsize", &ms, &len, NULL, 0);
    g_z->vmem_size = ms * 2;
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
        g_z->dedup_rev = (uint32_t*)mmap(NULL, (size_t)g_z->dedup_rev_size * 4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
        if (g_z->dedup_rev == MAP_FAILED) {
            munmap(g_z->pool, g_z->pool_size);
            munmap(g_z->vmem, g_z->vmem_size);
            munmap(g_z, sizeof(MemXZone3));
            g_z = NULL;
            pthread_mutex_unlock(&init_mutex);
            return;
        }
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
    g_z->free_bm = (uint64_t*)mmap(NULL, bm_sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (g_z->free_bm == MAP_FAILED) { munmap(g_z->meta, meta_sz); munmap(g_z->pool, g_z->pool_size); munmap(g_z->vmem, g_z->vmem_size); munmap(g_z, sizeof(MemXZone3)); g_z = NULL; pthread_mutex_unlock(&init_mutex); return; }
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
    if (pool_vault_native_enabled()) {
        g_z->pool_vault_native = 1;
        g_z->pool_detached = 1;
        g_z->pool_ghost = 1;
        if (pool_ensure_spill_fd_locked(g_z) != 0) {
            g_z->pool_vault_native = 0;
            g_z->pool_detached = 0;
        } else {
            (void)pool_vault_cache_init_locked(g_z);
        }
    }
    
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
            if (memx_pthread_spawn(&g_z->encode_threads[wi], tensor_encode_pool_worker, g_z) == 0)
                g_z->encode_nworkers++;
            else break;
        }
        if (g_z->encode_nworkers == 0) g_z->encode_pool_running = 0;
    }
    memx_pthread_spawn(&g_z->bg_thread, bg_compressor, g_z);
    if (g_z->async_pf_q) {
        for (int wi = 0; wi < ASYNC_PF_WORKERS; wi++) {
            if (memx_pthread_spawn(&g_z->async_pf_threads[wi], async_pf_worker, g_z) == 0)
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
            if (memx_pthread_spawn(&g_z->async_seal_threads[wi], async_seal_worker, g_z) == 0)
                g_z->async_seal_nworkers++;
            else break;
        }
        if (g_z->async_seal_nworkers == 0) g_z->async_seal_running = 0;
    } else {
        g_z->async_seal_running = 0;
    }
    shared_stats_init(g_z);
    
    fprintf(stderr, "[memx] ✅ memory expansion active (%llu MB virtual, %s, metal=lazy)\n",
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
    if (!g_z->phoenix_sealed && g_z->meta && g_z->vmem) {
        for (size_t i=0; i<g_z->npages; i++)
            if (g_z->meta[i].state==PAGE_COMPRESSED||g_z->meta[i].state==PAGE_NONE)
                mprotect((uint8_t*)g_z->vmem+i*PAGE_SZ, PAGE_SZ, PROT_READ|PROT_WRITE);
    }
    g_z->gpu_sb = nil; g_z->gpu_db = nil; g_z->gpu_zb = nil;
    size_t batch_bytes = g_z->batch_cap * PAGE_SZ;
    if (g_z->tmp_src && g_z->tmp_src != MAP_FAILED) munmap(g_z->tmp_src, batch_bytes);
    if (g_z->tmp_dst && g_z->tmp_dst != MAP_FAILED) munmap(g_z->tmp_dst, batch_bytes);
    if (g_z->tmp_sz && g_z->tmp_sz != MAP_FAILED) munmap(g_z->tmp_sz, g_z->batch_cap * 4);
    if (g_z->vmem && g_z->vmem != MAP_FAILED && g_z->vmem_size) munmap(g_z->vmem, g_z->vmem_size);
    if (g_z->pool && g_z->pool != MAP_FAILED && g_z->pool_size) munmap(g_z->pool, g_z->pool_size);
    if (g_z->meta && g_z->meta != MAP_FAILED && g_z->npages) munmap(g_z->meta, g_z->npages * sizeof(PageMeta));
    if (g_z->free_bm && g_z->free_bm != MAP_FAILED && g_z->free_bm_size) munmap(g_z->free_bm, g_z->free_bm_size * sizeof(uint64_t));
    if (g_z->hot_list && g_z->hot_cap) munmap(g_z->hot_list, g_z->hot_cap * 4);
    if (g_z->res_list && g_z->res_cap) munmap(g_z->res_list, g_z->res_cap * 4);
    if (g_z->dedup_hash && g_z->dedup_hash != MAP_FAILED) munmap(g_z->dedup_hash, DEDUP_HT_SIZE*8);
    if (g_z->dedup_off && g_z->dedup_off != MAP_FAILED) munmap(g_z->dedup_off, DEDUP_HT_SIZE*8);
    if (g_z->dedup_sz && g_z->dedup_sz != MAP_FAILED) munmap(g_z->dedup_sz, DEDUP_HT_SIZE*4);
    if (g_z->dedup_ref && g_z->dedup_ref != MAP_FAILED) munmap(g_z->dedup_ref, DEDUP_HT_SIZE*4);
    if (g_z->dedup_pending_free && g_z->dedup_pending_free != MAP_FAILED) munmap(g_z->dedup_pending_free, DEDUP_HT_SIZE);
    if (g_z->dedup_rev && g_z->dedup_rev != MAP_FAILED && g_z->dedup_rev_size) munmap(g_z->dedup_rev, (size_t)g_z->dedup_rev_size * 4);
    if (g_z->pool_free_off && g_z->pool_free_off != MAP_FAILED && g_z->pool_free_cap) munmap(g_z->pool_free_off, g_z->pool_free_cap * sizeof(uint64_t));
    if (g_z->pool_free_sz && g_z->pool_free_sz != MAP_FAILED && g_z->pool_free_cap) munmap(g_z->pool_free_sz, g_z->pool_free_cap * sizeof(uint32_t));
    if (g_z->sov_ents && g_z->sov_ents != MAP_FAILED && g_z->sov_bytes) munmap(g_z->sov_ents, (size_t)g_z->sov_bytes);
    if (g_z->sov_off_idx && g_z->sov_off_idx != MAP_FAILED && g_z->sov_off_idx_bytes) munmap(g_z->sov_off_idx, (size_t)g_z->sov_off_idx_bytes);
    if (g_z->sov_pidx_map && g_z->sov_pidx_map != MAP_FAILED && g_z->sov_pidx_map_bytes) munmap(g_z->sov_pidx_map, (size_t)g_z->sov_pidx_map_bytes);
    if (g_z->vault_cache && g_z->vault_cache != MAP_FAILED && g_z->vault_cache_bytes) munmap(g_z->vault_cache, (size_t)g_z->vault_cache_bytes);
    if (g_z->pool_spill_fd > 2) { close(g_z->pool_spill_fd); g_z->pool_spill_fd = -1; }
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
            if (npages > 0) meta_release_physical_range(g_z, sp, sp + npages - 1);
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
    if (st == PAGE_COMPRESSED) {
        if (!g_recompress_mode) return 1;
        {
            uint32_t thr = PAGE_SZ / 2;
            const char *env_keep = getenv("MEMX_RECOMPRESS_MAX_KEEP_PCT");
            if (env_keep && env_keep[0]) {
                int pct = atoi(env_keep);
                if (pct >= 10 && pct <= 95) thr = (PAGE_SZ * (uint32_t)pct) / 100;
            }
            if (m->comp_size > 0 && m->comp_size <= thr) return 1;
        }
        m->preferred_codec = MEMX_CODEC_TENSOR_FP16_ZLIB_SPLIT;
        m->codec_fail_streak = 0;
        if (!decompress_compressed_page(s, pidx, 0, 1)) return 0;
        st = m->state;
    }
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
    int deflate_prev = -1;
    if ((m->tensor_role == MEMX_TENSOR_ROLE_WEIGHT || m->tensor_role == MEMX_TENSOR_ROLE_EMBEDDING) &&
        (m->tensor_flags & (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY)) != 0) {
        int lv = g_recompress_mode ? 6 : 1;
        const char *env_lv = getenv("MEMX_WEIGHT_ZLIB_LEVEL");
        if (env_lv && env_lv[0]) {
            int v = atoi(env_lv);
            if (v >= 1 && v <= 9) lv = v;
        }
        if (g_recompress_mode) {
            const char *env_r = getenv("MEMX_FINAL_ZLIB_LEVEL");
            if (env_r && env_r[0]) {
                int v = atoi(env_r);
                if (v >= 1 && v <= 9) lv = v;
            }
            if (lv < 6) lv = 6;
        }
        if (lv > 1) deflate_prev = memx_deflate_level_push(lv);
    }
    if ((m->tensor_flags & MEMX_TENSOR_FLAG_SEQUENTIAL) != 0) {
        mprotect(pa, PAGE_SZ, PROT_NONE);
        __sync_synchronize();
        mprotect(pa, PAGE_SZ, PROT_READ);
        __sync_synchronize();
        if (m->dirty || m->write_seq != seq0) {
            if (deflate_prev >= 0) memx_deflate_level_push(deflate_prev);
            restore_compressing_page(s, pidx);
            return 0;
        }
    }
    memcpy(src, pa, PAGE_SZ);
    if (m->dirty || m->write_seq != seq0) {
        if (deflate_prev >= 0) memx_deflate_level_push(deflate_prev);
        restore_compressing_page(s, pidx);
        return 0;
    }
    uint32_t csz = 0;
    uint8_t codec = 0;
    encode_tensor_page_one(s, pidx, src, dst, tmp, &csz, &codec);
    if (deflate_prev >= 0) memx_deflate_level_push(deflate_prev);
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
    for (int probe = 0; probe < 24; probe++) {
        uint32_t s2 = (slot + probe) & DEDUP_HT_MASK;
        if (s->dedup_hash[s2] == 0) break;
        if (s->dedup_hash[s2] == h && s->dedup_sz[s2] == csz && s->dedup_ref[s2] > 0) {
            uint64_t existing_off = s->dedup_off[s2];
            if (pool_blob_eq_locked(s, existing_off, dst, csz)) {
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
        if (pool_store_blob_locked(s, off, dst, csz) != 0) {
            pool_free_insert_locked(s, off, csz);
            restore_compressing_page(s, pidx);
            pthread_mutex_unlock(&s->alloc_mutex);
            return 0;
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
        for (int probe = 0; probe < 24; probe++) {
            uint32_t s2 = (slot + probe) & DEDUP_HT_MASK;
            if (s->dedup_hash[s2] == 0 || s->dedup_ref[s2] == 0) {
                s->dedup_hash[s2] = h;
                s->dedup_off[s2] = off;
                s->dedup_sz[s2] = csz;
                s->dedup_ref[s2] = 1;
                if (s->dedup_rev && s->dedup_rev_size) s->dedup_rev[(uint32_t)(off / PAGE_SZ) & s->dedup_rev_mask] = (s2) + 1;
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
    for (size_t i = first; i <= last; i++) {
        PageMeta *m = &g_z->meta[i];
        if (m->state == PAGE_COMPRESSING) restore_compressing_page(g_z, i);
        if (m->state == PAGE_HOT && m->comp_size == 0) {
            uint8_t old = __sync_val_compare_and_swap(&m->state, PAGE_HOT, PAGE_RESIDENT);
            if (old == PAGE_HOT) {
                m->prefetched = 0;
                m->cooldown = 0;
            }
        }
        if (m->state == PAGE_RESIDENT || (m->state == PAGE_HOT && m->comp_size == 0)) {
            uint8_t *pa = (uint8_t *)g_z->vmem + i * PAGE_SZ;
            int prot = page_wants_write_protect(m) ? PROT_READ : (PROT_READ | PROT_WRITE);
            if (mprotect(pa, PAGE_SZ, prot) != 0) {
                (void)mmap(pa, PAGE_SZ, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            }
        }
    }
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
    mat_cache_invalidate();
    pthread_mutex_lock(&g_z->alloc_mutex);
    memx_runtime_reclaim_locked(g_z);
    release_all_compressed_physical_locked(g_z);
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

static void mat_cache_invalidate(void);

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
    if (phase == MEMX_EPOCH_FINAL) {
        mat_cache_invalidate();
    }
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


#define MEMX_ARCHIVE_MAGIC 0x4157584Du

typedef struct memx_archive_file_hdr {
    uint32_t magic;
    uint32_t version;
    uint32_t page_size;
    uint32_t flags;
    uint64_t nbytes;
    uint64_t page_count;
    uint32_t role;
    uint32_t dtype;
    uint32_t layout;
    uint32_t reserved0;
    uint64_t shape[4];
    uint64_t stride[4];
} memx_archive_file_hdr_t;

typedef struct memx_archive_page_ent {
    uint32_t comp_size;
    uint8_t codec;
    uint8_t kind;
    uint8_t reserved[2];
} memx_archive_page_ent_t;

static int archive_write_all(FILE *fp, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t off = 0;
    while (off < n) {
        size_t w = fwrite(p + off, 1, n - off, fp);
        if (w == 0) return -1;
        off += w;
    }
    return 0;
}

static int archive_read_all(FILE *fp, void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    size_t off = 0;
    while (off < n) {
        size_t r = fread(p + off, 1, n - off, fp);
        if (r == 0) return -1;
        off += r;
    }
    return 0;
}

static int install_precompressed_page(MemXZone3 *s, size_t pidx, const uint8_t *blob, uint32_t csz, uint8_t codec) {
    if (!s || !blob || csz == 0 || csz > PAGE_SZ || pidx >= s->npages) return -1;
    PageMeta *m = &s->meta[pidx];
    uint8_t *pa = (uint8_t *)s->vmem + pidx * PAGE_SZ;
    pthread_mutex_lock(&s->alloc_mutex);
    if (s->dedup_pending_free_count > 0) memx_runtime_reclaim_locked(s);
    if (m->comp_size != 0 && (m->state == PAGE_COMPRESSED || m->state == PAGE_HOT)) {
        dedup_decref(s, m->pool_offset, m->comp_size);
        m->pool_offset = 0;
        m->comp_size = 0;
        m->codec = 0;
        if (s->live_compressed_pages) __sync_fetch_and_sub(&s->live_compressed_pages, 1);
    }
    uint64_t off = 0;
    if (pool_alloc_extent_locked(s, csz, &off) != 0) {
        pthread_mutex_unlock(&s->alloc_mutex);
        return -1;
    }
    pool_prepare_write_range(s, off, csz);
    if (pool_store_blob_locked(s, off, blob, csz) != 0) {
        pool_free_insert_locked(s, off, csz);
        return ENOMEM;
    }
    {
        uint64_t start = off & ~((uint64_t)PAGE_SZ - 1);
        uint64_t end = (off + csz + PAGE_SZ - 1) & ~((uint64_t)PAGE_SZ - 1);
        if (end > s->pool_size) end = s->pool_size;
        if (end > start) mprotect(s->pool + start, (size_t)(end - start), PROT_READ);
    }
    uint64_t h = fnv1a_word(blob, csz);
    uint32_t slot = (uint32_t)(h & DEDUP_HT_MASK);
    for (int probe = 0; probe < 24; probe++) {
        uint32_t s2 = (slot + probe) & DEDUP_HT_MASK;
        if (s->dedup_hash[s2] == 0 || s->dedup_ref[s2] == 0) {
            s->dedup_hash[s2] = h;
            s->dedup_off[s2] = off;
            s->dedup_sz[s2] = csz;
            s->dedup_ref[s2] = 1;
            if (s->dedup_rev && s->dedup_rev_size) s->dedup_rev[(uint32_t)(off / PAGE_SZ) & s->dedup_rev_mask] = (s2) + 1;
            break;
        }
        if (s->dedup_hash[s2] == h && s->dedup_sz[s2] == csz && s->dedup_ref[s2] > 0 &&
            s->dedup_off[s2] < s->pool_size && pool_blob_eq_locked(s, s->dedup_off[s2], blob, csz)) {
            pool_free_insert_locked(s, off, csz);
            off = s->dedup_off[s2];
            __sync_fetch_and_add(&s->dedup_ref[s2], 1);
            break;
        }
    }
    uint8_t st = m->state;
    mprotect(pa, PAGE_SZ, PROT_NONE);
    m->pool_offset = off;
    m->comp_size = csz;
    m->codec = codec;
    m->preferred_codec = codec ? codec : m->preferred_codec;
    m->dirty = 0;
    m->codec_fail_streak = 0;
    __sync_synchronize();
    m->state = PAGE_COMPRESSED;
    if (st == PAGE_RESIDENT || st == PAGE_HOT || st == PAGE_COMPRESSING) {
        if (s->live_resident_pages) __sync_fetch_and_sub(&s->live_resident_pages, 1);
    }
    __sync_fetch_and_add(&s->live_compressed_pages, 1);
    __sync_fetch_and_add(&s->pool_used, csz);
    note_page_compressed(s, pidx, codec, csz);
    page_release_physical(s, pidx);
    pthread_mutex_unlock(&s->alloc_mutex);
    return 0;
}

static int install_raw_page(MemXZone3 *s, size_t pidx, const uint8_t *raw) {
    if (!s || !raw || pidx >= s->npages) return -1;
    PageMeta *m = &s->meta[pidx];
    uint8_t *pa = (uint8_t *)s->vmem + pidx * PAGE_SZ;
    if (m->state == PAGE_COMPRESSED || (m->state == PAGE_HOT && m->comp_size != 0)) {
        pthread_mutex_lock(&s->alloc_mutex);
        if (m->comp_size) {
            dedup_decref(s, m->pool_offset, m->comp_size);
            m->pool_offset = 0;
            m->comp_size = 0;
            m->codec = 0;
            if (s->live_compressed_pages) __sync_fetch_and_sub(&s->live_compressed_pages, 1);
        }
        pthread_mutex_unlock(&s->alloc_mutex);
    }
    mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
#if defined(MADV_FREE_REUSE)
    madvise(pa, PAGE_SZ, MADV_FREE_REUSE);
#endif
    memcpy(pa, raw, PAGE_SZ);
    m->dirty = 0;
    m->state = PAGE_RESIDENT;
    m->codec = 0;
    m->comp_size = 0;
    m->pool_offset = 0;
    int prot = page_wants_write_protect(m) ? PROT_READ : (PROT_READ | PROT_WRITE);
    mprotect(pa, PAGE_SZ, prot);
    res_list_add(s, (uint32_t)pidx);
    return 0;
}

int memx_runtime_context_export_archive(memx_runtime_context_t *ctx, void *ptr, const char *path, uint64_t *out_bytes) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !ptr || !path || !path[0]) return EINVAL;
    if (!g_z || !g_z->running || !is_ours(ptr)) return EINVAL;
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    if (sp >= g_z->npages || g_z->meta[sp].owner_tag != (uintptr_t)ctx || g_z->meta[sp].alloc_size == 0)
        return EINVAL;
    size_t nbytes = g_z->meta[sp].alloc_size;
    size_t npages = (nbytes + PAGE_SZ - 1) / PAGE_SZ;
    uint64_t dummy = 0;
    memx_runtime_context_force_compress_range(ctx, ptr, 0, nbytes, &dummy);

    memx_archive_page_ent_t *ents = (memx_archive_page_ent_t *)calloc(npages, sizeof(*ents));
    if (!ents) return ENOMEM;
    uint8_t **blobs = (uint8_t **)calloc(npages, sizeof(uint8_t *));
    if (!blobs) { free(ents); return ENOMEM; }

    int rc = 0;
    uint64_t payload = 0;
    for (size_t i = 0; i < npages; i++) {
        size_t pidx = sp + i;
        PageMeta *m = &g_z->meta[pidx];
        uint8_t st = m->state;
        if ((st == PAGE_COMPRESSED || st == PAGE_HOT) && m->comp_size > 0 && m->comp_size <= PAGE_SZ) {
            uint32_t csz = m->comp_size;
            uint64_t off = m->pool_offset;
            uint8_t *buf = (uint8_t *)malloc(csz);
            if (!buf) { rc = ENOMEM; break; }
            pthread_mutex_lock(&g_z->alloc_mutex);
            if (off + csz <= g_z->pool_size) {
                pool_prepare_write_range(g_z, off, csz);
                memcpy(buf, g_z->pool + off, csz);
                {
                    uint64_t start = off & ~((uint64_t)PAGE_SZ - 1);
                    uint64_t end = (off + csz + PAGE_SZ - 1) & ~((uint64_t)PAGE_SZ - 1);
                    if (end > g_z->pool_size) end = g_z->pool_size;
                    if (end > start) mprotect(g_z->pool + start, (size_t)(end - start), PROT_READ);
                }
            } else {
                free(buf);
                buf = NULL;
            }
            pthread_mutex_unlock(&g_z->alloc_mutex);
            if (!buf) {
                force_compress_page_now(g_z, pidx);
                m = &g_z->meta[pidx];
                if ((m->state == PAGE_COMPRESSED || m->state == PAGE_HOT) && m->comp_size > 0) {
                    csz = m->comp_size;
                    off = m->pool_offset;
                    buf = (uint8_t *)malloc(csz);
                    if (!buf) { rc = ENOMEM; break; }
                    pthread_mutex_lock(&g_z->alloc_mutex);
                    pool_prepare_write_range(g_z, off, csz);
                    memcpy(buf, g_z->pool + off, csz);
                    {
                        uint64_t start = off & ~((uint64_t)PAGE_SZ - 1);
                        uint64_t end = (off + csz + PAGE_SZ - 1) & ~((uint64_t)PAGE_SZ - 1);
                        if (end > g_z->pool_size) end = g_z->pool_size;
                        if (end > start) mprotect(g_z->pool + start, (size_t)(end - start), PROT_READ);
                    }
                    pthread_mutex_unlock(&g_z->alloc_mutex);
                }
            }
            if (buf) {
                ents[i].comp_size = csz;
                ents[i].codec = m->codec;
                ents[i].kind = 0;
                blobs[i] = buf;
                payload += csz;
                continue;
            }
        }
        uint8_t *raw = (uint8_t *)malloc(PAGE_SZ);
        if (!raw) { rc = ENOMEM; break; }
        if (m->state == PAGE_COMPRESSED || (m->state == PAGE_HOT && m->comp_size != 0)) {
            if (decompress_compressed_page(g_z, pidx, 0, 0) != 0) {
                free(raw);
                rc = EIO;
                break;
            }
        }
        uint8_t *pa = (uint8_t *)g_z->vmem + pidx * PAGE_SZ;
        mprotect(pa, PAGE_SZ, PROT_READ);
        memcpy(raw, pa, PAGE_SZ);
        ents[i].comp_size = PAGE_SZ;
        ents[i].codec = 0;
        ents[i].kind = 1;
        blobs[i] = raw;
        payload += PAGE_SZ;
    }

    if (rc == 0) {
        memx_archive_file_hdr_t hdr;
        memset(&hdr, 0, sizeof(hdr));
        hdr.magic = MEMX_ARCHIVE_MAGIC;
        hdr.version = MEMX_ARCHIVE_VERSION;
        hdr.page_size = (uint32_t)PAGE_SZ;
        hdr.flags = 0;
        hdr.nbytes = (uint64_t)nbytes;
        hdr.page_count = (uint64_t)npages;
        hdr.role = g_z->meta[sp].tensor_role;
        hdr.dtype = g_z->meta[sp].tensor_dtype;
        hdr.layout = g_z->meta[sp].tensor_layout;
        FILE *fp = fopen(path, "wb");
        if (!fp) rc = errno ? errno : EIO;
        else {
            if (archive_write_all(fp, &hdr, sizeof(hdr)) != 0) rc = EIO;
            if (rc == 0 && archive_write_all(fp, ents, npages * sizeof(*ents)) != 0) rc = EIO;
            for (size_t i = 0; rc == 0 && i < npages; i++) {
                if (archive_write_all(fp, blobs[i], ents[i].comp_size) != 0) rc = EIO;
            }
            if (fclose(fp) != 0 && rc == 0) rc = EIO;
            if (rc != 0) unlink(path);
        }
        if (rc == 0 && out_bytes) *out_bytes = (uint64_t)sizeof(hdr) + (uint64_t)npages * sizeof(*ents) + payload;
    }

    for (size_t i = 0; i < npages; i++) free(blobs[i]);
    free(blobs);
    free(ents);
    return rc;
}

int memx_runtime_context_import_archive(memx_runtime_context_t *ctx, const char *path, const memx_runtime_tensor_desc_t *desc_override, void **out_ptr, size_t *out_size) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !path || !path[0] || !out_ptr) return EINVAL;
    if (memx_runtime_init() != 0) return ENOMEM;
    FILE *fp = fopen(path, "rb");
    if (!fp) return errno ? errno : EIO;
    memx_archive_file_hdr_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    int rc = 0;
    if (archive_read_all(fp, &hdr, sizeof(hdr)) != 0) rc = EIO;
    if (rc == 0 && (hdr.magic != MEMX_ARCHIVE_MAGIC || hdr.version != MEMX_ARCHIVE_VERSION || hdr.page_size != PAGE_SZ))
        rc = EINVAL;
    if (rc == 0 && (hdr.nbytes == 0 || hdr.page_count == 0)) rc = EINVAL;
    size_t expect_pages = 0;
    if (rc == 0) {
        expect_pages = (size_t)((hdr.nbytes + PAGE_SZ - 1) / PAGE_SZ);
        if (hdr.page_count != (uint64_t)expect_pages) rc = EINVAL;
    }
    memx_archive_page_ent_t *ents = NULL;
    if (rc == 0) {
        ents = (memx_archive_page_ent_t *)calloc(expect_pages, sizeof(*ents));
        if (!ents) rc = ENOMEM;
        else if (archive_read_all(fp, ents, expect_pages * sizeof(*ents)) != 0) rc = EIO;
    }
    memx_runtime_tensor_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    if (desc_override && desc_override->struct_size >= sizeof(desc)) {
        desc = *desc_override;
        desc.struct_size = (uint32_t)sizeof(desc);
    } else {
        desc.struct_size = (uint32_t)sizeof(desc);
        desc.role = hdr.role ? hdr.role : MEMX_TENSOR_ROLE_WEIGHT;
        desc.dtype = hdr.dtype;
        desc.layout = hdr.layout ? hdr.layout : MEMX_TENSOR_LAYOUT_ROW_MAJOR;
        desc.flags = MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD;
        for (int i = 0; i < 4; i++) {
            desc.shape[i] = hdr.shape[i];
            desc.stride[i] = hdr.stride[i];
        }
    }
    void *ptr = NULL;
    if (rc == 0) {
        ptr = memx_runtime_context_malloc_tensor(ctx, (size_t)hdr.nbytes, &desc);
        if (!ptr) rc = ENOMEM;
    }
    if (rc == 0) {
        size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
        for (size_t i = 0; i < expect_pages; i++) {
            uint32_t csz = ents[i].comp_size;
            if (csz == 0 || csz > PAGE_SZ) { rc = EINVAL; break; }
            uint8_t *buf = (uint8_t *)malloc(csz);
            if (!buf) { rc = ENOMEM; break; }
            if (archive_read_all(fp, buf, csz) != 0) { free(buf); rc = EIO; break; }
            if (ents[i].kind == 1) {
                if (csz != PAGE_SZ) { free(buf); rc = EINVAL; break; }
                if (install_raw_page(g_z, sp + i, buf) != 0) { free(buf); rc = EIO; break; }
            } else {
                if (install_precompressed_page(g_z, sp + i, buf, csz, ents[i].codec) != 0) {
                    free(buf);
                    rc = EIO;
                    break;
                }
            }
            free(buf);
        }
        if (rc == 0) {
            memx_runtime_context_update_tensor_flags_range(
                ctx, ptr, 0, (size_t)hdr.nbytes,
                MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD);
        } else {
            memx_runtime_context_free(ctx, ptr);
            ptr = NULL;
        }
    }
    fclose(fp);
    free(ents);
    if (rc != 0) return rc;
    *out_ptr = ptr;
    if (out_size) *out_size = (size_t)hdr.nbytes;
    return 0;
}

static int ws_tile_collect_ranges(size_t rows, size_t cols, size_t elem, size_t col0, size_t coln, size_t nbytes,
                                  size_t *out_off, size_t *out_ln, int max_ranges, int *out_n) {
    if (!out_off || !out_ln || !out_n || max_ranges <= 0) return EINVAL;
    *out_n = 0;
    if (rows == 0 || cols == 0 || elem == 0 || coln == 0) return 0;
    if (col0 >= cols) return 0;
    if (col0 + coln > cols) coln = cols - col0;
    size_t row_bytes = cols * elem;
    size_t strip = coln * elem;
    size_t col_off = col0 * elem;
    size_t cur0 = (size_t)-1, cur1 = 0;
    for (size_t r = 0; r < rows; r++) {
        size_t b = r * row_bytes + col_off;
        size_t e = b + strip;
        if (b >= nbytes) break;
        if (e > nbytes) e = nbytes;
        size_t p0 = (b / PAGE_SZ) * PAGE_SZ;
        size_t p1 = ((e + PAGE_SZ - 1) / PAGE_SZ) * PAGE_SZ;
        if (p1 > nbytes) p1 = nbytes;
        if (p1 <= p0) continue;
        if (cur0 == (size_t)-1) {
            cur0 = p0; cur1 = p1;
        } else if (p0 <= cur1) {
            if (p1 > cur1) cur1 = p1;
        } else {
            if (*out_n >= max_ranges) return ENOMEM;
            out_off[*out_n] = cur0;
            out_ln[*out_n] = cur1 - cur0;
            (*out_n)++;
            cur0 = p0; cur1 = p1;
        }
    }
    if (cur0 != (size_t)-1) {
        if (*out_n >= max_ranges) return ENOMEM;
        out_off[*out_n] = cur0;
        out_ln[*out_n] = cur1 - cur0;
        (*out_n)++;
    }
    return 0;
}

int memx_runtime_context_ws_tile(memx_runtime_context_t *ctx, const memx_runtime_ws_tile_t *tile) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !tile || !tile->ptr) return EINVAL;
    if (tile->struct_size != 0 && tile->struct_size < sizeof(*tile)) return EINVAL;
    size_t sp = 0, nbytes = 0;
    if (ws_resolve_alloc(ctx, tile->ptr, &sp, &nbytes) != 0) return EINVAL;
    uint32_t flags = tile->flags ? tile->flags : (MEMX_WS_FLAG_HOT | MEMX_WS_FLAG_PREFETCH | MEMX_WS_FLAG_MARK_ACCESS);
    if (tile->retire_col_count > 0) {
        size_t roff[128], rln[128];
        int rn = 0;
        int rrc = ws_tile_collect_ranges(tile->rows, tile->cols, tile->elem_size,
                                         tile->retire_col_start, tile->retire_col_count, nbytes,
                                         roff, rln, 128, &rn);
        if (rrc == 0) {
            int sync = (flags & (MEMX_WS_FLAG_RETIRE_SYNC | MEMX_WS_FLAG_NO_ASYNC)) != 0;
            for (int i = 0; i < rn; i++) {
                if (flags & MEMX_WS_FLAG_RETIRE)
                    ws_retire_range(ctx, tile->ptr, roff[i], rln[i], sync);
                else
                    ws_cold_range(ctx, tile->ptr, roff[i], rln[i]);
            }
        }
    }
    size_t off[192], ln[192];
    int n = 0;
    int rc = ws_tile_collect_ranges(tile->rows, tile->cols, tile->elem_size,
                                    tile->col_start, tile->col_count, nbytes,
                                    off, ln, 192, &n);
    if (rc != 0) return rc;
    if (n == 0) return 0;
    size_t cover0 = off[0];
    size_t cover1 = off[n - 1] + ln[n - 1];
    size_t covered = 0;
    for (int i = 0; i < n; i++) covered += ln[i];
    double density = (cover1 > cover0) ? ((double)covered / (double)(cover1 - cover0)) : 1.0;
    if (density >= 0.55 || n <= 4) {
        size_t pref = 0;
        if ((flags & MEMX_WS_FLAG_PREFETCH) && tile->prefetch_cols > 0) {
            size_t poff[64], pln[64];
            int pn = 0;
            size_t pstart = tile->col_start + tile->col_count;
            if (ws_tile_collect_ranges(tile->rows, tile->cols, tile->elem_size,
                                       pstart, tile->prefetch_cols, nbytes,
                                       poff, pln, 64, &pn) == 0 && pn > 0) {
                pref = (poff[pn - 1] + pln[pn - 1]) > cover1 ? ((poff[pn - 1] + pln[pn - 1]) - cover1) : 0;
            }
        }
        return memx_runtime_context_ws_advance(ctx, tile->ptr, cover0, cover1 - cover0, pref, flags);
    }
    memx_runtime_ws_intent_t intents[192];
    memset(intents, 0, sizeof(intents));
    for (int i = 0; i < n; i++) {
        intents[i].struct_size = (uint32_t)sizeof(intents[i]);
        intents[i].flags = flags & ~MEMX_WS_FLAG_RETIRE & ~MEMX_WS_FLAG_RETIRE_SYNC;
        intents[i].ptr = tile->ptr;
        intents[i].offset = off[i];
        intents[i].length = ln[i];
        intents[i].prefetch_length = 0;
    }
    rc = memx_runtime_context_apply_ws(ctx, intents, (size_t)n);
    if (rc != 0) return rc;
    if ((flags & MEMX_WS_FLAG_PREFETCH) && tile->prefetch_cols > 0) {
        size_t poff[64], pln[64];
        int pn = 0;
        size_t pstart = tile->col_start + tile->col_count;
        if (ws_tile_collect_ranges(tile->rows, tile->cols, tile->elem_size,
                                   pstart, tile->prefetch_cols, nbytes,
                                   poff, pln, 64, &pn) == 0) {
            for (int i = 0; i < pn; i++)
                memx_runtime_context_prefetch_range(ctx, tile->ptr, poff[i], pln[i]);
        }
    }
    return 0;
}

static inline uint16_t memx_bf16_to_fp16_bits(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
#if defined(__aarch64__)
    __fp16 h = (__fp16)f;
    uint16_t o;
    memcpy(&o, &h, sizeof(o));
    return o;
#else
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t t = mant >> (1 - exp + 13);
        if ((mant >> (1 - exp + 12)) & 1u) t += 1u;
        return (uint16_t)(sign | t);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half += 1u;
    return (uint16_t)half;
#endif
}

static void memx_convert_bf16_to_fp16(const uint8_t *src, uint8_t *dst, size_t nbytes) {
    size_t n = nbytes >> 1;
    const uint16_t *s = (const uint16_t *)(const void *)src;
    uint16_t *d = (uint16_t *)(void *)dst;
    size_t i = 0;
#if defined(__aarch64__)
    for (; i + 16 <= n; i += 16) {
        d[i + 0] = memx_bf16_to_fp16_bits(s[i + 0]);
        d[i + 1] = memx_bf16_to_fp16_bits(s[i + 1]);
        d[i + 2] = memx_bf16_to_fp16_bits(s[i + 2]);
        d[i + 3] = memx_bf16_to_fp16_bits(s[i + 3]);
        d[i + 4] = memx_bf16_to_fp16_bits(s[i + 4]);
        d[i + 5] = memx_bf16_to_fp16_bits(s[i + 5]);
        d[i + 6] = memx_bf16_to_fp16_bits(s[i + 6]);
        d[i + 7] = memx_bf16_to_fp16_bits(s[i + 7]);
        d[i + 8] = memx_bf16_to_fp16_bits(s[i + 8]);
        d[i + 9] = memx_bf16_to_fp16_bits(s[i + 9]);
        d[i + 10] = memx_bf16_to_fp16_bits(s[i + 10]);
        d[i + 11] = memx_bf16_to_fp16_bits(s[i + 11]);
        d[i + 12] = memx_bf16_to_fp16_bits(s[i + 12]);
        d[i + 13] = memx_bf16_to_fp16_bits(s[i + 13]);
        d[i + 14] = memx_bf16_to_fp16_bits(s[i + 14]);
        d[i + 15] = memx_bf16_to_fp16_bits(s[i + 15]);
    }
#endif
    for (; i < n; i++) d[i] = memx_bf16_to_fp16_bits(s[i]);
}

#define MEMX_MAT_CACHE_N 256
#define MEMX_MAT_TLS_N 8
#define MEMX_MAT_ND_Q 1024
#define MEMX_MAT_ND_WORKERS 4

typedef struct {
    size_t pidx;
    uint32_t write_seq;
    volatile uint8_t valid;
    uint8_t page[PAGE_SZ];
} memx_mat_cache_ent_t;

static memx_mat_cache_ent_t *g_mat_cache = NULL;
static size_t g_mat_cache_map_bytes = 0;
static int g_mat_cache_heap = 0;
static pthread_mutex_t g_mat_slot_mu[MEMX_MAT_CACHE_N];
static pthread_once_t g_mat_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_mat_map_mu = PTHREAD_MUTEX_INITIALIZER;
static volatile int g_mat_nd_running = 0;
static uint32_t g_mat_nd_q[MEMX_MAT_ND_Q];
static volatile uint32_t g_mat_nd_head = 0;
static volatile uint32_t g_mat_nd_tail = 0;
static pthread_mutex_t g_mat_nd_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_mat_nd_cond = PTHREAD_COND_INITIALIZER;
static pthread_t g_mat_nd_threads[MEMX_MAT_ND_WORKERS];
static int g_mat_nd_nworkers = 0;

static __thread struct {
    size_t pidx[MEMX_MAT_TLS_N];
    uint32_t seq[MEMX_MAT_TLS_N];
    uint8_t valid[MEMX_MAT_TLS_N];
    uint8_t page[MEMX_MAT_TLS_N][PAGE_SZ];
    int clock;
} g_mat_tls;

static void mat_cache_map_alloc(void) {
    if (g_mat_cache) return;
    size_t bytes = (size_t)MEMX_MAT_CACHE_N * sizeof(memx_mat_cache_ent_t);
    void *m = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) {
        g_mat_cache = (memx_mat_cache_ent_t *)calloc(MEMX_MAT_CACHE_N, sizeof(memx_mat_cache_ent_t));
        g_mat_cache_map_bytes = 0;
        g_mat_cache_heap = 1;
    } else {
        g_mat_cache = (memx_mat_cache_ent_t *)m;
        g_mat_cache_map_bytes = bytes;
        g_mat_cache_heap = 0;
        memset(g_mat_cache, 0, bytes);
    }
    if (g_mat_cache) {
        for (int i = 0; i < MEMX_MAT_CACHE_N; i++) {
            g_mat_cache[i].valid = 0;
            g_mat_cache[i].pidx = (size_t)-1;
        }
    }
}

static void mat_cache_init_once(void) {
    for (int i = 0; i < MEMX_MAT_CACHE_N; i++) {
        pthread_mutex_init(&g_mat_slot_mu[i], NULL);
    }
    mat_cache_map_alloc();
}

static void mat_cache_ensure(void) {
    pthread_once(&g_mat_once, mat_cache_init_once);
    if (!g_mat_cache) {
        pthread_mutex_lock(&g_mat_map_mu);
        if (!g_mat_cache) mat_cache_map_alloc();
        pthread_mutex_unlock(&g_mat_map_mu);
    }
}

static const uint8_t *mat_tls_get_ptr(size_t pidx, uint32_t write_seq) {
    for (int i = 0; i < MEMX_MAT_TLS_N; i++) {
        if (g_mat_tls.valid[i] && g_mat_tls.pidx[i] == pidx && g_mat_tls.seq[i] == write_seq)
            return g_mat_tls.page[i];
    }
    return NULL;
}

static int mat_tls_get(size_t pidx, uint32_t write_seq, uint8_t *out_page) {
    const uint8_t *p = mat_tls_get_ptr(pidx, write_seq);
    if (!p) return 0;
    memcpy(out_page, p, PAGE_SZ);
    return 1;
}

static void mat_tls_put(size_t pidx, uint32_t write_seq, const uint8_t *page) {
    int slot = -1;
    for (int i = 0; i < MEMX_MAT_TLS_N; i++) {
        if (g_mat_tls.valid[i] && g_mat_tls.pidx[i] == pidx) { slot = i; break; }
    }
    if (slot < 0) {
        slot = g_mat_tls.clock % MEMX_MAT_TLS_N;
        g_mat_tls.clock++;
    }
    g_mat_tls.pidx[slot] = pidx;
    g_mat_tls.seq[slot] = write_seq;
    memcpy(g_mat_tls.page[slot], page, PAGE_SZ);
    g_mat_tls.valid[slot] = 1;
}

static int mat_cache_get(size_t pidx, uint32_t write_seq, uint8_t *out_page) {
    if (mat_tls_get(pidx, write_seq, out_page)) return 1;
    mat_cache_ensure();
    if (!g_mat_cache) return 0;
    uint32_t slot = (uint32_t)(pidx % MEMX_MAT_CACHE_N);
    pthread_mutex_lock(&g_mat_slot_mu[slot]);
    memx_mat_cache_ent_t *e = &g_mat_cache[slot];
    if (e->valid && e->pidx == pidx && e->write_seq == write_seq) {
        memcpy(out_page, e->page, PAGE_SZ);
        pthread_mutex_unlock(&g_mat_slot_mu[slot]);
        mat_tls_put(pidx, write_seq, out_page);
        return 1;
    }
    pthread_mutex_unlock(&g_mat_slot_mu[slot]);
    return 0;
}

static const uint8_t *mat_cache_get_ptr(size_t pidx, uint32_t write_seq) {
    const uint8_t *hit = mat_tls_get_ptr(pidx, write_seq);
    if (hit) return hit;
    mat_cache_ensure();
    if (!g_mat_cache) return NULL;
    uint32_t slot = (uint32_t)(pidx % MEMX_MAT_CACHE_N);
    pthread_mutex_lock(&g_mat_slot_mu[slot]);
    memx_mat_cache_ent_t *e = &g_mat_cache[slot];
    if (e->valid && e->pidx == pidx && e->write_seq == write_seq) {
        uint8_t tmp[PAGE_SZ];
        memcpy(tmp, e->page, PAGE_SZ);
        uint32_t seq = e->write_seq;
        pthread_mutex_unlock(&g_mat_slot_mu[slot]);
        mat_tls_put(pidx, seq, tmp);
        return mat_tls_get_ptr(pidx, seq);
    }
    pthread_mutex_unlock(&g_mat_slot_mu[slot]);
    return NULL;
}

static void mat_cache_put(size_t pidx, uint32_t write_seq, const uint8_t *page) {
    mat_tls_put(pidx, write_seq, page);
    mat_cache_ensure();
    if (!g_mat_cache) return;
    uint32_t slot = (uint32_t)(pidx % MEMX_MAT_CACHE_N);
    pthread_mutex_lock(&g_mat_slot_mu[slot]);
    memx_mat_cache_ent_t *e = &g_mat_cache[slot];
    e->pidx = pidx;
    e->write_seq = write_seq;
    memcpy(e->page, page, PAGE_SZ);
    __sync_synchronize();
    e->valid = 1;
    pthread_mutex_unlock(&g_mat_slot_mu[slot]);
}

static void mat_cache_invalidate(void) {
    mat_cache_ensure();
    if (g_mat_cache) {
        for (int i = 0; i < MEMX_MAT_CACHE_N; i++) {
            pthread_mutex_lock(&g_mat_slot_mu[i]);
            g_mat_cache[i].valid = 0;
            g_mat_cache[i].pidx = (size_t)-1;
            pthread_mutex_unlock(&g_mat_slot_mu[i]);
        }
    }
    for (int i = 0; i < MEMX_MAT_TLS_N; i++) {
        g_mat_tls.valid[i] = 0;
        g_mat_tls.pidx[i] = (size_t)-1;
    }
}
static void mat_cache_release_physical(void) {
    tca_pool_destroy();
    for (int i = 0; i < MEMX_MAT_TLS_N; i++) {
        if (g_mat_tls.valid[i]) {
#if defined(MADV_FREE_REUSABLE)
            madvise(g_mat_tls.page[i], PAGE_SZ, MADV_FREE_REUSABLE);
#endif
            madvise(g_mat_tls.page[i], PAGE_SZ, MADV_DONTNEED);
        }
        g_mat_tls.valid[i] = 0;
        g_mat_tls.pidx[i] = (size_t)-1;
    }
    pthread_mutex_lock(&g_mat_map_mu);
    if (g_mat_cache && g_mat_cache_map_bytes && !g_mat_cache_heap) {
        for (int i = 0; i < MEMX_MAT_CACHE_N; i++) {
            pthread_mutex_lock(&g_mat_slot_mu[i]);
            g_mat_cache[i].valid = 0;
            g_mat_cache[i].pidx = (size_t)-1;
            pthread_mutex_unlock(&g_mat_slot_mu[i]);
        }
        munmap(g_mat_cache, g_mat_cache_map_bytes);
        g_mat_cache = NULL;
        g_mat_cache_map_bytes = 0;
    } else if (g_mat_cache) {
        for (int i = 0; i < MEMX_MAT_CACHE_N; i++) {
            pthread_mutex_lock(&g_mat_slot_mu[i]);
            g_mat_cache[i].valid = 0;
            g_mat_cache[i].pidx = (size_t)-1;
#if defined(MADV_FREE_REUSABLE)
            madvise(g_mat_cache[i].page, PAGE_SZ, MADV_FREE_REUSABLE);
#endif
            madvise(g_mat_cache[i].page, PAGE_SZ, MADV_DONTNEED);
            (void)madvise(g_mat_cache[i].page, PAGE_SZ, MADV_PAGEOUT);
            pthread_mutex_unlock(&g_mat_slot_mu[i]);
        }
        if (g_mat_cache_heap) {
            free(g_mat_cache);
            g_mat_cache = NULL;
            g_mat_cache_heap = 0;
        }
    }
    pthread_mutex_unlock(&g_mat_map_mu);
}


static int materialize_page_bytes(MemXZone3 *s, size_t pidx, uint8_t *out_page, uint32_t flags);

static void *mat_nd_worker(void *arg) {
    (void)arg;
    in_memx = 1;
    uint8_t pagebuf[PAGE_SZ];
    while (g_mat_nd_running) {
        uint32_t pidx = UINT32_MAX;
        pthread_mutex_lock(&g_mat_nd_mu);
        while (g_mat_nd_running && g_mat_nd_tail == g_mat_nd_head)
            pthread_cond_wait(&g_mat_nd_cond, &g_mat_nd_mu);
        if (g_mat_nd_tail != g_mat_nd_head) {
            pidx = g_mat_nd_q[g_mat_nd_tail];
            g_mat_nd_tail = (g_mat_nd_tail + 1) % MEMX_MAT_ND_Q;
        }
        pthread_mutex_unlock(&g_mat_nd_mu);
        if (pidx == UINT32_MAX) continue;
        if (!g_z || !g_z->running || pidx >= g_z->npages) continue;
        (void)materialize_page_bytes(g_z, (size_t)pidx, pagebuf,
                                     MEMX_MATERIALIZE_KEEP_COMPRESSED | MEMX_MATERIALIZE_ALLOW_RESIDENT);
    }
    return NULL;
}

static void mat_nd_shutdown(void) {
    if (!g_mat_nd_running) return;
    g_mat_nd_running = 0;
    pthread_mutex_lock(&g_mat_nd_mu);
    pthread_cond_broadcast(&g_mat_nd_cond);
    pthread_mutex_unlock(&g_mat_nd_mu);
    for (int i = 0; i < g_mat_nd_nworkers; i++) {
        (void)pthread_join(g_mat_nd_threads[i], NULL);
    }
    g_mat_nd_nworkers = 0;
    g_mat_nd_head = 0;
    g_mat_nd_tail = 0;
}

static void mat_nd_ensure_workers(void) {
    if (g_mat_nd_running) return;
    mat_cache_ensure();
    g_mat_nd_running = 1;
    g_mat_nd_nworkers = 0;
    for (int i = 0; i < MEMX_MAT_ND_WORKERS; i++) {
        if (memx_pthread_spawn(&g_mat_nd_threads[i], mat_nd_worker, NULL) == 0)
            g_mat_nd_nworkers++;
    }
    if (g_mat_nd_nworkers == 0) g_mat_nd_running = 0;
}

static int mat_nd_enqueue_page(uint32_t pidx) {
    if (!g_z || pidx >= g_z->npages) return 0;
    mat_nd_ensure_workers();
    if (!g_mat_nd_running) return 0;
    pthread_mutex_lock(&g_mat_nd_mu);
    uint32_t next = (g_mat_nd_head + 1) % MEMX_MAT_ND_Q;
    if (next == g_mat_nd_tail) {
        pthread_mutex_unlock(&g_mat_nd_mu);
        return 0;
    }
    g_mat_nd_q[g_mat_nd_head] = pidx;
    g_mat_nd_head = next;
    pthread_cond_signal(&g_mat_nd_cond);
    pthread_mutex_unlock(&g_mat_nd_mu);
    return 1;
}

static int pool_copy_blob_locked(MemXZone3 *s, uint64_t d_off, uint32_t d_sz, uint8_t *payload) {
    if (!s || !payload || d_sz == 0 || d_sz > PAGE_SZ) return -1;
    if (d_off + (uint64_t)d_sz > s->pool_size) return -1;
    int have_spill = (s->pool_spill_fd > 2 && s->pool_spill_bytes >= d_off + (uint64_t)d_sz);
    if (have_spill && (pool_is_vault_native(s) || s->pool_detached || s->pool_ghost)) {
        if (pool_is_vault_native(s) && pool_vault_cache_get_locked(s, d_off, d_sz, payload)) {
            return 0;
        }
        if (pool_is_vault_native(s)) {
            (void)pool_vault_wbuf_flush_locked(s);
            const uint8_t *wp = pool_vault_window_ptr_locked(s, d_off, d_sz);
            if (wp) {
                memcpy(payload, wp, d_sz);
                s->pool_vault_reads++;
                pool_vault_cache_put_locked(s, d_off, d_sz, payload);
                return 0;
            }
        }
        ssize_t r = pread(s->pool_spill_fd, payload, d_sz, (off_t)d_off);
        if (r == (ssize_t)d_sz) {
            if (pool_is_vault_native(s)) {
                s->pool_vault_reads++;
                pool_vault_cache_put_locked(s, d_off, d_sz, payload);
            }
            return 0;
        }
        if (pool_is_vault_native(s) || s->pool_detached) return -1;
    }
    if (s->pool && !s->pool_detached) {
        uint64_t p0 = d_off & ~((uint64_t)PAGE_SZ - 1);
        uint64_t p1 = (d_off + d_sz + PAGE_SZ - 1) & ~((uint64_t)PAGE_SZ - 1);
        if (p1 > s->pool_size) p1 = s->pool_size;
        int ok = 1;
        if (p1 > p0) {
            size_t bytes = (size_t)(p1 - p0);
            if (mprotect(s->pool + p0, bytes, PROT_READ) != 0) {
                if (mprotect(s->pool + p0, bytes, PROT_READ | PROT_WRITE) != 0) ok = 0;
            }
        }
        if (ok) {
            memcpy(payload, s->pool + d_off, d_sz);
            return 0;
        }
    }
    if (have_spill) {
        ssize_t r = pread(s->pool_spill_fd, payload, d_sz, (off_t)d_off);
        if (r == (ssize_t)d_sz) return 0;
    }
    return -1;
}

static int mat_cache_get(size_t pidx, uint32_t write_seq, uint8_t *out_page);
static void mat_cache_put(size_t pidx, uint32_t write_seq, const uint8_t *page);

static int materialize_from_sov(MemXZone3 *s, size_t pidx, uint8_t *out_page);

static const uint8_t *materialize_from_sov_ptr(MemXZone3 *s, size_t pidx) {
    if (!s || !s->sovereign || !s->sov_ents || pidx >= s->npages) return NULL;
    sov_ent_t *e = sov_find_locked(s, (uint32_t)pidx);
    if (!e || e->csz == 0 || e->csz > PAGE_SZ) return NULL;
    const uint8_t *hit = mat_cache_get_ptr(pidx, e->seq);
    if (hit) {
        __sync_fetch_and_add(&s->sov_hits, 1);
        return hit;
    }
    uint8_t page[PAGE_SZ];
    if (materialize_from_sov(s, pidx, page) != 0) return NULL;
    return mat_cache_get_ptr(pidx, e->seq);
}

static int materialize_from_sov(MemXZone3 *s, size_t pidx, uint8_t *out_page) {
    if (!s || !out_page || !s->sovereign || !s->sov_ents) return -1;
    if (pidx >= s->npages) return -1;
    sov_ent_t *e = sov_find_locked(s, (uint32_t)pidx);
    if (!e || e->csz == 0 || e->csz > PAGE_SZ) return -1;
    if (mat_cache_get(pidx, e->seq, out_page)) {
        __sync_fetch_and_add(&s->sov_hits, 1);
        return 0;
    }
    uint8_t payload[PAGE_SZ];
    int rc = -1;
    int from_cache = 0;
    if (pool_is_vault_native(s)) {
        pthread_mutex_lock(&s->alloc_mutex);
        if (pool_vault_cache_get_locked(s, e->off, e->csz, payload)) {
            rc = 0;
            from_cache = 1;
        }
        pthread_mutex_unlock(&s->alloc_mutex);
    }
    if (rc != 0 && s->sovereign_frozen && s->pool_spill_fd > 2 &&
        s->pool_spill_bytes >= e->off + (uint64_t)e->csz) {
        ssize_t r = pread(s->pool_spill_fd, payload, e->csz, (off_t)e->off);
        if (r == (ssize_t)e->csz) rc = 0;
    }
    if (rc != 0) {
        pthread_mutex_lock(&s->alloc_mutex);
        if (pool_copy_blob_locked(s, e->off, e->csz, payload) != 0) {
            pthread_mutex_unlock(&s->alloc_mutex);
            return -1;
        }
        pthread_mutex_unlock(&s->alloc_mutex);
        rc = 0;
    } else if (!from_cache && pool_is_vault_native(s)) {
        pthread_mutex_lock(&s->alloc_mutex);
        pool_vault_cache_put_locked(s, e->off, e->csz, payload);
        if (!g_sov_batch_mode) sov_stream_inject_neighbors_locked(s, e);
        pthread_mutex_unlock(&s->alloc_mutex);
    }
    cpu_decompress(payload, e->csz, out_page);
    mat_cache_put(pidx, e->seq, out_page);
    __sync_fetch_and_add(&s->sov_hits, 1);
    __sync_fetch_and_add(&s->prefetch_hits, 1);
    return 0;
}

static int materialize_from_pool_blob(MemXZone3 *s, size_t pidx, uint8_t *out_page) {
    if (s && s->sovereign && s->sov_ents) {
        if (materialize_from_sov(s, pidx, out_page) == 0) return 0;
    }
    PageMeta *m = &s->meta[pidx];
    uint8_t payload[PAGE_SZ];
    uint32_t d_sz = 0;
    uint64_t d_off = 0;
    uint32_t seq0 = 0;
    pthread_mutex_lock(&s->alloc_mutex);
    if (m->state != PAGE_COMPRESSED) {
        pthread_mutex_unlock(&s->alloc_mutex);
        return -1;
    }
    d_sz = m->comp_size;
    d_off = m->pool_offset;
    seq0 = m->write_seq;
    if (d_sz == 0 || d_sz > PAGE_SZ || d_off + (uint64_t)d_sz > s->pool_size) {
        pthread_mutex_unlock(&s->alloc_mutex);
        return -1;
    }
    if (pool_copy_blob_locked(s, d_off, d_sz, payload) != 0) {
        pthread_mutex_unlock(&s->alloc_mutex);
        return -1;
    }
    pthread_mutex_unlock(&s->alloc_mutex);
    cpu_decompress(payload, d_sz, out_page);
    mat_cache_put(pidx, seq0, out_page);
    __sync_fetch_and_add(&s->prefetch_hits, 1);
    return 0;
}

static int materialize_resident_bytes(MemXZone3 *s, size_t pidx, uint8_t *out_page, uint32_t flags) {
    PageMeta *m = &s->meta[pidx];
    uint8_t *pa = (uint8_t *)s->vmem + pidx * PAGE_SZ;
    if (mprotect(pa, PAGE_SZ, PROT_READ) == 0 || mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE) == 0) {
        memcpy(out_page, pa, PAGE_SZ);
        if (page_wants_write_protect(m)) mprotect(pa, PAGE_SZ, PROT_READ);
        mat_cache_put(pidx, m->write_seq, out_page);
        return 0;
    }
    if (m->comp_size > 0 && m->comp_size <= PAGE_SZ) {
        if (decompress_compressed_page(s, pidx, 0, 2)) {
            if (mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE) != 0 &&
                mprotect(pa, PAGE_SZ, PROT_READ) != 0) return -1;
            memcpy(out_page, pa, PAGE_SZ);
            if (page_wants_write_protect(m)) mprotect(pa, PAGE_SZ, PROT_READ);
            mat_cache_put(pidx, m->write_seq, out_page);
            return 0;
        }
    }
    if ((flags & MEMX_MATERIALIZE_ALLOW_RESIDENT) == 0) return -1;
    return -1;
}

static int materialize_page_bytes(MemXZone3 *s, size_t pidx, uint8_t *out_page, uint32_t flags) {
    if (!s || !out_page || pidx >= s->npages) return -1;
    int keep = (flags & MEMX_MATERIALIZE_KEEP_COMPRESSED) != 0;
    if (keep && s->sovereign && s->sov_ents) {
        if (materialize_from_sov(s, pidx, out_page) == 0) return 0;
    }
    PageMeta *m = &s->meta[pidx];
    uint32_t seq_hint = m->write_seq;
    if (mat_cache_get(pidx, seq_hint, out_page)) return 0;
    for (int attempt = 0; attempt < 256; attempt++) {
        uint8_t st = m->state;
        seq_hint = m->write_seq;
        if (st == PAGE_HOT && m->comp_size != 0) {
            wait_decompress_complete(m);
            if (attempt >= 64 && m->state == PAGE_HOT && m->comp_size != 0) {
                if (decompress_compressed_page(s, pidx, 0, 2)) continue;
            }
            continue;
        }
        if (st == PAGE_COMPRESSING) {
            if (attempt >= 16) restore_compressing_page(s, pidx);
#if defined(__aarch64__)
            __asm__ __volatile__("yield");
#endif
            continue;
        }
        if (st == PAGE_RESIDENT || (st == PAGE_HOT && m->comp_size == 0)) {
            if (materialize_resident_bytes(s, pidx, out_page, flags) == 0) return 0;
            if (attempt + 1 >= 256) return -1;
#if defined(__aarch64__)
            __asm__ __volatile__("yield");
#endif
            continue;
        }
        if (st == PAGE_COMPRESSED) {
            if (keep) {
                if (materialize_from_pool_blob(s, pidx, out_page) == 0) return 0;
                if (attempt >= 32) {
                    if (decompress_compressed_page(s, pidx, 0, 2)) {
                        if (materialize_resident_bytes(s, pidx, out_page, flags) == 0) return 0;
                    }
                }
                continue;
            }
            if (decompress_compressed_page(s, pidx, 0, 2)) {
                if (materialize_resident_bytes(s, pidx, out_page, flags) == 0) return 0;
            }
            continue;
        }
        if (st == PAGE_NONE) {
            memset(out_page, 0, PAGE_SZ);
            return 0;
        }
#if defined(__aarch64__)
        __asm__ __volatile__("yield");
#endif
    }
    return -1;
}

static int materialize_page_ref(MemXZone3 *s, size_t pidx, const uint8_t **out_page, uint32_t flags) {
    if (!s || !out_page || pidx >= s->npages) return -1;
    PageMeta *m = &s->meta[pidx];
    uint32_t seq_hint = m->write_seq;
    const uint8_t *hit = mat_cache_get_ptr(pidx, seq_hint);
    if (hit) {
        *out_page = hit;
        return 0;
    }
    uint8_t pagebuf[PAGE_SZ];
    if (materialize_page_bytes(s, pidx, pagebuf, flags) != 0) return -1;
    mat_tls_put(pidx, m->write_seq, pagebuf);
    hit = mat_tls_get_ptr(pidx, m->write_seq);
    if (!hit) {
        for (int i = 0; i < MEMX_MAT_TLS_N; i++) {
            if (g_mat_tls.valid[i] && g_mat_tls.pidx[i] == pidx) {
                hit = g_mat_tls.page[i];
                break;
            }
        }
    }
    if (!hit) {
        mat_tls_put(pidx, m->write_seq, pagebuf);
        hit = mat_tls_get_ptr(pidx, m->write_seq);
    }
    if (!hit) return -1;
    *out_page = hit;
    return 0;
}

static void materialize_emit(const uint8_t *src, uint8_t *dst, size_t n, uint32_t flags) {
    if ((flags & MEMX_MATERIALIZE_BF16_TO_FP16) != 0 && (n & 1u) == 0) {
        memx_convert_bf16_to_fp16(src, dst, n);
    } else {
        memcpy(dst, src, n);
    }
}

int memx_runtime_context_materialize_range(memx_runtime_context_t *ctx, const void *ptr, size_t offset, size_t length, void *dst, size_t dst_cap, uint32_t flags) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !ptr || !dst) return EINVAL;
    if (length == 0) return 0;
    if (dst_cap < length) return EMSGSIZE;
    if (!g_z || !g_z->running || !is_ours((void *)ptr)) return EINVAL;
    size_t sp = 0, nbytes = 0;
    if (ws_resolve_alloc(ctx, (void *)ptr, &sp, &nbytes) != 0) return EINVAL;
    if (offset > nbytes) return EINVAL;
    if (offset + length > nbytes) length = nbytes - offset;
    if (length == 0) return 0;
    if (flags == 0) flags = MEMX_MATERIALIZE_KEEP_COMPRESSED | MEMX_MATERIALIZE_ALLOW_RESIDENT;

    uint8_t *out = (uint8_t *)dst;
    size_t end = offset + length;
    size_t p0 = offset / PAGE_SZ;
    size_t p1 = (end + PAGE_SZ - 1) / PAGE_SZ;
    int sov_fast = (g_z->sovereign && g_z->sov_ents && pool_is_vault_native(g_z));
    if (sov_fast && p1 > p0) {
        enum { RMAX = 2048 };
        size_t n = p1 - p0;
        if (n > RMAX) n = RMAX;
        size_t plist[RMAX];
        for (size_t i = 0; i < n; i++) plist[i] = sp + p0 + i;
        g_sov_batch_mode = 1;
        pthread_mutex_lock(&g_z->alloc_mutex);
        (void)sov_capsule_readv_warp_locked(g_z, plist, (int)n);
        pthread_mutex_unlock(&g_z->alloc_mutex);
        (void)sov_crw_direct_decomp_batch(g_z, plist, (int)n);
        g_sov_batch_mode = 0;
    }

    size_t cur = offset;
    uint8_t pagebuf[PAGE_SZ];
    while (cur < end) {
        size_t page_off = cur / PAGE_SZ;
        size_t pidx = sp + page_off;
        if (pidx >= g_z->npages) return EFAULT;
        size_t inside = cur - page_off * PAGE_SZ;
        size_t take = PAGE_SZ - inside;
        if (take > end - cur) take = end - cur;
        const uint8_t *srcp = NULL;
        if (sov_fast) srcp = materialize_from_sov_ptr(g_z, pidx);
        if (srcp) {
            materialize_emit(srcp + inside, out + (cur - offset), take, flags);
        } else {
            if (materialize_page_bytes(g_z, pidx, pagebuf, flags) != 0) return EIO;
            materialize_emit(pagebuf + inside, out + (cur - offset), take, flags);
        }
        cur += take;
    }
    return 0;
}

int memx_runtime_context_materialize_prefetch_range(memx_runtime_context_t *ctx, const void *ptr, size_t offset, size_t length, uint32_t flags) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !ptr) return EINVAL;
    if (length == 0) return 0;
    if (!g_z || !g_z->running || !is_ours((void *)ptr)) return EINVAL;
    size_t sp = 0, nbytes = 0;
    if (ws_resolve_alloc(ctx, (void *)ptr, &sp, &nbytes) != 0) return EINVAL;
    if (offset > nbytes) return 0;
    if (offset + length > nbytes) length = nbytes - offset;
    (void)flags;
    size_t end = offset + length;
    size_t cur = (offset / PAGE_SZ) * PAGE_SZ;
    int async_ok = 0;
    while (cur < end) {
        size_t page_off = cur / PAGE_SZ;
        size_t pidx = sp + page_off;
        if (pidx >= g_z->npages) break;
        if (mat_nd_enqueue_page((uint32_t)pidx)) async_ok = 1;
        else {
            uint8_t pagebuf[PAGE_SZ];
            (void)materialize_page_bytes(g_z, pidx, pagebuf,
                                         MEMX_MATERIALIZE_KEEP_COMPRESSED | MEMX_MATERIALIZE_ALLOW_RESIDENT);
        }
        cur += PAGE_SZ;
    }
    (void)async_ok;
    return 0;
}

static int sov_pidx_off_cmp(const void *a, const void *b) {
    size_t pa = *(const size_t *)a;
    size_t pb = *(const size_t *)b;
    uint64_t oa = 0, ob = 0;
    if (g_z && g_z->sovereign && g_z->sov_ents) {
        sov_ent_t *ea = sov_find_locked(g_z, (uint32_t)pa);
        sov_ent_t *eb = sov_find_locked(g_z, (uint32_t)pb);
        if (ea) oa = ea->off;
        else if (pa < g_z->npages) oa = g_z->meta[pa].pool_offset;
        if (eb) ob = eb->off;
        else if (pb < g_z->npages) ob = g_z->meta[pb].pool_offset;
    } else if (g_z) {
        if (pa < g_z->npages) oa = g_z->meta[pa].pool_offset;
        if (pb < g_z->npages) ob = g_z->meta[pb].pool_offset;
    }
    if (oa < ob) return -1;
    if (oa > ob) return 1;
    if (pa < pb) return -1;
    if (pa > pb) return 1;
    return 0;
}

int memx_runtime_context_materialize_tile(memx_runtime_context_t *ctx, const memx_runtime_ws_tile_t *tile, void *dst, size_t dst_cap, size_t dst_row_stride, uint32_t flags) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !tile || !tile->ptr || !dst) return EINVAL;
    if (tile->struct_size != 0 && tile->struct_size < sizeof(*tile)) return EINVAL;
    if (tile->rows == 0 || tile->col_count == 0 || tile->elem_size == 0 || tile->cols == 0) return 0;
    size_t sp = 0, nbytes = 0;
    if (ws_resolve_alloc(ctx, tile->ptr, &sp, &nbytes) != 0) return EINVAL;
    if (tile->col_start >= tile->cols) return EINVAL;
    size_t col_count = tile->col_count;
    if (tile->col_start + col_count > tile->cols) col_count = tile->cols - tile->col_start;
    size_t dense_row = col_count * tile->elem_size;
    if (dst_row_stride == 0) dst_row_stride = dense_row;
    if (dst_row_stride < dense_row) return EINVAL;
    size_t need = (tile->rows - 1) * dst_row_stride + dense_row;
    if (dst_cap < need) return EMSGSIZE;
    if (flags == 0) flags = MEMX_MATERIALIZE_KEEP_COMPRESSED | MEMX_MATERIALIZE_ALLOW_RESIDENT;

    size_t row_bytes = tile->cols * tile->elem_size;
    size_t col_off = tile->col_start * tile->elem_size;
    size_t strip = dense_row;
    uint8_t *out = (uint8_t *)dst;

    enum { UNIQ_MAX = 2048, UH_N = 4096 };
    size_t uniq[UNIQ_MAX];
    size_t *uh = (size_t *)malloc(sizeof(size_t) * UH_N);
    int nuniq = 0;
    uint8_t *tca_arena = NULL;
    size_t tca_abytes = 0;
    int tca_ok = 0;
    if (!uh) {
        for (size_t r = 0; r < tile->rows; r++) {
            size_t b = r * row_bytes + col_off;
            size_t e = b + strip;
            if (b >= nbytes) break;
            if (e > nbytes) e = nbytes;
            size_t cur = b;
            size_t dst_base = r * dst_row_stride;
            uint8_t pagebuf2[PAGE_SZ];
            while (cur < e) {
                size_t page_off = cur / PAGE_SZ;
                size_t pidx = sp + page_off;
                if (pidx >= g_z->npages) return EFAULT;
                if (materialize_page_bytes(g_z, pidx, pagebuf2, flags) != 0) return EIO;
                size_t inside = cur - page_off * PAGE_SZ;
                size_t take = PAGE_SZ - inside;
                if (take > e - cur) take = e - cur;
                materialize_emit(pagebuf2 + inside, out + dst_base + (cur - b), take, flags);
                cur += take;
            }
        }
        return 0;
    }
    for (int i = 0; i < UH_N; i++) uh[i] = (size_t)-1;
    for (size_t r = 0; r < tile->rows && nuniq < UNIQ_MAX; r++) {
        size_t b = r * row_bytes + col_off;
        size_t e = b + strip;
        if (b >= nbytes) break;
        if (e > nbytes) e = nbytes;
        size_t p0 = b / PAGE_SZ;
        size_t p1 = (e + PAGE_SZ - 1) / PAGE_SZ;
        for (size_t p = p0; p < p1 && nuniq < UNIQ_MAX; p++) {
            size_t pidx = sp + p;
            size_t h = (pidx * 11400714819323198485ull) & (UH_N - 1);
            int placed = 0;
            for (int probe = 0; probe < 64; probe++) {
                size_t si = (h + (size_t)probe) & (UH_N - 1);
                if (uh[si] == (size_t)-1) {
                    uh[si] = pidx;
                    uniq[nuniq++] = pidx;
                    placed = 1;
                    break;
                }
                if (uh[si] == pidx) {
                    placed = 1;
                    break;
                }
            }
            if (!placed && nuniq < UNIQ_MAX) {
                int found = 0;
                for (int i = 0; i < nuniq; i++) if (uniq[i] == pidx) { found = 1; break; }
                if (!found) uniq[nuniq++] = pidx;
            }
        }
    }

    if (nuniq >= 1 && g_z) {
        if (nuniq >= 2) {
            qsort(uniq, (size_t)nuniq, sizeof(size_t), sov_pidx_off_cmp);
        }
        int sov_fast = (g_z->sovereign && g_z->sov_ents && pool_is_vault_native(g_z));
        if (sov_fast && sov_tca_enabled() && nuniq >= 1) {
            g_sov_batch_mode = 1;
            int sticky_owned = 0;
            size_t hit_bytes = 0;
            uint8_t *hit = tca_sticky_try_hit(uniq, nuniq, &hit_bytes);
            if (hit) {
                tca_arena = hit;
                tca_abytes = hit_bytes;
                tca_ok = 1;
                sticky_owned = 1;
                __sync_fetch_and_add(&g_tca_sticky_hits, 1);
            } else {
                if (!sov_tca_novault()) {
                    pthread_mutex_lock(&g_z->alloc_mutex);
                    (void)sov_capsule_readv_warp_locked(g_z, uniq, nuniq);
                    pthread_mutex_unlock(&g_z->alloc_mutex);
                }
                int phits = 0;
                tca_sticky_slot_t *partial = NULL;
                if (tca_sticky_diff_enabled()) {
                    partial = tca_sticky_best_partial(uniq, nuniq, &phits);
                    if (partial && (phits * 2 < nuniq || phits < 2)) partial = NULL;
                }
                size_t tca_cap = 0;
                uint8_t *stick = tca_sticky_prepare(uniq, nuniq, &tca_cap, partial);
                uint8_t seed_stack[2048];
                uint8_t *seed = NULL;
                int seed_n = 0;
                if (stick) {
                    tca_arena = stick;
                    tca_abytes = tca_cap ? tca_cap : ((size_t)nuniq * (size_t)PAGE_SZ);
                    sticky_owned = 1;
                    if (partial && phits > 0 && nuniq <= 2048) {
                        memset(seed_stack, 0, (size_t)nuniq);
                        seed_n = tca_sticky_seed_from(partial, uniq, nuniq, stick, seed_stack);
                        if (seed_n > 0) {
                            seed = seed_stack;
                            __sync_fetch_and_add(&g_tca_sticky_partial, 1);
                            __sync_fetch_and_add(&g_tca_sticky_diff_pages, (uint64_t)seed_n);
                        }
                    }
                } else {
                    tca_abytes = (size_t)nuniq * (size_t)PAGE_SZ;
                    tca_arena = tca_arena_acquire(tca_abytes, &tca_cap);
                    if (tca_arena && tca_cap > tca_abytes) tca_abytes = tca_cap;
                }
                if (tca_arena) {
                    int filled = sov_tca_fill_arena_ex(g_z, uniq, nuniq, tca_arena, seed);
                    if (filled >= nuniq) {
                        tca_ok = 1;
                        if (sticky_owned) tca_sticky_commit();
                    } else {
                        if (!sticky_owned) tca_arena_release(tca_arena, tca_abytes, 0);
                        else tca_sticky_invalidate_active();
                        tca_arena = NULL;
                        tca_abytes = 0;
                        sticky_owned = 0;
                    }
                } else {
                    tca_arena = NULL;
                    tca_abytes = 0;
                }
            }
            (void)sticky_owned;
        }
        if (!tca_ok) {
            int direct_n = 0;
            if (sov_fast) {
                g_sov_batch_mode = 1;
                pthread_mutex_lock(&g_z->alloc_mutex);
                (void)sov_capsule_readv_warp_locked(g_z, uniq, nuniq);
                pthread_mutex_unlock(&g_z->alloc_mutex);
                direct_n = sov_crw_direct_decomp_batch(g_z, uniq, nuniq);
            }
            if (direct_n < nuniq) {
                if (nuniq >= 2) {
                    size_t *uniq_heap = (size_t *)malloc((size_t)nuniq * sizeof(size_t));
                    if (uniq_heap) {
                        memcpy(uniq_heap, uniq, (size_t)nuniq * sizeof(size_t));
                        int use_sov = sov_fast;
                        dispatch_apply((size_t)nuniq, DISPATCH_APPLY_AUTO, ^(size_t i) {
                            uint8_t pagebuf[PAGE_SZ];
                            if (use_sov) {
                                if (materialize_from_sov(g_z, uniq_heap[i], pagebuf) != 0)
                                    (void)materialize_page_bytes(g_z, uniq_heap[i], pagebuf, flags);
                            } else {
                                (void)materialize_page_bytes(g_z, uniq_heap[i], pagebuf, flags);
                            }
                        });
                        free(uniq_heap);
                    } else {
                        for (int i = 0; i < nuniq; i++) {
                            uint8_t pagebuf[PAGE_SZ];
                            if (sov_fast) (void)materialize_from_sov(g_z, uniq[i], pagebuf);
                            else {
                                const uint8_t *pagep = NULL;
                                (void)materialize_page_ref(g_z, uniq[i], &pagep, flags);
                            }
                        }
                    }
                } else if (nuniq == 1) {
                    uint8_t pagebuf[PAGE_SZ];
                    if (sov_fast) (void)materialize_from_sov(g_z, uniq[0], pagebuf);
                    else {
                        const uint8_t *pagep = NULL;
                        (void)materialize_page_ref(g_z, uniq[0], &pagep, flags);
                    }
                }
            }
        }
        if (sov_fast) {
            int skip_chrono = 0;
            if (tca_ok) {
                const char *es = getenv("MEMX_STICKY_NO_CHRONOS");
                if (!es || es[0] != '0') skip_chrono = 1;
            }
            size_t hi = uniq[0];
            for (int i = 1; i < nuniq; i++) if (uniq[i] > hi) hi = uniq[i];
            int hz = sov_chronos_horizon_pages();
            int async_c = 0;
            int prefer_vault_chrono = (tca_ok || sov_tca_enabled());
            {
                const char *ec = getenv("MEMX_TCA_CHRONOS");
                if (ec && ec[0] == '1') prefer_vault_chrono = 0;
            }
            const char *ef = getenv("MEMX_SOV_CHRONOS_ASYNC_FORCE");
            if (ef && ef[0] == '1') prefer_vault_chrono = 0;
            if (skip_chrono) prefer_vault_chrono = 1;
            const char *ea = getenv("MEMX_SOV_CHRONOS_ASYNC");
            if (!skip_chrono && (!ea || ea[0] != '0') && !prefer_vault_chrono) {
                sov_ent_t *ents = (sov_ent_t *)g_z->sov_ents;
                uint32_t lo = 0, hi2 = g_z->sov_count;
                uint32_t key = (uint32_t)hi;
                while (lo < hi2) {
                    uint32_t mid = lo + ((hi2 - lo) >> 1);
                    if (ents[mid].pidx <= key) lo = mid + 1;
                    else hi2 = mid;
                }
                size_t limit = hi + (size_t)hz;
                for (uint32_t i = lo; i < g_z->sov_count && async_c < 128; i++) {
                    if ((size_t)ents[i].pidx > limit) break;
                    if (mat_nd_enqueue_page(ents[i].pidx)) async_c++;
                }
            }
            if (!skip_chrono && async_c == 0) {
                pthread_mutex_lock(&g_z->alloc_mutex);
                sov_chronos_horizon_locked(g_z, hi, hz);
                pthread_mutex_unlock(&g_z->alloc_mutex);
            } else if (!skip_chrono) {
                g_z->sov_chronos_injects += (uint64_t)async_c;
            }
            g_sov_batch_mode = 0;
        }
    }
    free(uh);

    uint8_t pagebuf[PAGE_SZ];
    int sov_emit = (g_z && g_z->sovereign && g_z->sov_ents);
    size_t last_pidx = (size_t)-1;
    const uint8_t *last_page = NULL;
    enum { TCA_H = 4096 };
    size_t tca_hp[TCA_H];
    int tca_hi[TCA_H];
    int tca_map_ready = 0;
    if (tca_ok && tca_arena && nuniq > 0) {
        for (int i = 0; i < TCA_H; i++) { tca_hp[i] = (size_t)-1; tca_hi[i] = -1; }
        for (int i = 0; i < nuniq; i++) {
            size_t h = (uniq[i] * 11400714819323198485ull) & (TCA_H - 1);
            for (int probe = 0; probe < 64; probe++) {
                size_t si = (h + (size_t)probe) & (TCA_H - 1);
                if (tca_hp[si] == (size_t)-1 || tca_hp[si] == uniq[i]) {
                    tca_hp[si] = uniq[i];
                    tca_hi[si] = i;
                    break;
                }
            }
        }
        tca_map_ready = 1;
    }
    if (tca_map_ready && tile->rows >= 16) {
        size_t rows_n = tile->rows;
        size_t row_bytes_h = row_bytes;
        size_t col_off_h = col_off;
        size_t strip_h = strip;
        size_t nbytes_h = nbytes;
        size_t sp_h = sp;
        size_t dst_row_stride_h = dst_row_stride;
        uint8_t *out_h = out;
        uint8_t *tca_arena_h = tca_arena;
        size_t *tca_hp_h = tca_hp;
        int *tca_hi_h = tca_hi;
        uint32_t flags_h = flags;
        size_t npages_h = g_z->npages;
        __block volatile int emit_err = 0;
        dispatch_apply(rows_n, DISPATCH_APPLY_AUTO, ^(size_t r) {
            if (emit_err) return;
            size_t b = r * row_bytes_h + col_off_h;
            size_t e = b + strip_h;
            if (b >= nbytes_h) return;
            if (e > nbytes_h) e = nbytes_h;
            size_t cur = b;
            size_t dst_base = r * dst_row_stride_h;
            size_t last_p = (size_t)-1;
            const uint8_t *last_pgs = NULL;
            while (cur < e) {
                size_t page_off = cur / PAGE_SZ;
                size_t pidx = sp_h + page_off;
                if (pidx >= npages_h) { emit_err = EFAULT; return; }
                size_t inside = cur - page_off * PAGE_SZ;
                size_t take = PAGE_SZ - inside;
                if (take > e - cur) take = e - cur;
                const uint8_t *srcp = NULL;
                if (pidx == last_p && last_pgs) srcp = last_pgs;
                else {
                    int found = -1;
                    size_t h = (pidx * 11400714819323198485ull) & (TCA_H - 1);
                    for (int probe = 0; probe < 64; probe++) {
                        size_t si = (h + (size_t)probe) & (TCA_H - 1);
                        if (tca_hp_h[si] == (size_t)-1) break;
                        if (tca_hp_h[si] == pidx) { found = tca_hi_h[si]; break; }
                    }
                    if (found >= 0) {
                        srcp = tca_arena_h + (size_t)found * (size_t)PAGE_SZ;
                        last_p = pidx;
                        last_pgs = srcp;
                    }
                }
                if (!srcp) { emit_err = EIO; return; }
                materialize_emit(srcp + inside, out_h + dst_base + (cur - b), take, flags_h);
                cur += take;
            }
        });
        if (emit_err) {
            if (tca_arena && tca_abytes && !tca_sticky_owns(tca_arena)) tca_arena_release(tca_arena, tca_abytes, 0);
            return emit_err;
        }
    } else {
    for (size_t r = 0; r < tile->rows; r++) {
        size_t b = r * row_bytes + col_off;
        size_t e = b + strip;
        if (b >= nbytes) break;
        if (e > nbytes) e = nbytes;
        size_t cur = b;
        size_t dst_base = r * dst_row_stride;
        while (cur < e) {
            size_t page_off = cur / PAGE_SZ;
            size_t pidx = sp + page_off;
            if (pidx >= g_z->npages) {
                if (tca_arena && tca_abytes && !tca_sticky_owns(tca_arena)) tca_arena_release(tca_arena, tca_abytes, 0);
                return EFAULT;
            }
            size_t inside = cur - page_off * PAGE_SZ;
            size_t take = PAGE_SZ - inside;
            if (take > e - cur) take = e - cur;
            const uint8_t *srcp = NULL;
            if (tca_map_ready) {
                if (pidx == last_pidx && last_page) {
                    srcp = last_page;
                } else {
                    int found = -1;
                    size_t h = (pidx * 11400714819323198485ull) & (TCA_H - 1);
                    for (int probe = 0; probe < 64; probe++) {
                        size_t si = (h + (size_t)probe) & (TCA_H - 1);
                        if (tca_hp[si] == (size_t)-1) break;
                        if (tca_hp[si] == pidx) { found = tca_hi[si]; break; }
                    }
                    if (found >= 0) {
                        srcp = tca_arena + (size_t)found * (size_t)PAGE_SZ;
                        last_pidx = pidx;
                        last_page = srcp;
                    }
                }
            }
            if (!srcp && sov_emit) {
                if (pidx == last_pidx && last_page) srcp = last_page;
                else {
                    srcp = materialize_from_sov_ptr(g_z, pidx);
                    last_pidx = pidx;
                    last_page = srcp;
                }
            }
            if (srcp) {
                materialize_emit(srcp + inside, out + dst_base + (cur - b), take, flags);
            } else {
                if (materialize_page_bytes(g_z, pidx, pagebuf, flags) != 0) {
                    if (tca_arena && tca_abytes && !tca_sticky_owns(tca_arena)) tca_arena_release(tca_arena, tca_abytes, 0);
                    return EIO;
                }
                last_pidx = pidx;
                last_page = NULL;
                materialize_emit(pagebuf + inside, out + dst_base + (cur - b), take, flags);
            }
            cur += take;
        }
    }
    }
    if (tca_arena && tca_abytes) {
        if (!tca_sticky_owns(tca_arena))
            tca_arena_release(tca_arena, tca_abytes, 0);
        tca_arena = NULL;
        tca_abytes = 0;
    }
    if (tca_ok && g_z && g_z->vault_cache && g_z->vault_cache_bytes) {
        const char *er = getenv("MEMX_TCA_VAULT_DRIFT");
        int drift = (!er || er[0] != '0');
        if (drift) {
            pthread_mutex_lock(&g_z->alloc_mutex);
            if (g_z->vault_cache_next * 4ull >= g_z->vault_cache_bytes * 3ull) {
                pool_vault_cache_reset_locked(g_z);
                void *rm = mmap(g_z->vault_cache, (size_t)g_z->vault_cache_bytes,
                                PROT_READ | PROT_WRITE,
                                MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (rm == MAP_FAILED) {
#if defined(MADV_FREE_REUSABLE)
                    madvise(g_z->vault_cache, (size_t)g_z->vault_cache_bytes, MADV_FREE_REUSABLE);
#endif
                    madvise(g_z->vault_cache, (size_t)g_z->vault_cache_bytes, MADV_DONTNEED);
                }
                g_z->vault_cache_next = 0;
                g_z->vault_ring_reclaims++;
            }
            pthread_mutex_unlock(&g_z->alloc_mutex);
        }
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

static void sov_dissolve_cold_locked(MemXZone3 *s) {
    if (!s) return;
    if (s->sov_off_idx && s->sov_off_idx != MAP_FAILED && s->sov_off_idx_bytes) {
        munmap(s->sov_off_idx, (size_t)s->sov_off_idx_bytes);
        s->sov_off_idx = NULL;
        s->sov_off_idx_bytes = 0;
    }
    if (s->sov_ents && s->sov_ents != MAP_FAILED && s->sov_bytes) {
        munmap(s->sov_ents, (size_t)s->sov_bytes);
        s->sov_ents = NULL;
        s->sov_bytes = 0;
        s->sov_count = 0;
        s->sov_cap = 0;
    }
    sov_warp_buf_release_locked(s);
    pool_vault_cache_release_locked(s);
    pool_vault_windows_drop_locked(s);
    pool_vault_wbuf_release_locked(s);
    sov_release_structural_locked(s);
    hard_release_all_compressed_physical_locked(s);
    hard_release_all_compressed_physical_locked(s);
    if (s->pool && s->pool_next > 0 && (s->pool_vault_native || s->pool_detached || s->pool_ghost)) {
        pool_ghost_detach_range_locked(s, 0, s->pool_next);
        s->pool_detached = 1;
    }
}



static uint64_t memx_munmap_owned(void **pp, size_t bytes) {
    if (!pp || !*pp || *pp == MAP_FAILED || bytes == 0) return 0;
    void *p = *pp;
    *pp = NULL;
    munmap(p, bytes);
    return (uint64_t)bytes;
}

static int memx_phoenix_seal_locked(MemXZone3 *s, uint64_t *reclaimed) {
    if (!s || s->phoenix_sealed) return 0;
    uint64_t rec = 0;
    __sync_lock_test_and_set(&g_hard_quiesce, 1);
    if (s->attached) {
        sigaction(SIGSEGV, &old_segv, NULL);
        sigaction(SIGBUS, &old_bus, NULL);
        s->attached = 0;
    }
    if (s->pool_spill_fd > 2) {
        (void)pool_vault_wbuf_flush_locked(s);
        if (s->pool && s->pool != MAP_FAILED && s->pool_next > 0 &&
            !pool_is_vault_native(s) && s->pool_spill_bytes < s->pool_next) {
            (void)pool_ghost_pwrite_range_locked(s, 0, s->pool_next);
        }
        if (pool_is_vault_native(s) || s->pool_spill_bytes >= s->pool_next) {
            s->pool_detached = 1;
            s->pool_ghost = 1;
            if (s->pool_ghost_flushed < s->pool_next)
                s->pool_ghost_flushed = s->pool_next;
        }
    }
    pool_vault_cache_release_locked(s);
    pool_vault_windows_drop_locked(s);
    pool_vault_wbuf_release_locked(s);
    sov_warp_buf_release_locked(s);
    sov_release_structural_locked(s);
    sov_drop_locked(s);
    if (s->vmem && s->vmem != MAP_FAILED && s->vmem_size) {
        uint64_t charge = s->vmem_next ? s->vmem_next : 0;
        if (charge > s->vmem_size) charge = s->vmem_size;
        rec += charge;
        munmap(s->vmem, (size_t)s->vmem_size);
        s->vmem = NULL;
    }
    if (s->pool && s->pool != MAP_FAILED && s->pool_size) {
        uint64_t charge = s->pool_next ? s->pool_next : 0;
        if (charge > s->pool_size) charge = s->pool_size;
        rec += charge;
        munmap(s->pool, (size_t)s->pool_size);
        s->pool = NULL;
    }
    if (s->meta && s->meta != MAP_FAILED && s->npages) {
        rec += (uint64_t)s->npages * (uint64_t)sizeof(PageMeta);
        munmap(s->meta, s->npages * sizeof(PageMeta));
        s->meta = NULL;
    }
    if (s->free_bm && s->free_bm != MAP_FAILED && s->free_bm_size) {
        rec += (uint64_t)s->free_bm_size * (uint64_t)sizeof(uint64_t);
        munmap(s->free_bm, s->free_bm_size * sizeof(uint64_t));
        s->free_bm = NULL;
        s->free_bm_size = 0;
    }
    if (s->hot_list && s->hot_cap) {
        munmap(s->hot_list, (size_t)s->hot_cap * 4);
        s->hot_list = NULL;
        s->hot_count = 0;
        s->hot_cap = 0;
    }
    if (s->res_list && s->res_cap) {
        munmap(s->res_list, (size_t)s->res_cap * 4);
        s->res_list = NULL;
        s->res_count = 0;
        s->res_cap = 0;
    }
    if (s->dedup_hash && s->dedup_hash != MAP_FAILED) {
        munmap(s->dedup_hash, DEDUP_HT_SIZE * 8);
        s->dedup_hash = NULL;
    }
    if (s->dedup_off && s->dedup_off != MAP_FAILED) {
        munmap(s->dedup_off, DEDUP_HT_SIZE * 8);
        s->dedup_off = NULL;
    }
    if (s->dedup_sz && s->dedup_sz != MAP_FAILED) {
        munmap(s->dedup_sz, DEDUP_HT_SIZE * 4);
        s->dedup_sz = NULL;
    }
    if (s->dedup_ref && s->dedup_ref != MAP_FAILED) {
        munmap(s->dedup_ref, DEDUP_HT_SIZE * 4);
        s->dedup_ref = NULL;
    }
    if (s->dedup_pending_free && s->dedup_pending_free != MAP_FAILED) {
        munmap(s->dedup_pending_free, DEDUP_HT_SIZE);
        s->dedup_pending_free = NULL;
    }
    if (s->dedup_rev && s->dedup_rev != MAP_FAILED && s->dedup_rev_size) {
        munmap(s->dedup_rev, (size_t)s->dedup_rev_size * 4);
        s->dedup_rev = NULL;
        s->dedup_rev_size = 0;
    }
    if (s->pool_free_off && s->pool_free_off != MAP_FAILED && s->pool_free_cap) {
        munmap(s->pool_free_off, (size_t)s->pool_free_cap * sizeof(uint64_t));
        s->pool_free_off = NULL;
    }
    if (s->pool_free_sz && s->pool_free_sz != MAP_FAILED && s->pool_free_cap) {
        munmap(s->pool_free_sz, (size_t)s->pool_free_cap * sizeof(uint32_t));
        s->pool_free_sz = NULL;
        s->pool_free_cap = 0;
        s->pool_free_count = 0;
    }
    if (s->batch_cap) {
        size_t batch_bytes = s->batch_cap * PAGE_SZ;
        if (s->tmp_src && s->tmp_src != MAP_FAILED) {
            munmap(s->tmp_src, batch_bytes);
            s->tmp_src = NULL;
        }
        if (s->tmp_dst && s->tmp_dst != MAP_FAILED) {
            munmap(s->tmp_dst, batch_bytes);
            s->tmp_dst = NULL;
        }
        if (s->tmp_sz && s->tmp_sz != MAP_FAILED) {
            munmap(s->tmp_sz, s->batch_cap * 4);
            s->tmp_sz = NULL;
        }
    }
    s->vmem_next = 0;
    s->vmem_size = 0;
    s->pool_size = 0;
    s->pool_next = 0;
    s->pool_used = 0;
    s->npages = 0;
    s->phoenix_sealed = 1;
    s->pool_detached = 1;
    s->sovereign = 0;
    s->sovereign_frozen = 1;
    s->live_compressed_pages = 0;
    s->live_resident_pages = 0;
    s->live_hot_flag_pages = 0;
    s->live_nocomp_flag_pages = 0;
    if (reclaimed) *reclaimed += rec;
    {
        int close_spill = 1;
        const char *cs = getenv("MEMX_PHOENIX_CLOSE_SPILL");
        if (cs && cs[0] == '0') close_spill = 0;
        if (close_spill && s->pool_spill_fd > 2) {
            (void)fcntl(s->pool_spill_fd, F_NOCACHE, 1);
            close(s->pool_spill_fd);
            s->pool_spill_fd = -1;
            s->pool_spill_bytes = 0;
            s->pool_ghost_flushed = 0;
        }
    }
    fprintf(stderr, "[memx] phoenix_seal reclaimed~%lluMB spill_fd=%d sealed=1\n",
            (unsigned long long)(rec / (1024ull * 1024ull)), s->pool_spill_fd);
#if defined(__APPLE__)
    malloc_zone_pressure_relief(NULL, 0);
    malloc_zone_pressure_relief(NULL, 0);
#endif
    return 1;
}


#define MEMX_CAPSULE_MAGIC 0x4D584350u
#define MEMX_CAPSULE_VER 2u

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    uint64_t ent_count;
    uint64_t spill_bytes;
    uint64_t page_bytes;
    uint64_t page_sz;
    uint64_t reserved[4];
} memx_capsule_hdr_t;

typedef struct __attribute__((packed)) {
    uint32_t pidx;
    uint32_t csz;
    uint64_t off;
    uint32_t seq;
    uint8_t  codec;
    uint8_t  role;
    uint8_t  flags;
    uint8_t  _pad;
} memx_capsule_ent_t;

typedef struct {
    uint64_t off;
    uint32_t csz;
    uint32_t pidx;
    uint32_t slot;
} cap_batch_item_t;

typedef struct __attribute__((packed)) {
    uint32_t csz;
    uint32_t off_lo;
    uint32_t off_hi;
} memx_cap_rank_t;

typedef struct {
    int live;
    int spill_fd;
    int ledger_fd;
    int rank_fd;
    int dense;
    int lite;
    int has_rank;
    int export_clone;
    uint32_t dense_base;
    uint64_t spill_bytes;
    uint64_t ent_count;
    uint64_t page_sz;
    uint64_t page_bytes;
    uint64_t ledger_bytes;
    uint64_t materialize_pages;
    uint64_t materialize_bytes;
    uint64_t materialize_spans;
    uint64_t materialize_batch_pages;
    memx_capsule_ent_t *ents;
    size_t ents_bytes;
    char dir[512];
} memx_capsule_rt_t;

static memx_capsule_rt_t g_cap;
int memx_runtime_capsule_attach(const char *dirpath);

static int capsule_ent_cmp(const void *a, const void *b) {
    const memx_capsule_ent_t *x = (const memx_capsule_ent_t *)a;
    const memx_capsule_ent_t *y = (const memx_capsule_ent_t *)b;
    if (x->pidx < y->pidx) return -1;
    if (x->pidx > y->pidx) return 1;
    return 0;
}

static int capsule_batch_off_cmp(const void *a, const void *b) {
    const cap_batch_item_t *x = (const cap_batch_item_t *)a;
    const cap_batch_item_t *y = (const cap_batch_item_t *)b;
    if (x->off < y->off) return -1;
    if (x->off > y->off) return 1;
    if (x->pidx < y->pidx) return -1;
    if (x->pidx > y->pidx) return 1;
    return 0;
}

static int capsule_mkdir_p(const char *dirpath) {
    if (!dirpath || !dirpath[0]) return -1;
    if (mkdir(dirpath, 0700) == 0) return 0;
    if (errno == EEXIST) return 0;
    return -1;
}

static int capsule_copy_fd(int src_fd, int dst_fd, uint64_t bytes) {
    if (src_fd < 0 || dst_fd < 0) return -1;
    uint8_t buf[1 << 20];
    uint64_t off = 0;
    while (off < bytes) {
        size_t n = (size_t)((bytes - off) > sizeof(buf) ? sizeof(buf) : (bytes - off));
        ssize_t r = pread(src_fd, buf, n, (off_t)off);
        if (r <= 0) return -1;
        ssize_t w = pwrite(dst_fd, buf, (size_t)r, (off_t)off);
        if (w != r) return -1;
        off += (uint64_t)r;
    }
    return 0;
}

static int capsule_spill_publish(int src_fd, const char *dst_path, uint64_t bytes, int *out_cloned) {
    if (out_cloned) *out_cloned = 0;
    if (src_fd < 0 || !dst_path || !dst_path[0]) return -1;
    (void)unlink(dst_path);
#if defined(__APPLE__)
    char src_path[1024];
    memset(src_path, 0, sizeof(src_path));
    if (fcntl(src_fd, F_GETPATH, src_path) == 0 && src_path[0]) {
        if (clonefile(src_path, dst_path, 0) == 0) {
            if (out_cloned) *out_cloned = 1;
            return 0;
        }
        if (link(src_path, dst_path) == 0) {
            if (out_cloned) *out_cloned = 2;
            return 0;
        }
    }
    int dst_fd = open(dst_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (dst_fd < 0) return -1;
    if (fcopyfile(src_fd, dst_fd, NULL, COPYFILE_CLONE) == 0) {
        if (out_cloned) *out_cloned = 3;
        (void)fcntl(dst_fd, F_NOCACHE, 1);
        close(dst_fd);
        return 0;
    }
    if (fcopyfile(src_fd, dst_fd, NULL, COPYFILE_DATA | COPYFILE_CLONE) == 0) {
        if (out_cloned) *out_cloned = 4;
        (void)fcntl(dst_fd, F_NOCACHE, 1);
        close(dst_fd);
        return 0;
    }
    if (ftruncate(dst_fd, (off_t)bytes) != 0 || capsule_copy_fd(src_fd, dst_fd, bytes) != 0) {
        close(dst_fd);
        (void)unlink(dst_path);
        return -1;
    }
    (void)fcntl(dst_fd, F_NOCACHE, 1);
    close(dst_fd);
    return 0;
#else
    int dst_fd = open(dst_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (dst_fd < 0) return -1;
    if (ftruncate(dst_fd, (off_t)bytes) != 0 || capsule_copy_fd(src_fd, dst_fd, bytes) != 0) {
        close(dst_fd);
        (void)unlink(dst_path);
        return -1;
    }
    close(dst_fd);
    return 0;
#endif
}

static _Thread_local memx_capsule_ent_t g_cap_lite_ent;

static int capsule_read_ent_at(uint64_t rank, memx_capsule_ent_t *out) {
    if (!out || rank >= g_cap.ent_count) return -1;
    if (g_cap.ents) {
        *out = g_cap.ents[rank];
        return 0;
    }
    if (g_cap.ledger_fd < 0) return -1;
    off_t off = (off_t)(sizeof(memx_capsule_hdr_t) + rank * sizeof(memx_capsule_ent_t));
    if (pread(g_cap.ledger_fd, out, sizeof(*out), off) != (ssize_t)sizeof(*out)) return -1;
    return 0;
}

static const memx_capsule_ent_t *capsule_find_ent(uint32_t pidx) {
    if (!g_cap.live || g_cap.ent_count == 0) return NULL;
    if (g_cap.ents) {
        if (g_cap.dense) {
            if (pidx < g_cap.dense_base) return NULL;
            uint64_t i = (uint64_t)pidx - (uint64_t)g_cap.dense_base;
            if (i >= g_cap.ent_count) return NULL;
            const memx_capsule_ent_t *e = &g_cap.ents[i];
            if (e->pidx != pidx) return NULL;
            return e;
        }
        uint64_t lo = 0, hi = g_cap.ent_count;
        while (lo < hi) {
            uint64_t mid = lo + ((hi - lo) >> 1);
            uint32_t v = g_cap.ents[mid].pidx;
            if (v < pidx) lo = mid + 1;
            else hi = mid;
        }
        if (lo < g_cap.ent_count && g_cap.ents[lo].pidx == pidx) return &g_cap.ents[lo];
        return NULL;
    }
    if (g_cap.ledger_fd < 0) return NULL;
    if (g_cap.dense) {
        if (pidx < g_cap.dense_base) return NULL;
        uint64_t i = (uint64_t)pidx - (uint64_t)g_cap.dense_base;
        if (i >= g_cap.ent_count) return NULL;
        if (capsule_read_ent_at(i, &g_cap_lite_ent) != 0) return NULL;
        if (g_cap_lite_ent.pidx != pidx) return NULL;
        return &g_cap_lite_ent;
    }
    uint64_t lo = 0, hi = g_cap.ent_count;
    while (lo < hi) {
        uint64_t mid = lo + ((hi - lo) >> 1);
        memx_capsule_ent_t e;
        if (capsule_read_ent_at(mid, &e) != 0) return NULL;
        if (e.pidx < pidx) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= g_cap.ent_count) return NULL;
    if (capsule_read_ent_at(lo, &g_cap_lite_ent) != 0) return NULL;
    if (g_cap_lite_ent.pidx != pidx) return NULL;
    return &g_cap_lite_ent;
}

static void capsule_mark_dense(void) {
    g_cap.dense = 0;
    g_cap.dense_base = 0;
    if (!g_cap.ents || g_cap.ent_count == 0) return;
    uint32_t lo = g_cap.ents[0].pidx;
    uint32_t hi = g_cap.ents[g_cap.ent_count - 1].pidx;
    if ((uint64_t)hi < (uint64_t)lo) return;
    if ((uint64_t)hi - (uint64_t)lo + 1ull != g_cap.ent_count) return;
    for (uint64_t i = 0; i < g_cap.ent_count; i++) {
        if (g_cap.ents[i].pidx != (uint32_t)(lo + (uint32_t)i)) return;
    }
    g_cap.dense = 1;
    g_cap.dense_base = lo;
}

int memx_runtime_capsule_detach(void) {
    if (g_cap.ents && g_cap.ents != MAP_FAILED && g_cap.ents_bytes)
        munmap(g_cap.ents, g_cap.ents_bytes);
    if (g_cap.spill_fd > 2) close(g_cap.spill_fd);
    if (g_cap.ledger_fd > 2) close(g_cap.ledger_fd);
    if (g_cap.rank_fd > 2) close(g_cap.rank_fd);
    memset(&g_cap, 0, sizeof(g_cap));
    g_cap.spill_fd = -1;
    g_cap.ledger_fd = -1;
    g_cap.rank_fd = -1;
    return 0;
}

int memx_runtime_capsule_export(const char *dirpath, uint64_t *out_bytes) {
    if (!dirpath || !dirpath[0]) return EINVAL;
    if (!g_z) return ENOENT;
    if (g_z->phoenix_sealed) return EBUSY;
    pthread_mutex_lock(&g_z->alloc_mutex);
    (void)pool_vault_wbuf_flush_locked(g_z);
    if (g_z->pool_spill_fd < 0 || g_z->pool_spill_bytes == 0) {
        if (pool_ensure_spill_fd_locked(g_z) != 0) {
            pthread_mutex_unlock(&g_z->alloc_mutex);
            return EIO;
        }
        if (pool_is_vault_native(g_z)) {
            if (g_z->pool_spill_bytes < g_z->pool_next) {
                pthread_mutex_unlock(&g_z->alloc_mutex);
                return EIO;
            }
        } else {
            if (pool_ghost_pwrite_range_locked(g_z, 0, g_z->pool_next) != 0) {
                pthread_mutex_unlock(&g_z->alloc_mutex);
                return EIO;
            }
        }
    }
    size_t last = g_z->vmem_next / PAGE_SZ;
    if (last > g_z->npages) last = g_z->npages;
    if (last == 0) last = g_z->npages;
    uint64_t n = 0;
    for (size_t i = 0; i < last; i++) {
        PageMeta *m = &g_z->meta[i];
        if (m->state != PAGE_COMPRESSED) continue;
        if (m->comp_size == 0 || m->comp_size > PAGE_SZ) continue;
        if (m->pool_offset + (uint64_t)m->comp_size > g_z->pool_spill_bytes &&
            m->pool_offset + (uint64_t)m->comp_size > g_z->pool_next) continue;
        n++;
    }
    if (n == 0) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return ENOENT;
    }
    size_t ents_bytes = (size_t)n * sizeof(memx_capsule_ent_t);
    memx_capsule_ent_t *ents = (memx_capsule_ent_t *)mmap(NULL, ents_bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (ents == MAP_FAILED) {
        pthread_mutex_unlock(&g_z->alloc_mutex);
        return ENOMEM;
    }
    uint64_t w = 0;
    for (size_t i = 0; i < last && w < n; i++) {
        PageMeta *m = &g_z->meta[i];
        if (m->state != PAGE_COMPRESSED) continue;
        if (m->comp_size == 0 || m->comp_size > PAGE_SZ) continue;
        ents[w].pidx = (uint32_t)i;
        ents[w].csz = m->comp_size;
        ents[w].off = m->pool_offset;
        ents[w].seq = m->write_seq;
        ents[w].codec = m->codec;
        ents[w].role = (uint8_t)(m->tensor_role & 0xFF);
        ents[w].flags = 0;
        ents[w]._pad = 0;
        w++;
    }
    n = w;
    if (n > 1) qsort(ents, (size_t)n, sizeof(memx_capsule_ent_t), capsule_ent_cmp);
    uint64_t spill_bytes = g_z->pool_spill_bytes;
    if (spill_bytes < g_z->pool_next) spill_bytes = g_z->pool_next;
    int src_fd = g_z->pool_spill_fd;
    pthread_mutex_unlock(&g_z->alloc_mutex);

    if (capsule_mkdir_p(dirpath) != 0) {
        munmap(ents, ents_bytes);
        return EIO;
    }
    char spill_path[640];
    char led_path[640];
    snprintf(spill_path, sizeof(spill_path), "%s/spill.bin", dirpath);
    snprintf(led_path, sizeof(led_path), "%s/ledger.bin", dirpath);
    int cloned = 0;
    if (capsule_spill_publish(src_fd, spill_path, spill_bytes, &cloned) != 0) {
        munmap(ents, ents_bytes);
        return EIO;
    }

    memx_capsule_hdr_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = MEMX_CAPSULE_MAGIC;
    hdr.version = MEMX_CAPSULE_VER;
    hdr.ent_count = n;
    hdr.spill_bytes = spill_bytes;
    hdr.page_bytes = n * (uint64_t)PAGE_SZ;
    hdr.page_sz = PAGE_SZ;
    int lfd = open(led_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (lfd < 0) {
        munmap(ents, ents_bytes);
        return EIO;
    }
    if (pwrite(lfd, &hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)) {
        close(lfd);
        munmap(ents, ents_bytes);
        return EIO;
    }
    if (pwrite(lfd, ents, (size_t)n * sizeof(memx_capsule_ent_t), (off_t)sizeof(hdr)) != (ssize_t)((size_t)n * sizeof(memx_capsule_ent_t))) {
        close(lfd);
        munmap(ents, ents_bytes);
        return EIO;
    }
    close(lfd);
    int rank_written = 0;
    uint32_t dense_base = 0;
    int is_dense = 0;
    if (n > 0) {
        is_dense = 1;
        dense_base = ents[0].pidx;
        for (uint64_t i = 0; i < n; i++) {
            if (ents[i].pidx != (uint32_t)(dense_base + (uint32_t)i)) {
                is_dense = 0;
                break;
            }
        }
    }
    if (is_dense) {
        char rank_path[640];
        snprintf(rank_path, sizeof(rank_path), "%s/rank.map", dirpath);
        int rfd = open(rank_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (rfd >= 0) {
            uint32_t rm_magic = 0x4D58524Du;
            uint32_t rm_base = dense_base;
            uint64_t rm_n = n;
            if (pwrite(rfd, &rm_magic, 4, 0) == 4 &&
                pwrite(rfd, &rm_base, 4, 4) == 4 &&
                pwrite(rfd, &rm_n, 8, 8) == 8) {
                size_t rbytes = (size_t)n * sizeof(memx_cap_rank_t);
                memx_cap_rank_t *ranks = (memx_cap_rank_t *)mmap(NULL, rbytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
                if (ranks != MAP_FAILED) {
                    for (uint64_t i = 0; i < n; i++) {
                        ranks[i].csz = ents[i].csz;
                        ranks[i].off_lo = (uint32_t)(ents[i].off & 0xffffffffu);
                        ranks[i].off_hi = (uint32_t)(ents[i].off >> 32);
                    }
                    if (pwrite(rfd, ranks, rbytes, 16) == (ssize_t)rbytes) rank_written = 1;
                    munmap(ranks, rbytes);
                }
            }
            close(rfd);
            if (!rank_written) unlink(rank_path);
        }
    }
    munmap(ents, ents_bytes);
    uint64_t total = spill_bytes + sizeof(hdr) + n * sizeof(memx_capsule_ent_t);
    if (rank_written) total += 16ull + n * sizeof(memx_cap_rank_t);
    if (out_bytes) *out_bytes = total;
    g_cap.export_clone = cloned;
    fprintf(stderr, "[memx] capsule_export dir=%s ents=%llu spill=%lluMB page_logical=%lluMB total=%lluMB clone=%d rank=%d\n",
            dirpath,
            (unsigned long long)n,
            (unsigned long long)(spill_bytes / (1024ull * 1024ull)),
            (unsigned long long)((n * (uint64_t)PAGE_SZ) / (1024ull * 1024ull)),
            (unsigned long long)(total / (1024ull * 1024ull)),
            cloned,
            rank_written);
    {
        const char *hb = getenv("MEMX_CAPSULE_HOST_BIND");
        int do_bind = 1;
        if (hb && hb[0] == '0') do_bind = 0;
        if (do_bind) {
            setenv("MEMX_CAPSULE_LITE", "1", 0);
            (void)memx_runtime_capsule_attach(dirpath);
        }
    }
    return 0;
}


int memx_runtime_capsule_attach(const char *dirpath) {
    if (!dirpath || !dirpath[0]) return EINVAL;
    (void)memx_runtime_capsule_detach();
    char spill_path[640];
    char led_path[640];
    snprintf(spill_path, sizeof(spill_path), "%s/spill.bin", dirpath);
    snprintf(led_path, sizeof(led_path), "%s/ledger.bin", dirpath);
    int lfd = open(led_path, O_RDONLY);
    if (lfd < 0) return EIO;
    memx_capsule_hdr_t hdr;
    if (pread(lfd, &hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)) {
        close(lfd);
        return EIO;
    }
    if (hdr.magic != MEMX_CAPSULE_MAGIC || hdr.version != MEMX_CAPSULE_VER || hdr.ent_count == 0) {
        close(lfd);
        return EINVAL;
    }
    int lite = 0;
    const char *le = getenv("MEMX_CAPSULE_LITE");
    if (le && le[0] == '1') lite = 1;
    size_t ents_bytes = (size_t)hdr.ent_count * sizeof(memx_capsule_ent_t);
    memx_capsule_ent_t *ents = NULL;
    if (!lite) {
        void *m = mmap(NULL, ents_bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) {
            close(lfd);
            return ENOMEM;
        }
        if (pread(lfd, m, ents_bytes, (off_t)sizeof(hdr)) != (ssize_t)ents_bytes) {
            munmap(m, ents_bytes);
            close(lfd);
            return EIO;
        }
        ents = (memx_capsule_ent_t *)m;
    } else {
#if defined(__APPLE__)
        (void)fcntl(lfd, F_NOCACHE, 1);
#endif
    }
    int sfd = open(spill_path, O_RDONLY);
    if (sfd < 0) {
        if (ents) munmap(ents, ents_bytes);
        close(lfd);
        return EIO;
    }
#if defined(__APPLE__)
    (void)fcntl(sfd, F_NOCACHE, 1);
#endif
    g_cap.live = 1;
    g_cap.spill_fd = sfd;
    g_cap.ledger_fd = lite ? lfd : -1;
    if (!lite) close(lfd);
    g_cap.lite = lite;
    g_cap.rank_fd = -1;
    g_cap.has_rank = 0;
    g_cap.spill_bytes = hdr.spill_bytes;
    g_cap.ent_count = hdr.ent_count;
    g_cap.page_sz = hdr.page_sz ? hdr.page_sz : PAGE_SZ;
    g_cap.page_bytes = hdr.page_bytes;
    g_cap.ledger_bytes = sizeof(hdr) + ents_bytes;
    g_cap.ents = ents;
    g_cap.ents_bytes = lite ? 0 : ents_bytes;
    g_cap.materialize_pages = 0;
    g_cap.materialize_bytes = 0;
    g_cap.materialize_spans = 0;
    g_cap.materialize_batch_pages = 0;
    g_cap.export_clone = 0;
    snprintf(g_cap.dir, sizeof(g_cap.dir), "%s", dirpath);
    {
        char rank_path[640];
        snprintf(rank_path, sizeof(rank_path), "%s/rank.map", dirpath);
        int rfd = open(rank_path, O_RDONLY);
        if (rfd >= 0) {
            uint32_t rm_magic = 0, rm_base = 0;
            uint64_t rm_n = 0;
            if (pread(rfd, &rm_magic, 4, 0) == 4 && rm_magic == 0x4D58524Du &&
                pread(rfd, &rm_base, 4, 4) == 4 &&
                pread(rfd, &rm_n, 8, 8) == 8 && rm_n == hdr.ent_count) {
#if defined(__APPLE__)
                (void)fcntl(rfd, F_NOCACHE, 1);
#endif
                g_cap.rank_fd = rfd;
                g_cap.has_rank = 1;
                g_cap.dense = 1;
                g_cap.dense_base = rm_base;
                g_cap.ledger_bytes += 16ull + rm_n * sizeof(memx_cap_rank_t);
            } else {
                close(rfd);
            }
        }
    }
    if (!lite) capsule_mark_dense();
    else if (!g_cap.has_rank) {
        memx_capsule_ent_t first, last;
        g_cap.dense = 0;
        g_cap.dense_base = 0;
        if (capsule_read_ent_at(0, &first) == 0 && capsule_read_ent_at(hdr.ent_count - 1, &last) == 0) {
            if ((uint64_t)last.pidx >= (uint64_t)first.pidx &&
                (uint64_t)last.pidx - (uint64_t)first.pidx + 1ull == hdr.ent_count) {
                int ok = 1;
                if (hdr.ent_count > 2) {
                    memx_capsule_ent_t mid;
                    uint64_t mi = hdr.ent_count / 2;
                    if (capsule_read_ent_at(mi, &mid) != 0 || mid.pidx != (uint32_t)(first.pidx + (uint32_t)mi)) ok = 0;
                }
                if (ok) {
                    g_cap.dense = 1;
                    g_cap.dense_base = first.pidx;
                }
            }
        }
    }
    fprintf(stderr, "[memx] capsule_attach dir=%s ents=%llu spill=%lluMB dense=%d base=%u lite=%d rank=%d\n",
            dirpath,
            (unsigned long long)g_cap.ent_count,
            (unsigned long long)(g_cap.spill_bytes / (1024ull * 1024ull)),
            g_cap.dense,
            g_cap.dense_base,
            g_cap.lite,
            g_cap.has_rank);
    return 0;
}


static int capsule_rank_load(uint64_t rank, uint32_t *out_csz, uint64_t *out_off) {
    if (rank >= g_cap.ent_count) return -1;
    if (g_cap.has_rank && g_cap.rank_fd >= 0) {
        memx_cap_rank_t r;
        off_t off = (off_t)(16ull + rank * sizeof(memx_cap_rank_t));
        if (pread(g_cap.rank_fd, &r, sizeof(r), off) != (ssize_t)sizeof(r)) return -1;
        if (r.csz == 0 || r.csz > PAGE_SZ) return -1;
        uint64_t o = ((uint64_t)r.off_hi << 32) | (uint64_t)r.off_lo;
        if (o + (uint64_t)r.csz > g_cap.spill_bytes) return -1;
        if (out_csz) *out_csz = r.csz;
        if (out_off) *out_off = o;
        return 0;
    }
    memx_capsule_ent_t e;
    if (capsule_read_ent_at(rank, &e) != 0) return -1;
    if (e.csz == 0 || e.csz > PAGE_SZ) return -1;
    if (e.off + (uint64_t)e.csz > g_cap.spill_bytes) return -1;
    if (out_csz) *out_csz = e.csz;
    if (out_off) *out_off = e.off;
    return 0;
}

int memx_runtime_capsule_materialize_rank(uint64_t rank, void *dst, size_t dst_cap) {
    if (!dst || dst_cap < PAGE_SZ) return EINVAL;
    if (!g_cap.live || g_cap.spill_fd < 0) return ENOENT;
    uint32_t csz = 0;
    uint64_t off = 0;
    if (capsule_rank_load(rank, &csz, &off) != 0) return ENOENT;
    uint8_t payload[PAGE_SZ];
    if (pread(g_cap.spill_fd, payload, csz, (off_t)off) != (ssize_t)csz) return EIO;
    cpu_decompress(payload, csz, (uint8_t *)dst);
    g_cap.materialize_pages++;
    g_cap.materialize_bytes += PAGE_SZ;
    return 0;
}

int memx_runtime_capsule_materialize(uint32_t pidx, void *dst, size_t dst_cap) {
    if (!dst || dst_cap < PAGE_SZ) return EINVAL;
    if (!g_cap.live || g_cap.spill_fd < 0) return ENOENT;
    if (g_cap.dense) {
        if (pidx >= g_cap.dense_base) {
            uint64_t rank = (uint64_t)pidx - (uint64_t)g_cap.dense_base;
            if (rank < g_cap.ent_count)
                return memx_runtime_capsule_materialize_rank(rank, dst, dst_cap);
        }
    }
    const memx_capsule_ent_t *e = capsule_find_ent(pidx);
    if (!e) return ENOENT;
    if (e->csz == 0 || e->csz > PAGE_SZ) return EIO;
    if (e->off + (uint64_t)e->csz > g_cap.spill_bytes) return EIO;
    uint8_t payload[PAGE_SZ];
    if (pread(g_cap.spill_fd, payload, e->csz, (off_t)e->off) != (ssize_t)e->csz) return EIO;
    cpu_decompress(payload, e->csz, (uint8_t *)dst);
    g_cap.materialize_pages++;
    g_cap.materialize_bytes += PAGE_SZ;
    return 0;
}

int memx_runtime_capsule_materialize_v(const uint32_t *pidxs, uint32_t n, void *dst, size_t dst_stride) {
    if (!pidxs || n == 0 || !dst) return EINVAL;
    if (dst_stride < PAGE_SZ) return EINVAL;
    if (!g_cap.live || g_cap.spill_fd < 0) return ENOENT;
    enum { CAP_BATCH_MAX = 512 };
    if (n > CAP_BATCH_MAX) n = CAP_BATCH_MAX;
    cap_batch_item_t items[CAP_BATCH_MAX];
    uint32_t m = 0;
    for (uint32_t i = 0; i < n; i++) {
        const memx_capsule_ent_t *e = capsule_find_ent(pidxs[i]);
        if (!e || e->csz == 0 || e->csz > PAGE_SZ) continue;
        if (e->off + (uint64_t)e->csz > g_cap.spill_bytes) continue;
        items[m].off = e->off;
        items[m].csz = e->csz;
        items[m].pidx = e->pidx;
        items[m].slot = i;
        m++;
    }
    if (m == 0) return ENOENT;
    if (m > 1) qsort(items, (size_t)m, sizeof(items[0]), capsule_batch_off_cmp);
    uint8_t stack_payload[PAGE_SZ];
    uint8_t *span_buf = NULL;
    size_t span_cap = 0;
    uint32_t i = 0;
    while (i < m) {
        uint64_t base = items[i].off;
        uint64_t run_end = items[i].off + items[i].csz;
        uint32_t j = i + 1;
        while (j < m) {
            uint64_t o = items[j].off;
            uint32_t c = items[j].csz;
            uint64_t e2 = o + c;
            if (o < run_end) {
                if (e2 > run_end) run_end = e2;
                j++;
                continue;
            }
            if (o > run_end + 4096ull) break;
            if (e2 - base > (1ull << 20)) break;
            if (j - i >= 96) break;
            run_end = e2;
            j++;
        }
        size_t span = (size_t)(run_end - base);
        if (span == 0 || span > (1u << 20) || j == i + 1) {
            for (uint32_t k = i; k < j; k++) {
                if (pread(g_cap.spill_fd, stack_payload, items[k].csz, (off_t)items[k].off) != (ssize_t)items[k].csz)
                    continue;
                uint8_t *out = (uint8_t *)dst + (size_t)items[k].slot * dst_stride;
                cpu_decompress(stack_payload, items[k].csz, out);
                g_cap.materialize_pages++;
                g_cap.materialize_batch_pages++;
                g_cap.materialize_bytes += PAGE_SZ;
            }
            if (j > i) g_cap.materialize_spans++;
            i = j;
            continue;
        }
        if (span_cap < span) {
            if (span_buf && span_buf != MAP_FAILED) munmap(span_buf, span_cap);
            span_buf = (uint8_t *)mmap(NULL, span, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
            if (span_buf == MAP_FAILED) {
                span_buf = NULL;
                span_cap = 0;
                for (uint32_t k = i; k < j; k++) {
                    if (pread(g_cap.spill_fd, stack_payload, items[k].csz, (off_t)items[k].off) != (ssize_t)items[k].csz)
                        continue;
                    uint8_t *out = (uint8_t *)dst + (size_t)items[k].slot * dst_stride;
                    cpu_decompress(stack_payload, items[k].csz, out);
                    g_cap.materialize_pages++;
                    g_cap.materialize_batch_pages++;
                    g_cap.materialize_bytes += PAGE_SZ;
                }
                if (j > i) g_cap.materialize_spans++;
                i = j;
                continue;
            }
            span_cap = span;
        }
        if (pread(g_cap.spill_fd, span_buf, span, (off_t)base) != (ssize_t)span) {
            for (uint32_t k = i; k < j; k++) {
                if (pread(g_cap.spill_fd, stack_payload, items[k].csz, (off_t)items[k].off) != (ssize_t)items[k].csz)
                    continue;
                uint8_t *out = (uint8_t *)dst + (size_t)items[k].slot * dst_stride;
                cpu_decompress(stack_payload, items[k].csz, out);
                g_cap.materialize_pages++;
                g_cap.materialize_batch_pages++;
                g_cap.materialize_bytes += PAGE_SZ;
            }
        } else {
            g_cap.materialize_spans++;
            for (uint32_t k = i; k < j; k++) {
                uint64_t rel = items[k].off - base;
                uint8_t *out = (uint8_t *)dst + (size_t)items[k].slot * dst_stride;
                cpu_decompress(span_buf + rel, items[k].csz, out);
                g_cap.materialize_pages++;
                g_cap.materialize_batch_pages++;
                g_cap.materialize_bytes += PAGE_SZ;
            }
        }
        i = j;
    }
    if (span_buf && span_buf != MAP_FAILED) munmap(span_buf, span_cap);
    return 0;
}


int memx_runtime_capsule_pidx_at(uint64_t rank, uint32_t *out_pidx) {
    if (!out_pidx) return EINVAL;
    if (!g_cap.live || rank >= g_cap.ent_count) return ENOENT;
    if (g_cap.dense) {
        *out_pidx = (uint32_t)(g_cap.dense_base + (uint32_t)rank);
        return 0;
    }
    memx_capsule_ent_t e;
    if (capsule_read_ent_at(rank, &e) != 0) return EIO;
    *out_pidx = e.pidx;
    return 0;
}

int memx_runtime_capsule_stats(memx_runtime_capsule_stats_t *out_stats) {
    if (!out_stats) return EINVAL;
    memset(out_stats, 0, sizeof(*out_stats));
    out_stats->attached = g_cap.live ? 1 : 0;
    out_stats->ent_count = g_cap.ent_count;
    out_stats->spill_bytes = g_cap.spill_bytes;
    out_stats->page_bytes = g_cap.page_bytes;
    out_stats->ledger_bytes = g_cap.ledger_bytes;
    out_stats->materialize_pages = g_cap.materialize_pages;
    out_stats->materialize_bytes = g_cap.materialize_bytes;
    out_stats->materialize_spans = g_cap.materialize_spans;
    out_stats->materialize_batch_pages = g_cap.materialize_batch_pages;
    out_stats->dense = g_cap.dense;
    out_stats->export_clone = g_cap.export_clone;
    return 0;
}


static void memx_self_pageout_maps(MemXZone3 *s) {
    if (!s) return;
    if (s->sov_pidx_map && s->sov_pidx_map != MAP_FAILED && s->sov_pidx_map_bytes) {
        madvise(s->sov_pidx_map, (size_t)s->sov_pidx_map_bytes, MADV_DONTNEED);
        (void)madvise(s->sov_pidx_map, (size_t)s->sov_pidx_map_bytes, MADV_PAGEOUT);
    }
    if (s->sov_ents && s->sov_ents != MAP_FAILED && s->sov_bytes) {
#if defined(MADV_FREE_REUSABLE)
        madvise(s->sov_ents, (size_t)s->sov_bytes, MADV_FREE_REUSABLE);
#endif
        madvise(s->sov_ents, (size_t)s->sov_bytes, MADV_DONTNEED);
        (void)madvise(s->sov_ents, (size_t)s->sov_bytes, MADV_PAGEOUT);
    }
    if (s->sov_off_idx && s->sov_off_idx != MAP_FAILED && s->sov_off_idx_bytes) {
        madvise(s->sov_off_idx, (size_t)s->sov_off_idx_bytes, MADV_DONTNEED);
        (void)madvise(s->sov_off_idx, (size_t)s->sov_off_idx_bytes, MADV_PAGEOUT);
    }
    if (s->hot_list && s->hot_cap) {
        size_t bytes = (size_t)s->hot_cap * 4;
        madvise(s->hot_list, bytes, MADV_DONTNEED);
        (void)madvise(s->hot_list, bytes, MADV_PAGEOUT);
        s->hot_count = 0;
    }
    if (s->res_list && s->res_cap) {
        size_t bytes = (size_t)s->res_cap * 4;
        madvise(s->res_list, bytes, MADV_DONTNEED);
        (void)madvise(s->res_list, bytes, MADV_PAGEOUT);
        s->res_count = 0;
    }
    if (s->meta && s->npages) {
        size_t used = s->vmem_next / PAGE_SZ;
        if (used > s->npages) used = s->npages;
        if (used + 8 < s->npages)
            meta_release_physical_range(s, used + 8, s->npages - 1);
    }
}


int memx_runtime_trim(uint32_t flags, uint64_t *out_reclaimed_bytes) {
    if (!g_z) return ENOENT;
    if (g_z->phoenix_sealed) {
        if (out_reclaimed_bytes) *out_reclaimed_bytes = 0;
        return 0;
    }
    uint64_t reclaimed = 0;
    int hard = ((flags & 8u) != 0);
    int soft_sov_only = ((flags & 32u) != 0) && !hard && ((flags & (1u | 2u | 4u | 8u | 16u | 64u)) == 0);
    if (hard) __sync_lock_test_and_set(&g_hard_quiesce, 1);
    if (!soft_sov_only) mat_cache_release_physical();
    if ((flags & 8192u) != 0) {
        __sync_lock_test_and_set(&g_hard_quiesce, 1);
        mat_nd_shutdown();
        tca_pool_destroy();
        pthread_mutex_lock(&g_z->alloc_mutex);
        if ((flags & 512u) != 0 || (flags & 1024u) != 0 || (flags & 4096u) != 0) {
            if (g_z->vault_cache_bytes) reclaimed += g_z->vault_cache_bytes;
            pool_vault_cache_release_locked(g_z);
            sov_warp_buf_release_locked(g_z);
            if ((flags & 1024u) != 0) {
                uint64_t sb = g_z->sov_bytes + g_z->sov_off_idx_bytes + g_z->sov_pidx_map_bytes;
                sov_release_structural_locked(g_z);
                sov_drop_locked(g_z);
                reclaimed += sb;
            }
            if ((flags & 4096u) != 0 && g_z->vmem && g_z->vmem != MAP_FAILED && g_z->vmem_next > 0) {
                size_t bytes = (size_t)g_z->vmem_next;
                bytes = (bytes + PAGE_SZ - 1) & ~((size_t)PAGE_SZ - 1);
                if (bytes > 0 && bytes <= g_z->vmem_size) {
                    void *m = mmap(g_z->vmem, bytes, PROT_NONE,
                                   MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                    if (m != MAP_FAILED) reclaimed += bytes;
                }
            }
        }
        (void)memx_phoenix_seal_locked(g_z, &reclaimed);
        pthread_mutex_unlock(&g_z->alloc_mutex);
        tca_sticky_clear();
#if defined(__APPLE__)
        malloc_zone_pressure_relief(NULL, 0);
        malloc_zone_pressure_relief(NULL, 0);
#endif
        if (out_reclaimed_bytes) *out_reclaimed_bytes = reclaimed;
        return 0;
    }
    if ((flags & 512u) != 0) {
        mat_nd_shutdown();
        pthread_mutex_lock(&g_z->alloc_mutex);
        if (g_z->vault_cache_bytes) reclaimed += g_z->vault_cache_bytes;
        pool_vault_cache_release_locked(g_z);
        sov_warp_buf_release_locked(g_z);
        tca_pool_destroy();
        if ((flags & 1024u) != 0) {
            uint64_t sb = g_z->sov_bytes + g_z->sov_off_idx_bytes + g_z->sov_pidx_map_bytes;
            sov_release_structural_locked(g_z);
            sov_drop_locked(g_z);
            reclaimed += sb;
            g_z->sovereign = 0;
            g_z->sovereign_frozen = 1;
        }
        if ((flags & 4096u) != 0) {
            if (g_z->vmem && g_z->vmem != MAP_FAILED && g_z->vmem_next > 0) {
                size_t bytes = (size_t)g_z->vmem_next;
                bytes = (bytes + PAGE_SZ - 1) & ~((size_t)PAGE_SZ - 1);
                if (bytes > 0 && bytes <= g_z->vmem_size) {
                    void *m = mmap(g_z->vmem, bytes, PROT_NONE,
                                   MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                    if (m != MAP_FAILED) {
                        reclaimed += bytes;
                    } else {
#if defined(MADV_FREE_REUSABLE)
                        madvise(g_z->vmem, bytes, MADV_FREE_REUSABLE);
#endif
                        madvise(g_z->vmem, bytes, MADV_DONTNEED);
                        (void)madvise(g_z->vmem, bytes, MADV_PAGEOUT);
                        reclaimed += bytes;
                    }
                }
            }
            if (g_z->meta && g_z->npages) {
                size_t used = g_z->vmem_next / PAGE_SZ;
                if (used > g_z->npages) used = g_z->npages;
                if (used > 0) {
                    size_t mbytes = used * sizeof(PageMeta);
                    mbytes = (mbytes + PAGE_SZ - 1) & ~((size_t)PAGE_SZ - 1);
#if defined(MADV_FREE_REUSABLE)
                    madvise(g_z->meta, mbytes, MADV_FREE_REUSABLE);
#endif
                    madvise(g_z->meta, mbytes, MADV_DONTNEED);
                    (void)madvise(g_z->meta, mbytes, MADV_PAGEOUT);
                }
                if (used + 8 < g_z->npages)
                    meta_release_physical_range(g_z, used + 8, g_z->npages - 1);
            }
            memx_self_pageout_maps(g_z);
        }
        pthread_mutex_unlock(&g_z->alloc_mutex);
        tca_sticky_clear();
    }
    pthread_mutex_lock(&g_z->alloc_mutex);
    if (hard) {
        reclaimed += memx_runtime_reclaim_locked(g_z);
        int allow_compact = 1;
        const char *env_c = getenv("MEMX_HARD_COMPACT");
        if (env_c && env_c[0] == '0') allow_compact = 0;
        if (allow_compact) {
            uint64_t free_bytes = pool_free_extent_bytes_locked(g_z);
            uint64_t live = g_z->pool_next > free_bytes ? (g_z->pool_next - free_bytes) : 0;
            if (free_bytes >= (PAGE_SZ * 4096) ||
                (live > 0 && free_bytes >= (live / 3)) ||
                g_z->pool_free_count >= 64) {
                uint64_t moved = pool_compact_locked(g_z);
                if (moved) reclaimed += moved;
                reclaimed += memx_runtime_reclaim_locked(g_z);
            }
        }
    } else {
        reclaimed += memx_runtime_reclaim_and_compact_locked(g_z);
    }
    pool_release_all_free_extents_locked(g_z);
    release_all_compressed_physical_locked(g_z);
    if (hard) {
        g_z->hot_count = 0;
        g_z->res_count = 0;
        pool_hard_decommit_all_free_extents_locked(g_z);
        hard_release_all_compressed_physical_locked(g_z);
        if (g_z->pool && g_z->pool_next < g_z->pool_size)
            pool_hard_decommit_range(g_z, g_z->pool_next, g_z->pool_size - g_z->pool_next);
        {
            int want_spill = ((flags & 16u) != 0);
            const char *es = getenv("MEMX_POOL_SPILL");
            if (es && es[0] == '1') want_spill = 1;
            if (es && es[0] == '0') want_spill = 0;
            if (want_spill) g_pool_spill_force = 1;
            pool_pageout_live_locked(g_z);
            if (want_spill || pool_ghost_final_enabled() || pool_ghost_enabled() || g_z->pool_ghost) {
                int grc = 0;
                if (pool_ghost_final_enabled() || pool_ghost_enabled() || g_z->pool_ghost)
                    grc = pool_ghost_flush_locked(g_z);
                if (grc <= 0) {
                    int src = pool_spill_to_file_locked(g_z);
                    if (src > 0) grc = 1;
                }
                if (grc > 0) {
                    reclaimed += g_z->pool_spill_bytes ? g_z->pool_spill_bytes : g_z->pool_next;
                    pool_hard_decommit_all_free_extents_locked(g_z);
                    if (g_z->pool && g_z->pool_next < g_z->pool_size)
                        pool_hard_decommit_range(g_z, g_z->pool_next, g_z->pool_size - g_z->pool_next);
                    hard_release_all_compressed_physical_locked(g_z);
                    hard_release_all_compressed_physical_locked(g_z);
                    if (g_z->pool && g_z->pool_next > 0)
                        pool_ghost_detach_range_locked(g_z, 0, g_z->pool_next);
                    g_z->pool_detached = 1;
                    if (!g_z->pool_detached)
                        pool_pageout_live_locked(g_z);
                    (void)pool_vault_wbuf_flush_locked(g_z);
                    pool_vault_windows_drop_locked(g_z);
                    pool_vault_cache_release_locked(g_z);
                    pool_vault_wbuf_release_locked(g_z);
                    pool_vault_probe_locked(g_z);
                }
                g_pool_spill_force = 0;
            } else {
                pool_pageout_live_locked(g_z);
            }
        }
        if (g_z->tmp_src && g_z->tmp_src != MAP_FAILED && g_z->batch_cap) {
            size_t bytes = g_z->batch_cap * PAGE_SZ;
            void *m = mmap(g_z->tmp_src, bytes, PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (m == MAP_FAILED) madvise(g_z->tmp_src, bytes, MADV_DONTNEED);
        }
        if (g_z->tmp_dst && g_z->tmp_dst != MAP_FAILED && g_z->batch_cap) {
            size_t bytes = g_z->batch_cap * PAGE_SZ;
            void *m = mmap(g_z->tmp_dst, bytes, PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (m == MAP_FAILED) madvise(g_z->tmp_dst, bytes, MADV_DONTNEED);
        }
    }
    {
        int want_sov = ((flags & 32u) != 0) || (hard && sov_enabled_env());
        if (want_sov) {
            (void)pool_vault_wbuf_flush_locked(g_z);
            int ready = (g_z->pool_spill_fd > 2 && g_z->pool_spill_bytes >= g_z->pool_next && g_z->pool_next > 0);
            if (!ready && (g_z->pool_vault_native || g_z->pool_ghost || g_z->pool_detached)) {
                g_pool_spill_force = 1;
                if (pool_ghost_flush_locked(g_z) <= 0)
                    (void)pool_spill_to_file_locked(g_z);
                g_pool_spill_force = 0;
                ready = (g_z->pool_spill_fd > 2 && g_z->pool_spill_bytes >= g_z->pool_next && g_z->pool_next > 0);
            }
            if (ready || g_z->pool_next == 0) {
                if (sov_build_locked(g_z) == 0) {
                    hard_release_all_compressed_physical_locked(g_z);
                    hard_release_all_compressed_physical_locked(g_z);
                    int hard_sov = ((flags & 64u) != 0);
                    const char *eh = getenv("MEMX_SOVEREIGN_HARD");
                    if (eh && eh[0] == '1') hard_sov = 1;
                    if (eh && eh[0] == '0') hard_sov = 0;
                    if (hard_sov) sov_release_structural_locked(g_z);
                    if (soft_sov_only) {
                        sov_warm_vault_stream_locked(g_z);
                        reclaimed += g_z->sov_warm_bytes;
                        pool_vault_cache_avcs_locked(g_z);
                    }
                    if (((flags & 128u) != 0) && !soft_sov_only) {
                        sov_dissolve_cold_locked(g_z);
                        reclaimed += g_z->sov_bytes ? g_z->sov_bytes : 0;
                    }
                    reclaimed += g_z->sov_count ? ((uint64_t)g_z->sov_count * 24ull) : 0;
                    pool_vault_probe_locked(g_z);
                }
            }
        }
    }
    if ((flags & 1u) != 0) {
        size_t cap = g_z->batch_cap;
        size_t bytes = cap * PAGE_SZ;
        if (!hard) {
            if (g_z->tmp_src && g_z->tmp_src != MAP_FAILED && bytes > 0) {
                void *m = mmap(g_z->tmp_src, bytes, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (m == MAP_FAILED) {
#if defined(MADV_FREE_REUSABLE)
                    madvise(g_z->tmp_src, bytes, MADV_FREE_REUSABLE);
#endif
                    madvise(g_z->tmp_src, bytes, MADV_DONTNEED);
                }
            }
            if (g_z->tmp_dst && g_z->tmp_dst != MAP_FAILED && bytes > 0) {
                void *m = mmap(g_z->tmp_dst, bytes, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (m == MAP_FAILED) {
#if defined(MADV_FREE_REUSABLE)
                    madvise(g_z->tmp_dst, bytes, MADV_FREE_REUSABLE);
#endif
                    madvise(g_z->tmp_dst, bytes, MADV_DONTNEED);
                }
            }
            if (g_z->tmp_sz && g_z->tmp_sz != MAP_FAILED && cap > 0) {
                size_t zbytes = cap * 4;
                void *m = mmap(g_z->tmp_sz, zbytes, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (m == MAP_FAILED) madvise(g_z->tmp_sz, zbytes, MADV_DONTNEED);
            }
        } else {
            if (g_z->tmp_src && g_z->tmp_src != MAP_FAILED && bytes > 0) {
                void *m = mmap(g_z->tmp_src, bytes, PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (m == MAP_FAILED) madvise(g_z->tmp_src, bytes, MADV_DONTNEED);
            }
            if (g_z->tmp_dst && g_z->tmp_dst != MAP_FAILED && bytes > 0) {
                void *m = mmap(g_z->tmp_dst, bytes, PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (m == MAP_FAILED) madvise(g_z->tmp_dst, bytes, MADV_DONTNEED);
            }
            if (g_z->tmp_sz && g_z->tmp_sz != MAP_FAILED && cap > 0) {
                size_t zbytes = cap * 4;
                void *m = mmap(g_z->tmp_sz, zbytes, PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (m == MAP_FAILED) madvise(g_z->tmp_sz, zbytes, MADV_DONTNEED);
            }
        }
        if ((flags & 2u) != 0) {
            g_z->gpu_sb = nil;
            g_z->gpu_db = nil;
            g_z->gpu_zb = nil;
            if ((flags & 4u) != 0) {
                g_z->comp_pipe = nil;
                g_z->decomp_pipe = nil;
                g_z->queue = nil;
                g_z->device = nil;
            }
        } else {
            if (g_z->gpu_sb) {
                void *p = [g_z->gpu_sb contents];
                if (p) {
#if defined(MADV_FREE_REUSABLE)
                    madvise(p, bytes, MADV_FREE_REUSABLE);
#endif
                    madvise(p, bytes, MADV_DONTNEED);
                }
            }
            if (g_z->gpu_db) {
                void *p = [g_z->gpu_db contents];
                if (p) {
#if defined(MADV_FREE_REUSABLE)
                    madvise(p, bytes, MADV_FREE_REUSABLE);
#endif
                    madvise(p, bytes, MADV_DONTNEED);
                }
            }
        }
        if (g_z->hot_list && g_z->hot_cap > 0) {
            size_t used = (size_t)g_z->hot_count * 4;
            size_t oldb = (size_t)g_z->hot_cap * 4;
            size_t start = (used + PAGE_SZ - 1) & ~((size_t)PAGE_SZ - 1);
            if (start < oldb) {
                void *m = mmap((uint8_t*)g_z->hot_list + start, oldb - start, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (m == MAP_FAILED) madvise((uint8_t*)g_z->hot_list + start, oldb - start, MADV_DONTNEED);
            }
        }
        if (g_z->res_list && g_z->res_cap > 0) {
            size_t used = (size_t)g_z->res_count * 4;
            size_t oldb = (size_t)g_z->res_cap * 4;
            size_t start = (used + PAGE_SZ - 1) & ~((size_t)PAGE_SZ - 1);
            if (start < oldb) {
                void *m = mmap((uint8_t*)g_z->res_list + start, oldb - start, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (m == MAP_FAILED) madvise((uint8_t*)g_z->res_list + start, oldb - start, MADV_DONTNEED);
            }
        }
        if (g_z->pool && g_z->pool_next < g_z->pool_size)
            pool_release_physical_range(g_z, g_z->pool_next, g_z->pool_size - g_z->pool_next);
        pool_release_all_free_extents_locked(g_z);
        release_all_compressed_physical_locked(g_z);
        if (hard) {
            if (g_z->pool && g_z->pool_next < g_z->pool_size)
                pool_hard_decommit_range(g_z, g_z->pool_next, g_z->pool_size - g_z->pool_next);
            pool_hard_decommit_all_free_extents_locked(g_z);
            hard_release_all_compressed_physical_locked(g_z);
            pool_pageout_live_locked(g_z);
        }
        if (g_z->meta && g_z->npages > 0) {
            size_t used_pages = g_z->vmem_next / PAGE_SZ;
            if (used_pages + 64 < g_z->npages) {
                size_t first = used_pages + 64;
                first = (first + 63) & ~((size_t)63);
                if (first < g_z->npages)
                    meta_release_physical_range(g_z, first, g_z->npages - 1);
            }
            if (used_pages > 0) {
                uintptr_t a0 = ((uintptr_t)&g_z->meta[0]) & ~((uintptr_t)PAGE_SZ - 1);
                uintptr_t a1 = ((uintptr_t)&g_z->meta[used_pages - 1] + sizeof(PageMeta) + PAGE_SZ - 1) & ~((uintptr_t)PAGE_SZ - 1);
                if (a1 > a0) {
                    (void)madvise((void*)a0, a1 - a0, MADV_PAGEOUT);
                }
            }
        }
        if (g_z->dedup_rev && g_z->dedup_rev != MAP_FAILED && g_z->dedup_rev_size > 0) {
            size_t rbytes = (size_t)g_z->dedup_rev_size * 4;
            size_t keep = 0;
            if (g_z->pool_next > 0) {
                keep = ((g_z->pool_next / PAGE_SZ) + 1) * 4;
                keep = (keep + PAGE_SZ - 1) & ~((size_t)PAGE_SZ - 1);
            }
            if (keep + PAGE_SZ < rbytes) {
                void *m = mmap((uint8_t*)g_z->dedup_rev + keep, rbytes - keep, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (m == MAP_FAILED) madvise((uint8_t*)g_z->dedup_rev + keep, rbytes - keep, MADV_DONTNEED);
            }
        }
        if (g_z->free_bm && g_z->free_bm_size > 0) {
            size_t used_pages = g_z->vmem_next / PAGE_SZ;
            if (used_pages < g_z->npages) {
                size_t word0 = (used_pages + 63) / 64;
                size_t start = word0 * sizeof(uint64_t);
                start = (start + PAGE_SZ - 1) & ~((size_t)PAGE_SZ - 1);
                size_t oldb = g_z->free_bm_size * sizeof(uint64_t);
                if (start < oldb) {
                    void *m = mmap((uint8_t*)g_z->free_bm + start, oldb - start, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                    if (m == MAP_FAILED) madvise((uint8_t*)g_z->free_bm + start, oldb - start, MADV_DONTNEED);
                }
            }
        }
    }
    if ((flags & 2048u) != 0) {
        memx_self_pageout_maps(g_z);
        tca_sticky_clear();
    }
    pthread_mutex_unlock(&g_z->alloc_mutex);
#if defined(__APPLE__)
    malloc_zone_pressure_relief(NULL, 0);
    if (hard) malloc_zone_pressure_relief(NULL, 0);
#endif
    if (hard) {
        __sync_lock_release(&g_hard_quiesce);
    }
    if (out_reclaimed_bytes) *out_reclaimed_bytes = reclaimed;
    return 0;
}


int memx_runtime_recompress_begin(uint32_t zlib_level) {
    if (!g_z) return ENOENT;
    int lv = (zlib_level >= 1 && zlib_level <= 9) ? (int)zlib_level : 6;
    (void)memx_deflate_level_push(lv);
    g_recompress_mode = 1;
    return 0;
}

int memx_runtime_recompress_end(uint64_t *out_reclaimed_bytes) {
    if (!g_z) return ENOENT;
    g_recompress_mode = 0;
    memx_deflate_level_push(1);
    mat_cache_invalidate();
    __sync_lock_test_and_set(&g_hard_quiesce, 1);
    pthread_mutex_lock(&g_z->alloc_mutex);
    uint64_t reclaimed = memx_runtime_reclaim_and_compact_locked(g_z);
    pool_release_all_free_extents_locked(g_z);
    release_all_compressed_physical_locked(g_z);
    hard_release_all_compressed_physical_locked(g_z);
    pool_hard_decommit_all_free_extents_locked(g_z);
    if (g_z->pool && g_z->pool_next < g_z->pool_size)
        pool_hard_decommit_range(g_z, g_z->pool_next, g_z->pool_size - g_z->pool_next);
    pool_pageout_live_locked(g_z);
    pthread_mutex_unlock(&g_z->alloc_mutex);
    __sync_lock_release(&g_hard_quiesce);
#if defined(__APPLE__)
    malloc_zone_pressure_relief(NULL, 0);
#endif
    if (out_reclaimed_bytes) *out_reclaimed_bytes = reclaimed;
    return 0;
}

int memx_runtime_context_recompress_range(memx_runtime_context_t *ctx, void *ptr, size_t offset, size_t length, uint64_t *out_pages) {
    if (!ctx || ctx->magic != MEMX_CONTEXT_MAGIC || !ptr || !g_z || !is_ours(ptr)) return EINVAL;
    if (length == 0) {
        if (out_pages) *out_pages = 0;
        return 0;
    }
    size_t sp = ((uintptr_t)ptr - (uintptr_t)g_z->vmem) / PAGE_SZ;
    if (sp >= g_z->npages || g_z->meta[sp].owner_tag != (uintptr_t)ctx || g_z->meta[sp].alloc_size == 0)
        return EINVAL;
    size_t size = g_z->meta[sp].alloc_size;
    if (offset >= size || length > size - offset) return EINVAL;
    size_t first = sp + offset / PAGE_SZ;
    size_t last = sp + (offset + length - 1) / PAGE_SZ;
    int prev = g_recompress_mode;
    g_recompress_mode = 1;
    uint64_t done = 0;
    for (size_t i = first; i <= last; i++) {
        PageMeta *m = &g_z->meta[i];
        if (m->tensor_role != MEMX_TENSOR_ROLE_WEIGHT && m->tensor_role != MEMX_TENSOR_ROLE_EMBEDDING)
            continue;
        m->tensor_flags |= (MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY);
        m->tensor_flags &= ~MEMX_TENSOR_FLAG_HOT;
        if (m->state == PAGE_COMPRESSED && m->comp_size > 0 && m->comp_size <= (PAGE_SZ / 2))
            continue;
        if (force_compress_page_now(g_z, i)) done++;
    }
    g_recompress_mode = prev;
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
    if (out_pages) *out_pages = done;
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
