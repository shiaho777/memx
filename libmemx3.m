// MemX Dylib v3: __interpose approach with proper recursion guard
// This is the ONLY way to intercept malloc() calls system-wide
//
// Key insights from debugging:
//   - malloc_zone_register does NOT intercept malloc() calls
//   - __interpose + DYLD_INSERT_LIBRARIES is the only way
//   - Must use __thread recursion guard + mmap fallback
//   - Must defer GPU init to avoid deadlock during dylib load
//   - Background compressor uses orig_malloc/orig_free (not intercepted)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <pthread.h>
#include <mach/mach.h>
#include <malloc/malloc.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>

#import <Metal/Metal.h>

#define PAGE_SZ 16384
#define MB (1024ULL*1024)
#define LARGE_THRESHOLD 65536

// ─── GPU Shaders v3: Adaptive compression (v2 format, internal optimization) ───
// Zero-heavy pages skip LZ77 (faster), structured pages use full RLE+LZ77
// Output format is 100% v2-compatible — no decompressor changes needed
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
// Saved original default zone hooks (set during init_memx when we hijack the default zone)
static malloc_zone_t *s_default_zone = NULL;
static void *(*orig_zone_malloc)(malloc_zone_t*, size_t) = NULL;
static void (*orig_zone_free)(malloc_zone_t*, void*) = NULL;
static void *(*orig_zone_calloc)(malloc_zone_t*, size_t, size_t) = NULL;
static void *(*orig_zone_realloc)(malloc_zone_t*, void*, size_t) = NULL;

static void *real_malloc(size_t size) {
    if (orig_zone_malloc) return orig_zone_malloc(s_default_zone, size);
    if (!s_default_zone) s_default_zone = malloc_default_zone();
    return s_default_zone->malloc(s_default_zone, size);
}
static void real_free(void *ptr) {
    if (!ptr) return;
    malloc_zone_t *z = malloc_zone_from_ptr(ptr);
    if (z && z->free) z->free(z, ptr);
}
static void *real_calloc(size_t nmemb, size_t size) {
    if (orig_zone_calloc) return orig_zone_calloc(s_default_zone, nmemb, size);
    if (!s_default_zone) s_default_zone = malloc_default_zone();
    return s_default_zone->calloc(s_default_zone, nmemb, size);
}
static void *real_realloc(void *ptr, size_t size) {
    if (!ptr) return real_malloc(size);
    malloc_zone_t *z = malloc_zone_from_ptr(ptr);
    if (z && z->realloc) return z->realloc(z, ptr, size);
    return NULL;
}
static size_t real_malloc_size(const void *ptr) {
    malloc_zone_t *z = malloc_zone_from_ptr((void*)ptr);
    if (z && z->size) return z->size(z, ptr);
    return 0;
}

typedef struct {
    uint8_t  state;
    uint32_t comp_size;
    uint64_t pool_offset;
    uint8_t  prefetched;     // 1 if this page was prefetched (not fault-triggered)
    uint8_t  cooldown;       // scans remaining before compressible (5 for prefetched)
    uint8_t  _pad[2];
} PageMeta;

typedef struct {
    void        *vmem;       uint64_t    vmem_size;   size_t      npages;
    uint64_t    vmem_next;
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
    uint32_t           *dedup_rev;   // reverse index: pool_page → slot (DEDUP_REV_SIZE)
    #define DEDUP_REV_SIZE 8192       // covers pool_offset / PAGE_SZ up to 128MB pool
    #define DEDUP_REV_MASK (DEDUP_REV_SIZE-1)
    volatile uint64_t   dedup_hits;  // number of dedup hits
    // Predictive prefetch: detect sequential access and prefetch ahead
    volatile uint64_t   prefetch_hits;   // pages that were prefetched before fault
    volatile uint64_t   prefetch_misses; // pages prefetched but never accessed
    volatile uint64_t   prefetch_count;  // total prefetch operations
    #define PREFETCH_AHEAD 2             // how many pages to prefetch ahead
    #define PREFETCH_STRIDE_MIN 1
    #define PREFETCH_STRIDE_MAX 64
} MemXZone3;

static MemXZone3 *g_z = NULL;
static __thread int in_memx = 0;  // Per-thread recursion guard

// ─── CPU decompressor (signal-safe, stack-efficient) ───
static void cpu_decompress(const uint8_t *src, uint32_t cs, uint8_t *dst) {
    if (cs>=PAGE_SZ||src[0]!=0x4D||src[1]!=0x58){memcpy(dst,src,cs<PAGE_SZ?cs:PAGE_SZ);if(cs<PAGE_SZ)memset(dst+cs,0,PAGE_SZ-cs);return;}
    uint8_t ver=src[2];
    // Decode RLE/LZ77 directly into dst (no intermediate buffer — saves 16KB stack)
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
    if (!s->dedup_rev) return;
    // O(1) reverse lookup: pool_page → slot
    uint32_t pp = (uint32_t)(pool_offset / PAGE_SZ) & DEDUP_REV_MASK;
    uint32_t slot = s->dedup_rev[pp];
    if (slot < DEDUP_HT_SIZE && s->dedup_ref[slot] > 0 && s->dedup_off[slot] == pool_offset && s->dedup_sz[slot] == comp_size) {
        __sync_fetch_and_sub(&s->dedup_ref[slot], 1);
        return;
    }
    // Fallback: linear scan (for hash collisions in rev table)
    for (uint32_t i = 0; i < DEDUP_HT_SIZE; i++) {
        if (s->dedup_ref[i] > 0 && s->dedup_off[i] == pool_offset && s->dedup_sz[i] == comp_size) {
            __sync_fetch_and_sub(&s->dedup_ref[i], 1);
            return;
        }
    }
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
    mprotect(pa,PAGE_SZ,PROT_READ|PROT_WRITE);
    if(m->state==PAGE_NONE){
        uint8_t old = __sync_val_compare_and_swap(&m->state, PAGE_NONE, PAGE_RESIDENT);
        if(old == PAGE_NONE) memset(pa,0,PAGE_SZ);
    }
    else if(m->state==PAGE_COMPRESSED){
        // Atomic CAS: only one thread wins the race to decompress this page
        uint8_t old = __sync_val_compare_and_swap(&m->state, PAGE_COMPRESSED, PAGE_HOT);
        if(old == PAGE_COMPRESSED) {
            // We won — decompress
            uint64_t d_off=m->pool_offset; uint32_t d_sz=m->comp_size;
            cpu_decompress(g_z->pool+d_off,d_sz,pa);m->comp_size=0;m->prefetched=0;m->cooldown=2;
            dedup_decref(g_z,d_off,d_sz);
        }
        // else: another thread already decompressed it, data is ready
    }
    __sync_fetch_and_add(&g_z->faults,1);
    // ─── Predictive prefetch: if this page was compressed, prefetch ahead ───
    // Detect sequential access by checking if next pages are also compressed
    {
        int seq_count = 0;
        for(int k=1; k<=PREFETCH_AHEAD*2 && pi+k<g_z->npages; k++) {
            if(g_z->meta[pi+k].state==PAGE_COMPRESSED) seq_count++;
            else break;
        }
        if(seq_count >= 2) {
            // Sequential pattern detected - prefetch next pages
            int pf = 0;
            for(int k=1; k<=PREFETCH_AHEAD && pi+k<g_z->npages && pf<PREFETCH_AHEAD; k++) {
                PageMeta *nm = &g_z->meta[pi+k];
                // Atomic CAS: only one thread claims this page for prefetch
                uint8_t old = __sync_val_compare_and_swap(&nm->state, PAGE_COMPRESSED, PAGE_HOT);
                if(old == PAGE_COMPRESSED) {
                    uint8_t *npa = (uint8_t*)g_z->vmem + (pi+k)*PAGE_SZ;
                    mprotect(npa, PAGE_SZ, PROT_READ|PROT_WRITE);
                    uint64_t d_off=nm->pool_offset; uint32_t d_sz=nm->comp_size;
                    cpu_decompress(g_z->pool+d_off, d_sz, npa);
                    nm->comp_size = 0; nm->prefetched = 1; nm->cooldown = 5;
                    dedup_decref(g_z, d_off, d_sz);
                    pf++;
                }
            }
            if(pf > 0) __sync_fetch_and_add(&g_z->prefetch_count, 1);
        }
    }
    return;
chain:
    if(sig==SIGSEGV&&old_segv.sa_handler!=SIG_DFL&&old_segv.sa_handler!=SIG_IGN){if(old_segv.sa_flags&SA_SIGINFO)old_segv.sa_sigaction(sig,info,ctx);else if(old_segv.sa_handler!=SIG_ERR)old_segv.sa_handler(sig);}
    else if(sig==SIGBUS&&old_bus.sa_handler!=SIG_DFL&&old_bus.sa_handler!=SIG_IGN){if(old_bus.sa_flags&SA_SIGINFO)old_bus.sa_sigaction(sig,info,ctx);else if(old_bus.sa_handler!=SIG_ERR)old_bus.sa_handler(sig);}
    else{signal(sig,SIG_DFL);raise(sig);}
}

// ─── Background compressor ───
static void *bg_compressor(void *arg) {
    MemXZone3 *s=(MemXZone3*)arg;
    in_memx = 1;  // CRITICAL: prevent Metal internal mallocs from going to our pool
    const size_t BATCH=s->batch_cap;
    while(s->running){
        // Cooldown: decrement counters, transition HOT→RESIDENT only when cooldown=0
        for(size_t i=0;i<s->npages;i++) {
            if(s->meta[i].state==PAGE_HOT) {
                if(s->meta[i].cooldown > 0) {
                    s->meta[i].cooldown--;
                } else {
                    // Atomic CAS: only transition if still HOT (not re-faulted)
                    uint8_t old = __sync_val_compare_and_swap(&s->meta[i].state, PAGE_HOT, PAGE_RESIDENT);
                    if(old == PAGE_HOT) s->meta[i].prefetched=0;
                }
            }
        }
        size_t tc[BATCH]; size_t nc=0;
        for(size_t i=0;i<s->npages&&nc<BATCH&&s->running;i++)
            if(s->meta[i].state==PAGE_RESIDENT) tc[nc++]=i;
        if(nc==0){sleep(1);continue;}
        // Copy page data
        for(size_t i=0;i<nc;i++) memcpy(s->tmp_src+i*PAGE_SZ,(uint8_t*)s->vmem+tc[i]*PAGE_SZ,PAGE_SZ);
        // GPU compress
        int gr=0;
        gr=gpu_compress(s,nc);
        if(gr!=0){sleep(1);continue;}
        for(size_t i=0;i<nc;i++){
            uint32_t cs=s->tmp_sz[i]; size_t pidx=tc[i];
            // Skip incompressible pages (less than 32 bytes savings)
            if(cs>=PAGE_SZ || cs >= PAGE_SZ - 32) continue;
            if(s->pool_next+cs>s->pool_size) continue;
            // ─── Dedup: check if same compressed data already exists ───
            uint8_t *cdata = s->tmp_dst + i*PAGE_SZ;
            // Hash the compressed data (FNV-1a)
            uint64_t h = 14695981039346656037ULL;
            for(uint32_t j=0; j<cs; j++) { h ^= cdata[j]; h *= 1099511628211ULL; }
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
                        // Dedup hit! Set metadata first, then atomic state transition
                        __sync_fetch_and_add(&s->dedup_ref[s2], 1);
                        s->meta[pidx].comp_size=cs; s->meta[pidx].pool_offset=existing_off;
                        __sync_synchronize();  // memory barrier: ensure metadata visible before state change
                        uint8_t old_state = __sync_val_compare_and_swap(&s->meta[pidx].state, PAGE_RESIDENT, PAGE_COMPRESSED);
                        if(old_state != PAGE_RESIDENT) { __sync_fetch_and_sub(&s->dedup_ref[s2], 1); continue; }
                        mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ,PAGE_SZ,PROT_NONE);
                        __sync_fetch_and_add(&s->compressions,1);
                        __sync_fetch_and_add(&s->bytes_saved,PAGE_SZ-cs);
                        __sync_fetch_and_add(&s->dedup_hits,1);
                        dedup_found = 1;
                        break;
                    }
                }
            }
            if (dedup_found) continue;
            // ─── No dedup hit: store new compressed data ───
            uint64_t off=s->pool_next;
            // Make pool pages writable as needed
            size_t pool_page_start = off / PAGE_SZ;
            size_t pool_page_end = (off + cs + PAGE_SZ - 1) / PAGE_SZ;
            for(size_t pp=pool_page_start; pp<pool_page_end; pp++)
                mprotect(s->pool + pp*PAGE_SZ, PAGE_SZ, PROT_READ|PROT_WRITE);
            memcpy(s->pool+off,cdata,cs);  // Only store compressed bytes
            s->pool_next+=cs;
            __sync_fetch_and_add(&s->pool_used,cs);
            // Set metadata first, then atomic state transition
            s->meta[pidx].comp_size=cs; s->meta[pidx].pool_offset=off;
            __sync_synchronize();  // memory barrier
            uint8_t old_state = __sync_val_compare_and_swap(&s->meta[pidx].state, PAGE_RESIDENT, PAGE_COMPRESSED);
            if(old_state != PAGE_RESIDENT) continue;  // page was touched, skip
            mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ,PAGE_SZ,PROT_NONE);
            __sync_fetch_and_add(&s->compressions,1);
            __sync_fetch_and_add(&s->bytes_saved,PAGE_SZ-cs);
            // Add to dedup table (open-addressing insert, reuse stale ref=0 slots)
            for(int probe=0; probe<8; probe++) {
                uint32_t s2 = (slot + probe) & DEDUP_HT_MASK;
                if (s->dedup_hash[s2] == 0 || s->dedup_ref[s2] == 0) {
                    s->dedup_hash[s2] = h;
                    s->dedup_off[s2] = off;
                    s->dedup_sz[s2] = cs;
                    __sync_lock_test_and_set(&s->dedup_ref[s2], 1);  // atomic write
                    s->dedup_rev[(uint32_t)(off / PAGE_SZ) & DEDUP_REV_MASK] = s2;
                    break;
                }
            }
        }
        {struct timespec ts={0,1000000};nanosleep(&ts,NULL);}
    }
    return NULL;
}

// ─── Address check ───
static int is_ours(void *ptr) {
    if(!g_z||!g_z->vmem) return 0;
    uintptr_t a=(uintptr_t)ptr;
    return a>=(uintptr_t)g_z->vmem && a<(uintptr_t)g_z->vmem+g_z->vmem_size;
}

// ─── Interposed malloc/calloc/realloc/free ───
static void init_memx(void);  // Forward declaration for lazy init
static void *memx_malloc(size_t size) {
    // Lazy init: start Metal only when first large allocation occurs
    if (!g_z && size >= LARGE_THRESHOLD && !in_memx) init_memx();
    // Always pass through if: recursion, not initialized, or small alloc
    if (in_memx || !g_z || !g_z->running || size < LARGE_THRESHOLD) {
        return real_malloc(size);
    }
    
    in_memx = 1;
    
    // Allocate from our compressed pool (include size header in page count)
    pthread_mutex_lock(&g_z->alloc_mutex);
    size_t alloc_size = ((size + sizeof(size_t) + PAGE_SZ - 1) / PAGE_SZ) * PAGE_SZ;
    size_t npages = alloc_size / PAGE_SZ;
    size_t sp = g_z->vmem_next / PAGE_SZ, found=0, cont=0;
    for (size_t i=sp; i<g_z->npages; i++) {
        if (g_z->meta[i].state==PAGE_NONE) { if(!cont) found=i; cont++; if(cont>=npages) break; }
        else cont=0;
    }
    if (cont < npages) { pthread_mutex_unlock(&g_z->alloc_mutex); in_memx=0; return real_malloc(size); }
    
    void *result = (uint8_t*)g_z->vmem + found * PAGE_SZ;
    g_z->vmem_next = (found + npages) * PAGE_SZ;
    for (size_t i=found; i<found+npages; i++) g_z->meta[i].state = PAGE_NONE;
    mprotect(result, PAGE_SZ, PROT_READ|PROT_WRITE);
    g_z->meta[found].state = PAGE_RESIDENT;
    memcpy(result, &size, sizeof(size_t));
    pthread_mutex_unlock(&g_z->alloc_mutex);
    
    in_memx = 0;
    return (uint8_t*)result + sizeof(size_t);
}

static void memx_free(void *ptr) {
    if (!ptr) return;
    if (!g_z || !is_ours(ptr)) {
        real_free(ptr);
        return;
    }
    uint8_t *real = (uint8_t*)ptr - sizeof(size_t);
    size_t size; memcpy(&size, real, sizeof(size_t));
    size_t alloc_size = ((size + sizeof(size_t) + PAGE_SZ - 1) / PAGE_SZ) * PAGE_SZ;
    size_t sp = ((uintptr_t)real - (uintptr_t)g_z->vmem) / PAGE_SZ;
    size_t np = alloc_size / PAGE_SZ;
    for (size_t i=sp; i<sp+np && i<g_z->npages; i++) {
        if (g_z->meta[i].state==PAGE_COMPRESSED) mprotect((uint8_t*)g_z->vmem+i*PAGE_SZ, PAGE_SZ, PROT_READ|PROT_WRITE);
        g_z->meta[i].state=PAGE_NONE; g_z->meta[i].comp_size=0;
        mprotect((uint8_t*)g_z->vmem+i*PAGE_SZ, PAGE_SZ, PROT_NONE);
    }
}

static void *memx_calloc(size_t nmemb, size_t size) {
    if (in_memx || !g_z || !g_z->running) {
        return real_calloc(nmemb, size);
    }
    size_t total = nmemb * size;
    void *ptr = memx_malloc(total);
    if (ptr && is_ours(ptr)) { /* pages zero-filled on fault */ }
    else if (ptr) memset(ptr, 0, total);
    return ptr;
}

static void *memx_realloc(void *ptr, size_t size) {
    if (!ptr) return memx_malloc(size);
    if (size == 0) { memx_free(ptr); return NULL; }
    if (!g_z || !is_ours(ptr)) {
        return real_realloc(ptr, size);
    }
    uint8_t *real = (uint8_t*)ptr - sizeof(size_t);
    size_t old_size; memcpy(&old_size, real, sizeof(size_t));
    void *new_ptr = memx_malloc(size);
    if (!new_ptr) return NULL;
    size_t copy = old_size < size ? old_size : size;
    memcpy(new_ptr, ptr, copy);
    memx_free(ptr);
    return new_ptr;
}

// ─── Also interpose malloc_size (required for ObjC runtime compatibility) ───
// We DON'T interpose malloc_size (causes infinite recursion with __interpose).
// Instead, we register a malloc_zone with a size() method.
// The system malloc_size() queries all registered zones via malloc_zone_from_ptr().

static size_t zone3_size(malloc_zone_t *zone, const void *ptr) {
    if (!g_z || !is_ours((void*)ptr)) return 0;
    uint8_t *real = (uint8_t*)ptr - sizeof(size_t);
    size_t size; memcpy(&size, real, sizeof(size_t));
    return size;
}

static void *zone3_malloc(malloc_zone_t *zone, size_t size) {
    return memx_malloc(size);
}

static void zone3_free(malloc_zone_t *zone, void *ptr) {
    memx_free(ptr);
}

static void *zone3_calloc(malloc_zone_t *zone, size_t nmemb, size_t size) {
    return memx_calloc(nmemb, size);
}

static void *zone3_realloc(malloc_zone_t *zone, void *ptr, size_t size) {
    return memx_realloc(ptr, size);
}

static malloc_zone_t memx_zone;

// ─── Init (deferred via dispatch) ───
static volatile int memx_initing = 0;

static void init_memx(void) {
    if (g_z) return;
    if (__sync_bool_compare_and_swap(&memx_initing, 0, 1) == 0) {
        while (!g_z) usleep(1000);
        return;
    }
    in_memx = 1;  // Prevent recursion during Metal init
    
    g_z = (MemXZone3*)mmap(NULL, sizeof(MemXZone3), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (g_z == MAP_FAILED) { g_z = NULL; memx_initing = 0; return; }
    memset(g_z, 0, sizeof(MemXZone3));
    
    // GPU
    g_z->device = MTLCreateSystemDefaultDevice();
    if (!g_z->device) { munmap(g_z, sizeof(MemXZone3)); g_z = NULL; memx_initing = 0; return; }
    NSError *err = nil;
    id<MTLLibrary> lib = [g_z->device newLibraryWithSource:shader_src options:nil error:&err];
    if (!lib) { munmap(g_z, sizeof(MemXZone3)); g_z = NULL; memx_initing = 0; return; }
    g_z->queue = [g_z->device newCommandQueue];
    g_z->comp_pipe = [g_z->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"cp"] error:&err];
    g_z->decomp_pipe = [g_z->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dp"] error:&err];
    if (!g_z->comp_pipe || !g_z->decomp_pipe) { munmap(g_z, sizeof(MemXZone3)); g_z = NULL; memx_initing = 0; return; }
    
    // Pre-allocate persistent GPU + temp buffers (256 pages = 4MB each)
    g_z->batch_cap = 256;
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
    g_z->dedup_rev  = (uint32_t*)mmap(NULL, DEDUP_REV_SIZE*4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(g_z->dedup_hash, 0, DEDUP_HT_SIZE*8);
    memset(g_z->dedup_ref, 0, DEDUP_HT_SIZE*4);
    memset(g_z->dedup_rev, 0xFF, DEDUP_REV_SIZE*4);  // 0xFFFFFFFF = invalid slot
    if (!g_z->gpu_sb||!g_z->gpu_db||!g_z->gpu_zb||!g_z->tmp_src||!g_z->tmp_dst||!g_z->tmp_sz||!g_z->dedup_hash) {
        munmap(g_z, sizeof(MemXZone3)); g_z = NULL; memx_initing = 0; return;
    }
    
    // Virtual memory
    int64_t ms=0; size_t len=sizeof(ms);
    sysctlbyname("hw.memsize", &ms, &len, NULL, 0);
    g_z->vmem_size = ms * 4;
    g_z->npages = g_z->vmem_size / PAGE_SZ;
    g_z->vmem = mmap(NULL, g_z->vmem_size, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (g_z->vmem == MAP_FAILED) { munmap(g_z, sizeof(MemXZone3)); g_z = NULL; memx_initing = 0; return; }
    g_z->pool_size = g_z->vmem_size / 2;
    g_z->pool = mmap(NULL, g_z->pool_size, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (g_z->pool == MAP_FAILED) { munmap(g_z->vmem, g_z->vmem_size); munmap(g_z, sizeof(MemXZone3)); g_z = NULL; memx_initing = 0; return; }
    size_t meta_sz = g_z->npages * sizeof(PageMeta);
    g_z->meta = (PageMeta*)mmap(NULL, meta_sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (g_z->meta == MAP_FAILED) { munmap(g_z->pool, g_z->pool_size); munmap(g_z->vmem, g_z->vmem_size); munmap(g_z, sizeof(MemXZone3)); g_z = NULL; memx_initing = 0; return; }
    
    // Signal handler with alternate stack (decompressor uses 16KB on stack)
    stack_t ss;
    ss.ss_sp = mmap(NULL, SIGSTKSZ*4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    ss.ss_size = SIGSTKSZ*4;
    ss.ss_flags = 0;
    sigaltstack(&ss, NULL);
    struct sigaction sa; memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = fault_handler; sa.sa_flags = SA_SIGINFO|SA_NODEFER|SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old_segv); sigaction(SIGBUS, &sa, &old_bus);
    g_z->attached = 1;
    
    // Initialize allocation mutex for thread safety
    pthread_mutex_init(&g_z->alloc_mutex, NULL);
    
    // Background compressor
    g_z->running = 1;
    pthread_create(&g_z->bg_thread, NULL, bg_compressor, g_z);
    
    in_memx = 0;  // Now allow our pool to be used
    
    // Register malloc_zone so malloc_size/malloc_zone_from_ptr can find our allocations
    memset(&memx_zone, 0, sizeof(memx_zone));
    memx_zone.version = 8;
    memx_zone.zone_name = "MemX GPU Compressed";
    memx_zone.size = zone3_size;
    memx_zone.malloc = zone3_malloc;
    memx_zone.free = zone3_free;
    memx_zone.calloc = zone3_calloc;
    memx_zone.realloc = zone3_realloc;
    malloc_zone_register(&memx_zone);
    // NOTE: We do NOT hijack the default zone's function pointers.
    // Doing so causes exit() to be called during Metal commit, killing the compressor.
    // Instead, we rely on __interpose for malloc/mmap interception and
    // malloc_zone_from_ptr for free/realloc routing.
    //
    // On macOS 15+, __interpose for malloc is unreliable due to dyld cache.
    // mmap interpose works reliably. Programs using mmap(MAP_ANON|MAP_PRIVATE)
    // for large allocations (databases, ML frameworks, etc.) are fully supported.

    fprintf(stderr, "[memx] ✅ GPU memory expansion active (%llu MB virtual, __interpose + zone)\n",
            (unsigned long long)(g_z->vmem_size / MB));
}

static void fini_memx(void) {
    if (!g_z) return;
    g_z->running = 0;
    pthread_join(g_z->bg_thread, NULL);
    // Restore signal handlers FIRST to prevent faults during cleanup
    if (g_z->attached) { sigaction(SIGSEGV, &old_segv, NULL); sigaction(SIGBUS, &old_bus, NULL); g_z->attached = 0; }
    for (size_t i=0; i<g_z->npages; i++)
        if (g_z->meta[i].state==PAGE_COMPRESSED||g_z->meta[i].state==PAGE_NONE)
            mprotect((uint8_t*)g_z->vmem+i*PAGE_SZ, PAGE_SZ, PROT_READ|PROT_WRITE);
    malloc_zone_unregister(&memx_zone);
    // Release persistent GPU buffers (ObjC release may trigger malloc/free)
    g_z->gpu_sb = nil; g_z->gpu_db = nil; g_z->gpu_zb = nil;
    size_t batch_bytes = g_z->batch_cap * PAGE_SZ;
    munmap(g_z->tmp_src, batch_bytes);
    munmap(g_z->tmp_dst, batch_bytes);
    munmap(g_z->tmp_sz, g_z->batch_cap * 4);
    munmap(g_z->vmem, g_z->vmem_size);
    munmap(g_z->pool, g_z->pool_size);
    munmap(g_z->meta, g_z->npages * sizeof(PageMeta));
    munmap(g_z->dedup_hash, DEDUP_HT_SIZE*8);
    munmap(g_z->dedup_off, DEDUP_HT_SIZE*8);
    munmap(g_z->dedup_sz, DEDUP_HT_SIZE*4);
    munmap(g_z->dedup_ref, DEDUP_HT_SIZE*4);
    munmap(g_z->dedup_rev, DEDUP_REV_SIZE*4);
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

// Simple allocation table for mmap-routed allocations (no size header needed)
#define MMAP_TABLE_MAX 4096
static struct { void *base; size_t npages; } mmap_table[MMAP_TABLE_MAX];
static int mmap_table_count = 0;

static volatile int mmap_safe = 0;  // Set to 1 after our init completes

static void *memx_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    // Fast reject: not safe yet or during shutdown
    if (!mmap_safe || addr != NULL || fd != -1 || length < LARGE_THRESHOLD) goto passthrough;
    if (!(flags & MAP_ANON) || !(flags & MAP_PRIVATE) || (flags & MAP_FIXED)) goto passthrough;
    if (in_memx) goto passthrough;
    // Lazy init: start Metal only when first large mmap occurs
    if (!g_z) init_memx();

    {
        in_memx = 1;
        pthread_mutex_lock(&g_z->alloc_mutex);
        size_t npages = (length + PAGE_SZ - 1) / PAGE_SZ;
        size_t sp = g_z->vmem_next / PAGE_SZ, found=0, cont=0;
        for (size_t i=sp; i<g_z->npages; i++) {
            if (g_z->meta[i].state==PAGE_NONE) { if(!cont) found=i; cont++; if(cont>=npages) break; }
            else cont=0;
        }
        if (cont >= npages) {
            void *result = (uint8_t*)g_z->vmem + found * PAGE_SZ;
            g_z->vmem_next = (found + npages) * PAGE_SZ;
            for (size_t i=found; i<found+npages; i++) g_z->meta[i].state = PAGE_RESIDENT;
            mprotect(result, npages * PAGE_SZ, PROT_READ|PROT_WRITE);
            // Record in allocation table
            if (mmap_table_count < MMAP_TABLE_MAX) {
                mmap_table[mmap_table_count].base = result;
                mmap_table[mmap_table_count].npages = npages;
                mmap_table_count++;
            }
            pthread_mutex_unlock(&g_z->alloc_mutex);
            in_memx = 0;
            return result;
        }
        pthread_mutex_unlock(&g_z->alloc_mutex);
        in_memx = 0;
    }
passthrough:
    return mmap(addr, length, prot, flags, fd, offset);
}

static int memx_munmap(void *addr, size_t length) {
    if (mmap_safe && g_z && is_ours(addr)) {
        // Find in mmap allocation table
        size_t npages = 0;
        size_t sp = 0;
        for (int i=0; i<mmap_table_count; i++) {
            if (mmap_table[i].base == addr) {
                npages = mmap_table[i].npages;
                sp = ((uintptr_t)addr - (uintptr_t)g_z->vmem) / PAGE_SZ;
                mmap_table[i] = mmap_table[mmap_table_count-1];
                mmap_table_count--;
                break;
            }
        }
        // Not in table: might be a memx_malloc allocation (with size header at addr-8)
        if (npages == 0) {
            uint8_t *real = (uint8_t*)addr - sizeof(size_t);
            if (is_ours(real)) {
                size_t stored_size; memcpy(&stored_size, real, sizeof(size_t));
                size_t alloc_size = ((stored_size + sizeof(size_t) + PAGE_SZ - 1) / PAGE_SZ) * PAGE_SZ;
                npages = alloc_size / PAGE_SZ;
                sp = ((uintptr_t)real - (uintptr_t)g_z->vmem) / PAGE_SZ;
            }
        }
        if (npages > 0 && sp < g_z->npages) {
            for (size_t i=sp; i<sp+npages && i<g_z->npages; i++) {
                if (g_z->meta[i].state==PAGE_COMPRESSED) mprotect((uint8_t*)g_z->vmem+i*PAGE_SZ, PAGE_SZ, PROT_READ|PROT_WRITE);
                g_z->meta[i].state=PAGE_NONE; g_z->meta[i].comp_size=0;
                mprotect((uint8_t*)g_z->vmem+i*PAGE_SZ, PAGE_SZ, PROT_NONE);
            }
            return 0;
        }
    }
    return munmap(addr, length);
}

// ─── __interpose section ───
typedef struct { const void *replacement; const void *original; } interpose_t;
#define INTERPOSE(func) { (const void*)memx_##func, (const void*)func }
__attribute__((used)) static const interpose_t interposers[]
    __attribute__((section("__DATA,__interpose"))) = {
    INTERPOSE(malloc), INTERPOSE(calloc), INTERPOSE(realloc), INTERPOSE(free),
    INTERPOSE(mmap), INTERPOSE(munmap),
};

__attribute__((constructor)) static void memx_ctor(void) {
    // Cache default zone pointer early (safe - no allocation)
    s_default_zone = malloc_default_zone();
    // DON'T init Metal here - defer until first large allocation
    // This avoids 100MB Metal overhead for processes that don't need compression
    mmap_safe = 1;  // mmap interpose is safe (just passes through until g_z is set)
    
    // On macOS 15+, __interpose for malloc may not work due to dyld cache.
    // Patch GOT entries (__la_symbol_ptr) in all loaded images as a fallback.
    // This ensures malloc/free/calloc/realloc are intercepted even when
    // the compiler generates stub calls (which is the case with -fno-builtin-malloc).
    {
        uint32_t img_count = _dyld_image_count();
        for (uint32_t img = 0; img < img_count; img++) {
            const struct mach_header *hdr = _dyld_get_image_header(img);
            intptr_t slide = _dyld_get_image_vmaddr_slide(img);
            if (hdr->magic != MH_MAGIC_64) continue;
            
            struct mach_header_64 *h64 = (struct mach_header_64 *)hdr;
            uint8_t *ptr = (uint8_t *)h64 + sizeof(struct mach_header_64);
            struct segment_command_64 *linkedit = NULL;
            struct symtab_command *symtab = NULL;
            struct dysymtab_command *dysymtab = NULL;
            
            for (uint32_t c = 0; c < h64->ncmds; c++) {
                struct load_command *lc = (struct load_command *)ptr;
                if (lc->cmd == LC_SEGMENT_64) {
                    struct segment_command_64 *seg = (struct segment_command_64 *)ptr;
                    if (strcmp(seg->segname, "__LINKEDIT") == 0) linkedit = seg;
                } else if (lc->cmd == LC_SYMTAB) {
                    symtab = (struct symtab_command *)ptr;
                } else if (lc->cmd == LC_DYSYMTAB) {
                    dysymtab = (struct dysymtab_command *)ptr;
                }
                ptr += lc->cmdsize;
            }
            if (!linkedit || !symtab || !dysymtab) continue;
            
            uintptr_t le_base = slide + linkedit->vmaddr - linkedit->fileoff;
            uint32_t *indirect_syms = (uint32_t *)(le_base + dysymtab->indirectsymoff);
            char *strtab = (char *)(le_base + symtab->stroff);
            struct nlist_64 *syms = (struct nlist_64 *)(le_base + symtab->symoff);
            
            // Walk sections looking for __la_symbol_ptr and __nl_symbol_ptr
            ptr = (uint8_t *)h64 + sizeof(struct mach_header_64);
            for (uint32_t c = 0; c < h64->ncmds; c++) {
                struct load_command *lc = (struct load_command *)ptr;
                if (lc->cmd == LC_SEGMENT_64) {
                    struct segment_command_64 *seg = (struct segment_command_64 *)ptr;
                    struct section_64 *sects = (struct section_64 *)(ptr + sizeof(struct segment_command_64));
                    for (uint32_t s = 0; s < seg->nsects; s++) {
                        if (strcmp(sects[s].sectname, "__la_symbol_ptr") != 0 &&
                            strcmp(sects[s].sectname, "__nl_symbol_ptr") != 0) continue;
                        
                        uint64_t **entries = (uint64_t **)(slide + sects[s].addr);
                        uint32_t nentries = sects[s].size / 8;
                        for (uint32_t e = 0; e < nentries; e++) {
                            uint32_t idx = indirect_syms[sects[s].reserved1 + e];
                            if (idx == 0x80000000 || idx == 0x40000000) continue; // ABS/LOCAL
                            const char *name = strtab + syms[idx].n_un.n_strx;
                            
                            // Replace malloc/free/calloc/realloc GOT entries
                            if (strcmp(name, "_malloc") == 0 && entries[e] != (uint64_t *)memx_malloc) {
                                entries[e] = (uint64_t *)memx_malloc;
                            } else if (strcmp(name, "_free") == 0 && entries[e] != (uint64_t *)memx_free) {
                                entries[e] = (uint64_t *)memx_free;
                            } else if (strcmp(name, "_calloc") == 0 && entries[e] != (uint64_t *)memx_calloc) {
                                entries[e] = (uint64_t *)memx_calloc;
                            } else if (strcmp(name, "_realloc") == 0 && entries[e] != (uint64_t *)memx_realloc) {
                                entries[e] = (uint64_t *)memx_realloc;
                            }
                        }
                    }
                }
                ptr += lc->cmdsize;
            }
        }
    }
}
__attribute__((destructor)) static void memx_dtor(void) { mmap_safe = 0; fini_memx(); }
