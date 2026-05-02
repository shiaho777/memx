// MemX Dylib v2: malloc_zone override approach
// Much simpler and safer than __interpose
//
// USAGE: DYLD_INSERT_LIBRARIES=./libmemx2.dylib <any_app>
//
// Strategy:
//   - Create a custom malloc_zone for large allocations (>64KB)
//   - Small allocations go to default zone (zero overhead)
//   - Large allocations go to our GPU-compressed zone
//   - On access to compressed page → SIGSEGV → CPU decompress → resume
//   - Background thread GPU-compresses cold pages

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <pthread.h>
#include <mach/mach_time.h>
#include <malloc/malloc.h>

#import <Metal/Metal.h>

#define PAGE_SZ 16384
#define MB (1024ULL*1024)
#define LARGE_THRESHOLD 65536

// ─── GPU Shaders ───
static NSString *const shader_src = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"constant uint PS=16384;\n"
"uint h4(threadgroup const uchar* p){return ((uint)p[0]|((uint)p[1]<<8)|((uint)p[2]<<16)|((uint)p[3]<<24))*2654435761u;}\n"
"kernel void cp(device const uchar* s[[buffer(0)]],device uchar* d[[buffer(1)]],device uint* z[[buffer(2)]],uint t[[thread_position_in_threadgroup]],uint pg[[threadgroup_position_in_grid]],uint ts[[threads_per_threadgroup]]){threadgroup uchar dp[16384];threadgroup uint hk[2048],hv[2048];uint po=pg*PS;if(t<256){if(t==0){dp[0]=s[po];for(uint i=1;i<64;i++)dp[i]=s[po+i]-s[po+i-1];}else{dp[t*64]=s[po+t*64]-s[po+t*64-1];for(uint i=1;i<64;i++)dp[t*64+i]=s[po+t*64+i]-s[po+t*64+i-1];}}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=t;i<2048;i+=ts){hk[i]=0xFFFFFFFFu;hv[i]=0;}threadgroup_barrier(mem_flags::mem_threadgroup);if(t==0){uint db=pg*PS,so=4,si=0;while(si<PS&&so<PS){if(si+4<=PS){uint h=h4(dp+si)&2047;uint pp=hv[h],pk=hk[h];uint ck=(uint)dp[si]|((uint)dp[si+1]<<8)|((uint)dp[si+2]<<16)|((uint)dp[si+3]<<24);hk[h]=ck;hv[h]=si;if(pk==ck&&pp<si&&(si-pp)<4096){uint ml=0;while(ml<258&&si+ml<PS&&dp[si+ml]==dp[pp+ml])ml++;if(ml>=4){so+=5;si+=ml;continue;}}}if(dp[si]==0xFF||dp[si]==0xFE)so+=2;else so++;si++;}if(so>=PS){z[pg]=PS;}else{for(uint i=0;i<2048;i++){hk[i]=0xFFFFFFFFu;hv[i]=0;}d[db]=0x4D;d[db+1]=0x58;d[db+2]=1;d[db+3]=0;uint ip=0,op=4;while(ip<PS&&op<PS-6){if(ip+4<=PS){uint h=h4(dp+ip)&2047;uint pp=hv[h],pk=hk[h];uint ck=(uint)dp[ip]|((uint)dp[ip+1]<<8)|((uint)dp[ip+2]<<16)|((uint)dp[ip+3]<<24);hk[h]=ck;hv[h]=ip;if(pk==ck&&pp<ip&&(ip-pp)<4096){uint ml=0,off=ip-pp;while(ml<258&&ip+ml<PS&&dp[ip+ml]==dp[pp+ml])ml++;if(ml>=4){d[db+op++]=0xFF;d[db+op++]=(uchar)(off&0xFF);d[db+op++]=(uchar)((off>>8)&0xFF);d[db+op++]=(uchar)(ml&0xFF);d[db+op++]=(uchar)((ml>>8)&0xFF);ip+=ml;continue;}}}if(dp[ip]==0xFF){d[db+op++]=0xFE;d[db+op++]=0xFF;}else if(dp[ip]==0xFE){d[db+op++]=0xFE;d[db+op++]=0xFE;}else{d[db+op++]=dp[ip];}ip++;}z[pg]=op;}}threadgroup_barrier(mem_flags::mem_threadgroup);if(z[pg]==PS){for(uint i=t;i<PS;i+=ts)d[pg*PS+i]=s[po+i];}}\n"
"kernel void dp(device const uchar* s[[buffer(0)]],device uchar* d[[buffer(1)]],device const uint* z[[buffer(2)]],uint t[[thread_position_in_threadgroup]],uint pg[[threadgroup_position_in_grid]],uint ts[[threads_per_threadgroup]]){uint po=pg*PS,sb=pg*PS,cs=z[pg];if(cs==PS){for(uint i=t;i<PS;i+=ts)d[po+i]=s[sb+i];return;}if(s[sb]!=0x4D||s[sb+1]!=0x58){for(uint i=t;i<PS;i+=ts)d[po+i]=s[sb+i];return;}threadgroup uchar db[16384];if(t==0){uint ip=4,op=0;while(ip<cs&&op<PS){uchar b=s[sb+ip];if(b==0xFF&&ip+4<cs){ip++;uint off=(uint)s[sb+ip]|(((uint)s[sb+ip+1])<<8);ip+=2;uint ml=(uint)s[sb+ip]|(((uint)s[sb+ip+1])<<8);ip+=2;uint ms=op-off;for(uint i=0;i<ml&&op<PS;i++)db[op++]=db[ms+i];}else if(b==0xFE&&ip+1<cs){ip++;db[op++]=s[sb+ip++];}else{db[op++]=b;ip++;}}}threadgroup_barrier(mem_flags::mem_threadgroup);if(t==0){d[po]=db[0];for(uint i=1;i<PS;i++)d[po+i]=d[po+i-1]+db[i];}}\n";

// ─── Page states ───
#define PAGE_NONE       0
#define PAGE_RESIDENT   1
#define PAGE_COMPRESSED 2

typedef struct { uint8_t state; uint32_t comp_size; uint64_t pool_offset; } PageMeta;

typedef struct {
    void        *vmem;
    uint64_t    vmem_size;
    size_t      npages;
    uint64_t    vmem_next;
    
    uint8_t    *pool;
    uint64_t    pool_size;
    uint64_t    pool_used;
    uint64_t    pool_next;
    
    PageMeta   *meta;
    
    id<MTLDevice>                   device;
    id<MTLCommandQueue>             queue;
    id<MTLComputePipelineState>     comp_pipe;
    id<MTLComputePipelineState>     decomp_pipe;
    
    volatile uint64_t  faults;
    volatile uint64_t  compressions;
    volatile uint64_t  bytes_saved;
    volatile int       running;
    volatile int       attached;
    pthread_t          bg_thread;
} MemXZone;

static MemXZone *g_zone = NULL;

// ─── CPU decompressor (signal-safe) ───
static void cpu_decompress_page(const uint8_t *src, uint32_t cs, uint8_t *dst) {
    if (cs >= PAGE_SZ || src[0] != 0x4D || src[1] != 0x58) { memcpy(dst, src, PAGE_SZ); return; }
    uint8_t db[PAGE_SZ]; uint32_t ip=4, op=0;
    while (ip<cs && op<PAGE_SZ) {
        uint8_t b=src[ip];
        if (b==0xFF && ip+4<cs) { ip++; uint32_t o=(uint32_t)src[ip]|((uint32_t)src[ip+1]<<8); ip+=2; uint32_t m=(uint32_t)src[ip]|((uint32_t)src[ip+1]<<8); ip+=2; uint32_t s=op-o; for(uint32_t i=0;i<m&&op<PAGE_SZ;i++) db[op++]=db[s+i]; }
        else if (b==0xFE && ip+1<cs) { ip++; db[op++]=src[ip++]; }
        else { db[op++]=b; ip++; }
    }
    if (op>0) { dst[0]=db[0]; for(uint32_t i=1;i<op;i++) dst[i]=dst[i-1]+db[i]; if(op<PAGE_SZ) memset(dst+op,0,PAGE_SZ-op); }
    else memset(dst,0,PAGE_SZ);
}

// ─── GPU compress ───
static int gpu_compress(MemXZone *s, size_t count, uint8_t *src, uint8_t *dst, uint32_t *sizes) {
    size_t bytes = count * PAGE_SZ;
    id<MTLBuffer> sb=[s->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> db=[s->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> zb=[s->device newBufferWithLength:count*4 options:MTLResourceStorageModeShared];
    if (!sb||!db||!zb) return -1;
    memcpy([sb contents],src,bytes);
    id<MTLCommandBuffer> cb=[s->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:s->comp_pipe];
    [enc setBuffer:sb offset:0 atIndex:0]; [enc setBuffer:db offset:0 atIndex:1]; [enc setBuffer:zb offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(dst,[db contents],bytes); memcpy(sizes,[zb contents],count*4);
    return 0;
}

// ─── Signal handler ───
static struct sigaction old_segv, old_bus;

static void fault_handler(int sig, siginfo_t *info, void *ctx) {
    if (!g_zone || !g_zone->running) goto chain;
    uintptr_t fa = (uintptr_t)info->si_addr;
    uintptr_t vs = (uintptr_t)g_zone->vmem, ve = vs + g_zone->vmem_size;
    if (fa < vs || fa >= ve) goto chain;
    
    size_t pi = (fa - vs) / PAGE_SZ;
    uint8_t *pa = (uint8_t*)g_zone->vmem + pi * PAGE_SZ;
    PageMeta *m = &g_zone->meta[pi];
    
    mprotect(pa, PAGE_SZ, PROT_READ | PROT_WRITE);
    if (m->state == PAGE_NONE) { memset(pa, 0, PAGE_SZ); m->state = PAGE_RESIDENT; }
    else if (m->state == PAGE_COMPRESSED) { cpu_decompress_page(g_zone->pool+m->pool_offset, m->comp_size, pa); m->state=PAGE_RESIDENT; m->comp_size=0; }
    __sync_fetch_and_add(&g_zone->faults, 1);
    return;
    
chain:
    if (sig==SIGSEGV && old_segv.sa_handler!=SIG_DFL && old_segv.sa_handler!=SIG_IGN) {
        if (old_segv.sa_flags & SA_SIGINFO) old_segv.sa_sigaction(sig,info,ctx);
        else if (old_segv.sa_handler != SIG_ERR) old_segv.sa_handler(sig);
    } else if (sig==SIGBUS && old_bus.sa_handler!=SIG_DFL && old_bus.sa_handler!=SIG_IGN) {
        if (old_bus.sa_flags & SA_SIGINFO) old_bus.sa_sigaction(sig,info,ctx);
        else if (old_bus.sa_handler != SIG_ERR) old_bus.sa_handler(sig);
    } else { signal(sig, SIG_DFL); raise(sig); }
}

// ─── Background compressor ───
static void *bg_compressor(void *arg) {
    MemXZone *s = (MemXZone *)arg;
    const size_t BATCH = 256;  // Larger batch for GPU efficiency
    while (s->running) {
        size_t tc[BATCH]; size_t nc = 0;
        for (size_t i=0; i<s->npages && nc<BATCH && s->running; i++)
            if (s->meta[i].state == PAGE_RESIDENT) tc[nc++] = i;
        if (nc == 0) { sleep(1); continue; }
        
        uint8_t *sb = (uint8_t*)malloc(nc*PAGE_SZ);
        uint8_t *db = (uint8_t*)malloc(nc*PAGE_SZ);
        uint32_t *sz = (uint32_t*)malloc(nc*4);
        if (!sb||!db||!sz) { free(sb); free(db); free(sz); sleep(1); continue; }
        
        for (size_t i=0; i<nc; i++) memcpy(sb+i*PAGE_SZ, (uint8_t*)s->vmem+tc[i]*PAGE_SZ, PAGE_SZ);
        if (gpu_compress(s,nc,sb,db,sz)!=0) { free(sb); free(db); free(sz); sleep(1); continue; }
        
        for (size_t i=0; i<nc; i++) {
            uint32_t cs=sz[i]; size_t pidx=tc[i];
            if (cs>=PAGE_SZ) continue;
            if (s->pool_next+cs > s->pool_size) continue;
            uint64_t off=s->pool_next;
            memcpy(s->pool+off, db+i*PAGE_SZ, cs);  // Only store compressed bytes!
            s->pool_next += cs;  // Advance by compressed size, not PAGE_SZ
            __sync_fetch_and_add(&s->pool_used, cs);
            s->meta[pidx].state=PAGE_COMPRESSED; s->meta[pidx].comp_size=cs; s->meta[pidx].pool_offset=off;
            mprotect((uint8_t*)s->vmem+pidx*PAGE_SZ, PAGE_SZ, PROT_NONE);
            __sync_fetch_and_add(&s->compressions, 1);
            __sync_fetch_and_add(&s->bytes_saved, PAGE_SZ-cs);
        }
        free(sb); free(db); free(sz);
        usleep(10000);  // 10ms between batches - fast compression
    }
    return NULL;
}

// ─── Check if address is in our pool ───
static int is_memx_ptr(void *ptr) {
    if (!g_zone || !g_zone->vmem) return 0;
    uintptr_t a = (uintptr_t)ptr;
    return a >= (uintptr_t)g_zone->vmem && a < (uintptr_t)g_zone->vmem + g_zone->vmem_size;
}

// ─── Custom malloc_zone ───
// This is the KEY: we register a custom malloc_zone
// The system will route large allocations to us automatically
// Small allocations go to the default zone (zero overhead)

static size_t zone_size(malloc_zone_t *zone, const void *ptr) {
    if (!g_zone || !is_memx_ptr(ptr)) return 0;
    uint8_t *real = (uint8_t*)ptr - sizeof(size_t);
    size_t size; memcpy(&size, real, sizeof(size_t));
    return size;
}

static void *zone_malloc(malloc_zone_t *zone, size_t size) {
    if (!g_zone || size < LARGE_THRESHOLD)
        return malloc_default_zone()->malloc(malloc_default_zone(), size);
    
    size_t alloc_size = ((size + PAGE_SZ - 1) / PAGE_SZ) * PAGE_SZ;
    size_t npages = alloc_size / PAGE_SZ;
    size_t sp = g_zone->vmem_next / PAGE_SZ, found=0, cont=0;
    for (size_t i=sp; i<g_zone->npages; i++) {
        if (g_zone->meta[i].state==PAGE_NONE) { if(!cont) found=i; cont++; if(cont>=npages) break; }
        else cont=0;
    }
    if (cont < npages) return malloc_default_zone()->malloc(malloc_default_zone(), size);
    
    void *result = (uint8_t*)g_zone->vmem + found * PAGE_SZ;
    g_zone->vmem_next = (found + npages) * PAGE_SZ;
    for (size_t i=found; i<found+npages; i++) g_zone->meta[i].state = PAGE_NONE;
    mprotect(result, PAGE_SZ, PROT_READ|PROT_WRITE);
    g_zone->meta[found].state = PAGE_RESIDENT;
    memcpy(result, &size, sizeof(size_t));
    return (uint8_t*)result + sizeof(size_t);
}

static void zone_free(malloc_zone_t *zone, void *ptr) {
    if (!ptr) return;
    if (!g_zone || !is_memx_ptr(ptr)) { malloc_default_zone()->free(malloc_default_zone(), ptr); return; }
    
    uint8_t *real = (uint8_t*)ptr - sizeof(size_t);
    size_t size; memcpy(&size, real, sizeof(size_t));
    size_t alloc_size = ((size + sizeof(size_t) + PAGE_SZ - 1) / PAGE_SZ) * PAGE_SZ;
    size_t sp = ((uintptr_t)real - (uintptr_t)g_zone->vmem) / PAGE_SZ;
    size_t np = alloc_size / PAGE_SZ;
    for (size_t i=sp; i<sp+np && i<g_zone->npages; i++) {
        if (g_zone->meta[i].state==PAGE_COMPRESSED) mprotect((uint8_t*)g_zone->vmem+i*PAGE_SZ, PAGE_SZ, PROT_READ|PROT_WRITE);
        g_zone->meta[i].state=PAGE_NONE; g_zone->meta[i].comp_size=0;
        mprotect((uint8_t*)g_zone->vmem+i*PAGE_SZ, PAGE_SZ, PROT_NONE);
    }
}

static void *zone_calloc(malloc_zone_t *zone, size_t nmemb, size_t size) {
    size_t total = nmemb * size;
    void *ptr = zone_malloc(zone, total);
    if (ptr && is_memx_ptr(ptr)) { /* pages zero-filled on fault */ }
    else if (ptr) memset(ptr, 0, total);
    return ptr;
}

static void *zone_realloc(malloc_zone_t *zone, void *ptr, size_t size) {
    if (!ptr) return zone_malloc(zone, size);
    if (size == 0) { zone_free(zone, ptr); return NULL; }
    if (!is_memx_ptr(ptr)) return malloc_default_zone()->realloc(malloc_default_zone(), ptr, size);
    
    uint8_t *real = (uint8_t*)ptr - sizeof(size_t);
    size_t old_size; memcpy(&old_size, real, sizeof(size_t));
    void *new_ptr = zone_malloc(zone, size);
    if (!new_ptr) return NULL;
    size_t copy = old_size < size ? old_size : size;
    memcpy(new_ptr, ptr, copy);
    zone_free(zone, ptr);
    return new_ptr;
}

static malloc_zone_t memx_zone;

// ─── Init (lazy, called from constructor) ───
static void init_memx(void) {
    if (g_zone) return;
    
    g_zone = (MemXZone*)mmap(NULL, sizeof(MemXZone), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (g_zone == MAP_FAILED) { g_zone = NULL; return; }
    memset(g_zone, 0, sizeof(MemXZone));
    
    // GPU
    g_zone->device = MTLCreateSystemDefaultDevice();
    if (!g_zone->device) { munmap(g_zone, sizeof(MemXZone)); g_zone=NULL; return; }
    
    NSError *err = nil;
    id<MTLLibrary> lib = [g_zone->device newLibraryWithSource:shader_src options:nil error:&err];
    if (!lib) { munmap(g_zone, sizeof(MemXZone)); g_zone=NULL; return; }
    
    g_zone->queue = [g_zone->device newCommandQueue];
    g_zone->comp_pipe = [g_zone->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"cp"] error:&err];
    g_zone->decomp_pipe = [g_zone->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dp"] error:&err];
    if (!g_zone->comp_pipe || !g_zone->decomp_pipe) { munmap(g_zone, sizeof(MemXZone)); g_zone=NULL; return; }
    
    // Virtual memory
    int64_t ms=0; size_t len=sizeof(ms);
    sysctlbyname("hw.memsize", &ms, &len, NULL, 0);
    g_zone->vmem_size = ms * 4;
    g_zone->npages = g_zone->vmem_size / PAGE_SZ;
    
    g_zone->vmem = mmap(NULL, g_zone->vmem_size, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (g_zone->vmem == MAP_FAILED) { munmap(g_zone, sizeof(MemXZone)); g_zone=NULL; return; }
    
    g_zone->pool_size = g_zone->vmem_size / 2;
    g_zone->pool = mmap(NULL, g_zone->pool_size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (g_zone->pool == MAP_FAILED) { munmap(g_zone->vmem, g_zone->vmem_size); munmap(g_zone, sizeof(MemXZone)); g_zone=NULL; return; }
    
    size_t meta_sz = g_zone->npages * sizeof(PageMeta);
    g_zone->meta = (PageMeta*)mmap(NULL, meta_sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (g_zone->meta == MAP_FAILED) { munmap(g_zone->pool, g_zone->pool_size); munmap(g_zone->vmem, g_zone->vmem_size); munmap(g_zone, sizeof(MemXZone)); g_zone=NULL; return; }
    
    // Signal handler
    struct sigaction sa; memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = fault_handler;
    sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old_segv);
    sigaction(SIGBUS, &sa, &old_bus);
    g_zone->attached = 1;
    
    // Background compressor
    g_zone->running = 1;
    pthread_create(&g_zone->bg_thread, NULL, bg_compressor, g_zone);
    
    // Register custom malloc zone
    memset(&memx_zone, 0, sizeof(memx_zone));
    memx_zone.version = 8;
    memx_zone.zone_name = "MemX GPU Compressed";
    memx_zone.malloc = zone_malloc;
    memx_zone.free = zone_free;
    memx_zone.calloc = zone_calloc;
    memx_zone.realloc = zone_realloc;
    memx_zone.size = zone_size;
    malloc_zone_register(&memx_zone);
    
    fprintf(stderr, "[memx] ✅ GPU memory expansion active (%llu MB virtual, zone registered)\n",
            (unsigned long long)(g_zone->vmem_size / MB));
}

static void fini_memx(void) {
    if (!g_zone) return;
    g_zone->running = 0;
    pthread_join(g_zone->bg_thread, NULL);
    for (size_t i=0; i<g_zone->npages; i++)
        if (g_zone->meta[i].state==PAGE_COMPRESSED || g_zone->meta[i].state==PAGE_NONE)
            mprotect((uint8_t*)g_zone->vmem+i*PAGE_SZ, PAGE_SZ, PROT_READ|PROT_WRITE);
    if (g_zone->attached) { sigaction(SIGSEGV, &old_segv, NULL); sigaction(SIGBUS, &old_bus, NULL); }
    malloc_zone_unregister(&memx_zone);
    munmap(g_zone->vmem, g_zone->vmem_size);
    munmap(g_zone->pool, g_zone->pool_size);
    munmap(g_zone->meta, g_zone->npages * sizeof(PageMeta));
    munmap(g_zone, sizeof(MemXZone));
    g_zone = NULL;
}

__attribute__((constructor)) static void memx_ctor(void) {
    // Defer init slightly to avoid dylib load-time deadlocks
    // dispatch_after works even without a runloop (uses a helper thread)
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 100 * NSEC_PER_MSEC),
                   dispatch_get_global_queue(0, 0), ^{
        init_memx();
    });
}
__attribute__((destructor)) static void memx_dtor(void) { fini_memx(); }
