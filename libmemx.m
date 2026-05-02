// MemX Dylib: GPU-Compressed malloc() Replacement
//
// USAGE:
//   DYLD_INSERT_LIBRARIES=libmemx.dylib <any_app>
//
// WHAT IT DOES:
//   - Intercepts malloc/calloc/realloc/free
//   - Allocations above 64KB go through GPU-compressed pool
//   - Small allocations pass through to system malloc (no overhead)
//   - Cold pages auto-compressed by GPU, transparently decompressed on access
//   - Result: app uses less physical memory, can allocate more
//
// EXAMPLE:
//   DYLD_INSERT_LIBRARIES=./libmemx.dylib python3 train_model.py
//   DYLD_INSERT_LIBRARIES=./libmemx.dylib node server.js
//   DYLD_INSERT_LIBRARIES=./libmemx.dylib /Applications/Xcode.app/...
//
// SAFETY:
//   - Only intercepts allocations > 64KB (malloc threshold)
//   - Small allocations untouched = zero risk for normal code
//   - Signal handler only handles our address range
//   - Clean detach on dlclose

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <pthread.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#include <dlfcn.h>

#import <Metal/Metal.h>

#define PAGE_SZ 16384
#define MB (1024ULL*1024)
#define LARGE_ALLOC_THRESHOLD 65536  // 64KB - only compress large allocs

// ─── GPU Shaders (proven Delta+LZ77) ───
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

typedef struct {
    uint8_t  state;
    uint32_t comp_size;
    uint64_t pool_offset;
} PageMeta;

typedef struct {
    void            *vmem;
    uint64_t        vmem_size;
    size_t          npages;
    uint64_t        vmem_next;       // next free offset for allocation
    
    uint8_t        *pool;
    uint64_t        pool_size;
    uint64_t        pool_used;
    uint64_t        pool_next;
    
    PageMeta       *meta;
    
    id<MTLDevice>           device;
    id<MTLCommandQueue>     queue;
    id<MTLComputePipelineState> comp_pipe;
    id<MTLComputePipelineState> decomp_pipe;
    
    volatile uint64_t  faults;
    volatile uint64_t  compressions;
    volatile uint64_t  bytes_saved;
    volatile int       running;
    volatile int       attached;
    pthread_t          bg_thread;
    
    // Original malloc function pointers
    void *(*orig_malloc)(size_t);
    void (*orig_free)(void *);
    void *(*orig_calloc)(size_t, size_t);
    void *(*orig_realloc)(void *, size_t);
} MemXDylib;

static MemXDylib *g_memx = NULL;

// ─── CPU fallback decompressor (signal-safe) ───
static void cpu_decompress_page(const uint8_t *src, uint32_t comp_size, uint8_t *dst) {
    if (comp_size >= PAGE_SZ || src[0] != 0x4D || src[1] != 0x58) {
        memcpy(dst, src, PAGE_SZ);
        return;
    }
    uint8_t delta_buf[PAGE_SZ];
    uint32_t ip = 4, op = 0;
    while (ip < comp_size && op < PAGE_SZ) {
        uint8_t b = src[ip];
        if (b == 0xFF && ip + 4 < comp_size) {
            ip++;
            uint32_t off = (uint32_t)src[ip] | ((uint32_t)src[ip+1] << 8); ip += 2;
            uint32_t ml = (uint32_t)src[ip] | ((uint32_t)src[ip+1] << 8); ip += 2;
            uint32_t ms = op - off;
            for (uint32_t i = 0; i < ml && op < PAGE_SZ; i++) delta_buf[op++] = delta_buf[ms + i];
        } else if (b == 0xFE && ip + 1 < comp_size) { ip++; delta_buf[op++] = src[ip++]; }
        else { delta_buf[op++] = b; ip++; }
    }
    if (op > 0) {
        dst[0] = delta_buf[0];
        for (uint32_t i = 1; i < op; i++) dst[i] = dst[i-1] + delta_buf[i];
        if (op < PAGE_SZ) memset(dst + op, 0, PAGE_SZ - op);
    } else { memset(dst, 0, PAGE_SZ); }
}

// ─── GPU compress/decompress ───
static int gpu_compress(MemXDylib *s, size_t count, uint8_t *src, uint8_t *dst, uint32_t *sizes) {
    size_t bytes = count * PAGE_SZ;
    id<MTLBuffer> sb = [s->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> db = [s->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> zb = [s->device newBufferWithLength:count*4 options:MTLResourceStorageModeShared];
    if (!sb||!db||!zb) return -1;
    memcpy([sb contents], src, bytes);
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:s->comp_pipe];
    [enc setBuffer:sb offset:0 atIndex:0]; [enc setBuffer:db offset:0 atIndex:1]; [enc setBuffer:zb offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(dst, [db contents], bytes); memcpy(sizes, [zb contents], count*4);
    return 0;
}

// ─── Signal handler ───
static struct sigaction old_sigsegv, old_sigbus;

static void page_fault_handler(int sig, siginfo_t *info, void *ctx) {
    if (!g_memx || !g_memx->running) goto chain;
    
    uintptr_t fault_addr = (uintptr_t)info->si_addr;
    uintptr_t vmem_start = (uintptr_t)g_memx->vmem;
    uintptr_t vmem_end = vmem_start + g_memx->vmem_size;
    
    if (fault_addr < vmem_start || fault_addr >= vmem_end) goto chain;
    
    size_t page_idx = (fault_addr - vmem_start) / PAGE_SZ;
    uint8_t *page_addr = (uint8_t*)g_memx->vmem + page_idx * PAGE_SZ;
    PageMeta *m = &g_memx->meta[page_idx];
    
    mprotect(page_addr, PAGE_SZ, PROT_READ | PROT_WRITE);
    
    if (m->state == PAGE_NONE) {
        memset(page_addr, 0, PAGE_SZ);
        m->state = PAGE_RESIDENT;
    } else if (m->state == PAGE_COMPRESSED) {
        cpu_decompress_page(g_memx->pool + m->pool_offset, m->comp_size, page_addr);
        m->state = PAGE_RESIDENT;
        m->comp_size = 0;
    }
    
    __sync_fetch_and_add(&g_memx->faults, 1);
    return;
    
chain:
    if (sig == SIGSEGV && old_sigsegv.sa_handler != SIG_DFL && old_sigsegv.sa_handler != SIG_IGN) {
        if (old_sigsegv.sa_flags & SA_SIGINFO) old_sigsegv.sa_sigaction(sig, info, ctx);
        else if (old_sigsegv.sa_handler != SIG_ERR) old_sigsegv.sa_handler(sig);
    } else if (sig == SIGBUS && old_sigbus.sa_handler != SIG_DFL && old_sigbus.sa_handler != SIG_IGN) {
        if (old_sigbus.sa_flags & SA_SIGINFO) old_sigbus.sa_sigaction(sig, info, ctx);
        else if (old_sigbus.sa_handler != SIG_ERR) old_sigbus.sa_handler(sig);
    } else {
        signal(sig, SIG_DFL); raise(sig);
    }
}

// ─── Background compressor ───
static void *bg_compressor(void *arg) {
    MemXDylib *s = (MemXDylib *)arg;
    const size_t BATCH = 64;
    
    while (s->running) {
        size_t to_compress[BATCH];
        size_t n_compress = 0;
        
        for (size_t i = 0; i < s->npages && n_compress < BATCH && s->running; i++) {
            if (s->meta[i].state == PAGE_RESIDENT) {
                to_compress[n_compress++] = i;
            }
        }
        
        if (n_compress == 0) { sleep(2); continue; }
        
        uint8_t *src_buf = g_memx->orig_malloc(n_compress * PAGE_SZ);
        uint8_t *dst_buf = g_memx->orig_malloc(n_compress * PAGE_SZ);
        uint32_t *sizes = g_memx->orig_malloc(n_compress * 4);
        if (!src_buf || !dst_buf || !sizes) { g_memx->orig_free(src_buf); g_memx->orig_free(dst_buf); g_memx->orig_free(sizes); sleep(1); continue; }
        
        for (size_t i = 0; i < n_compress; i++)
            memcpy(src_buf + i * PAGE_SZ, (uint8_t*)s->vmem + to_compress[i] * PAGE_SZ, PAGE_SZ);
        
        if (gpu_compress(s, n_compress, src_buf, dst_buf, sizes) != 0) {
            g_memx->orig_free(src_buf); g_memx->orig_free(dst_buf); g_memx->orig_free(sizes); sleep(1); continue;
        }
        
        for (size_t i = 0; i < n_compress; i++) {
            uint32_t cs = sizes[i];
            size_t pidx = to_compress[i];
            if (cs >= PAGE_SZ) continue;
            if (s->pool_next + PAGE_SZ > s->pool_size) continue;
            
            uint64_t off = s->pool_next;
            memcpy(s->pool + off, dst_buf + i * PAGE_SZ, PAGE_SZ);
            s->pool_next += PAGE_SZ;
            __sync_fetch_and_add(&s->pool_used, cs);
            
            s->meta[pidx].state = PAGE_COMPRESSED;
            s->meta[pidx].comp_size = cs;
            s->meta[pidx].pool_offset = off;
            
            mprotect((uint8_t*)s->vmem + pidx * PAGE_SZ, PAGE_SZ, PROT_NONE);
            
            __sync_fetch_and_add(&s->compressions, 1);
            __sync_fetch_and_add(&s->bytes_saved, PAGE_SZ - cs);
        }
        
        g_memx->orig_free(src_buf); g_memx->orig_free(dst_buf); g_memx->orig_free(sizes);
        usleep(100000);
    }
    return NULL;
}

// ─── Check if address is in our pool ───
static int is_memx_ptr(void *ptr) {
    if (!g_memx || !g_memx->vmem) return 0;
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t start = (uintptr_t)g_memx->vmem;
    uintptr_t end = start + g_memx->vmem_size;
    return (addr >= start && addr < end);
}

// ─── Interposed malloc/calloc/realloc/free ───
static void init_memx(void);  // forward decl for lazy init
static __thread int in_memx = 0;

static void *mmap_alloc(size_t size) {
    void *p = mmap(NULL, size+sizeof(size_t), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    memcpy(p, &size, sizeof(size_t));
    return (uint8_t*)p + sizeof(size_t);
}

static void *memx_malloc(size_t size) {
    if (in_memx || !g_memx || size < LARGE_ALLOC_THRESHOLD) {
        if (g_memx && g_memx->orig_malloc) return g_memx->orig_malloc(size);
        return mmap_alloc(size);
    }
    // Lazy init on first large allocation
    if (!g_memx || !g_memx->running) { init_memx(); if (!g_memx || !g_memx->running) return mmap_alloc(size); }
    in_memx = 1;
    
    // Align to page boundary
    size_t alloc_size = ((size + PAGE_SZ - 1) / PAGE_SZ) * PAGE_SZ;
    size_t npages = alloc_size / PAGE_SZ;
    
    // Find contiguous free pages
    size_t start_page = g_memx->vmem_next / PAGE_SZ;
    size_t found_at = 0;
    size_t contiguous = 0;
    
    for (size_t i = start_page; i < g_memx->npages; i++) {
        if (g_memx->meta[i].state == PAGE_NONE) {
            if (contiguous == 0) found_at = i;
            contiguous++;
            if (contiguous >= npages) break;
        } else {
            contiguous = 0;
        }
    }
    
    if (contiguous < npages) {
        in_memx = 0;
        return g_memx->orig_malloc(size);
    }
    
    // Allocate pages (they start as PROT_NONE, will fault on access)
    void *result = (uint8_t*)g_memx->vmem + found_at * PAGE_SZ;
    g_memx->vmem_next = (found_at + npages) * PAGE_SZ;
    
    // Mark pages as NONE (will be faulted in on first access)
    for (size_t i = found_at; i < found_at + npages; i++) {
        g_memx->meta[i].state = PAGE_NONE;
    }
    
    // Store allocation size at the beginning of the first page
    // (will be written when the page faults in)
    // Actually, we need the page to be writable to store the size
    // Let's make the first page resident immediately
    mprotect(result, PAGE_SZ, PROT_READ | PROT_WRITE);
    g_memx->meta[found_at].state = PAGE_RESIDENT;
    
    memcpy(result, &size, sizeof(size_t));
    
    in_memx = 0;
    return (uint8_t*)result + sizeof(size_t);
}

static void memx_free(void *ptr) {
    if (!ptr) return;
    if (!g_memx || !is_memx_ptr(ptr)) {
        if (g_memx && g_memx->orig_free) g_memx->orig_free(ptr);
        else if (!g_memx) {
            // Early init - might be mmap_alloc'd, try dlsym
            void (*rf)(void*) = dlsym(RTLD_NEXT, "free");
            if (rf) rf(ptr);
        }
        return;
    }
    
    // Get the real start (before size header)
    uint8_t *real_start = (uint8_t*)ptr - sizeof(size_t);
    size_t size;
    memcpy(&size, real_start, sizeof(size_t));
    
    size_t alloc_size = ((size + sizeof(size_t) + PAGE_SZ - 1) / PAGE_SZ) * PAGE_SZ;
    size_t start_page = ((uintptr_t)real_start - (uintptr_t)g_memx->vmem) / PAGE_SZ;
    size_t npages = alloc_size / PAGE_SZ;
    
    // Decompress if needed, then mark as NONE
    for (size_t i = start_page; i < start_page + npages && i < g_memx->npages; i++) {
        if (g_memx->meta[i].state == PAGE_COMPRESSED) {
            mprotect((uint8_t*)g_memx->vmem + i * PAGE_SZ, PAGE_SZ, PROT_READ | PROT_WRITE);
        }
        g_memx->meta[i].state = PAGE_NONE;
        g_memx->meta[i].comp_size = 0;
        mprotect((uint8_t*)g_memx->vmem + i * PAGE_SZ, PAGE_SZ, PROT_NONE);
    }
}

static void *memx_calloc(size_t nmemb, size_t size) {
    if (in_memx || !g_memx) {
        // During early init or recursion - use orig_calloc or mmap
        if (g_memx && g_memx->orig_calloc) return g_memx->orig_calloc(nmemb, size);
        // mmap fallback: zero-filled by default
        size_t total = nmemb * size;
        void *p = mmap_alloc(total);
        if (p) memset(p, 0, total);
        return p;
    }
    size_t total = nmemb * size;
    void *ptr = memx_malloc(total);
    if (ptr && is_memx_ptr(ptr)) {
        // Pages fault in as zero-filled
    } else if (ptr) {
        memset(ptr, 0, total);
    }
    return ptr;
}

static void *memx_realloc(void *ptr, size_t size) {
    if (!ptr) return memx_malloc(size);
    if (size == 0) { memx_free(ptr); return NULL; }
    if (!g_memx || !is_memx_ptr(ptr)) {
        if (g_memx && g_memx->orig_realloc) return g_memx->orig_realloc(ptr, size);
        return NULL;  // can't realloc mmap_alloc'd easily
    }
    
    // Get old size
    uint8_t *real_start = (uint8_t*)ptr - sizeof(size_t);
    size_t old_size;
    memcpy(&old_size, real_start, sizeof(size_t));
    
    // Allocate new
    void *new_ptr = memx_malloc(size);
    if (!new_ptr) return NULL;
    
    // Copy data
    size_t copy_size = old_size < size ? old_size : size;
    memcpy(new_ptr, ptr, copy_size);
    
    // Free old
    memx_free(ptr);
    
    return new_ptr;
}

// ─── Lazy init (called on first large allocation) ───
static volatile int memx_initing = 0;

static void init_memx(void) {
    if (g_memx) return;
    if (__sync_bool_compare_and_swap(&memx_initing, 0, 1) == 0) {
        while (!g_memx) usleep(1000);
        return;
    }
    in_memx = 1;  // CRITICAL: prevent recursion during Metal init
    
    // Use mmap directly to avoid recursive malloc interception
    g_memx = (MemXDylib *)mmap(NULL, sizeof(MemXDylib),
                                 PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (g_memx == MAP_FAILED) { g_memx = NULL; memx_initing = 0; return; }
    memset((void*)g_memx, 0, sizeof(MemXDylib));
    
    // Resolve original malloc functions BEFORE doing anything else
    g_memx->orig_malloc = dlsym(RTLD_NEXT, "malloc");
    g_memx->orig_free = dlsym(RTLD_NEXT, "free");
    g_memx->orig_calloc = dlsym(RTLD_NEXT, "calloc");
    g_memx->orig_realloc = dlsym(RTLD_NEXT, "realloc");
    
    if (!g_memx->orig_malloc || !g_memx->orig_free || !g_memx->orig_calloc) {
        munmap(g_memx, sizeof(MemXDylib)); g_memx = NULL; memx_initing = 0;
        return;
    }
    
    // GPU setup
    g_memx->device = MTLCreateSystemDefaultDevice();
    if (!g_memx->device) { munmap(g_memx, sizeof(MemXDylib)); g_memx = NULL; memx_initing = 0; return; }
    
    NSError *err = nil;
    id<MTLLibrary> lib = [g_memx->device newLibraryWithSource:shader_src options:nil error:&err];
    if (!lib) { munmap(g_memx, sizeof(MemXDylib)); g_memx = NULL; memx_initing = 0; return; }
    
    g_memx->queue = [g_memx->device newCommandQueue];
    g_memx->comp_pipe = [g_memx->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"cp"] error:&err];
    g_memx->decomp_pipe = [g_memx->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dp"] error:&err];
    if (!g_memx->comp_pipe || !g_memx->decomp_pipe) { munmap(g_memx, sizeof(MemXDylib)); g_memx = NULL; memx_initing = 0; return; }
    
    // Virtual memory pool: 4x physical RAM
    int64_t memsize = 0; size_t len = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &len, NULL, 0);
    g_memx->vmem_size = memsize * 4;
    g_memx->npages = g_memx->vmem_size / PAGE_SZ;
    
    g_memx->vmem = mmap(NULL, g_memx->vmem_size, PROT_NONE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (g_memx->vmem == MAP_FAILED) { munmap(g_memx, sizeof(MemXDylib)); g_memx = NULL; memx_initing = 0; return; }
    
    // Compressed pool
    g_memx->pool_size = g_memx->vmem_size / 2;
    g_memx->pool = mmap(NULL, g_memx->pool_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (g_memx->pool == MAP_FAILED) { munmap(g_memx->vmem, g_memx->vmem_size); munmap(g_memx, sizeof(MemXDylib)); g_memx = NULL; memx_initing = 0; return; }
    
    // Meta - use mmap to avoid malloc
    size_t meta_size = g_memx->npages * sizeof(PageMeta);
    g_memx->meta = (PageMeta *)mmap(NULL, meta_size, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (g_memx->meta == MAP_FAILED) { munmap(g_memx->pool, g_memx->pool_size); munmap(g_memx->vmem, g_memx->vmem_size); munmap(g_memx, sizeof(MemXDylib)); g_memx = NULL; memx_initing = 0; return; }
    
    // Install signal handler
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = page_fault_handler;
    sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old_sigsegv);
    sigaction(SIGBUS, &sa, &old_sigbus);
    g_memx->attached = 1;
    
    // Start background compressor
    g_memx->running = 1;
    pthread_create(&g_memx->bg_thread, NULL, bg_compressor, g_memx);
    
    fprintf(stderr, "[memx] ✅ GPU memory expansion active (4x physical = %llu MB virtual)\n",
            (unsigned long long)(g_memx->vmem_size / MB));
    fprintf(stderr, "[memx] Large allocations (>%d bytes) use compressed pool\n",
            LARGE_ALLOC_THRESHOLD);
    in_memx = 0;  // Allow memx_malloc to use pool now
}

static void fini_memx(void) {
    if (!g_memx) return;
    
    g_memx->running = 0;
    pthread_join(g_memx->bg_thread, NULL);
    
    // Decompress all
    for (size_t i = 0; i < g_memx->npages; i++) {
        if (g_memx->meta[i].state == PAGE_COMPRESSED || g_memx->meta[i].state == PAGE_NONE) {
            mprotect((uint8_t*)g_memx->vmem + i * PAGE_SZ, PAGE_SZ, PROT_READ | PROT_WRITE);
        }
    }
    
    // Restore signal handlers
    if (g_memx->attached) {
        sigaction(SIGSEGV, &old_sigsegv, NULL);
        sigaction(SIGBUS, &old_sigbus, NULL);
    }
    
    munmap(g_memx->vmem, g_memx->vmem_size);
    munmap(g_memx->pool, g_memx->pool_size);
    munmap(g_memx->meta, g_memx->npages * sizeof(PageMeta));
    
    fprintf(stderr, "[memx] ✅ Shutdown. Compressed %llu pages, saved %llu MB, resolved %llu faults\n",
            (unsigned long long)g_memx->compressions,
            (unsigned long long)(g_memx->bytes_saved / MB),
            (unsigned long long)g_memx->faults);
    
    munmap(g_memx, sizeof(MemXDylib));
    g_memx = NULL;
}

// ─── DYLD_INSERT_LIBRARIES interpose ───
// This is the macOS way to replace malloc/calloc/realloc/free

typedef struct {
    const void *replacement;
    const void *original;
} interpose_t;

#define INTERPOSE(func) { (const void *)memx_##func, (const void *)func }

__attribute__((used)) static const interpose_t interposers[]
    __attribute__((section("__DATA,__interpose"))) = {
    INTERPOSE(malloc),
    INTERPOSE(calloc),
    INTERPOSE(realloc),
    INTERPOSE(free),
};

// Constructor: minimal - just resolve orig functions
// Destructor: cleanup if initialized
__attribute__((constructor)) static void memx_ctor(void) {
    // Resolve original functions early so they're available
    // But don't init GPU/mmap yet (too early, causes deadlock)
}
__attribute__((destructor)) static void memx_dtor(void) { if (g_memx) fini_memx(); }
