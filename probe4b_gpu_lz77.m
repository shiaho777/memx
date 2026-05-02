// Probe 4 v2: GPU LZ77 compression - fits in 32KB threadgroup memory
// Strategy: Hash table in threadgroup (8KB), output to device memory directly
// Each threadgroup = 1 page (16KB), sequential compress with parallel hash updates
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/mach_vm.h>

#import <Metal/Metal.h>
#include <zlib.h>

#define GB (1024ULL*1024*1024)
#define MB (1024ULL*1024)
#define KB (1024ULL)

static double ns_per_tick;
static void init_time(void) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    ns_per_tick = (double)info.numer / (double)info.denom;
}
#define NS(ticks) ((double)(ticks) * ns_per_tick)

static NSString *const lz_shader = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"constant uint PAGE_SIZE = 16384;\n"
"constant uint HT_SIZE = 2048; // power of 2, fits in 16KB\n"
"constant uint MIN_MATCH = 4;\n"
"constant uint MAX_MATCH = 64;\n"
"\n"
"uint hash4(device const uchar* p) {\n"
"    return ((uint)p[0] | ((uint)p[1]<<8) | ((uint)p[2]<<16) | ((uint)p[3]<<24)) * 2654435761u;\n"
"}\n"
"\n"
"// Each threadgroup compresses one page\n"
"// Thread 0 drives sequential LZ77, others assist with hash table\n"
"kernel void lz77_compress(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device uchar* dst_pages [[buffer(1)]],\n"
"    device uint* comp_sizes [[buffer(2)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    // Hash table: 2048 entries x (4+4) bytes = 16KB\n"
"    threadgroup uint ht_keys[HT_SIZE];\n"
"    threadgroup uint ht_vals[HT_SIZE];\n"
"    \n"
"    for (uint i = tid; i < HT_SIZE; i += tg_size) {\n"
"        ht_keys[i] = 0xFFFFFFFFu;\n"
"        ht_vals[i] = 0;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    device const uchar* src = src_pages + page_off;\n"
"    device uchar* dst = dst_pages + page_off;\n"
"    \n"
"    // Thread 0 does the sequential compression\n"
"    if (tid == 0) {\n"
"        uint ip = 0;  // input position\n"
"        uint op = 0;  // output position\n"
"        \n"
"        while (ip < PAGE_SIZE && op < PAGE_SIZE - 8) {\n"
"            if (ip + MIN_MATCH <= PAGE_SIZE) {\n"
"                uint h = hash4(src + ip) & (HT_SIZE - 1);\n"
"                uint prev_pos = ht_vals[h];\n"
"                uint prev_key = ht_keys[h];\n"
"                uint cur_key = (uint)src[ip]|((uint)src[ip+1]<<8)|((uint)src[ip+2]<<16)|((uint)src[ip+3]<<24);\n"
"                ht_keys[h] = cur_key;\n"
"                ht_vals[h] = ip;\n"
"                \n"
"                if (prev_key == cur_key && prev_pos < ip && (ip - prev_pos) < 4096) {\n"
"                    uint ml = 0;\n"
"                    uint off = ip - prev_pos;\n"
"                    while (ml < MAX_MATCH && ip+ml < PAGE_SIZE && src[ip+ml] == src[prev_pos+ml]) ml++;\n"
"                    \n"
"                    if (ml >= MIN_MATCH) {\n"
"                        // Emit match: [0xFF] [offset_lo] [offset_hi] [length-MIN_MATCH]\n"
"                        dst[op++] = 0xFF;\n"
"                        dst[op++] = (uchar)(off & 0xFF);\n"
"                        dst[op++] = (uchar)(off >> 8);\n"
"                        dst[op++] = (uchar)(ml - MIN_MATCH);\n"
"                        ip += ml;\n"
"                        continue;\n"
"                    }\n"
"                }\n"
"            }\n"
"            \n"
"            // Literal\n"
"            dst[op++] = src[ip++];\n"
"        }\n"
"        comp_sizes[page_id] = op;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"}\n"
"\n"
"// Decompressor\n"
"kernel void lz77_decompress(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device uchar* dst_pages [[buffer(1)]],\n"
"    device const uint* comp_sizes [[buffer(2)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    device const uchar* src = src_pages + page_off;\n"
"    device uchar* dst = dst_pages + page_off;\n"
"    uint comp_size = comp_sizes[page_id];\n"
"    \n"
"    if (tid == 0) {\n"
"        uint ip = 0;\n"
"        uint op = 0;\n"
"        \n"
"        while (ip < comp_size && op < PAGE_SIZE) {\n"
"            if (src[ip] == 0xFF && ip + 3 < comp_size) {\n"
"                // Match\n"
"                ip++;\n"
"                uint off = (uint)src[ip] | ((uint)src[ip+1] << 8);\n"
"                ip += 2;\n"
"                uint ml = (uint)src[ip++] + MIN_MATCH;\n"
"                uint match_src = op - off;\n"
"                for (uint i = 0; i < ml && op < PAGE_SIZE; i++) {\n"
"                    dst[op++] = dst[match_src + i];\n"
"                }\n"
"            } else {\n"
"                dst[op++] = src[ip++];\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"// Page dedup hash\n"
"kernel void hash_pages(\n"
"    device const uchar* src [[buffer(0)]],\n"
"    device uint* hashes [[buffer(1)]],\n"
"    uint page_id [[thread_position_in_grid]]\n"
") {\n"
"    uint offset = page_id * PAGE_SIZE;\n"
"    uint h = 2166136261u;\n"
"    for (uint i = 0; i < PAGE_SIZE; i++) {\n"
"        h ^= src[offset + i];\n"
"        h *= 16777619u;\n"
"    }\n"
"    hashes[page_id] = h;\n"
"}\n";

int main(void) {
    init_time();

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { printf("No Metal!\n"); return 1; }

    printf("=== PROBE 4v2: GPU LZ77 Page Compression ===\n");
    printf("Device: %s\n\n", [[device name] UTF8String]);

    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:lz_shader options:nil error:&error];
    if (!lib) { printf("Shader error: %s\n", [[error localizedDescription] UTF8String]); return 1; }

    id<MTLCommandQueue> queue = [device newCommandQueue];

    struct { const char *name; int type; } patterns[] = {
        {"All zeros", 0},
        {"Repeated 4B", 1},
        {"Sparse (90% zero)", 2},
        {"JSON-like", 3},
        {"Random", 4},
        {"Mixed (30%zero+30%repeat+40%rand)", 5}
    };
    int npat = 6;
    size_t total = 64 * MB;
    size_t npages = total / 16384;

    for (int pi = 0; pi < npat; pi++) {
        printf("--- %s ---\n", patterns[pi].name);

        void *data = mmap(NULL, total, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
        if (!data) continue;

        switch (patterns[pi].type) {
        case 0: memset(data, 0, total); break;
        case 1: for(size_t i=0;i<total;i+=4) ((uint32_t*)data)[i/4]=(uint32_t)(i/16384); break;
        case 2: memset(data,0,total); for(size_t i=0;i<total;i+=160) ((char*)data)[i]=(char)(i&0xFF); break;
        case 3: {
            const char *t="{\"key\":\"value\",\"num\":12345,\"arr\":[1,2,3,4,5],\"name\":\"test_data_item\"}";
            size_t tl=strlen(t);
            for(size_t o=0;o<total;o+=tl) memcpy((char*)data+o, t, tl<(total-o)?tl:(total-o));
            break;
        }
        case 4: for(size_t i=0;i<total;i++) ((char*)data)[i]=(char)((i*6364136223846793005ULL)>>33); break;
        case 5:
            memset(data,0,total*3/10);
            for(size_t i=total*3/10;i<total*6/10;i+=4) ((uint32_t*)data)[i/4]=0x42424242;
            for(size_t i=total*6/10;i<total;i++) ((char*)data)[i]=(char)((i*6364136223846793005ULL)>>33);
            break;
        }

        id<MTLBuffer> src_buf = [device newBufferWithBytesNoCopy:data length:total options:MTLResourceStorageModeShared deallocator:^(void *p,NSUInteger l){munmap(p,l);}];
        if (!src_buf) {
            src_buf = [device newBufferWithLength:total options:MTLResourceStorageModeShared];
            memcpy([src_buf contents], data, total);
            munmap(data, total);
        }

        id<MTLBuffer> dst_buf = [device newBufferWithLength:total options:MTLResourceStorageModeShared];
        id<MTLBuffer> size_buf = [device newBufferWithLength:npages*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> verify_buf = [device newBufferWithLength:total options:MTLResourceStorageModeShared];

        id<MTLFunction> comp_func = [lib newFunctionWithName:@"lz77_compress"];
        id<MTLComputePipelineState> comp_pipe = [device newComputePipelineStateWithFunction:comp_func error:&error];
        if (!comp_pipe) { printf("  comp error: %s\n", [[error localizedDescription] UTF8String]); continue; }

        // Warmup
        for (int w=0;w<3;w++) {
            id<MTLCommandBuffer> cb=[queue commandBuffer];
            id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:comp_pipe];
            [enc setBuffer:src_buf offset:0 atIndex:0];
            [enc setBuffer:dst_buf offset:0 atIndex:1];
            [enc setBuffer:size_buf offset:0 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }

        uint64_t t0=mach_absolute_time();
        id<MTLCommandBuffer> cb=[queue commandBuffer];
        id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:comp_pipe];
        [enc setBuffer:src_buf offset:0 atIndex:0];
        [enc setBuffer:dst_buf offset:0 atIndex:1];
        [enc setBuffer:size_buf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        uint64_t t1=mach_absolute_time();
        double comp_ms=NS(t1-t0)/1e6;

        uint32_t *sizes=(uint32_t*)[size_buf contents];
        size_t comp_total=0;
        for(size_t i=0;i<npages;i++) comp_total+=sizes[i];
        double ratio = comp_total > 0 ? (double)total/comp_total : 999;

        printf("  GPU LZ77: %.2fms (%.2fGB/s), %.2fx, %llu->%lluMB\n",
               comp_ms, (double)total/(1ULL*1024*1024*1024)/(comp_ms/1000.0), ratio,
               (unsigned long long)(total/MB), (unsigned long long)(comp_total/MB));

        // Decompress
        id<MTLFunction> decomp_func = [lib newFunctionWithName:@"lz77_decompress"];
        id<MTLComputePipelineState> decomp_pipe = [device newComputePipelineStateWithFunction:decomp_func error:&error];
        if (!decomp_pipe) { printf("  decomp error: %s\n", [[error localizedDescription] UTF8String]); continue; }

        for (int w=0;w<3;w++) {
            id<MTLCommandBuffer> cb2=[queue commandBuffer];
            id<MTLComputeCommandEncoder> enc2=[cb2 computeCommandEncoder];
            [enc2 setComputePipelineState:decomp_pipe];
            [enc2 setBuffer:dst_buf offset:0 atIndex:0];
            [enc2 setBuffer:verify_buf offset:0 atIndex:1];
            [enc2 setBuffer:size_buf offset:0 atIndex:2];
            [enc2 dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc2 endEncoding]; [cb2 commit]; [cb2 waitUntilCompleted];
        }

        t0=mach_absolute_time();
        cb=[queue commandBuffer];
        enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:decomp_pipe];
        [enc setBuffer:dst_buf offset:0 atIndex:0];
        [enc setBuffer:verify_buf offset:0 atIndex:1];
        [enc setBuffer:size_buf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        t1=mach_absolute_time();
        double decomp_ms=NS(t1-t0)/1e6;

        int mismatch = memcmp([src_buf contents], [verify_buf contents], total);
        printf("  GPU decomp: %.2fms (%.2fGB/s), integrity: %s\n",
               decomp_ms, (double)total/(1ULL*1024*1024*1024)/(decomp_ms/1000.0),
               mismatch==0 ? "PERFECT" : "MISMATCH");

        // CPU zlib
        size_t zlib_total=0;
        t0=mach_absolute_time();
        for(size_t p=0;p<npages;p++) {
            uLongf dlen=compressBound(16384);
            unsigned char *tmp=malloc(dlen);
            compress2(tmp,&dlen,(unsigned char*)[src_buf contents]+p*16384,16384,1);
            zlib_total+=dlen; free(tmp);
        }
        t1=mach_absolute_time();
        double zlib_ms=NS(t1-t0)/1e6;
        double zlib_ratio = zlib_total>0 ? (double)total/zlib_total : 999;

        printf("  CPU zlib:  %.2fms (%.2fGB/s), %.2fx\n", zlib_ms, (double)total/(1ULL*1024*1024*1024)/(zlib_ms/1000.0), zlib_ratio);
        printf("  Speedup: %.2fx, ratio gap: %.2fx\n\n", zlib_ms/comp_ms, zlib_ratio/ratio);
    }

    // Single page decompress latency
    printf("=== Single Page Decompress Latency ===\n");
    {
        void *data = mmap(NULL, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
        memset(data, 0x42, 16384);
        id<MTLBuffer> src = [device newBufferWithBytesNoCopy:data length:16384 options:MTLResourceStorageModeShared deallocator:^(void *p,NSUInteger l){munmap(p,l);}];
        if (!src) { src=[device newBufferWithLength:16384 options:MTLResourceStorageModeShared]; memset([src contents],0x42,16384); }
        id<MTLBuffer> dst=[device newBufferWithLength:16384 options:MTLResourceStorageModeShared];
        id<MTLBuffer> sz=[device newBufferWithLength:4 options:MTLResourceStorageModeShared];

        id<MTLFunction> cf=[lib newFunctionWithName:@"lz77_compress"];
        id<MTLComputePipelineState> cp=[device newComputePipelineStateWithFunction:cf error:&error];
        id<MTLCommandBuffer> cb=[queue commandBuffer];
        id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:cp];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        [enc setBuffer:sz offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

        id<MTLFunction> df=[lib newFunctionWithName:@"lz77_decompress"];
        id<MTLComputePipelineState> dp=[device newComputePipelineStateWithFunction:df error:&error];
        id<MTLBuffer> vb=[device newBufferWithLength:16384 options:MTLResourceStorageModeShared];

        // Warmup
        for(int w=0;w<10;w++) {
            cb=[queue commandBuffer]; enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:dp];
            [enc setBuffer:dst offset:0 atIndex:0];
            [enc setBuffer:vb offset:0 atIndex:1];
            [enc setBuffer:sz offset:0 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }

        uint64_t t0=mach_absolute_time();
        for(int r=0;r<1000;r++) {
            cb=[queue commandBuffer]; enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:dp];
            [enc setBuffer:dst offset:0 atIndex:0];
            [enc setBuffer:vb offset:0 atIndex:1];
            [enc setBuffer:sz offset:0 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
        uint64_t t1=mach_absolute_time();
        double single_ns = NS(t1-t0)/1000;
        printf("Single page decompress: %.0f ns (%.1f us)\n", single_ns, single_ns/1000);
        printf("SSD random read: ~100000 ns (100 us)\n");
        printf("DRAM page fault: ~5000 ns (5 us)\n");
        if (single_ns < 100000) printf("*** GPU decompress FASTER than SSD! ***\n");
        if (single_ns < 10000) printf("*** GPU decompress approaches DRAM speed! ***\n");
    }

    printf("\n=== PROBE 4v2 COMPLETE ===\n");
    return 0;
}
