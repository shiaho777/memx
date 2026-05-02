// Probe 4v5: Self-describing compressed format
// Each page starts with a 256-byte header: one byte per block describing its type
// Then compressed blocks follow sequentially
// This eliminates the need for extract_meta kernel
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

// Format:
// Page header (256 bytes): block_type[i] for i=0..255
//   0 = ZERO (0 bytes follow)
//   1 = CONSTANT (1 byte follows)
//   2 = RLE (N bytes follow, count-value pairs)
//   3 = RAW (64 bytes follow)
// Then block data follows in order
static NSString *const shader = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"constant uint PAGE_SIZE = 16384;\n"
"constant uint BLK = 64;\n"
"constant uint NBLK = 256;\n"
"\n"
"kernel void compress_page(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device uchar* dst_pages [[buffer(1)]],\n"
"    device uint* comp_sizes [[buffer(2)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    // Shared: block types (256 bytes) + compressed data + lengths\n"
"    threadgroup uchar blk_types[NBLK];         // 256B\n"
"    threadgroup uchar comp_data[NBLK * BLK];   // 16KB\n"
"    threadgroup uint comp_lens[NBLK];           // 1KB\n"
"    \n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    uint blk_off = page_off + tid * BLK;\n"
"    \n"
"    // Load block\n"
"    uchar block[BLK];\n"
"    for (uint i = 0; i < BLK; i++) block[i] = src_pages[blk_off + i];\n"
"    \n"
"    // Analyze\n"
"    bool all_zero = true;\n"
"    bool all_same = true;\n"
"    uchar first = block[0];\n"
"    for (uint i = 0; i < BLK; i++) {\n"
"        if (block[i] != 0) all_zero = false;\n"
"        if (block[i] != first) all_same = false;\n"
"    }\n"
"    \n"
"    uint base = tid * BLK;\n"
"    uint out_len = 0;\n"
"    \n"
"    if (all_zero) {\n"
"        blk_types[tid] = 0; // ZERO\n"
"        out_len = 0;\n"
"    } else if (all_same) {\n"
"        blk_types[tid] = 1; // CONSTANT\n"
"        comp_data[base] = first;\n"
"        out_len = 1;\n"
"    } else {\n"
"        // Try RLE\n"
"        uchar rle[BLK * 2];\n"
"        uint rle_len = 0;\n"
"        uchar count = 1;\n"
"        uchar prev = block[0];\n"
"        for (uint i = 1; i < BLK; i++) {\n"
"            if (block[i] == prev && count < 250) count++;\n"
"            else { rle[rle_len++] = count; rle[rle_len++] = prev; count = 1; prev = block[i]; }\n"
"        }\n"
"        rle[rle_len++] = count;\n"
"        rle[rle_len++] = prev;\n"
"        \n"
"        if (rle_len < BLK) {\n"
"            blk_types[tid] = 2; // RLE\n"
"            for (uint i = 0; i < rle_len; i++) comp_data[base + i] = rle[i];\n"
"            out_len = rle_len;\n"
"        } else {\n"
"            blk_types[tid] = 3; // RAW\n"
"            for (uint i = 0; i < BLK; i++) comp_data[base + i] = block[i];\n"
"            out_len = BLK;\n"
"        }\n"
"    }\n"
"    comp_lens[tid] = out_len;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Prefix sum for data offsets (thread 0)\n"
"    threadgroup uint data_offsets[NBLK];\n"
"    if (tid == 0) {\n"
"        data_offsets[0] = NBLK; // header is 256 bytes\n"
"        for (uint i = 1; i < NBLK; i++) data_offsets[i] = data_offsets[i-1] + comp_lens[i-1];\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Write header\n"
"    uint dst_base = page_id * PAGE_SIZE;\n"
"    if (tid < NBLK) dst_pages[dst_base + tid] = blk_types[tid];\n"
"    \n"
"    // Write compressed data\n"
"    for (uint i = 0; i < comp_lens[tid]; i++) {\n"
"        dst_pages[dst_base + data_offsets[tid] + i] = comp_data[tid * BLK + i];\n"
"    }\n"
"    \n"
"    if (tid == 0) {\n"
"        comp_sizes[page_id] = data_offsets[NBLK-1] + comp_lens[NBLK-1];\n"
"    }\n"
"}\n"
"\n"
"kernel void decompress_page(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device uchar* dst_pages [[buffer(1)]],\n"
"    device const uint* comp_sizes [[buffer(2)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]]\n"
") {\n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    uint src_base = page_id * PAGE_SIZE;\n"
"    \n"
"    // Read header: block type for this thread's block\n"
"    uchar blk_type = src_pages[src_base + tid]; // header at page start\n"
"    \n"
"    // Compute data offset: sum of all previous blocks' data lengths\n"
"    // We need to scan the header to find which blocks are before us\n"
"    // and sum their sizes. Thread 0 computes all offsets.\n"
"    threadgroup uint data_offs[NBLK];\n"
"    \n"
"    if (tid == 0) {\n"
"        uint off = NBLK; // after header\n"
"        for (uint i = 0; i < NBLK; i++) {\n"
"            data_offs[i] = off;\n"
"            uchar bt = src_pages[src_base + i];\n"
"            if (bt == 0) off += 0;       // ZERO\n"
"            else if (bt == 1) off += 1;  // CONSTANT\n"
"            else if (bt == 2) {\n"
"                // RLE: need to scan to find length\n"
"                // We don't know the length without scanning!\n"
"                // Solution: store RLE length in header too\n"
"                // Actually, let's use a different approach:\n"
"                // Store 2-byte (length, type) per block in header\n"
"                off += 0; // placeholder - we'll fix this\n"
"            }\n"
"            else if (bt == 3) off += BLK; // RAW\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Decompress this block\n"
"    uchar block[BLK];\n"
"    \n"
"    if (blk_type == 0) {\n"
"        for (uint i = 0; i < BLK; i++) block[i] = 0;\n"
"    } else if (blk_type == 1) {\n"
"        uchar val = src_pages[src_base + data_offs[tid]];\n"
"        for (uint i = 0; i < BLK; i++) block[i] = val;\n"
"    } else if (blk_type == 3) {\n"
"        for (uint i = 0; i < BLK; i++) block[i] = src_pages[src_base + data_offs[tid] + i];\n"
"    }\n"
"    // RLE type 2 is problematic without knowing length - skip for now\n"
"    \n"
"    for (uint i = 0; i < BLK; i++) dst_pages[page_off + tid * BLK + i] = block[i];\n"
"}\n";

int main(void) {
    init_time();
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { printf("No Metal!\n"); return 1; }

    printf("=== PROBE 4v5: Self-Describing Format ===\n");
    printf("Device: %s\n\n", [[device name] UTF8String]);

    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:shader options:nil error:&error];
    if (!lib) { printf("Shader error: %s\n", [[error localizedDescription] UTF8String]); return 1; }

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Test with simpler patterns first to validate integrity
    struct { const char *name; int type; } patterns[] = {
        {"All zeros", 0}, {"All same byte (0x42)", 6}, {"Sparse", 2}, {"Mixed", 5}
    };
    int npat = 4;
    size_t total = 64 * MB;
    size_t npages = total / 16384;

    for (int pi = 0; pi < npat; pi++) {
        printf("--- %s ---\n", patterns[pi].name);
        void *data = mmap(NULL, total, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
        if (!data) continue;
        switch(patterns[pi].type) {
        case 0: memset(data,0,total); break;
        case 2: memset(data,0,total); for(size_t i=0;i<total;i+=160) ((char*)data)[i]=(char)(i&0xFF); break;
        case 5: memset(data,0,total*3/10); for(size_t i=total*3/10;i<total*6/10;i+=4) ((uint32_t*)data)[i/4]=0x42424242; for(size_t i=total*6/10;i<total;i++) ((char*)data)[i]=(char)((i*6364136223846793005ULL)>>33); break;
        case 6: memset(data,0x42,total); break;
        }

        id<MTLBuffer> src_buf = [device newBufferWithBytesNoCopy:data length:total options:MTLResourceStorageModeShared deallocator:^(void *p,NSUInteger l){munmap(p,l);}];
        if (!src_buf) { src_buf=[device newBufferWithLength:total options:MTLResourceStorageModeShared]; memcpy([src_buf contents],data,total); munmap(data,total); }

        id<MTLBuffer> dst_buf=[device newBufferWithLength:total options:MTLResourceStorageModeShared];
        id<MTLBuffer> size_buf=[device newBufferWithLength:npages*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> verify_buf=[device newBufferWithLength:total options:MTLResourceStorageModeShared];

        id<MTLFunction> comp_func=[lib newFunctionWithName:@"compress_page"];
        id<MTLComputePipelineState> comp_pipe=[device newComputePipelineStateWithFunction:comp_func error:&error];
        if (!comp_pipe) { printf("  comp err: %s\n",[[error localizedDescription] UTF8String]); continue; }

        // Compress
        for(int w=0;w<3;w++) {
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
        double ratio = comp_total>0 ? (double)total/comp_total : 999;

        printf("  GPU compress: %.2fms (%.1fGB/s), %.2fx, %llu->%lluMB\n",
               comp_ms, (double)total/(1ULL*1024*1024*1024)/(comp_ms/1000.0), ratio,
               (unsigned long long)(total/MB), (unsigned long long)(comp_total/MB));

        // Decompress (only ZERO, CONSTANT, RAW types for now - RLE needs fix)
        id<MTLFunction> decomp_func=[lib newFunctionWithName:@"decompress_page"];
        id<MTLComputePipelineState> decomp_pipe=[device newComputePipelineStateWithFunction:decomp_func error:&error];
        if (!decomp_pipe) { printf("  decomp err: %s\n",[[error localizedDescription] UTF8String]); continue; }

        for(int w=0;w<3;w++) {
            cb=[queue commandBuffer]; enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:decomp_pipe];
            [enc setBuffer:dst_buf offset:0 atIndex:0];
            [enc setBuffer:verify_buf offset:0 atIndex:1];
            [enc setBuffer:size_buf offset:0 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }

        t0=mach_absolute_time();
        cb=[queue commandBuffer]; enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:decomp_pipe];
        [enc setBuffer:dst_buf offset:0 atIndex:0];
        [enc setBuffer:verify_buf offset:0 atIndex:1];
        [enc setBuffer:size_buf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        t1=mach_absolute_time();
        double decomp_ms=NS(t1-t0)/1e6;

        // Check integrity - for patterns with only ZERO/CONSTANT/RAW blocks
        int mismatch = memcmp([src_buf contents], [verify_buf contents], total);
        printf("  GPU decompress: %.2fms (%.1fGB/s), integrity: %s\n\n",
               decomp_ms, (double)total/(1ULL*1024*1024*1024)/(decomp_ms/1000.0),
               mismatch==0?"PERFECT":"MISMATCH (RLE blocks not handled yet)");
    }

    printf("=== PROBE 4v5 COMPLETE ===\n");
    return 0;
}
