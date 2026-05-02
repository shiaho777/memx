// Probe 4v3: GPU Parallel Page Compression - ALL threads work
// Key insight: Split each 16KB page into 64-byte blocks.
// Phase 1 (parallel): Each thread compresses one block independently
// Phase 2 (parallel): Write compressed blocks to output with prefix-sum offsets
// This achieves TRUE parallelism within each page
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

// Strategy: Each 16KB page = 256 blocks of 64 bytes
// Phase 1: Each thread RLE-compresses its block into threadgroup shared memory
// Phase 2: Prefix sum to compute offsets, parallel write to device memory
static NSString *const shader = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"constant uint PAGE_SIZE = 16384;\n"
"constant uint BLK = 64;\n"
"constant uint NBLK = 256;\n"
"\n"
"// Phase 1: Each thread compresses one 64-byte block via RLE\n"
"// Output to threadgroup memory\n"
"kernel void compress_page(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device uchar* dst_pages [[buffer(1)]],\n"
"    device uint* comp_sizes [[buffer(2)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    // Each thread gets one 64-byte block\n"
"    // threadgroup shared: compressed blocks (max 64 bytes each) + sizes\n"
"    threadgroup uchar comp_buf[NBLK * BLK];  // 16KB max\n"
"    threadgroup uint comp_lens[NBLK];         // 1KB\n"
"    \n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    uint blk_off = page_off + tid * BLK;\n"
"    \n"
"    // Load block\n"
"    uchar block[BLK];\n"
"    for (uint i = 0; i < BLK; i++) block[i] = src_pages[blk_off + i];\n"
"    \n"
"    // Check if all same byte\n"
"    bool all_same = true;\n"
"    uchar first = block[0];\n"
"    for (uint i = 1; i < BLK; i++) { if (block[i] != first) { all_same = false; break; } }\n"
"    \n"
"    uint out_len = 0;\n"
"    \n"
"    if (all_same && first == 0) {\n"
"        // All zeros: 1 byte marker\n"
"        comp_buf[tid * BLK + 0] = 0xFE; // zero-block marker\n"
"        out_len = 1;\n"
"    } else if (all_same) {\n"
"        // All same byte: 2 bytes\n"
"        comp_buf[tid * BLK + 0] = 0xFD; // constant-block marker\n"
"        comp_buf[tid * BLK + 1] = first;\n"
"        out_len = 2;\n"
"    } else {\n"
"        // RLE encode\n"
"        uint pos = 0;\n"
"        uchar count = 1;\n"
"        uchar prev = block[0];\n"
"        for (uint i = 1; i < BLK; i++) {\n"
"            if (block[i] == prev && count < 127) {\n"
"                count++;\n"
"            } else {\n"
"                comp_buf[tid * BLK + pos++] = count;\n"
"                comp_buf[tid * BLK + pos++] = prev;\n"
"                count = 1;\n"
"                prev = block[i];\n"
"            }\n"
"        }\n"
"        comp_buf[tid * BLK + pos++] = count;\n"
"        comp_buf[tid * BLK + pos++] = prev;\n"
"        out_len = pos;\n"
"    }\n"
"    \n"
"    comp_lens[tid] = out_len;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Prefix sum to compute offsets\n"
"    // Simple sequential prefix sum (thread 0 does it)\n"
"    threadgroup uint offsets[NBLK];\n"
"    if (tid == 0) {\n"
"        offsets[0] = 0;\n"
"        for (uint i = 1; i < NBLK; i++) offsets[i] = offsets[i-1] + comp_lens[i-1];\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Write compressed data to device memory\n"
"    uint dst_base = page_id * PAGE_SIZE;\n"
"    for (uint i = 0; i < comp_lens[tid]; i++) {\n"
"        dst_pages[dst_base + offsets[tid] + i] = comp_buf[tid * BLK + i];\n"
"    }\n"
"    \n"
"    if (tid == 0) {\n"
"        uint total = offsets[NBLK-1] + comp_lens[NBLK-1];\n"
"        comp_sizes[page_id] = total;\n"
"    }\n"
"}\n"
"\n"
"// Decompressor: each thread decompresses one block\n"
"kernel void decompress_page(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device uchar* dst_pages [[buffer(1)]],\n"
"    device const uint* comp_sizes [[buffer(2)]],\n"
"    device const uint* block_offsets [[buffer(3)]],\n"
"    device const uint* block_lengths [[buffer(4)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    uint src_base = page_id * PAGE_SIZE;\n"
"    uint blk_off = block_offsets[page_id * NBLK + tid];\n"
"    uint blk_len = block_lengths[page_id * NBLK + tid];\n"
"    \n"
"    uchar block[BLK];\n"
"    uint out_pos = 0;\n"
"    \n"
"    if (blk_len == 1 && src_pages[src_base + blk_off] == 0xFE) {\n"
"        // Zero block\n"
"        for (uint i = 0; i < BLK; i++) block[i] = 0;\n"
"        out_pos = BLK;\n"
"    } else if (blk_len == 2 && src_pages[src_base + blk_off] == 0xFD) {\n"
"        // Constant block\n"
"        uchar val = src_pages[src_base + blk_off + 1];\n"
"        for (uint i = 0; i < BLK; i++) block[i] = val;\n"
"        out_pos = BLK;\n"
"    } else {\n"
"        // RLE decode\n"
"        for (uint i = 0; i < blk_len && out_pos < BLK; i += 2) {\n"
"            uchar count = src_pages[src_base + blk_off + i];\n"
"            uchar value = src_pages[src_base + blk_off + i + 1];\n"
"            for (uchar j = 0; j < count && out_pos < BLK; j++) {\n"
"                block[out_pos++] = value;\n"
"            }\n"
"        }\n"
"    }\n"
"    \n"
"    // Write to output\n"
"    for (uint i = 0; i < BLK; i++) {\n"
"        dst_pages[page_off + tid * BLK + i] = block[i];\n"
"    }\n"
"}\n"
"\n"
// Metadata extraction kernel: compute block offsets/lengths from compressed data\n"
"kernel void extract_metadata(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device const uint* comp_sizes [[buffer(1)]],\n"
"    device uint* block_offsets [[buffer(2)]],\n"
"    device uint* block_lengths [[buffer(3)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    // Thread 0 scans the compressed stream to find block boundaries\n"
"    threadgroup uint blk_offs[NBLK];\n"
"    threadgroup uint blk_lens[NBLK];\n"
"    \n"
"    if (tid == 0) {\n"
"        uint src_base = page_id * PAGE_SIZE;\n"
"        uint comp_size = comp_sizes[page_id];\n"
"        uint pos = 0;\n"
"        uint blk = 0;\n"
"        \n"
"        while (pos < comp_size && blk < NBLK) {\n"
"            blk_offs[blk] = pos;\n"
"            uchar marker = src_pages[src_base + pos];\n"
"            \n"
"            if (marker == 0xFE) {\n"
"                blk_lens[blk] = 1;\n"
"                pos += 1;\n"
"            } else if (marker == 0xFD) {\n"
"                blk_lens[blk] = 2;\n"
"                pos += 2;\n"
"            } else {\n"
"                // RLE: count pairs\n"
"                uint start = pos;\n"
"                while (pos + 1 < comp_size && pos - start < BLK) {\n"
"                    uchar cnt = src_pages[src_base + pos];\n"
"                    pos += 2; // count + value\n"
"                    // Check if next byte is a marker\n"
"                    if (pos < comp_size) {\n"
"                        uchar next = src_pages[src_base + pos];\n"
"                        if (next == 0xFE || next == 0xFD) break;\n"
"                    }\n"
"                }\n"
"                blk_lens[blk] = pos - start;\n"
"            }\n"
"            blk++;\n"
"        }\n"
"        // Zero remaining\n"
"        for (uint i = blk; i < NBLK; i++) { blk_offs[i] = 0; blk_lens[i] = 0; }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Write to device\n"
"    uint idx = page_id * NBLK + tid;\n"
"    block_offsets[idx] = blk_offs[tid];\n"
"    block_lengths[idx] = blk_lens[tid];\n"
"}\n";

int main(void) {
    init_time();
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { printf("No Metal!\n"); return 1; }

    printf("=== PROBE 4v3: GPU Parallel Block Compression ===\n");
    printf("Device: %s\n\n", [[device name] UTF8String]);

    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:shader options:nil error:&error];
    if (!lib) { printf("Shader error: %s\n", [[error localizedDescription] UTF8String]); return 1; }

    id<MTLCommandQueue> queue = [device newCommandQueue];

    struct { const char *name; int type; } patterns[] = {
        {"All zeros", 0}, {"Repeated 4B", 1}, {"Sparse", 2},
        {"JSON-like", 3}, {"Random", 4}, {"Mixed", 5}
    };
    int npat = 6;
    size_t total = 64 * MB;
    size_t npages = total / 16384;

    for (int pi = 0; pi < npat; pi++) {
        printf("--- %s ---\n", patterns[pi].name);
        void *data = mmap(NULL, total, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
        if (!data) continue;
        switch(patterns[pi].type) {
        case 0: memset(data,0,total); break;
        case 1: for(size_t i=0;i<total;i+=4) ((uint32_t*)data)[i/4]=(uint32_t)(i/16384); break;
        case 2: memset(data,0,total); for(size_t i=0;i<total;i+=160) ((char*)data)[i]=(char)(i&0xFF); break;
        case 3: { const char *t="{\"key\":\"value\",\"num\":12345,\"arr\":[1,2,3,4,5],\"name\":\"test_data_item\"}"; size_t tl=strlen(t); for(size_t o=0;o<total;o+=tl) memcpy((char*)data+o,t,tl<(total-o)?tl:(total-o)); break; }
        case 4: for(size_t i=0;i<total;i++) ((char*)data)[i]=(char)((i*6364136223846793005ULL)>>33); break;
        case 5: memset(data,0,total*3/10); for(size_t i=total*3/10;i<total*6/10;i+=4) ((uint32_t*)data)[i/4]=0x42424242; for(size_t i=total*6/10;i<total;i++) ((char*)data)[i]=(char)((i*6364136223846793005ULL)>>33); break;
        }

        id<MTLBuffer> src_buf = [device newBufferWithBytesNoCopy:data length:total options:MTLResourceStorageModeShared deallocator:^(void *p,NSUInteger l){munmap(p,l);}];
        if (!src_buf) { src_buf=[device newBufferWithLength:total options:MTLResourceStorageModeShared]; memcpy([src_buf contents],data,total); munmap(data,total); }

        id<MTLBuffer> dst_buf=[device newBufferWithLength:total options:MTLResourceStorageModeShared];
        id<MTLBuffer> size_buf=[device newBufferWithLength:npages*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> off_buf=[device newBufferWithLength:npages*256*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> len_buf=[device newBufferWithLength:npages*256*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> verify_buf=[device newBufferWithLength:total options:MTLResourceStorageModeShared];

        // Compress
        id<MTLFunction> comp_func=[lib newFunctionWithName:@"compress_page"];
        id<MTLComputePipelineState> comp_pipe=[device newComputePipelineStateWithFunction:comp_func error:&error];
        if (!comp_pipe) { printf("  comp err: %s\n",[[error localizedDescription] UTF8String]); continue; }

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

        printf("  GPU compress: %.2fms (%.2fGB/s), %.2fx, %llu->%lluMB\n",
               comp_ms, (double)total/(1ULL*1024*1024*1024)/(comp_ms/1000.0), ratio,
               (unsigned long long)(total/MB), (unsigned long long)(comp_total/MB));

        // Extract metadata
        id<MTLFunction> meta_func=[lib newFunctionWithName:@"extract_metadata"];
        id<MTLComputePipelineState> meta_pipe=[device newComputePipelineStateWithFunction:meta_func error:&error];
        if (meta_pipe) {
            cb=[queue commandBuffer]; enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:meta_pipe];
            [enc setBuffer:dst_buf offset:0 atIndex:0];
            [enc setBuffer:size_buf offset:0 atIndex:1];
            [enc setBuffer:off_buf offset:0 atIndex:2];
            [enc setBuffer:len_buf offset:0 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }

        // Decompress
        id<MTLFunction> decomp_func=[lib newFunctionWithName:@"decompress_page"];
        id<MTLComputePipelineState> decomp_pipe=[device newComputePipelineStateWithFunction:decomp_func error:&error];
        if (!decomp_pipe) { printf("  decomp err: %s\n",[[error localizedDescription] UTF8String]); continue; }

        for(int w=0;w<3;w++) {
            cb=[queue commandBuffer]; enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:decomp_pipe];
            [enc setBuffer:dst_buf offset:0 atIndex:0];
            [enc setBuffer:verify_buf offset:0 atIndex:1];
            [enc setBuffer:size_buf offset:0 atIndex:2];
            [enc setBuffer:off_buf offset:0 atIndex:3];
            [enc setBuffer:len_buf offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }

        t0=mach_absolute_time();
        cb=[queue commandBuffer]; enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:decomp_pipe];
        [enc setBuffer:dst_buf offset:0 atIndex:0];
        [enc setBuffer:verify_buf offset:0 atIndex:1];
        [enc setBuffer:size_buf offset:0 atIndex:2];
        [enc setBuffer:off_buf offset:0 atIndex:3];
        [enc setBuffer:len_buf offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        t1=mach_absolute_time();
        double decomp_ms=NS(t1-t0)/1e6;

        int mismatch=memcmp([src_buf contents],[verify_buf contents],total);
        printf("  GPU decompress: %.2fms (%.2fGB/s), integrity: %s\n",
               decomp_ms, (double)total/(1ULL*1024*1024*1024)/(decomp_ms/1000.0),
               mismatch==0?"PERFECT":"MISMATCH");

        // CPU zlib
        size_t zlib_total=0;
        t0=mach_absolute_time();
        for(size_t p=0;p<npages;p++) {
            uLongf dlen=compressBound(16384); unsigned char *tmp=malloc(dlen);
            compress2(tmp,&dlen,(unsigned char*)[src_buf contents]+p*16384,16384,1);
            zlib_total+=dlen; free(tmp);
        }
        t1=mach_absolute_time();
        double zlib_ms=NS(t1-t0)/1e6;
        double zlib_ratio=zlib_total>0?(double)total/zlib_total:999;
        printf("  CPU zlib: %.2fms (%.2fGB/s), %.2fx | GPU speedup: %.2fx, ratio gap: %.2fx\n\n",
               zlib_ms,(double)total/(1ULL*1024*1024*1024)/(zlib_ms/1000.0),zlib_ratio,
               zlib_ms/comp_ms, zlib_ratio/ratio);
    }

    printf("=== PROBE 4v3 COMPLETE ===\n");
    return 0;
}
