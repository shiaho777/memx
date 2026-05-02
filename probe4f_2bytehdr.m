// Probe 4v6: Fixed 2-byte header per block (type + data_length)
// This makes decompression straightforward - no scanning needed
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

// Format per page:
//   Header: 512 bytes = 256 x (type:1byte, data_len:1byte)
//   Data: compressed blocks sequentially
// Block types:
//   0 = ZERO (data_len=0)
//   1 = CONSTANT (data_len=1, 1 byte value)
//   2 = RLE (data_len=N, count-value pairs)
//   3 = RAW (data_len=64)
static NSString *const shader = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"constant uint PAGE_SIZE = 16384;\n"
"constant uint BLK = 64;\n"
"constant uint NBLK = 256;\n"
"constant uint HDR_SIZE = 512; // 256 * 2\n"
"\n"
"struct BlockHdr { uchar type; uchar len; };\n"
"\n"
"kernel void compress_page(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device uchar* dst_pages [[buffer(1)]],\n"
"    device uint* comp_sizes [[buffer(2)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    threadgroup BlockHdr headers[NBLK];       // 512B\n"
"    threadgroup uchar comp_data[NBLK * BLK];   // 16KB\n"
"    \n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    uint blk_off = page_off + tid * BLK;\n"
"    \n"
"    uchar block[BLK];\n"
"    for (uint i = 0; i < BLK; i++) block[i] = src_pages[blk_off + i];\n"
"    \n"
"    bool all_zero = true, all_same = true;\n"
"    uchar first = block[0];\n"
"    for (uint i = 0; i < BLK; i++) {\n"
"        if (block[i] != 0) all_zero = false;\n"
"        if (block[i] != first) all_same = false;\n"
"    }\n"
"    \n"
"    uint base = tid * BLK;\n"
"    \n"
"    if (all_zero) {\n"
"        headers[tid].type = 0;\n"
"        headers[tid].len = 0;\n"
"    } else if (all_same) {\n"
"        headers[tid].type = 1;\n"
"        headers[tid].len = 1;\n"
"        comp_data[base] = first;\n"
"    } else {\n"
"        uchar rle[BLK * 2];\n"
"        uint rle_len = 0;\n"
"        uchar count = 1, prev = block[0];\n"
"        for (uint i = 1; i < BLK; i++) {\n"
"            if (block[i] == prev && count < 250) count++;\n"
"            else { rle[rle_len++] = count; rle[rle_len++] = prev; count = 1; prev = block[i]; }\n"
"        }\n"
"        rle[rle_len++] = count; rle[rle_len++] = prev;\n"
"        \n"
"        if (rle_len < BLK) {\n"
"            headers[tid].type = 2;\n"
"            headers[tid].len = (uchar)rle_len;\n"
"            for (uint i = 0; i < rle_len; i++) comp_data[base + i] = rle[i];\n"
"        } else {\n"
"            headers[tid].type = 3;\n"
"            headers[tid].len = BLK;\n"
"            for (uint i = 0; i < BLK; i++) comp_data[base + i] = block[i];\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Compute data offset via prefix sum\n"
"    threadgroup uint data_offs[NBLK];\n"
"    if (tid == 0) {\n"
"        data_offs[0] = HDR_SIZE;\n"
"        for (uint i = 1; i < NBLK; i++) data_offs[i] = data_offs[i-1] + headers[i-1].len;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Write header (2 bytes per block)\n"
"    uint dst_base = page_id * PAGE_SIZE;\n"
"    \n"
"    // Check if compressed would exceed original\n"
"    // Thread 0 computes total size first\n"
"    threadgroup uint total_comp_size;\n"
"    if (tid == 0) {\n"
"        total_comp_size = data_offs[NBLK-1] + headers[NBLK-1].len;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    if (total_comp_size > PAGE_SIZE) {\n"
"        // Incompressible - write raw page\n"
"        for (uint i = tid; i < PAGE_SIZE; i += tg_size) {\n"
"            dst_pages[dst_base + i] = src_pages[page_off + i];\n"
"        }\n"
"        if (tid == 0) comp_sizes[page_id] = PAGE_SIZE;\n"
"    } else {\n"
"        // Write compressed format\n"
"        dst_pages[dst_base + tid * 2] = headers[tid].type;\n"
"        dst_pages[dst_base + tid * 2 + 1] = headers[tid].len;\n"
"        for (uint i = 0; i < headers[tid].len; i++) {\n"
"            dst_pages[dst_base + data_offs[tid] + i] = comp_data[tid * BLK + i];\n"
"        }\n"
"        if (tid == 0) comp_sizes[page_id] = total_comp_size;\n"
"    }\n"
"}\n"
"\n"
"kernel void decompress_page(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device uchar* dst_pages [[buffer(1)]],\n"
"    device const uint* comp_sizes [[buffer(2)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    uint src_base = page_id * PAGE_SIZE;\n"
"    uint comp_size = comp_sizes[page_id];\n"
"    \n"
"    // Check if page is incompressible (stored raw)\n"
"    if (comp_size == PAGE_SIZE) {\n"
"        // Raw page - just copy\n"
"        for (uint i = tid; i < PAGE_SIZE; i += tg_size) {\n"
"            dst_pages[page_off + i] = src_pages[src_base + i];\n"
"        }\n"
"        return;\n"
"    }\n"
"    \n"
"    // Read this block's header\n"
"    uchar blk_type = src_pages[src_base + tid * 2];\n"
"    uchar blk_len = src_pages[src_base + tid * 2 + 1];\n"
"    \n"
"    // Compute data offset: HDR_SIZE + sum of all previous blocks' lengths\n"
"    // Thread 0 computes all offsets\n"
"    threadgroup uint data_offs[NBLK];\n"
"    if (tid == 0) {\n"
"        data_offs[0] = HDR_SIZE;\n"
"        for (uint i = 1; i < NBLK; i++) {\n"
"            data_offs[i] = data_offs[i-1] + src_pages[src_base + (i-1) * 2 + 1];\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Decompress\n"
"    uchar block[BLK];\n"
"    uint data_start = data_offs[tid];\n"
"    \n"
"    if (blk_type == 0) {\n"
"        for (uint i = 0; i < BLK; i++) block[i] = 0;\n"
"    } else if (blk_type == 1) {\n"
"        uchar val = src_pages[src_base + data_start];\n"
"        for (uint i = 0; i < BLK; i++) block[i] = val;\n"
"    } else if (blk_type == 2) {\n"
"        // RLE decode\n"
"        uint pos = 0;\n"
"        uint out_pos = 0;\n"
"        while (pos < blk_len && out_pos < BLK) {\n"
"            uchar count = src_pages[src_base + data_start + pos];\n"
"            pos++;\n"
"            uchar value = src_pages[src_base + data_start + pos];\n"
"            pos++;\n"
"            for (uchar j = 0; j < count && out_pos < BLK; j++) {\n"
"                block[out_pos++] = value;\n"
"            }\n"
"        }\n"
"    } else { // type 3 = RAW\n"
"        for (uint i = 0; i < BLK; i++) block[i] = src_pages[src_base + data_start + i];\n"
"    }\n"
"    \n"
"    for (uint i = 0; i < BLK; i++) dst_pages[page_off + tid * BLK + i] = block[i];\n"
"}\n";

int main(void) {
    init_time();
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { printf("No Metal!\n"); return 1; }

    printf("=== PROBE 4v6: 2-Byte Header Format (Lossless) ===\n");
    printf("Device: %s\n\n", [[device name] UTF8String]);

    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:shader options:nil error:&error];
    if (!lib) { printf("Shader error: %s\n", [[error localizedDescription] UTF8String]); return 1; }

    id<MTLCommandQueue> queue = [device newCommandQueue];

    struct { const char *name; int type; } patterns[] = {
        {"All zeros", 0}, {"All same (0x42)", 6}, {"Repeated 4B", 1},
        {"Sparse", 2}, {"JSON-like", 3}, {"Random", 4}, {"Mixed", 5}
    };
    int npat = 7;
    size_t total = 64 * MB;
    size_t npages = total / 16384;

    double sum_gpu_speed=0, sum_zlib_speed=0, sum_gpu_ratio=0, sum_zlib_ratio=0;
    int perfect_count = 0;

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
        double gpu_gbps = (double)total/(1ULL*1024*1024*1024)/(comp_ms/1000.0);

        printf("  GPU compress: %.2fms (%.1fGB/s), %.2fx, %llu->%lluMB\n",
               comp_ms, gpu_gbps, ratio,
               (unsigned long long)(total/MB), (unsigned long long)(comp_total/MB));

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

        int mismatch=memcmp([src_buf contents],[verify_buf contents],total);
        printf("  GPU decompress: %.2fms (%.1fGB/s), integrity: %s\n",
               decomp_ms, (double)total/(1ULL*1024*1024*1024)/(decomp_ms/1000.0),
               mismatch==0?"** PERFECT **":"MISMATCH");

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
        double zlib_gbps=(double)total/(1ULL*1024*1024*1024)/(zlib_ms/1000.0);

        printf("  CPU zlib: %.2fms (%.1fGB/s), %.2fx | speedup: %.1fx\n\n",
               zlib_ms, zlib_gbps, zlib_ratio, zlib_ms/comp_ms);

        if (mismatch==0) {
            sum_gpu_speed+=gpu_gbps; sum_zlib_speed+=zlib_gbps;
            sum_gpu_ratio+=ratio; sum_zlib_ratio+=zlib_ratio;
            perfect_count++;
        }
    }

    printf("========================================\n");
    printf("SUMMARY (%d/%d patterns with PERFECT integrity)\n", perfect_count, npat);
    if (perfect_count>0) {
        printf("  Avg GPU: %.1f GB/s, %.2fx compression\n", sum_gpu_speed/perfect_count, sum_gpu_ratio/perfect_count);
        printf("  Avg zlib: %.1f GB/s, %.2fx compression\n", sum_zlib_speed/perfect_count, sum_zlib_ratio/perfect_count);
        printf("  GPU is %.1fx faster, zlib is %.1fx better compression\n",
               sum_gpu_speed/sum_zlib_speed, sum_zlib_ratio/sum_gpu_ratio);
    }
    printf("\n=== PROBE 4v6 COMPLETE ===\n");
    return 0;
}
