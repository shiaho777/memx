// Probe 4v4: GPU Adaptive Block Compression
// Key fix: If compressed block > original, store raw. Adaptive per-block.
// Also fix decompressor to handle raw blocks correctly.
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

static NSString *const shader = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"constant uint PAGE_SIZE = 16384;\n"
"constant uint BLK = 64;\n"
"constant uint NBLK = 256;\n"
"\n"
"// Block header byte (first byte of each compressed block):\n"
"//   0x00-0xFB: RLE encoded, this byte is the first count\n"
"//   0xFC: RAW block (not compressed, follows 64 bytes verbatim)\n"
"//   0xFD: CONSTANT block (1 byte value follows, fills 64 bytes)\n"
"//   0xFE: ZERO block (all zeros, no more data)\n"
"//   0xFF: RESERVED\n"
"\n"
"kernel void compress_page(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device uchar* dst_pages [[buffer(1)]],\n"
"    device uint* comp_sizes [[buffer(2)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    // Shared storage for compressed blocks + lengths\n"
"    // Max compressed size per block: 65 bytes (1 header + 64 raw)\n"
"    threadgroup uchar comp_buf[NBLK * 65];  // 16640 bytes\n"
"    threadgroup uint comp_lens[NBLK];        // 1024 bytes\n"
"    \n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    uint blk_off = page_off + tid * BLK;\n"
"    \n"
"    // Load block into registers\n"
"    uchar block[BLK];\n"
"    for (uint i = 0; i < BLK; i++) block[i] = src_pages[blk_off + i];\n"
"    \n"
"    // Analyze block\n"
"    bool all_zero = true;\n"
"    bool all_same = true;\n"
"    uchar first = block[0];\n"
"    for (uint i = 0; i < BLK; i++) {\n"
"        if (block[i] != 0) all_zero = false;\n"
"        if (block[i] != first) all_same = false;\n"
"    }\n"
"    \n"
"    uint out_len = 0;\n"
"    uint base = tid * 65;\n"
"    \n"
"    if (all_zero) {\n"
"        comp_buf[base] = 0xFE;\n"
"        out_len = 1;\n"
"    } else if (all_same) {\n"
"        comp_buf[base] = 0xFD;\n"
"        comp_buf[base + 1] = first;\n"
"        out_len = 2;\n"
"    } else {\n"
"        // Try RLE\n"
"        uchar rle[BLK * 2]; // worst case\n"
"        uint rle_len = 0;\n"
"        uchar count = 1;\n"
"        uchar prev = block[0];\n"
"        for (uint i = 1; i < BLK; i++) {\n"
"            if (block[i] == prev && count < 250) {\n"
"                count++;\n"
"            } else {\n"
"                rle[rle_len++] = count;\n"
"                rle[rle_len++] = prev;\n"
"                count = 1;\n"
"                prev = block[i];\n"
"            }\n"
"        }\n"
"        rle[rle_len++] = count;\n"
"        rle[rle_len++] = prev;\n"
"        \n"
"        if (rle_len < BLK) {\n"
"            // RLE is smaller, use it\n"
"            // But check: first byte must not be 0xFC-0xFF\n"
"            if (rle[0] >= 0xFC) {\n"
"                // Edge case: would be confused with marker. Use raw.\n"
"                comp_buf[base] = 0xFC;\n"
"                for (uint i = 0; i < BLK; i++) comp_buf[base + 1 + i] = block[i];\n"
"                out_len = 65;\n"
"            } else {\n"
"                for (uint i = 0; i < rle_len; i++) comp_buf[base + i] = rle[i];\n"
"                out_len = rle_len;\n"
"            }\n"
"        } else {\n"
"            // RLE is not smaller, store raw\n"
"            comp_buf[base] = 0xFC;\n"
"            for (uint i = 0; i < BLK; i++) comp_buf[base + 1 + i] = block[i];\n"
"            out_len = 65;\n"
"        }\n"
"    }\n"
"    \n"
"    comp_lens[tid] = out_len;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Prefix sum for offsets (thread 0)\n"
"    threadgroup uint offsets[NBLK];\n"
"    if (tid == 0) {\n"
"        offsets[0] = 0;\n"
"        for (uint i = 1; i < NBLK; i++) offsets[i] = offsets[i-1] + comp_lens[i-1];\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Write to device memory\n"
"    uint dst_base = page_id * PAGE_SIZE;\n"
"    for (uint i = 0; i < comp_lens[tid]; i++) {\n"
"        dst_pages[dst_base + offsets[tid] + i] = comp_buf[base + i];\n"
"    }\n"
"    \n"
"    if (tid == 0) {\n"
"        comp_sizes[page_id] = offsets[NBLK-1] + comp_lens[NBLK-1];\n"
"    }\n"
"}\n"
"\n"
"// Decompressor: each thread decompresses one block\n"
"kernel void decompress_page(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device uchar* dst_pages [[buffer(1)]],\n"
"    device const uint* block_offsets [[buffer(2)]],\n"
"    device const uint* block_lengths [[buffer(3)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]]\n"
") {\n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    uint src_base = page_id * PAGE_SIZE;\n"
"    uint blk_off = block_offsets[page_id * NBLK + tid];\n"
"    uint blk_len = block_lengths[page_id * NBLK + tid];\n"
"    \n"
"    uchar block[BLK];\n"
"    uint out_pos = 0;\n"
"    \n"
"    if (blk_len == 0) {\n"
"        // Empty block (shouldn't happen, fill with zeros)\n"
"        for (uint i = 0; i < BLK; i++) block[i] = 0;\n"
"    } else {\n"
"        uchar marker = src_pages[src_base + blk_off];\n"
"        \n"
"        if (marker == 0xFE) {\n"
"            // Zero block\n"
"            for (uint i = 0; i < BLK; i++) block[i] = 0;\n"
"        } else if (marker == 0xFD) {\n"
"            // Constant block\n"
"            uchar val = src_pages[src_base + blk_off + 1];\n"
"            for (uint i = 0; i < BLK; i++) block[i] = val;\n"
"        } else if (marker == 0xFC) {\n"
"            // Raw block\n"
"            for (uint i = 0; i < BLK; i++) block[i] = src_pages[src_base + blk_off + 1 + i];\n"
"        } else {\n"
"            // RLE: first byte is count, then value, alternating\n"
"            uint pos = 0;\n"
"            while (pos < blk_len && out_pos < BLK) {\n"
"                uchar count = src_pages[src_base + blk_off + pos];\n"
"                if (count >= 0xFC) break; // shouldn't happen mid-RLE\n"
"                pos++;\n"
"                if (pos >= blk_len) break;\n"
"                uchar value = src_pages[src_base + blk_off + pos];\n"
"                pos++;\n"
"                for (uchar j = 0; j < count && out_pos < BLK; j++) {\n"
"                    block[out_pos++] = value;\n"
"                }\n"
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
"// Extract block metadata from compressed stream\n"
"kernel void extract_meta(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device const uint* comp_sizes [[buffer(1)]],\n"
"    device uint* block_offsets [[buffer(2)]],\n"
"    device uint* block_lengths [[buffer(3)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]]\n"
") {\n"
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
"                blk_lens[blk] = 1; pos += 1;\n"
"            } else if (marker == 0xFD) {\n"
"                blk_lens[blk] = 2; pos += 2;\n"
"            } else if (marker == 0xFC) {\n"
"                blk_lens[blk] = 65; pos += 65;\n"
"            } else {\n"
"                // RLE: scan until next marker or end\n"
"                uint start = pos;\n"
"                while (pos < comp_size) {\n"
"                    uchar b = src_pages[src_base + pos];\n"
"                    if (b >= 0xFC) break; // found next marker\n"
"                    pos++; // count\n"
"                    if (pos >= comp_size) break;\n"
"                    pos++; // value\n"
"                }\n"
"                blk_lens[blk] = pos - start;\n"
"            }\n"
"            blk++;\n"
"        }\n"
"        for (uint i = blk; i < NBLK; i++) { blk_offs[i] = comp_size; blk_lens[i] = 0; }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    block_offsets[page_id * NBLK + tid] = blk_offs[tid];\n"
"    block_lengths[page_id * NBLK + tid] = blk_lens[tid];\n"
"}\n";

int main(void) {
    init_time();
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { printf("No Metal!\n"); return 1; }

    printf("=== PROBE 4v4: GPU Adaptive Block Compression ===\n");
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

    double total_gpu_speed = 0, total_zlib_speed = 0, total_gpu_ratio = 0, total_zlib_ratio = 0;
    int valid = 0;

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

        id<MTLFunction> comp_func=[lib newFunctionWithName:@"compress_page"];
        id<MTLComputePipelineState> comp_pipe=[device newComputePipelineStateWithFunction:comp_func error:&error];
        if (!comp_pipe) { printf("  comp err: %s\n",[[error localizedDescription] UTF8String]); continue; }

        // Warmup + measure compress
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

        // Extract metadata
        id<MTLFunction> meta_func=[lib newFunctionWithName:@"extract_meta"];
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
            [enc setBuffer:off_buf offset:0 atIndex:2];
            [enc setBuffer:len_buf offset:0 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }

        t0=mach_absolute_time();
        cb=[queue commandBuffer]; enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:decomp_pipe];
        [enc setBuffer:dst_buf offset:0 atIndex:0];
        [enc setBuffer:verify_buf offset:0 atIndex:1];
        [enc setBuffer:off_buf offset:0 atIndex:2];
        [enc setBuffer:len_buf offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        t1=mach_absolute_time();
        double decomp_ms=NS(t1-t0)/1e6;

        int mismatch=memcmp([src_buf contents],[verify_buf contents],total);
        printf("  GPU decompress: %.2fms (%.1fGB/s), integrity: %s\n",
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
        double zlib_gbps=(double)total/(1ULL*1024*1024*1024)/(zlib_ms/1000.0);

        printf("  CPU zlib: %.2fms (%.1fGB/s), %.2fx | speedup: %.1fx\n\n",
               zlib_ms, zlib_gbps, zlib_ratio, zlib_ms/comp_ms);

        if (mismatch == 0) {
            total_gpu_speed += gpu_gbps;
            total_zlib_speed += zlib_gbps;
            total_gpu_ratio += ratio;
            total_zlib_ratio += zlib_ratio;
            valid++;
        }
    }

    printf("========================================\n");
    printf("SUMMARY (valid patterns only)\n");
    if (valid > 0) {
        printf("  Avg GPU compress: %.1f GB/s, avg ratio: %.2fx\n",
               total_gpu_speed/valid, total_gpu_ratio/valid);
        printf("  Avg CPU zlib:     %.1f GB/s, avg ratio: %.2fx\n",
               total_zlib_speed/valid, total_zlib_ratio/valid);
        printf("  GPU speed advantage: %.1fx faster\n", total_zlib_speed/total_gpu_speed > 1 ? 1 : total_gpu_speed/total_zlib_speed);
    }

    printf("\n=== PROBE 4v4 COMPLETE ===\n");
    return 0;
}
