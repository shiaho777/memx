// MemX v0.3: Delta+RLE compression - paradigm shift for real data
// Key insight: Real data has HIGH spatial correlation.
// Delta encoding turns correlated data into mostly-zeros, which RLE crushes.
// This is what makes FLAC, PNG, and other codecs work.
//
// Pipeline: [Delta encode 64B block] → [RLE encode deltas] → [store]
// Reverse:  [RLE decode] → [Delta decode (prefix sum)] → [original]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/mach_time.h>

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
"constant uint PAGE_SIZE = 16384;\n"
"constant uint BLK = 64;\n"
"constant uint NBLK = 256;\n"
"constant uint HDR_SIZE = 512;\n"
"struct BlockHdr { uchar type; uchar len; };\n"
"\n"
"// Delta encode a block: delta[0]=block[0], delta[i]=block[i]-block[i-1]\n"
"void delta_encode(uchar block[BLK], uchar deltas[BLK]) {\n"
"    deltas[0] = block[0];\n"
"    for (uint i = 1; i < BLK; i++) deltas[i] = block[i] - block[i-1];\n"
"}\n"
"\n"
"kernel void compress_page(\n"
"    device const uchar* src [[buffer(0)]],\n"
"    device uchar* dst [[buffer(1)]],\n"
"    device uint* sizes [[buffer(2)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    threadgroup BlockHdr headers[NBLK];\n"
"    threadgroup uchar comp_data[NBLK * BLK];\n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    uint blk_off = page_off + tid * BLK;\n"
"    \n"
"    uchar block[BLK];\n"
"    for (uint i = 0; i < BLK; i++) block[i] = src[blk_off + i];\n"
"    \n"
"    // Delta encode\n"
"    uchar deltas[BLK];\n"
"    delta_encode(block, deltas);\n"
"    \n"
"    // Analyze deltas\n"
"    bool all_zero = true, all_same = true;\n"
"    uchar first = deltas[0];\n"
"    for (uint i = 0; i < BLK; i++) { if (deltas[i]!=0) all_zero=false; if (deltas[i]!=first) all_same=false; }\n"
"    \n"
"    uint base = tid * BLK;\n"
"    \n"
"    if (all_zero) {\n"
"        // Constant block: all values same → deltas are [val, 0, 0, ...]\n"
"        headers[tid].type = 0; // ZERO deltas = constant original\n"
"        headers[tid].len = 0;\n"
"        // We still need to store the first value!\n"
"        // Use type 1 instead\n"
"        headers[tid].type = 1;\n"
"        headers[tid].len = 1;\n"
"        comp_data[base] = block[0]; // store original first value\n"
"    } else if (all_same) {\n"
"        // Linear progression: block[i] = block[0] + i*step\n"
"        headers[tid].type = 2; // LINEAR: store first value + step\n"
"        headers[tid].len = 2;\n"
"        comp_data[base] = block[0];\n"
"        comp_data[base+1] = deltas[1]; // step = delta[1]\n"
"    } else {\n"
"        // RLE the deltas\n"
"        uchar rle[BLK * 2];\n"
"        uint rle_len = 0;\n"
"        uchar count = 1, prev = deltas[0];\n"
"        for (uint i = 1; i < BLK; i++) {\n"
"            if (deltas[i] == prev && count < 250) count++;\n"
"            else { rle[rle_len++] = count; rle[rle_len++] = prev; count = 1; prev = deltas[i]; }\n"
"        }\n"
"        rle[rle_len++] = count; rle[rle_len++] = prev;\n"
"        \n"
"        if (rle_len < BLK) {\n"
"            headers[tid].type = 3; // RLE-delta\n"
"            headers[tid].len = (uchar)rle_len;\n"
"            for (uint i = 0; i < rle_len; i++) comp_data[base + i] = rle[i];\n"
"        } else {\n"
"            // Try raw delta (might be smaller than raw original if deltas are small)\n"
"            // Check if raw deltas are worth it vs raw original\n"
"            // For now, just store raw original\n"
"            headers[tid].type = 4; // RAW\n"
"            headers[tid].len = BLK;\n"
"            for (uint i = 0; i < BLK; i++) comp_data[base + i] = block[i];\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    threadgroup uint data_offs[NBLK];\n"
"    if (tid==0) { data_offs[0]=HDR_SIZE; for(uint i=1;i<NBLK;i++) data_offs[i]=data_offs[i-1]+headers[i-1].len; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    threadgroup uint total_comp_size;\n"
"    if (tid==0) total_comp_size = data_offs[NBLK-1]+headers[NBLK-1].len;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    uint dst_base = page_id * PAGE_SIZE;\n"
"    if (total_comp_size > PAGE_SIZE) {\n"
"        for (uint i=tid; i<PAGE_SIZE; i+=tg_size) dst[dst_base+i] = src[page_off+i];\n"
"        if (tid==0) sizes[page_id] = PAGE_SIZE;\n"
"    } else {\n"
"        dst[dst_base+tid*2] = headers[tid].type;\n"
"        dst[dst_base+tid*2+1] = headers[tid].len;\n"
"        for (uint i=0; i<headers[tid].len; i++) dst[dst_base+data_offs[tid]+i] = comp_data[tid*BLK+i];\n"
"        if (tid==0) sizes[page_id] = total_comp_size;\n"
"    }\n"
"}\n"
"\n"
"kernel void decompress_page(\n"
"    device const uchar* src [[buffer(0)]],\n"
"    device uchar* dst [[buffer(1)]],\n"
"    device const uint* sizes [[buffer(2)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    uint src_base = page_id * PAGE_SIZE;\n"
"    uint comp_size = sizes[page_id];\n"
"    if (comp_size == PAGE_SIZE) {\n"
"        for (uint i=tid; i<PAGE_SIZE; i+=tg_size) dst[page_off+i] = src[src_base+i];\n"
"        return;\n"
"    }\n"
"    uchar blk_type = src[src_base + tid*2];\n"
"    uchar blk_len = src[src_base + tid*2 + 1];\n"
"    threadgroup uint data_offs[NBLK];\n"
"    if (tid==0) { data_offs[0]=HDR_SIZE; for(uint i=1;i<NBLK;i++) data_offs[i]=data_offs[i-1]+src[src_base+(i-1)*2+1]; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    uchar block[BLK];\n"
"    uint ds = data_offs[tid];\n"
"    \n"
"    if (blk_type == 1) {\n"
"        // CONSTANT: all values = first byte\n"
"        uchar val = src[src_base + ds];\n"
"        for (uint i = 0; i < BLK; i++) block[i] = val;\n"
"    } else if (blk_type == 2) {\n"
"        // LINEAR: block[i] = first + i * step\n"
"        uchar first = src[src_base + ds];\n"
"        uchar step = src[src_base + ds + 1];\n"
"        block[0] = first;\n"
"        for (uint i = 1; i < BLK; i++) block[i] = block[i-1] + step;\n"
"    } else if (blk_type == 3) {\n"
"        // RLE-delta: decode RLE → deltas → prefix sum → original\n"
"        uchar deltas[BLK];\n"
"        uint p = 0, o = 0;\n"
"        while (p < blk_len && o < BLK) {\n"
"            uchar c = src[src_base + ds + p]; p++;\n"
"            uchar v = src[src_base + ds + p]; p++;\n"
"            for (uchar j = 0; j < c && o < BLK; j++) deltas[o++] = v;\n"
"        }\n"
"        // Delta decode (prefix sum)\n"
"        block[0] = deltas[0];\n"
"        for (uint i = 1; i < BLK; i++) block[i] = block[i-1] + deltas[i];\n"
"    } else { // type 4 = RAW\n"
"        for (uint i = 0; i < BLK; i++) block[i] = src[src_base + ds + i];\n"
"    }\n"
"    \n"
"    for (uint i = 0; i < BLK; i++) dst[page_off + tid * BLK + i] = block[i];\n"
"}\n";

int main(void) {
    init_time();
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { printf("No Metal!\n"); return 1; }

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  MemX v0.3: Delta+RLE Compression Evaluation     ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");
    printf("Device: %s\n\n", [[device name] UTF8String]);

    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:shader options:nil error:&error];
    if (!lib) { printf("Shader error: %s\n", [[error localizedDescription] UTF8String]); return 1; }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLFunction> comp_func = [lib newFunctionWithName:@"compress_page"];
    id<MTLComputePipelineState> comp_pipe = [device newComputePipelineStateWithFunction:comp_func error:&error];
    id<MTLFunction> decomp_func = [lib newFunctionWithName:@"decompress_page"];
    id<MTLComputePipelineState> decomp_pipe = [device newComputePipelineStateWithFunction:decomp_func error:&error];

    // Real-world data patterns
    struct { const char *name; int type; } patterns[] = {
        {"1. App heap (zeros+pointers)", 0},
        {"2. JSON documents", 1},
        {"3. Database rows", 2},
        {"4. RGBA pixel data (gradient)", 3},
        {"5. Source code", 4},
        {"6. Browser tabs (mixed)", 5},
        {"7. All zeros", 6},
        {"8. Log files (repeated text)", 7},
    };
    int npat = 8;
    size_t total = 256 * MB;
    size_t npages = total / 16384;

    double sum_ratio = 0, sum_speed = 0, sum_zlib_ratio = 0, sum_zlib_speed = 0;
    int perfect_count = 0;

    for (int pi = 0; pi < npat; pi++) {
        printf("--- %s ---\n", patterns[pi].name);

        void *data = calloc(1, total);
        if (!data) continue;

        switch (patterns[pi].type) {
        case 0: // App heap
            for (size_t i = total*6/10; i < total*8/10; i += 8) {
                uint64_t ptr = 0x0000000100000000ULL + (i & 0xFFFF);
                memcpy((char*)data+i, &ptr, 8);
            }
            for (size_t i = total*8/10; i < total; i += 4) {
                uint32_t v = (uint32_t)(i % 256);
                memcpy((char*)data+i, &v, 4);
            }
            break;
        case 1: // JSON
        {
            const char *json = "{\"id\":12345,\"name\":\"user_name_here\",\"email\":\"test@example.com\",\"active\":true,\"score\":98.6,\"tags\":[\"admin\",\"user\"],\"meta\":{\"k1\":\"v1\",\"k2\":42}}";
            size_t jl = strlen(json);
            for (size_t o = 0; o < total; o += jl)
                memcpy((char*)data+o, json, jl < (total-o) ? jl : (total-o));
            break;
        }
        case 2: // Database rows
        {
            char row[128];
            for (int c = 0; c < 128; c++) row[c] = (c < 4) ? c : (c < 20) ? 'A'+(c%26) : (c < 100) ? 0 : (c%16);
            for (size_t o = 0; o < total; o += 128)
                memcpy((char*)data+o, row, 128);
            break;
        }
        case 3: // RGBA gradient
            for (size_t i = 0; i < total; i += 4) {
                ((unsigned char*)data)[i+0] = (unsigned char)((i/4) & 0xFF);
                ((unsigned char*)data)[i+1] = (unsigned char)(((i/4)>>8) & 0xFF);
                ((unsigned char*)data)[i+2] = (unsigned char)(((i/4)>>16) & 0xFF);
                ((unsigned char*)data)[i+3] = 0xFF;
            }
            break;
        case 4: // Source code
        {
            const char *code = "int main(int argc, char *argv[]) {\n    printf(\"Hello, World!\\n\");\n    for (int i = 0; i < 10; i++) {\n        result += process_item(data[i]);\n    }\n    return 0;\n}\n";
            size_t cl = strlen(code);
            for (size_t o = 0; o < total; o += cl)
                memcpy((char*)data+o, code, cl < (total-o) ? cl : (total-o));
            break;
        }
        case 5: // Browser tabs
        {
            const char *html = "<!DOCTYPE html><html><head><title>Page</title></head><body><div class=\"content\"><p>Hello world</p></div></body></html>";
            size_t hl = strlen(html);
            for (size_t o = total*3/10; o < total*7/10; o += hl)
                memcpy((char*)data+o, html, hl < (total*7/10-o) ? hl : (total*7/10-o));
            for (size_t i = total*7/10; i < total; i++)
                ((char*)data)[i] = (char)((i * 6364136223846793005ULL + 1442695040888963407ULL) >> 33);
            break;
        }
        case 6: // All zeros
            memset(data, 0, total);
            break;
        case 7: // Log files
        {
            const char *log = "[2024-01-15 10:23:45] INFO  [main] Processing request from 192.168.1.100: user_id=12345 action=login status=success latency=42ms\n";
            size_t ll = strlen(log);
            for (size_t o = 0; o < total; o += ll)
                memcpy((char*)data+o, log, ll < (total-o) ? ll : (total-o));
            break;
        }
        }

        id<MTLBuffer> src_buf = [device newBufferWithLength:total options:MTLResourceStorageModeShared];
        memcpy([src_buf contents], data, total);
        free(data);

        id<MTLBuffer> dst_buf = [device newBufferWithLength:total options:MTLResourceStorageModeShared];
        id<MTLBuffer> size_buf = [device newBufferWithLength:npages*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> verify_buf = [device newBufferWithLength:total options:MTLResourceStorageModeShared];

        // GPU compress
        uint64_t t0 = mach_absolute_time();
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:comp_pipe];
        [enc setBuffer:src_buf offset:0 atIndex:0];
        [enc setBuffer:dst_buf offset:0 atIndex:1];
        [enc setBuffer:size_buf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        uint64_t t1 = mach_absolute_time();
        double comp_ms = NS(t1-t0)/1e6;

        uint32_t *sizes = (uint32_t*)[size_buf contents];
        size_t comp_total = 0;
        for (size_t i = 0; i < npages; i++) comp_total += sizes[i];
        double ratio = (double)total / comp_total;
        double speed = (double)total / (1ULL*1024*1024*1024) / (comp_ms/1000.0);

        // GPU decompress
        cb = [queue commandBuffer];
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:decomp_pipe];
        [enc setBuffer:dst_buf offset:0 atIndex:0];
        [enc setBuffer:verify_buf offset:0 atIndex:1];
        [enc setBuffer:size_buf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

        int mismatch = memcmp([src_buf contents], [verify_buf contents], total);

        // CPU zlib
        size_t zlib_total = 0;
        t0 = mach_absolute_time();
        for (size_t p = 0; p < npages; p++) {
            uLongf dlen = compressBound(16384);
            unsigned char *tmp = malloc(dlen);
            compress2(tmp, &dlen, (unsigned char*)[src_buf contents]+p*16384, 16384, 1);
            zlib_total += dlen;
            free(tmp);
        }
        t1 = mach_absolute_time();
        double zlib_ms = NS(t1-t0)/1e6;
        double zlib_ratio = (double)total / zlib_total;
        double zlib_speed = (double)total / (1ULL*1024*1024*1024) / (zlib_ms/1000.0);

        printf("  GPU: %.2fx @ %.1f GB/s | zlib: %.2fx @ %.2f GB/s | %s",
               ratio, speed, zlib_ratio, zlib_speed,
               mismatch==0 ? "PERFECT ✅" : "MISMATCH ❌");
        if (mismatch == 0 && ratio > 1.0)
            printf(" | GPU %.1fx faster, zlib %.1fx better ratio", speed/zlib_speed, zlib_ratio/ratio);
        printf("\n");

        if (mismatch == 0) {
            sum_ratio += ratio;
            sum_speed += speed;
            sum_zlib_ratio += zlib_ratio;
            sum_zlib_speed += zlib_speed;
            perfect_count++;
        }
    }

    // Batch decompress latency (amortized)
    printf("\n═══ Batch Decompress Latency ═══\n");
    {
        // Create compressed data for 1000 pages
        size_t bench_pages = 1000;
        size_t bench_size = bench_pages * 16384;
        void *bench_data = malloc(bench_size);
        for (size_t i = 0; i < bench_size; i++)
            ((char*)bench_data)[i] = (char)((i * 6364136223846793005ULL) >> 33);
        // Make it semi-compressible
        for (size_t i = 0; i < bench_size; i += 32)
            memset((char*)bench_data + i, 0, 16);

        id<MTLBuffer> bsrc = [device newBufferWithLength:bench_size options:MTLResourceStorageModeShared];
        memcpy([bsrc contents], bench_data, bench_size);
        id<MTLBuffer> bdst = [device newBufferWithLength:bench_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsz = [device newBufferWithLength:bench_pages*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bvb = [device newBufferWithLength:bench_size options:MTLResourceStorageModeShared];
        free(bench_data);

        // Compress
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:comp_pipe];
        [enc setBuffer:bsrc offset:0 atIndex:0];
        [enc setBuffer:bdst offset:0 atIndex:1];
        [enc setBuffer:bsz offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(bench_pages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

        // Measure batch decompress for different batch sizes
        int batch_sizes[] = {1, 10, 100, 1000};
        for (int bi = 0; bi < 4; bi++) {
            int bs = batch_sizes[bi];
            int iters = 1000 / bs;
            uint64_t t0 = mach_absolute_time();
            for (int r = 0; r < iters; r++) {
                cb = [queue commandBuffer]; enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:decomp_pipe];
                [enc setBuffer:bdst offset:0 atIndex:0];
                [enc setBuffer:bvb offset:0 atIndex:1];
                [enc setBuffer:bsz offset:0 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(bs,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }
            uint64_t t1 = mach_absolute_time();
            double per_page_us = NS(t1-t0) / (iters * bs) / 1000;
            printf("  Batch %4d pages: %.1f μs/page amortized\n", bs, per_page_us);
        }
        printf("  SSD random read: ~100 μs/page\n");
        printf("  DRAM page fault: ~5 μs\n");
    }

    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║              DELTA+RLE SUMMARY                   ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    if (perfect_count > 0) {
        printf("║  GPU avg compression: %.2fx                       ║\n", sum_ratio/perfect_count);
        printf("║  GPU avg speed:       %.1f GB/s                   ║\n", sum_speed/perfect_count);
        printf("║  zlib avg compression:%.2fx                       ║\n", sum_zlib_ratio/perfect_count);
        printf("║  zlib avg speed:      %.2f GB/s                   ║\n", sum_zlib_speed/perfect_count);
        printf("║                                                  ║\n");
        printf("║  GPU is %.1fx faster, zlib is %.1fx better ratio  ║\n",
               sum_speed/sum_zlib_speed, sum_zlib_ratio/sum_ratio);
        printf("║  24GB → %.0fGB (GPU) vs %.0fGB (zlib)           ║\n",
               24.0 * sum_ratio/perfect_count,
               24.0 * sum_zlib_ratio/perfect_count);
    }
    printf("╚══════════════════════════════════════════════════╝\n");

    return 0;
}
