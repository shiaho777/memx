// Probe 4: GPU-Native Lossless Page Compression
// Goal: Achieve high compression ratio (like zlib) at GPU speed (6+ GB/s)
// Key insight: Each threadgroup = 1 page, shared memory = hash table for LZ matches
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
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

// GPU LZ77 compressor: each threadgroup compresses one 16KB page
// Uses threadgroup memory as a 4KB hash table for match finding
static NSString *const lz_shader = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"constant uint PAGE_SIZE = 16384;\n"
"constant uint HASH_TABLE_SIZE = 4096; // must be power of 2\n"
"constant uint MIN_MATCH = 4;\n"
"constant uint MAX_MATCH = 64;\n"
"\n"
"// 4-byte hash for match finding\n"
"uint hash4(device const uchar* p) {\n"
"    return ((uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24)) * 2654435761u;\n"
"}\n"
"\n"
"// Token format:\n"
"//   literal run: [0x00..0x7F] [len-1] followed by len literal bytes\n"
"//   match:       [0x80..0xFF] [offset_lo] [offset_hi] [length-4]\n"
"//   offset = ((token & 0x7F) << 8) | offset_lo, with offset_hi for extended\n"
"\n"
"kernel void lz77_compress_page(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device uchar* dst_pages [[buffer(1)]],\n"
"    device uint* page_compressed_sizes [[buffer(2)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    // Shared hash table in threadgroup memory\n"
"    threadgroup uint hash_keys[HASH_TABLE_SIZE];\n"
"    threadgroup uint hash_vals[HASH_TABLE_SIZE];\n"
"    \n"
"    // Initialize hash table\n"
"    for (uint i = tid; i < HASH_TABLE_SIZE; i += tg_size) {\n"
"        hash_keys[i] = 0xFFFFFFFFu;\n"
"        hash_vals[i] = 0;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    device const uchar* src = src_pages + page_off;\n"
"    \n"
"    // Output buffer in threadgroup memory (avoid global mem conflicts)\n"
"    threadgroup uchar out_buf[PAGE_SIZE + 256]; // max compressed = slightly larger\n"
"    threadgroup uint out_pos;\n"
"    threadgroup uint src_pos;\n"
"    \n"
"    if (tid == 0) {\n"
"        out_pos = 0;\n"
"        src_pos = 0;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Sequential compression with parallel hash table updates\n"
"    // Thread 0 drives the compressor, others update hash table\n"
"    if (tid == 0) {\n"
"        while (src_pos < PAGE_SIZE && out_pos < PAGE_SIZE - 4) {\n"
"            uint pos = src_pos;\n"
"            \n"
"            if (pos + MIN_MATCH <= PAGE_SIZE) {\n"
"                uint h = hash4(src + pos) & (HASH_TABLE_SIZE - 1);\n"
"                uint prev_pos = hash_vals[h];\n"
"                uint prev_key = hash_keys[h];\n"
"                uint cur_key = (uint)src[pos] | ((uint)src[pos+1]<<8) | ((uint)src[pos+2]<<16) | ((uint)src[pos+3]<<24);\n"
"                \n"
"                // Update hash table\n"
"                hash_keys[h] = cur_key;\n"
"                hash_vals[h] = pos;\n"
"                \n"
"                // Check for match\n"
"                if (prev_key == cur_key && prev_pos < pos && (pos - prev_pos) < 65535) {\n"
"                    // Found a potential match - verify and count length\n"
"                    uint match_len = 0;\n"
"                    uint match_off = pos - prev_pos;\n"
"                    while (match_len < MAX_MATCH && pos + match_len < PAGE_SIZE && prev_pos + match_len < PAGE_SIZE) {\n"
"                        if (src[pos + match_len] == src[prev_pos + match_len]) match_len++;\n"
"                        else break;\n"
"                    }\n"
"                    \n"
"                    if (match_len >= MIN_MATCH) {\n"
"                        // Emit match token\n"
"                        uint off = match_off;\n"
"                        if (off < 128) {\n"
"                            out_buf[out_pos++] = 0x80 | (off >> 8);\n"
"                            out_buf[out_pos++] = off & 0xFF;\n"
"                        } else {\n"
"                            out_buf[out_pos++] = 0x80 | (off >> 16);\n"
"                            out_buf[out_pos++] = (off >> 8) & 0xFF;\n"
"                            out_buf[out_pos++] = off & 0xFF;\n"
"                        }\n"
"                        out_buf[out_pos++] = (uchar)(match_len - MIN_MATCH);\n"
"                        src_pos = pos + match_len;\n"
"                        continue;\n"
"                    }\n"
"                }\n"
"            }\n"
"            \n"
"            // No match - emit literal\n"
"            // Collect literal run\n"
"            uint lit_start = pos;\n"
"            uint lit_len = 1;\n"
"            \n"
"            while (lit_len < 127 && pos + lit_len < PAGE_SIZE && out_pos + lit_len + 2 < PAGE_SIZE) {\n"
"                uint next_pos = pos + lit_len;\n"
"                if (next_pos + MIN_MATCH <= PAGE_SIZE) {\n"
"                    uint h = hash4(src + next_pos) & (HASH_TABLE_SIZE - 1);\n"
"                    uint prev_p = hash_vals[h];\n"
"                    uint pk = hash_keys[h];\n"
"                    uint ck = (uint)src[next_pos] | ((uint)src[next_pos+1]<<8) | ((uint)src[next_pos+2]<<16) | ((uint)src[next_pos+3]<<24);\n"
"                    if (pk == ck && prev_p < next_pos && (next_pos - prev_p) < 65535) {\n"
"                        // Quick match check\n"
"                        uint ml = 0;\n"
"                        while (ml < MIN_MATCH && next_pos + ml < PAGE_SIZE) {\n"
"                            if (src[next_pos+ml] == src[prev_p+ml]) ml++; else break;\n"
"                        }\n"
"                        if (ml >= MIN_MATCH) break; // stop literal run\n"
"                    }\n"
"                    // Update hash\n"
"                    hash_keys[h] = ck;\n"
"                    hash_vals[h] = next_pos;\n"
"                }\n"
"                lit_len++;\n"
"            }\n"
"            \n"
"            // Emit literal token\n"
"            out_buf[out_pos++] = (uchar)(lit_len - 1); // 0x00..0x7E = 1..127 literals\n"
"            for (uint i = 0; i < lit_len; i++) {\n"
"                out_buf[out_pos++] = src[lit_start + i];\n"
"            }\n"
"            src_pos = pos + lit_len;\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Copy from threadgroup to global memory\n"
"    uint final_size = out_pos;\n"
"    for (uint i = tid; i < final_size; i += tg_size) {\n"
"        dst_pages[page_id * PAGE_SIZE + i] = out_buf[i];\n"
"    }\n"
"    \n"
"    if (tid == 0) {\n"
"        page_compressed_sizes[page_id] = final_size;\n"
"    }\n"
"}\n"
"\n"
"// Decompressor: each threadgroup decompresses one page\n"
"kernel void lz77_decompress_page(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device uchar* dst_pages [[buffer(1)]],\n"
"    device const uint* page_compressed_sizes [[buffer(2)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    device const uchar* src = src_pages + page_off;\n"
"    uint comp_size = page_compressed_sizes[page_id];\n"
"    \n"
"    threadgroup uchar out_buf[PAGE_SIZE];\n"
"    threadgroup uint out_pos;\n"
"    threadgroup uint in_pos;\n"
"    \n"
"    if (tid == 0) {\n"
"        out_pos = 0;\n"
"        in_pos = 0;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    if (tid == 0) {\n"
"        while (in_pos < comp_size && out_pos < PAGE_SIZE) {\n"
"            uchar token = src[in_pos++];\n"
"            \n"
"            if (token & 0x80) {\n"
"                // Match\n"
"                uint off_hi = (token & 0x7F);\n"
"                uint off_lo = src[in_pos++];\n"
"                uint offset = (off_hi << 8) | off_lo;\n"
"                if (offset == 0 && off_hi > 0) {\n"
"                    // Extended offset\n"
"                    offset = (off_hi << 16) | ((uint)src[in_pos] << 8) | (uint)src[in_pos+1];\n"
"                    in_pos += 2;\n"
"                }\n"
"                uint length = (uint)src[in_pos++] + MIN_MATCH;\n"
"                \n"
"                // Copy match (may overlap!)\n"
"                uint match_src = out_pos - offset;\n"
"                for (uint i = 0; i < length && out_pos < PAGE_SIZE; i++) {\n"
"                    out_buf[out_pos++] = out_buf[match_src + i];\n"
"                }\n"
"            } else {\n"
"                // Literal run\n"
"                uint lit_len = (uint)token + 1;\n"
"                for (uint i = 0; i < lit_len && out_pos < PAGE_SIZE && in_pos < comp_size; i++) {\n"
"                    out_buf[out_pos++] = src[in_pos++];\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Copy to global\n"
"    for (uint i = tid; i < PAGE_SIZE; i += tg_size) {\n"
"        dst_pages[page_off + i] = out_buf[i];\n"
"    }\n"
"}\n"
"\n"
"// Fast page dedup: hash + compare\n"
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
    if (!device) { printf("No Metal device!\n"); return 1; }

    printf("=== PROBE 4: GPU-Native Lossless Page Compression ===\n");
    printf("Device: %s, Unified: %s\n\n", [[device name] UTF8String],
           [device hasUnifiedMemory] ? "YES" : "NO");

    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:lz_shader options:nil error:&error];
    if (!lib) { printf("Shader error: %s\n", [[error localizedDescription] UTF8String]); return 1; }

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Test with different data patterns
    struct pattern { const char *name; int type; };
    struct pattern patterns[] = {
        {"All zeros", 0},
        {"Repeated 4-byte pattern", 1},
        {"Sparse data (90% zeros)", 2},
        {"Semi-structured (JSON-like)", 3},
        {"Random data", 4},
        {"Mixed real-world", 5}
    };
    int npatterns = sizeof(patterns)/sizeof(patterns[0]);

    size_t total = 64 * MB;
    size_t npages = total / 16384;

    for (int pi = 0; pi < npatterns; pi++) {
        printf("--- Pattern: %s ---\n", patterns[pi].name);

        void *data = mmap(NULL, total, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
        if (data == MAP_FAILED) continue;

        // Generate pattern
        switch (patterns[pi].type) {
        case 0: // all zeros
            memset(data, 0, total);
            break;
        case 1: // repeated 4-byte
            for (size_t i = 0; i < total; i += 4) {
                ((uint32_t*)data)[i/4] = (uint32_t)(i / 16384); // same per page
            }
            break;
        case 2: // sparse
            memset(data, 0, total);
            for (size_t i = 0; i < total; i += 160) { // 1 in 10 cache lines
                ((char*)data)[i] = (char)(i & 0xFF);
            }
            break;
        case 3: // JSON-like
            for (size_t p = 0; p < npages; p++) {
                char *page = (char*)data + p * 16384;
                const char *json_template = "{\"key\":\"value\",\"num\":12345,\"arr\":[1,2,3,4,5],\"name\":\"test_data_item\"}";
                size_t tlen = strlen(json_template);
                for (size_t off = 0; off < 16384; off += tlen) {
                    memcpy(page + off, json_template, tlen < 16384 - off ? tlen : 16384 - off);
                }
            }
            break;
        case 4: // random
            for (size_t i = 0; i < total; i++) {
                ((char*)data)[i] = (char)((i * 6364136223846793005ULL + 1442695040888963407ULL) >> 33);
            }
            break;
        case 5: // mixed: 30% zero, 30% repeated, 40% random
            memset(data, 0, total * 3 / 10);
            for (size_t i = total*3/10; i < total*6/10; i += 4) {
                ((uint32_t*)data)[i/4] = 0x42424242;
            }
            for (size_t i = total*6/10; i < total; i++) {
                ((char*)data)[i] = (char)((i * 6364136223846793005ULL + 1442695040888963407ULL) >> 33);
            }
            break;
        }

        // Create zero-copy Metal buffer
        id<MTLBuffer> src_buf = [device newBufferWithBytesNoCopy:data
                                                          length:total
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:^(void *p, NSUInteger l) { munmap(p,l); }];
        if (!src_buf) {
            src_buf = [device newBufferWithLength:total options:MTLResourceStorageModeShared];
            memcpy([src_buf contents], data, total);
            munmap(data, total);
        }

        id<MTLBuffer> dst_buf = [device newBufferWithLength:total options:MTLResourceStorageModeShared];
        id<MTLBuffer> size_buf = [device newBufferWithLength:npages * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> verify_buf = [device newBufferWithLength:total options:MTLResourceStorageModeShared];

        // GPU Compress
        id<MTLFunction> comp_func = [lib newFunctionWithName:@"lz77_compress_page"];
        id<MTLComputePipelineState> comp_pipe = [device newComputePipelineStateWithFunction:comp_func error:&error];
        if (!comp_pipe) { printf("  compress pipeline error: %s\n", [[error localizedDescription] UTF8String]); continue; }

        // Warmup
        for (int w = 0; w < 3; w++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:comp_pipe];
            [enc setBuffer:src_buf offset:0 atIndex:0];
            [enc setBuffer:dst_buf offset:0 atIndex:1];
            [enc setBuffer:size_buf offset:0 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        uint64_t t0 = mach_absolute_time();
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:comp_pipe];
        [enc setBuffer:src_buf offset:0 atIndex:0];
        [enc setBuffer:dst_buf offset:0 atIndex:1];
        [enc setBuffer:size_buf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        uint64_t t1 = mach_absolute_time();
        double comp_ms = NS(t1-t0)/1e6;

        // Calculate compression ratio
        uint32_t *sizes = (uint32_t*)[size_buf contents];
        size_t total_compressed = 0;
        for (size_t i = 0; i < npages; i++) {
            total_compressed += sizes[i];
        }
        double ratio = total_compressed > 0 ? (double)total / total_compressed : 999.0;

        printf("  GPU LZ77 compress: %.2f ms (%.2f GB/s), ratio %.2fx, %lluMB -> %lluMB\n",
               comp_ms, (double)total/(1024*1024*1024)/(comp_ms/1000.0), ratio,
               (unsigned long long)(total/MB), (unsigned long long)(total_compressed/MB));

        // GPU Decompress
        id<MTLFunction> decomp_func = [lib newFunctionWithName:@"lz77_decompress_page"];
        id<MTLComputePipelineState> decomp_pipe = [device newComputePipelineStateWithFunction:decomp_func error:&error];
        if (!decomp_pipe) { printf("  decompress pipeline error: %s\n", [[error localizedDescription] UTF8String]); continue; }

        // Warmup
        for (int w = 0; w < 3; w++) {
            id<MTLCommandBuffer> cb2 = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoder];
            [enc2 setComputePipelineState:decomp_pipe];
            [enc2 setBuffer:dst_buf offset:0 atIndex:0];
            [enc2 setBuffer:verify_buf offset:0 atIndex:1];
            [enc2 setBuffer:size_buf offset:0 atIndex:2];
            [enc2 dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc2 endEncoding];
            [cb2 commit];
            [cb2 waitUntilCompleted];
        }

        t0 = mach_absolute_time();
        cb = [queue commandBuffer];
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:decomp_pipe];
        [enc setBuffer:dst_buf offset:0 atIndex:0];
        [enc setBuffer:verify_buf offset:0 atIndex:1];
        [enc setBuffer:size_buf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        t1 = mach_absolute_time();
        double decomp_ms = NS(t1-t0)/1e6;

        // Verify
        int mismatch = memcmp([src_buf contents], [verify_buf contents], total);
        printf("  GPU LZ77 decompress: %.2f ms (%.2f GB/s), integrity: %s\n",
               decomp_ms, (double)total/(1024*1024*1024)/(decomp_ms/1000.0),
               mismatch == 0 ? "PERFECT" : "MISMATCH");

        // CPU zlib comparison
        size_t zlib_total = 0;
        t0 = mach_absolute_time();
        for (size_t p = 0; p < npages; p++) {
            uLongf dlen = compressBound(16384);
            unsigned char *tmp = malloc(dlen);
            compress2(tmp, &dlen, (unsigned char*)[src_buf contents] + p * 16384, 16384, 1);
            zlib_total += dlen;
            free(tmp);
        }
        t1 = mach_absolute_time();
        double zlib_ms = NS(t1-t0)/1e6;
        double zlib_ratio = zlib_total > 0 ? (double)total / zlib_total : 999.0;

        printf("  CPU zlib:            %.2f ms (%.2f GB/s), ratio %.2fx\n",
               zlib_ms, (double)total/(1024*1024*1024)/(zlib_ms/1000.0), zlib_ratio);
        printf("  GPU speedup: %.2fx, ratio gap: %.2fx\n\n",
               zlib_ms / comp_ms, zlib_ratio / ratio);
    }

    // ================================================================
    // Key experiment: End-to-end effective memory expansion
    // ================================================================
    printf("=== End-to-End Memory Expansion Simulation ===\n");
    {
        // Simulate a realistic memory workload:
        // - 8GB "hot" data that must stay uncompressed
        // - 16GB "warm" data that can be compressed
        // - GPU compresses warm data in background

        size_t hot_size = 1 * GB;  // scaled down for test
        size_t warm_size = 1 * GB;
        size_t total_test = hot_size + warm_size;

        printf("Simulating: %lluMB hot + %lluMB warm data\n",
               (unsigned long long)(hot_size/MB), (unsigned long long)(warm_size/MB));

        // Allocate and fill
        void *all_data = mmap(NULL, total_test, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
        if (all_data == MAP_FAILED) { printf("mmap failed\n"); return 1; }

        // Hot data: random-ish (incompressible)
        for (size_t i = 0; i < hot_size; i++) {
            ((char*)all_data)[i] = (char)((i * 6364136223846793005ULL) >> 33);
        }
        // Warm data: semi-compressible (typical app data)
        char *warm = (char*)all_data + hot_size;
        for (size_t p = 0; p < warm_size / 16384; p++) {
            char *page = warm + p * 16384;
            // 50% repeated patterns, 50% sparse
            for (size_t i = 0; i < 8192; i += 4) {
                ((uint32_t*)page)[i/4] = (uint32_t)(p * 0x42424242);
            }
            memset(page + 8192, 0, 8192);
        }

        // Create Metal buffers
        id<MTLBuffer> all_buf = [device newBufferWithBytesNoCopy:all_data
                                                          length:total_test
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:^(void *p, NSUInteger l) { munmap(p,l); }];
        if (!all_buf) {
            all_buf = [device newBufferWithLength:total_test options:MTLResourceStorageModeShared];
            memcpy([all_buf contents], all_data, total_test);
            munmap(all_data, total_test);
        }

        size_t warm_npages = warm_size / 16384;
        id<MTLBuffer> comp_buf = [device newBufferWithLength:warm_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> size_buf = [device newBufferWithLength:warm_npages * 4 options:MTLResourceStorageModeShared];

        id<MTLFunction> comp_func = [lib newFunctionWithName:@"lz77_compress_page"];
        id<MTLComputePipelineState> comp_pipe = [device newComputePipelineStateWithFunction:comp_func error:&error];

        // Compress warm data
        uint64_t t0 = mach_absolute_time();
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:comp_pipe];
        [enc setBuffer:all_buf offset:hot_size atIndex:0]; // warm data starts at hot_size
        [enc setBuffer:comp_buf offset:0 atIndex:1];
        [enc setBuffer:size_buf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(warm_npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        uint64_t t1 = mach_absolute_time();

        uint32_t *sizes = (uint32_t*)[size_buf contents];
        size_t compressed_warm = 0;
        for (size_t i = 0; i < warm_npages; i++) compressed_warm += sizes[i];

        double comp_time = NS(t1-t0)/1e6;
        double effective_total = hot_size + compressed_warm;
        double expansion = (double)total_test / effective_total;

        printf("Warm data compressed: %lluMB -> %lluMB (%.2fx) in %.2f ms\n",
               (unsigned long long)(warm_size/MB), (unsigned long long)(compressed_warm/MB),
               (double)warm_size/compressed_warm, comp_time);
        printf("Effective memory: %lluMB (was %lluMB)\n",
               (unsigned long long)(effective_total/MB), (unsigned long long)(total_test/MB));
        printf("MEMORY EXPANSION: %.2fx\n", expansion);
        printf("Compression throughput: %.2f GB/s\n",
               (double)warm_size/(1024*1024*1024)/(comp_time/1000.0));

        // Now test decompress on demand (simulate page fault handling)
        id<MTLFunction> decomp_func = [lib newFunctionWithName:@"lz77_decompress_page"];
        id<MTLComputePipelineState> decomp_pipe = [device newComputePipelineStateWithFunction:decomp_func error:&error];
        id<MTLBuffer> decomp_buf = [device newBufferWithLength:warm_size options:MTLResourceStorageModeShared];

        // Decompress a single page (worst case latency)
        size_t single_page = 1;
        t0 = mach_absolute_time();
        for (int r = 0; r < 100; r++) {
            cb = [queue commandBuffer];
            enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:decomp_pipe];
            [enc setBuffer:comp_buf offset:0 atIndex:0];
            [enc setBuffer:decomp_buf offset:0 atIndex:1];
            [enc setBuffer:size_buf offset:0 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(single_page,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        t1 = mach_absolute_time();
        double single_decomp_ns = NS(t1-t0) / 100;

        printf("\nSingle page decompress latency: %.0f ns (%.0f us)\n",
               single_decomp_ns, single_decomp_ns/1000);
        printf("Compare: SSD random read ~100us, DRAM page fault ~5us\n");
        if (single_decomp_ns < 100000) {
            printf("*** GPU decompress is FASTER than SSD! ***\n");
        }
        if (single_decomp_ns < 10000) {
            printf("*** GPU decompress approaches DRAM speed! ***\n");
        }
    }

    printf("\n=== PROBE 4 COMPLETE ===\n");
    return 0;
}
