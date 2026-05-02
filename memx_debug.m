// Debug: single page Delta+LZ77 compress/decompress with byte-level verification
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach.h>
#include <mach/mach_time.h>

#import <Metal/Metal.h>

#define PAGE_SIZE 16384

static double ns_per_tick;
static void init_time(void) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    ns_per_tick = (double)info.numer / (double)info.denom;
}
#define NS(ticks) ((double)(ticks) * ns_per_tick)

// Same shader as v0.4
static NSString *const shader = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"constant uint PAGE_SIZE = 16384;\n"
"constant uint BLK = 64;\n"
"constant uint NBLK = 256;\n"
"constant uint HT_SIZE = 4096;\n"
"constant uint MIN_MATCH = 4;\n"
"constant uint MAX_MATCH = 258;\n"
"\n"
"uint hash4_tg(threadgroup const uchar* p) {\n"
"    return ((uint)p[0] | ((uint)p[1]<<8) | ((uint)p[2]<<16) | ((uint)p[3]<<24)) * 2654435761u;\n"
"}\n"
"\n"
"kernel void compress_page(\n"
"    device const uchar* src_pages [[buffer(0)]],\n"
"    device uchar* dst_pages [[buffer(1)]],\n"
"    device uint* comp_sizes [[buffer(2)]],\n"
"    uint tid [[thread_position_in_threadgroup]],\n"
"    uint page_id [[threadgroup_position_in_grid]],\n"
"    uint tg_size [[threads_per_threadgroup]]\n"
") {\n"
"    threadgroup uchar delta_page[PAGE_SIZE];\n"
"    threadgroup uint ht_keys[2048];\n"
"    threadgroup uint ht_vals[2048];\n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    if (tid < NBLK) {\n"
"        if (tid == 0) {\n"
"            delta_page[0] = src_pages[page_off];\n"
"            for (uint i = 1; i < BLK; i++)\n"
"                delta_page[i] = src_pages[page_off + i] - src_pages[page_off + i - 1];\n"
"        } else {\n"
"            // Cross-block delta: first byte = this_block[0] - prev_block[63]\n"
"            delta_page[tid * BLK] = src_pages[page_off + tid * BLK] - src_pages[page_off + tid * BLK - 1];\n"
"            for (uint i = 1; i < BLK; i++)\n"
"                delta_page[tid * BLK + i] = src_pages[page_off + tid * BLK + i] - src_pages[page_off + tid * BLK + i - 1];\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint i = tid; i < 2048; i += tg_size) { ht_keys[i] = 0xFFFFFFFFu; ht_vals[i] = 0; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) {\n"
"        uint ip = 0, op = 4;\n"
"        uint dst_base = page_id * PAGE_SIZE;\n"
"        dst_pages[dst_base] = 0x4D; dst_pages[dst_base+1] = 0x58;\n"
"        dst_pages[dst_base+2] = 1; dst_pages[dst_base+3] = 0;\n"
"        while (ip < PAGE_SIZE && op < PAGE_SIZE - 6) {\n"
"            if (ip + MIN_MATCH <= PAGE_SIZE) {\n"
"                uint h = hash4_tg(delta_page + ip) & 2047;\n"
"                uint prev_pos = ht_vals[h];\n"
"                uint prev_key = ht_keys[h];\n"
"                uint cur_key = (uint)delta_page[ip]|((uint)delta_page[ip+1]<<8)|((uint)delta_page[ip+2]<<16)|((uint)delta_page[ip+3]<<24);\n"
"                ht_keys[h] = cur_key; ht_vals[h] = ip;\n"
"                if (prev_key == cur_key && prev_pos < ip && (ip - prev_pos) < 4096) {\n"
"                    uint ml = 0, off = ip - prev_pos;\n"
"                    while (ml < MAX_MATCH && ip+ml < PAGE_SIZE && delta_page[ip+ml] == delta_page[prev_pos+ml]) ml++;\n"
"                    if (ml >= MIN_MATCH) {\n"
"                        dst_pages[dst_base+op++] = 0xFF;\n"
"                        dst_pages[dst_base+op++] = (uchar)(off & 0xFF);\n"
"                        dst_pages[dst_base+op++] = (uchar)((off >> 8) & 0xFF);\n"
"                        dst_pages[dst_base+op++] = (uchar)(ml & 0xFF);\n"
"                        dst_pages[dst_base+op++] = (uchar)((ml >> 8) & 0xFF);\n"
"                        ip += ml; continue;\n"
"                    }\n"
"                }\n"
"            }\n"
"            if (delta_page[ip] == 0xFF) { dst_pages[dst_base+op++] = 0xFE; dst_pages[dst_base+op++] = 0xFF; }\n"
"            else if (delta_page[ip] == 0xFE) { dst_pages[dst_base+op++] = 0xFE; dst_pages[dst_base+op++] = 0xFE; }\n"
"            else { dst_pages[dst_base+op++] = delta_page[ip]; }\n"
"            ip++;\n"
"        }\n"
"        if (op >= PAGE_SIZE) comp_sizes[page_id] = PAGE_SIZE;\n"
"        else comp_sizes[page_id] = op;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (comp_sizes[page_id] == PAGE_SIZE) {\n"
"        for (uint i = tid; i < PAGE_SIZE; i += tg_size)\n"
"            dst_pages[page_id * PAGE_SIZE + i] = src_pages[page_off + i];\n"
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
"    if (comp_size == PAGE_SIZE) {\n"
"        for (uint i = tid; i < PAGE_SIZE; i += tg_size)\n"
"            dst_pages[page_off + i] = src_pages[src_base + i];\n"
"        return;\n"
"    }\n"
"    if (src_pages[src_base] != 0x4D || src_pages[src_base+1] != 0x58) {\n"
"        for (uint i = tid; i < PAGE_SIZE; i += tg_size)\n"
"            dst_pages[page_off + i] = src_pages[src_base + i];\n"
"        return;\n"
"    }\n"
"    threadgroup uchar delta_buf[PAGE_SIZE];\n"
"    if (tid == 0) {\n"
"        uint ip = 4, op = 0;\n"
"        while (ip < comp_size && op < PAGE_SIZE) {\n"
"            uchar b = src_pages[src_base + ip];\n"
"            if (b == 0xFF && ip + 4 < comp_size) {\n"
"                ip++;\n"
"                uint off = (uint)src_pages[src_base+ip] | (((uint)src_pages[src_base+ip+1]) << 8);\n"
"                ip += 2;\n"
"                uint ml = (uint)src_pages[src_base+ip] | (((uint)src_pages[src_base+ip+1]) << 8);\n"
"                ip += 2;\n"
"                uint match_src = op - off;\n"
"                for (uint i = 0; i < ml && op < PAGE_SIZE; i++)\n"
"                    delta_buf[op++] = delta_buf[match_src + i];\n"
"            } else if (b == 0xFE && ip + 1 < comp_size) {\n"
"                ip++;\n"
"                delta_buf[op++] = src_pages[src_base + ip++];\n"
"            } else {\n"
"                delta_buf[op++] = b;\n"
"                ip++;\n"
"            }\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) {\n"
"        dst_pages[page_off] = delta_buf[0];\n"
"        for (uint i = 1; i < PAGE_SIZE; i++)\n"
"            dst_pages[page_off + i] = dst_pages[page_off + i - 1] + delta_buf[i];\n"
"    }\n"
"}\n";

int main(void) {
    init_time();
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    printf("=== Delta+LZ77 Debug (Single Page) ===\n\n");

    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:shader options:nil error:&error];
    if (!lib) { printf("Shader error: %s\n", [[error localizedDescription] UTF8String]); return 1; }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLComputePipelineState> comp_pipe = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"compress_page"] error:&error];
    id<MTLComputePipelineState> decomp_pipe = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"decompress_page"] error:&error];

    // Test patterns
    const char *names[] = {"JSON", "Source code", "Log file", "App heap", "All zeros"};
    int types[] = {1, 4, 7, 0, 6};

    for (int t = 0; t < 5; t++) {
        printf("--- %s ---\n", names[t]);
        unsigned char *page = calloc(1, PAGE_SIZE);

        switch(types[t]) {
        case 1: {
            const char *json = "{\"id\":12345,\"name\":\"user_name_here\",\"email\":\"test@example.com\",\"active\":true,\"score\":98.6,\"tags\":[\"admin\",\"user\"],\"meta\":{\"k1\":\"v1\",\"k2\":42}}";
            size_t jl = strlen(json);
            for (size_t o = 0; o < PAGE_SIZE; o += jl)
                memcpy(page+o, json, jl < (PAGE_SIZE-o) ? jl : (PAGE_SIZE-o));
            break;
        }
        case 4: {
            const char *code = "int main(int argc, char *argv[]) {\n    printf(\"Hello, World!\\n\");\n    for (int i = 0; i < 10; i++) {\n        result += process_item(data[i]);\n    }\n    return 0;\n}\n";
            size_t cl = strlen(code);
            for (size_t o = 0; o < PAGE_SIZE; o += cl)
                memcpy(page+o, code, cl < (PAGE_SIZE-o) ? cl : (PAGE_SIZE-o));
            break;
        }
        case 7: {
            const char *log = "[2024-01-15 10:23:45] INFO  [main] Processing request from 192.168.1.100: user_id=12345 action=login status=success latency=42ms\n";
            size_t ll = strlen(log);
            for (size_t o = 0; o < PAGE_SIZE; o += ll)
                memcpy(page+o, log, ll < (PAGE_SIZE-o) ? ll : (PAGE_SIZE-o));
            break;
        }
        case 0:
            for (size_t i = PAGE_SIZE*6/10; i < PAGE_SIZE*8/10; i += 8) {
                uint64_t p = 0x0000000100000000ULL + (i & 0xFFFF);
                memcpy(page+i, &p, 8);
            }
            for (size_t i = PAGE_SIZE*8/10; i < PAGE_SIZE; i += 4) {
                uint32_t v = (uint32_t)(i % 256);
                memcpy(page+i, &v, 4);
            }
            break;
        case 6:
            memset(page, 0, PAGE_SIZE);
            break;
        }

        id<MTLBuffer> src_buf = [device newBufferWithLength:PAGE_SIZE options:MTLResourceStorageModeShared];
        memcpy([src_buf contents], page, PAGE_SIZE);
        id<MTLBuffer> dst_buf = [device newBufferWithLength:PAGE_SIZE options:MTLResourceStorageModeShared];
        id<MTLBuffer> size_buf = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> verify_buf = [device newBufferWithLength:PAGE_SIZE options:MTLResourceStorageModeShared];

        // Compress
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:comp_pipe];
        [enc setBuffer:src_buf offset:0 atIndex:0];
        [enc setBuffer:dst_buf offset:0 atIndex:1];
        [enc setBuffer:size_buf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

        uint32_t comp_size = *(uint32_t*)[size_buf contents];
        printf("  Compressed: %u -> %u bytes (%.2fx)\n", PAGE_SIZE, comp_size, (double)PAGE_SIZE/comp_size);

        // Dump first 32 bytes of compressed output
        unsigned char *comp_data = (unsigned char*)[dst_buf contents];
        printf("  First 32 bytes: ");
        for (int i = 0; i < 32 && i < (int)comp_size; i++) printf("%02X ", comp_data[i]);
        printf("\n");

        // Decompress
        cb = [queue commandBuffer]; enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:decomp_pipe];
        [enc setBuffer:dst_buf offset:0 atIndex:0];
        [enc setBuffer:verify_buf offset:0 atIndex:1];
        [enc setBuffer:size_buf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

        unsigned char *decomp_data = (unsigned char*)[verify_buf contents];
        int mismatch = memcmp(page, decomp_data, PAGE_SIZE);
        printf("  Integrity: %s\n", mismatch == 0 ? "PERFECT ✅" : "MISMATCH ❌");

        if (mismatch) {
            // Find first mismatch
            for (int i = 0; i < PAGE_SIZE; i++) {
                if (page[i] != decomp_data[i]) {
                    printf("  First mismatch at byte %d: expected 0x%02X, got 0x%02X\n",
                           i, page[i], decomp_data[i]);
                    // Show context
                    printf("  Context (expected): ");
                    for (int j = (i>8?i-8:0); j < i+8 && j < PAGE_SIZE; j++) printf("%02X ", page[j]);
                    printf("\n  Context (got):      ");
                    for (int j = (i>8?i-8:0); j < i+8 && j < PAGE_SIZE; j++) printf("%02X ", decomp_data[j]);
                    printf("\n");
                    break;
                }
            }
        }
        printf("\n");
        free(page);
    }

    return 0;
}
