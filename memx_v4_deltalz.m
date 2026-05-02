// MemX v0.4: Hybrid Delta+RLE + Cross-block LZ77
// Key innovation: Use 4KB sub-pages (64 blocks of 64B) with cross-block LZ77
// Each threadgroup handles one 4KB sub-page with a shared hash table
// This allows finding matches ACROSS blocks within the sub-page
//
// Strategy per 16KB page (4 sub-pages of 4KB):
//   Phase 1: Delta encode each 64B block (parallel, 64 threads)
//   Phase 2: LZ77 compress delta-encoded sub-page (sequential, thread 0)
//   Phase 3: If LZ77 doesn't help, fall back to RLE per block
//   Phase 4: If nothing helps, store raw
//
// This should dramatically improve compression on text data because:
//   - Delta encoding turns repeated patterns into zeros
//   - LZ77 can then find repeated delta patterns across blocks
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
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

// New approach: Each 16KB page = 1 threadgroup of 256 threads
// All 256 threads cooperate on one page
// Phase 1 (parallel): Each thread delta-encodes its 64B block
// Phase 2 (thread 0): LZ77 compress the entire delta-encoded page
//   using a hash table in threadgroup memory
// Phase 3: Store compressed or raw

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
"uint hash4(device const uchar* p) {\n"
"    return ((uint)p[0] | ((uint)p[1]<<8) | ((uint)p[2]<<16) | ((uint)p[3]<<24)) * 2654435761u;\n"
"}\n"
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
"    // Threadgroup memory layout:\n"
"    //   delta_page[16384] = delta-encoded page (16KB)\n"
"    //   ht_keys[4096] = hash table keys (16KB)\n"
"    //   ht_vals[4096] = hash table values (16KB)\n"
"    //   Total = 48KB > 32KB limit!\n"
"    // \n"
"    // Solution: Use smaller hash table (2048 entries = 16KB)\n"
"    //   delta_page[16384] = 16KB\n"
"    //   ht_keys[2048] = 8KB\n"
"    //   ht_vals[2048] = 8KB\n"
"    //   Total = 32KB exactly!\n"
"    \n"
"    threadgroup uchar delta_page[PAGE_SIZE];  // 16KB\n"
"    threadgroup uint ht_keys[2048];           // 8KB\n"
"    threadgroup uint ht_vals[2048];           // 8KB\n"
"    \n"
"    uint page_off = page_id * PAGE_SIZE;\n"
"    \n"
"    // Phase 1: Delta encode (parallel)\n"
"    // Each thread handles one 64B block\n"
"    uint blk_off = page_off + tid * BLK;\n"
"    if (tid < NBLK) {\n"
"        if (tid == 0) {\n"
"            delta_page[0] = src_pages[page_off];\n"
"            for (uint i = 1; i < BLK; i++)\n"
"                delta_page[i] = src_pages[page_off + i] - src_pages[page_off + i - 1];\n"
"        } else {\n"
"            delta_page[tid * BLK] = src_pages[page_off + tid * BLK] - src_pages[page_off + tid * BLK - 1];\n"
"            for (uint i = 1; i < BLK; i++)\n"
"                delta_page[tid * BLK + i] = src_pages[page_off + tid * BLK + i] - src_pages[page_off + tid * BLK + i - 1];\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Phase 2: LZ77 compress the delta page (thread 0)\n"
"    // Initialize hash table\n"
"    for (uint i = tid; i < 2048; i += tg_size) {\n"
"        ht_keys[i] = 0xFFFFFFFFu;\n"
"        ht_vals[i] = 0;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Output buffer in threadgroup (max PAGE_SIZE + small header)\n"
"    // We'll write directly to device memory to save threadgroup space\n"
"    \n"
"    if (tid == 0) {\n"
"        uint ip = 0, op = 4;\n"
"        uint dst_base = page_id * PAGE_SIZE;\n"
"        \n"
"        // First pass: just count output size to check if compressible\n"
"        // We simulate compression without writing to dst\n"
"        // This avoids corrupting dst if page is incompressible\n"
"        uint sim_op = 4; // simulated output position\n"
"        uint sim_ip = 0;\n"
"        while (sim_ip < PAGE_SIZE && sim_op < PAGE_SIZE) {\n"
"            if (sim_ip + MIN_MATCH <= PAGE_SIZE) {\n"
"                uint h = hash4_tg(delta_page + sim_ip) & 2047;\n"
"                uint prev_pos = ht_vals[h];\n"
"                uint prev_key = ht_keys[h];\n"
"                uint cur_key = (uint)delta_page[sim_ip]|((uint)delta_page[sim_ip+1]<<8)|((uint)delta_page[sim_ip+2]<<16)|((uint)delta_page[sim_ip+3]<<24);\n"
"                ht_keys[h] = cur_key; ht_vals[h] = sim_ip;\n"
"                if (prev_key == cur_key && prev_pos < sim_ip && (sim_ip - prev_pos) < 4096) {\n"
"                    uint ml = 0, off = sim_ip - prev_pos;\n"
"                    while (ml < MAX_MATCH && sim_ip+ml < PAGE_SIZE && delta_page[sim_ip+ml] == delta_page[prev_pos+ml]) ml++;\n"
"                    if (ml >= MIN_MATCH) { sim_op += 5; sim_ip += ml; continue; }\n"
"                }\n"
"            }\n"
"            if (delta_page[sim_ip] == 0xFF || delta_page[sim_ip] == 0xFE) sim_op += 2;\n"
"            else sim_op++;\n"
"            sim_ip++;\n"
"        }\n"
"        \n"
"        if (sim_op >= PAGE_SIZE) {\n"
"            comp_sizes[page_id] = PAGE_SIZE; // incompressible\n"
"        } else {\n"
"            // Second pass: actually compress (reset hash table)\n"
"            // Hash table was modified in sim pass, but that's ok -\n"
"            // we'll just use the final state, which is actually better\n"
"            // since it has more history. But we need to reset ip.\n"
"            // Actually, the hash table state from the simulation IS the final state.\n"
"            // We can't reuse it because positions are from sim pass.\n"
"            // Reset hash table and do real compression.\n"
"            for (uint i = 0; i < 2048; i++) { ht_keys[i] = 0xFFFFFFFFu; ht_vals[i] = 0; }\n"
"            \n"
"            dst_pages[dst_base] = 0x4D; dst_pages[dst_base+1] = 0x58;\n"
"            dst_pages[dst_base+2] = 1; dst_pages[dst_base+3] = 0;\n"
"            while (ip < PAGE_SIZE && op < PAGE_SIZE - 6) {\n"
"                if (ip + MIN_MATCH <= PAGE_SIZE) {\n"
"                    uint h = hash4_tg(delta_page + ip) & 2047;\n"
"                    uint prev_pos = ht_vals[h];\n"
"                    uint prev_key = ht_keys[h];\n"
"                    uint cur_key = (uint)delta_page[ip]|((uint)delta_page[ip+1]<<8)|((uint)delta_page[ip+2]<<16)|((uint)delta_page[ip+3]<<24);\n"
"                    ht_keys[h] = cur_key; ht_vals[h] = ip;\n"
"                    if (prev_key == cur_key && prev_pos < ip && (ip - prev_pos) < 4096) {\n"
"                        uint ml = 0, off = ip - prev_pos;\n"
"                        while (ml < MAX_MATCH && ip+ml < PAGE_SIZE && delta_page[ip+ml] == delta_page[prev_pos+ml]) ml++;\n"
"                        if (ml >= MIN_MATCH) {\n"
"                            dst_pages[dst_base+op++] = 0xFF;\n"
"                            dst_pages[dst_base+op++] = (uchar)(off & 0xFF);\n"
"                            dst_pages[dst_base+op++] = (uchar)((off >> 8) & 0xFF);\n"
"                            dst_pages[dst_base+op++] = (uchar)(ml & 0xFF);\n"
"                            dst_pages[dst_base+op++] = (uchar)((ml >> 8) & 0xFF);\n"
"                            ip += ml; continue;\n"
"                        }\n"
"                    }\n"
"                }\n"
"                if (delta_page[ip] == 0xFF) { dst_pages[dst_base+op++] = 0xFE; dst_pages[dst_base+op++] = 0xFF; }\n"
"                else if (delta_page[ip] == 0xFE) { dst_pages[dst_base+op++] = 0xFE; dst_pages[dst_base+op++] = 0xFE; }\n"
"                else { dst_pages[dst_base+op++] = delta_page[ip]; }\n"
"                ip++;\n"
"            }\n"
"            comp_sizes[page_id] = op;\n"
"        }\n"
"    }\n"
"    \n"
"    // If incompressible, copy raw (all threads help)\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    // Read comp_sizes from device memory (written by thread 0)\n"
"    if (comp_sizes[page_id] == PAGE_SIZE) {\n"
"        for (uint i = tid; i < PAGE_SIZE; i += tg_size) {\n"
"            dst_pages[page_id * PAGE_SIZE + i] = src_pages[page_off + i];\n"
"        }\n"
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
"    if (comp_size == PAGE_SIZE) {\n"
"        // Raw page\n"
"        for (uint i = tid; i < PAGE_SIZE; i += tg_size)\n"
"            dst_pages[page_off + i] = src_pages[src_base + i];\n"
"        return;\n"
"    }\n"
"    \n"
"    // Check format marker\n"
"    if (src_pages[src_base] != 0x4D || src_pages[src_base+1] != 0x58) {\n"
"        // Unknown format, copy raw\n"
"        for (uint i = tid; i < PAGE_SIZE; i += tg_size)\n"
"            dst_pages[page_off + i] = src_pages[src_base + i];\n"
"        return;\n"
"    }\n"
"    \n"
"    // LZ77 decompress to delta_page, then delta decode\n"
"    threadgroup uchar delta_buf[PAGE_SIZE];\n"
"    \n"
"    if (tid == 0) {\n"
"        uint ip = 4; // skip header\n"
"        uint op = 0;\n"
"        \n"
"        while (ip < comp_size && op < PAGE_SIZE) {\n"
"            uchar b = src_pages[src_base + ip];\n"
"            if (b == 0xFF && ip + 4 < comp_size) {\n"
"                // Match\n"
"                ip++;\n"
"                uint off = (uint)src_pages[src_base+ip] | (((uint)src_pages[src_base+ip+1]) << 8);\n"
"                ip += 2;\n"
"                uint ml = (uint)src_pages[src_base+ip] | (((uint)src_pages[src_base+ip+1]) << 8);\n"
"                ip += 2;\n"
"                uint match_src = op - off;\n"
"                for (uint i = 0; i < ml && op < PAGE_SIZE; i++) {\n"
"                    delta_buf[op++] = delta_buf[match_src + i];\n"
"                }\n"
"            } else if (b == 0xFE && ip + 1 < comp_size) {\n"
"                // Escaped literal\n"
"                ip++;\n"
"                delta_buf[op++] = src_pages[src_base + ip++];\n"
"            } else {\n"
"                // Normal literal\n"
"                delta_buf[op++] = b;\n"
"                ip++;\n"
"            }\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    \n"
"    // Delta decode (sequential - thread 0 does prefix sum across entire page)\n"
"    // This is necessary because delta decode has cross-block dependencies\n"
"    if (tid == 0) {\n"
"        dst_pages[page_off] = delta_buf[0];\n"
"        for (uint i = 1; i < PAGE_SIZE; i++) {\n"
"            dst_pages[page_off + i] = dst_pages[page_off + i - 1] + delta_buf[i];\n"
"        }\n"
"    }\n"
"}\n";

int main(void) {
    init_time();
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { printf("No Metal!\n"); return 1; }

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  MemX v0.4: Delta+LZ77 Hybrid Compression       ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");
    printf("Device: %s\n\n", [[device name] UTF8String]);

    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:shader options:nil error:&error];
    if (!lib) { printf("Shader error: %s\n", [[error localizedDescription] UTF8String]); return 1; }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLFunction> comp_func = [lib newFunctionWithName:@"compress_page"];
    id<MTLComputePipelineState> comp_pipe = [device newComputePipelineStateWithFunction:comp_func error:&error];
    if (!comp_pipe) { printf("comp error: %s\n", [[error localizedDescription] UTF8String]); return 1; }
    id<MTLFunction> decomp_func = [lib newFunctionWithName:@"decompress_page"];
    id<MTLComputePipelineState> decomp_pipe = [device newComputePipelineStateWithFunction:decomp_func error:&error];
    if (!decomp_pipe) { printf("decomp error: %s\n", [[error localizedDescription] UTF8String]); return 1; }

    struct { const char *name; int type; } patterns[] = {
        {"1. App heap", 0}, {"2. JSON", 1}, {"3. Database", 2},
        {"4. RGBA gradient", 3}, {"5. Source code", 4}, {"6. Browser tabs", 5},
        {"7. All zeros", 6}, {"8. Log files", 7},
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
        switch(patterns[pi].type) {
        case 0:
            for(size_t i=total*6/10;i<total*8/10;i+=8){uint64_t p=0x0000000100000000ULL+(i&0xFFFF);memcpy((char*)data+i,&p,8);}
            for(size_t i=total*8/10;i<total;i+=4){uint32_t v=(uint32_t)(i%256);memcpy((char*)data+i,&v,4);}
            break;
        case 1: {const char*j="{\"id\":12345,\"name\":\"user_name_here\",\"email\":\"test@example.com\",\"active\":true,\"score\":98.6,\"tags\":[\"admin\",\"user\"],\"meta\":{\"k1\":\"v1\",\"k2\":42}}";size_t jl=strlen(j);for(size_t o=0;o<total;o+=jl)memcpy((char*)data+o,j,jl<(total-o)?jl:(total-o));break;}
        case 2: {char r[128];for(int c=0;c<128;c++)r[c]=(c<4)?c:(c<20)?'A'+(c%26):(c<100)?0:(c%16);for(size_t o=0;o<total;o+=128)memcpy((char*)data+o,r,128);break;}
        case 3: for(size_t i=0;i<total;i+=4){((unsigned char*)data)[i]=(unsigned char)((i/4)&0xFF);((unsigned char*)data)[i+1]=(unsigned char)(((i/4)>>8)&0xFF);((unsigned char*)data)[i+2]=(unsigned char)(((i/4)>>16)&0xFF);((unsigned char*)data)[i+3]=0xFF;}break;
        case 4: {const char*c="int main(int argc, char *argv[]) {\n    printf(\"Hello, World!\\n\");\n    for (int i = 0; i < 10; i++) {\n        result += process_item(data[i]);\n    }\n    return 0;\n}\n";size_t cl=strlen(c);for(size_t o=0;o<total;o+=cl)memcpy((char*)data+o,c,cl<(total-o)?cl:(total-o));break;}
        case 5: {const char*h="<!DOCTYPE html><html><head><title>Page</title></head><body><div class=\"content\"><p>Hello world</p></div></body></html>";size_t hl=strlen(h);for(size_t o=total*3/10;o<total*7/10;o+=hl)memcpy((char*)data+o,h,hl<(total*7/10-o)?hl:(total*7/10-o));for(size_t i=total*7/10;i<total;i++)((char*)data)[i]=(char)((i*6364136223846793005ULL+1442695040888963407ULL)>>33);break;}
        case 6: memset(data,0,total);break;
        case 7: {const char*l="[2024-01-15 10:23:45] INFO  [main] Processing request from 192.168.1.100: user_id=12345 action=login status=success latency=42ms\n";size_t ll=strlen(l);for(size_t o=0;o<total;o+=ll)memcpy((char*)data+o,l,ll<(total-o)?ll:(total-o));break;}
        }

        id<MTLBuffer> src_buf = [device newBufferWithLength:total options:MTLResourceStorageModeShared];
        memcpy([src_buf contents], data, total);
        free(data);
        id<MTLBuffer> dst_buf = [device newBufferWithLength:total options:MTLResourceStorageModeShared];
        id<MTLBuffer> size_buf = [device newBufferWithLength:npages*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> verify_buf = [device newBufferWithLength:total options:MTLResourceStorageModeShared];

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

        cb = [queue commandBuffer]; enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:decomp_pipe];
        [enc setBuffer:dst_buf offset:0 atIndex:0];
        [enc setBuffer:verify_buf offset:0 atIndex:1];
        [enc setBuffer:size_buf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(npages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

        int mismatch = memcmp([src_buf contents], [verify_buf contents], total);

        size_t zlib_total = 0;
        t0 = mach_absolute_time();
        for (size_t p = 0; p < npages; p++) {
            uLongf dlen = compressBound(16384); unsigned char *tmp = malloc(dlen);
            compress2(tmp, &dlen, (unsigned char*)[src_buf contents]+p*16384, 16384, 1);
            zlib_total += dlen; free(tmp);
        }
        t1 = mach_absolute_time();
        double zlib_ms = NS(t1-t0)/1e6;
        double zlib_ratio = (double)total / zlib_total;
        double zlib_speed = (double)total / (1ULL*1024*1024*1024) / (zlib_ms/1000.0);

        printf("  GPU: %.2fx @ %.1f GB/s | zlib: %.2fx @ %.2f GB/s | %s\n",
               ratio, speed, zlib_ratio, zlib_speed,
               mismatch==0 ? "PERFECT ✅" : "MISMATCH ❌");

        if (mismatch == 0) {
            sum_ratio += ratio; sum_speed += speed;
            sum_zlib_ratio += zlib_ratio; sum_zlib_speed += zlib_speed;
            perfect_count++;
        }
    }

    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║         DELTA+LZ77 SUMMARY                       ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    if (perfect_count > 0) {
        printf("║  GPU avg: %.2fx @ %.1f GB/s                    ║\n", sum_ratio/perfect_count, sum_speed/perfect_count);
        printf("║  zlib avg: %.2fx @ %.2f GB/s                  ║\n", sum_zlib_ratio/perfect_count, sum_zlib_speed/perfect_count);
        printf("║  GPU %.1fx faster, zlib %.1fx better ratio     ║\n",
               sum_speed/sum_zlib_speed, sum_zlib_ratio/sum_ratio);
    }
    printf("╚══════════════════════════════════════════════════╝\n");
    return 0;
}
