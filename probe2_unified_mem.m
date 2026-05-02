// Probe 2: Can GPU directly access CPU memory on Apple Silicon?
// This tests the unified memory hypothesis - the foundation for paradigm shift
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <mach/mach.h>
#include <mach/mach_time.h>

#import <Metal/Metal.h>

#define MB (1024ULL*1024)
#define KB (1024ULL)

static double ns_per_tick;
static void init_time(void) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    ns_per_tick = (double)info.numer / (double)info.denom;
}
#define NS(ticks) ((double)(ticks) * ns_per_tick)

int main(void) {
    init_time();

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("No Metal device!\n");
        return 1;
    }
    printf("=== PROBE 2: Unified Memory GPU Access ===\n");
    printf("GPU: %s\n", [[device name] UTF8String]);
    printf("RegistryID: %llu\n", [device registryID]);
    printf("Max buffer size: %llu MB\n", [device maxBufferLength] / MB);
    printf("Has unified memory: %s\n", [device hasUnifiedMemory] ? "YES" : "NO");
    printf("Max threads per threadgroup: %lu\n", (unsigned long)[device maxThreadsPerThreadgroup].width);

    // Experiment A: Allocate with malloc, create MTLBuffer with noCopy
    // This is the KEY test - can GPU read CPU-allocated memory WITHOUT copying?
    printf("\n--- Experiment A: CPU malloc -> GPU noCopy buffer ---\n");
    size_t buf_size = 64 * MB;
    void *cpu_buf = malloc(buf_size);
    if (!cpu_buf) { printf("malloc failed\n"); return 1; }

    // Fill with known pattern
    memset(cpu_buf, 0x42, buf_size);

    // Create no-copy buffer - GPU shares the same physical pages
    id<MTLBuffer> gpu_buf = [device newBufferWithBytesNoCopy:cpu_buf
                                                       length:buf_size
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:^(void *ptr, NSUInteger length) {
                                                      free(ptr);
                                                  }];
    if (!gpu_buf) {
        printf("noCopy buffer FAILED - trying MTLResourceStorageModeShared...\n");
        // Fall back to shared buffer
        gpu_buf = [device newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
        if (gpu_buf) {
            memcpy([gpu_buf contents], cpu_buf, buf_size);
            printf("Shared buffer: SUCCESS (but requires copy)\n");
        }
    } else {
        printf("noCopy buffer: SUCCESS! GPU shares CPU physical pages!\n");
    }

    // Experiment B: Create shared buffer, modify from CPU, read from GPU
    printf("\n--- Experiment B: Shared buffer bidirectional access ---\n");
    id<MTLBuffer> shared_buf = [device newBufferWithLength:buf_size
                                                   options:MTLResourceStorageModeShared];
    if (shared_buf) {
        printf("Shared buffer created: %p (%zu MB)\n", [shared_buf contents], buf_size/MB);

        // Write from CPU
        uint64_t t0 = mach_absolute_time();
        memset([shared_buf contents], 0xAA, buf_size);
        uint64_t t1 = mach_absolute_time();
        printf("CPU memset: %.2f GB/s\n", (double)buf_size / (1024*1024*1024) / (NS(t1-t0)/1e9));

        // Create compute pipeline to read from GPU
        NSString *shader_src = @""
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void verify(device uchar* buf [[buffer(0)]],\n"
            "                   device atomic_uint* result [[buffer(1)]],\n"
            "                   uint gid [[thread_position_in_grid]]) {\n"
            "    if (buf[gid] != 0xAA) atomic_fetch_add_explicit(result, 1, memory_order_relaxed);\n"
            "}\n";
        NSError *error = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:shader_src options:nil error:&error];
        if (lib) {
            id<MTLFunction> func = [lib newFunctionWithName:@"verify"];
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&error];
            if (pipeline) {
                // Result buffer
                id<MTLBuffer> result_buf = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
                memset([result_buf contents], 0, 4);

                id<MTLCommandQueue> queue = [device newCommandQueue];
                id<MTLCommandBuffer> cmdbuf = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [cmdbuf computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:shared_buf offset:0 atIndex:0];
                [encoder setBuffer:result_buf offset:0 atIndex:1];

                MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
                MTLSize gridSize = MTLSizeMake(buf_size, 1, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
                [cmdbuf commit];
                [cmdbuf waitUntilCompleted];

                uint32_t mismatches = *(uint32_t*)[result_buf contents];
                printf("GPU verification: %u mismatches out of %zu bytes\n", mismatches, buf_size);
                if (mismatches == 0) {
                    printf("*** GPU READ CPU DATA WITH ZERO COPY - UNIFIED MEMORY CONFIRMED! ***\n");
                }
            } else {
                printf("Pipeline failed: %s\n", [[error localizedDescription] UTF8String]);
            }
        } else {
            printf("Library failed: %s\n", [[error localizedDescription] UTF8String]);
        }
    }

    // Experiment C: GPU write -> CPU read bandwidth
    printf("\n--- Experiment C: GPU write -> CPU read bandwidth ---\n");
    NSString *write_shader = @""
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void fill_buf(device uchar* buf [[buffer(0)]],\n"
        "                     device const uint32_t* pattern [[buffer(1)]],\n"
        "                     uint gid [[thread_position_in_grid]]) {\n"
        "    buf[gid] = (uchar)(*pattern);\n"
        "}\n";
    NSError *error2 = nil;
    id<MTLLibrary> lib2 = [device newLibraryWithSource:write_shader options:nil error:&error2];
    if (lib2) {
        id<MTLFunction> func2 = [lib2 newFunctionWithName:@"fill_buf"];
        id<MTLComputePipelineState> pipeline2 = [device newComputePipelineStateWithFunction:func2 error:&error2];
        if (pipeline2) {
            id<MTLBuffer> test_buf = [device newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> pattern_buf = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
            *(uint32_t*)[pattern_buf contents] = 0xBB;

            id<MTLCommandQueue> queue2 = [device newCommandQueue];

            // Warmup
            for (int w = 0; w < 3; w++) {
                id<MTLCommandBuffer> cb = [queue2 commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipeline2];
                [enc setBuffer:test_buf offset:0 atIndex:0];
                [enc setBuffer:pattern_buf offset:0 atIndex:1];
                MTLSize tg = MTLSizeMake(256, 1, 1);
                MTLSize gr = MTLSizeMake(buf_size, 1, 1);
                [enc dispatchThreads:gr threadsPerThreadgroup:tg];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            // Measure GPU write time
            uint64_t t0 = mach_absolute_time();
            id<MTLCommandBuffer> cb = [queue2 commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline2];
            [enc setBuffer:test_buf offset:0 atIndex:0];
            [enc setBuffer:pattern_buf offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(buf_size,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            uint64_t t1 = mach_absolute_time();
            double gpu_write_ns = NS(t1 - t0);
            printf("GPU fill %zuMB: %.2f ms (%.2f GB/s)\n", buf_size/MB, gpu_write_ns/1e6, (double)buf_size/(1024*1024*1024)/(gpu_write_ns/1e9));

            // CPU read back
            t0 = mach_absolute_time();
            volatile char sink = 0;
            volatile char *p = (volatile char*)[test_buf contents];
            for (size_t i = 0; i < buf_size; i += 64) sink += p[i];
            t1 = mach_absolute_time();
            double cpu_read_ns = NS(t1 - t0);
            printf("CPU read GPU data: %.2f ms (%.2f GB/s)\n", cpu_read_ns/1e6, (double)buf_size/(1024*1024*1024)/(cpu_read_ns/1e9));
            printf("Data consistent: %s\n", ((unsigned char*)[test_buf contents])[0] == 0xBB ? "YES" : "NO");
            if (sink == 127) printf("impossible\n");
        }
    }

    // Experiment D: Measure GPU compute throughput on memory-bound operations
    printf("\n--- Experiment D: GPU memory throughput (copy kernel) ---\n");
    NSString *copy_shader = @""
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void mem_copy(device const uchar* src [[buffer(0)]],\n"
        "                     device uchar* dst [[buffer(1)]],\n"
        "                     uint gid [[thread_position_in_grid]]) {\n"
        "    dst[gid] = src[gid];\n"
        "}\n";
    NSError *error3 = nil;
    id<MTLLibrary> lib3 = [device newLibraryWithSource:copy_shader options:nil error:&error3];
    if (lib3) {
        id<MTLFunction> func3 = [lib3 newFunctionWithName:@"mem_copy"];
        id<MTLComputePipelineState> pipeline3 = [device newComputePipelineStateWithFunction:func3 error:&error3];
        if (pipeline3) {
            size_t copy_size = 256 * MB;
            id<MTLBuffer> src_buf = [device newBufferWithLength:copy_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> dst_buf = [device newBufferWithLength:copy_size options:MTLResourceStorageModeShared];
            memset([src_buf contents], 0x55, copy_size);

            id<MTLCommandQueue> queue3 = [device newCommandQueue];

            // Warmup
            for (int w = 0; w < 3; w++) {
                id<MTLCommandBuffer> cb = [queue3 commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipeline3];
                [enc setBuffer:src_buf offset:0 atIndex:0];
                [enc setBuffer:dst_buf offset:0 atIndex:1];
                [enc dispatchThreads:MTLSizeMake(copy_size,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            // Measure
            uint64_t t0 = mach_absolute_time();
            for (int r = 0; r < 10; r++) {
                id<MTLCommandBuffer> cb = [queue3 commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipeline3];
                [enc setBuffer:src_buf offset:0 atIndex:0];
                [enc setBuffer:dst_buf offset:0 atIndex:1];
                [enc dispatchThreads:MTLSizeMake(copy_size,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
            uint64_t t1 = mach_absolute_time();
            double total_ns = NS(t1 - t0);
            double total_bytes = (double)copy_size * 10;
            printf("GPU copy %zuMB x10: %.2f ms total, %.2f GB/s\n",
                   copy_size/MB, total_ns/1e6, total_bytes/(1024*1024*1024)/(total_ns/1e9));
        }
    }

    printf("\n=== PROBE 2 COMPLETE ===\n");
    return 0;
}
