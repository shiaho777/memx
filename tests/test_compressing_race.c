#include "memx_runtime.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define MB (1024ULL * 1024ULL)
#define PAGE_SZ 16384ULL
#define HALF_PER_PAGE (PAGE_SZ / 2)
#define GENERATIONS 4U

typedef struct writer_arg {
    uint8_t *ptr;
    size_t page_count;
    atomic_int done;
    atomic_uint generation;
    atomic_ulong pages_written;
} writer_arg_t;

static uint16_t value_for(size_t page, size_t half, uint32_t generation) {
    uint16_t lo = (uint16_t)((half * 37 + page * 11 + generation * 53 + half / 257) & 0x00FF);
    uint16_t hi = (uint16_t)(0x3A + ((half / 1024 + page + generation) & 3));
    return (uint16_t)((hi << 8) | lo);
}

static void write_u16(uint8_t *ptr, size_t byte_offset, uint16_t value) {
    ptr[byte_offset] = (uint8_t)(value & 0xFF);
    ptr[byte_offset + 1] = (uint8_t)(value >> 8);
}

static void write_page_half(uint8_t *ptr, size_t page, size_t first_half, size_t last_half, uint32_t generation) {
    size_t page_offset = page * PAGE_SZ;
    for (size_t h = first_half; h < last_half; h++) {
        write_u16(ptr, page_offset + h * 2, value_for(page, h, generation));
    }
}

static void *writer_main(void *opaque) {
    writer_arg_t *arg = (writer_arg_t *)opaque;
    for (uint32_t gen = 1; gen <= GENERATIONS; gen++) {
        atomic_store_explicit(&arg->generation, gen, memory_order_release);
        for (size_t page = 0; page < arg->page_count; page++) {
            write_page_half(arg->ptr, page, 0, HALF_PER_PAGE / 2, gen);
            if ((page & 3) == 0) usleep(80);
            write_page_half(arg->ptr, page, HALF_PER_PAGE / 2, HALF_PER_PAGE, gen);
            atomic_store_explicit(&arg->pages_written, page + 1, memory_order_release);
            if ((page & 7) == 0) usleep(120);
        }
    }
    atomic_store_explicit(&arg->done, 1, memory_order_release);
    return NULL;
}

static int verify_generation(const uint8_t *ptr, size_t page_count, uint32_t generation) {
    for (size_t page = 0; page < page_count; page++) {
        size_t page_offset = page * PAGE_SZ;
        for (size_t h = 0; h < HALF_PER_PAGE; h++) {
            uint16_t got = (uint16_t)ptr[page_offset + h * 2] | ((uint16_t)ptr[page_offset + h * 2 + 1] << 8);
            uint16_t expected = value_for(page, h, generation);
            if (got != expected) {
                fprintf(stderr, "race mismatch page=%zu half=%zu got=0x%04x expected=0x%04x\n", page, h, got, expected);
                return -1;
            }
        }
    }
    return 0;
}

static int wait_for_compressed(const uint8_t *ptr, uint64_t minimum_pages, unsigned tenths, uint64_t *out_pages) {
    uint64_t best = 0;
    for (unsigned i = 0; i < tenths; i++) {
        memx_runtime_allocation_info_t info;
        if (memx_runtime_get_allocation_info(ptr, &info) == 0) {
            if (info.compressed_pages > best) best = info.compressed_pages;
            if (info.compressed_pages >= minimum_pages) {
                if (out_pages) *out_pages = info.compressed_pages;
                return 0;
            }
        }
        usleep(100000);
    }
    if (out_pages) *out_pages = best;
    return -1;
}

int main(void) {
    memx_runtime_context_t *ctx = NULL;
    pthread_t writer;
    writer_arg_t arg;
    memx_runtime_stats_t stats;
    uint64_t compressed_before_write = 0;
    uint64_t max_during_write = 0;
    uint64_t final_compressed = 0;
    size_t size = 48 * MB;
    size_t page_count = size / PAGE_SZ;

    if (memx_runtime_context_create("compressing-race", &ctx) != 0 || !ctx) {
        fprintf(stderr, "memx_runtime_context_create failed\n");
        return 1;
    }
    if (memx_runtime_context_set_quota(ctx, 256 * MB) != 0) {
        fprintf(stderr, "memx_runtime_context_set_quota failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 2;
    }

    memx_runtime_tensor_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.struct_size = sizeof(desc);
    desc.role = MEMX_TENSOR_ROLE_WEIGHT;
    desc.dtype = MEMX_TENSOR_DTYPE_FP16;
    desc.layout = MEMX_TENSOR_LAYOUT_ROW_MAJOR;
    desc.flags = MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_SEQUENTIAL | MEMX_TENSOR_FLAG_COLD;
    desc.rank = 2;
    desc.shape[0] = page_count;
    desc.shape[1] = HALF_PER_PAGE;
    desc.stride[0] = HALF_PER_PAGE;
    desc.stride[1] = 1;

    uint8_t *ptr = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, size, &desc);
    if (!ptr) {
        fprintf(stderr, "memx_runtime_context_malloc_tensor failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 3;
    }

    write_page_half(ptr, 0, 0, 0, 0);
    memset(ptr, 0, size);
    if (wait_for_compressed(ptr, page_count / 4, 80, &compressed_before_write) != 0) {
        fprintf(stderr, "initial compression wait failed compressed=%llu\n", (unsigned long long)compressed_before_write);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 4;
    }

    arg.ptr = ptr;
    arg.page_count = page_count;
    atomic_init(&arg.done, 0);
    atomic_init(&arg.generation, 0);
    atomic_init(&arg.pages_written, 0);
    if (pthread_create(&writer, NULL, writer_main, &arg) != 0) {
        fprintf(stderr, "pthread_create failed\n");
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 5;
    }

    while (!atomic_load_explicit(&arg.done, memory_order_acquire)) {
        memx_runtime_allocation_info_t info;
        if (memx_runtime_get_allocation_info(ptr, &info) == 0 && info.compressed_pages > max_during_write) {
            max_during_write = info.compressed_pages;
        }
        usleep(50000);
    }

    pthread_join(writer, NULL);

    if (max_during_write == 0) {
        fprintf(stderr, "no compression visible during writer initial=%llu\n", (unsigned long long)compressed_before_write);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 6;
    }

    if (wait_for_compressed(ptr, page_count / 4, 80, &final_compressed) != 0) {
        fprintf(stderr, "post-write compression wait failed compressed=%llu\n", (unsigned long long)final_compressed);
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 7;
    }

    if (verify_generation(ptr, page_count, GENERATIONS) != 0) {
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 8;
    }

    if (memx_runtime_get_stats(&stats) != 0) {
        fprintf(stderr, "memx_runtime_get_stats failed\n");
        memx_runtime_context_free(ctx, ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 9;
    }

    printf("compressing race: initial=%llu during=%llu final=%llu faults=%llu tensor=%llu bitplane=%llu\n",
           (unsigned long long)compressed_before_write,
           (unsigned long long)max_during_write,
           (unsigned long long)final_compressed,
           (unsigned long long)stats.faults,
           (unsigned long long)stats.tensor_codec_pages,
           (unsigned long long)stats.tensor_bitplane_pages);

    memx_runtime_context_free(ctx, ptr);
    memx_runtime_context_destroy(ctx);
    memx_runtime_shutdown();
    return 0;
}
