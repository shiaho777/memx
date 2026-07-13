#include "memx_runtime.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MB (1024ULL * 1024ULL)
#define PAGE_SZ 16384ULL

typedef struct tensor_case {
    const char *name;
    uint8_t *ptr;
    size_t size_bytes;
    memx_runtime_tensor_desc_t desc;
    memx_runtime_allocation_info_t info;
} tensor_case_t;

static int refresh_infos(tensor_case_t *cases, size_t count);

static uint16_t fp16_weight_value(size_t index, size_t page) {
    uint16_t lo = (uint16_t)((index * 37 + page * 11 + index / 257) & 0x03FF);
    uint16_t hi = (uint16_t)(0x3A + ((index / 1024 + page) & 3));
    return (uint16_t)((hi << 8) | (lo & 0xFF));
}

static uint16_t kv_value(size_t index, size_t page) {
    uint16_t base = (uint16_t)(0x3000 + (page & 7));
    return (uint16_t)(base ^ ((index / 64) & 15) ^ ((index * 13 + page) & 0x003F));
}

static uint8_t prefix_byte(size_t offset) {
    size_t in_page = offset & (PAGE_SZ - 1);
    if ((in_page % 512) == 0) return (uint8_t)((in_page / 512) & 0xFF);
    if ((in_page % 4096) == 13) return 0x5A;
    return 0;
}

static uint8_t sparse_byte(size_t offset) {
    size_t in_page = offset & (PAGE_SZ - 1);
    if ((in_page % 1024) == 17) return (uint8_t)(0x80 | ((offset / PAGE_SZ) & 0x3F));
    if ((in_page % 4096) == 93) return (uint8_t)(offset / 4096);
    return 0;
}

static void write_u16(uint8_t *ptr, size_t index, uint16_t value) {
    ptr[index * 2] = (uint8_t)(value & 0xFF);
    ptr[index * 2 + 1] = (uint8_t)(value >> 8);
}

static void fill_weights(uint8_t *ptr, size_t size_bytes) {
    size_t count = size_bytes / 2;
    for (size_t i = 0; i < count; i++) {
        size_t page = (i * 2) / PAGE_SZ;
        write_u16(ptr, i, fp16_weight_value(i, page));
    }
}

static void fill_kv(uint8_t *ptr, size_t size_bytes) {
    size_t count = size_bytes / 2;
    for (size_t i = 0; i < count; i++) {
        size_t page = (i * 2) / PAGE_SZ;
        write_u16(ptr, i, kv_value(i, page));
    }
}

static void fill_prefix(uint8_t *ptr, size_t size_bytes) {
    for (size_t off = 0; off < size_bytes; off++) {
        ptr[off] = prefix_byte(off);
    }
}

static void fill_sparse(uint8_t *ptr, size_t size_bytes) {
    for (size_t off = 0; off < size_bytes; off++) {
        ptr[off] = sparse_byte(off);
    }
}

static int verify_weights(const uint8_t *ptr, size_t size_bytes) {
    size_t count = size_bytes / 2;
    for (size_t i = 0; i < count; i += 263) {
        size_t page = (i * 2) / PAGE_SZ;
        uint16_t expected = fp16_weight_value(i, page);
        uint16_t got = (uint16_t)ptr[i * 2] | ((uint16_t)ptr[i * 2 + 1] << 8);
        if (got != expected) {
            fprintf(stderr, "weight mismatch half=%zu page=%zu got=0x%04x expected=0x%04x\n", i, page, got, expected);
            return -1;
        }
    }
    return 0;
}

static int verify_kv(const uint8_t *ptr, size_t size_bytes) {
    size_t count = size_bytes / 2;
    for (size_t i = 0; i < count; i += 251) {
        size_t page = (i * 2) / PAGE_SZ;
        uint16_t expected = kv_value(i, page);
        uint16_t got = (uint16_t)ptr[i * 2] | ((uint16_t)ptr[i * 2 + 1] << 8);
        if (got != expected) {
            fprintf(stderr, "kv mismatch half=%zu page=%zu got=0x%04x expected=0x%04x\n", i, page, got, expected);
            return -1;
        }
    }
    return 0;
}

static int verify_prefix(const uint8_t *ptr, size_t size_bytes) {
    for (size_t off = 0; off < size_bytes; off += 197) {
        uint8_t expected = prefix_byte(off);
        if (ptr[off] != expected) {
            fprintf(stderr, "prefix mismatch off=%zu page=%llu got=0x%02x expected=0x%02x\n", off, (unsigned long long)(off / PAGE_SZ), ptr[off], expected);
            return -1;
        }
    }
    return 0;
}

static int verify_sparse(const uint8_t *ptr, size_t size_bytes) {
    for (size_t off = 0; off < size_bytes; off += 131) {
        uint8_t expected = sparse_byte(off);
        if (ptr[off] != expected) {
            fprintf(stderr, "sparse mismatch off=%zu page=%llu got=0x%02x expected=0x%02x\n", off, (unsigned long long)(off / PAGE_SZ), ptr[off], expected);
            return -1;
        }
    }
    for (size_t page = 0; page * PAGE_SZ < size_bytes; page++) {
        size_t off = page * PAGE_SZ + 17;
        uint8_t expected = sparse_byte(off);
        if (off < size_bytes && ptr[off] != expected) {
            fprintf(stderr, "sparse explicit mismatch off=%zu page=%zu got=0x%02x expected=0x%02x\n", off, page, ptr[off], expected);
            return -1;
        }
    }
    return 0;
}

static int wait_for_capacity_signals(tensor_case_t *cases, size_t count, unsigned seconds) {
    for (unsigned i = 0; i < seconds * 10; i++) {
        memx_runtime_stats_t stats;
        if (refresh_infos(cases, count) == 0 &&
            memx_runtime_get_stats(&stats) == 0 &&
            cases[0].info.compressed_pages > 0 &&
            cases[1].info.compressed_pages > 0 &&
            cases[2].info.compressed_pages > 0 &&
            cases[3].info.compressed_pages > 0 &&
            stats.weight_compressed_pages > 0 &&
            stats.kv_cache_compressed_pages > 0 &&
            stats.tensor_sparse_pages > 0 &&
            stats.dedup_hits > 0) {
            return 0;
        }
        usleep(100000);
    }
    return -1;
}

static int refresh_infos(tensor_case_t *cases, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (memx_runtime_get_allocation_info(cases[i].ptr, &cases[i].info) != 0) return -1;
    }
    return 0;
}

static uint64_t logical_bytes(const tensor_case_t *cases, size_t count) {
    uint64_t total = 0;
    for (size_t i = 0; i < count; i++) total += cases[i].size_bytes;
    return total;
}

static uint64_t compressed_bytes_with_duplicates(const tensor_case_t *cases, size_t count) {
    uint64_t total = 0;
    for (size_t i = 0; i < count; i++) total += cases[i].info.compressed_bytes;
    return total;
}

static uint64_t resident_bytes_from_infos(const tensor_case_t *cases, size_t count) {
    uint64_t total = 0;
    for (size_t i = 0; i < count; i++) {
        uint64_t resident_pages = cases[i].info.page_count - cases[i].info.compressed_pages;
        total += resident_pages * PAGE_SZ;
    }
    return total;
}

static uint64_t stored_bytes_before_dedup(const tensor_case_t *cases, size_t count) {
    return compressed_bytes_with_duplicates(cases, count) + resident_bytes_from_infos(cases, count);
}

static uint64_t physical_estimate_bytes(const tensor_case_t *cases, size_t count, const memx_runtime_stats_t *stats) {
    uint64_t resident = resident_bytes_from_infos(cases, count);
    return resident + stats->pool_used_bytes;
}

static uint64_t projected_logical_bytes(double ratio, uint64_t physical_bytes, double usable_fraction) {
    return (uint64_t)((double)physical_bytes * usable_fraction * ratio);
}

static void print_capacity_projection(const char *name, uint64_t physical_bytes, double ratio) {
    uint64_t full = projected_logical_bytes(ratio, physical_bytes, 1.0);
    uint64_t safe = projected_logical_bytes(ratio, physical_bytes, 0.80);
    printf("  projection_%s: full=%5.1fGB safe80=%5.1fGB target2x=%s target4x=%s safe80_target4x=%s\n",
           name,
           (double)full / 1024.0 / (double)MB,
           (double)safe / 1024.0 / (double)MB,
           full >= physical_bytes * 2 ? "pass" : "fail",
           full >= physical_bytes * 4 ? "pass" : "fail",
           safe >= physical_bytes * 4 ? "pass" : "fail");
}

static void print_tensor_case(const tensor_case_t *tc) {
    uint64_t resident_pages = tc->info.page_count - tc->info.compressed_pages;
    uint64_t stored = tc->info.compressed_bytes + resident_pages * PAGE_SZ;
    double ratio = stored ? (double)tc->size_bytes / (double)stored : 0.0;
    printf("  %-18s logical=%5.1fMB compressed_pages=%5llu/%-5llu compressed=%5.1fMB resident=%5.1fMB stored_ratio=%5.2fx codec=0x%02x tensor_codec_pages=%llu flags=0x%x\n",
           tc->name,
           (double)tc->size_bytes / (double)MB,
           (unsigned long long)tc->info.compressed_pages,
           (unsigned long long)tc->info.page_count,
           (double)tc->info.compressed_bytes / (double)MB,
           (double)(resident_pages * PAGE_SZ) / (double)MB,
           ratio,
           tc->info.primary_codec,
           (unsigned long long)tc->info.tensor_codec_pages,
           tc->info.tensor_flags);
}

static memx_runtime_tensor_desc_t desc_for(uint32_t role, uint32_t dtype, uint32_t flags, uint32_t layer) {
    memx_runtime_tensor_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.struct_size = sizeof(desc);
    desc.role = role;
    desc.dtype = dtype;
    desc.layout = MEMX_TENSOR_LAYOUT_ROW_MAJOR;
    desc.flags = flags;
    desc.rank = 2;
    desc.shape[0] = 8192;
    desc.shape[1] = 8192;
    desc.stride[0] = 8192;
    desc.stride[1] = 1;
    desc.layer_index = layer;
    return desc;
}

static int allocate_case(memx_runtime_context_t *ctx, tensor_case_t *tc, void (*fill)(uint8_t *, size_t)) {
    errno = 0;
    tc->ptr = (uint8_t *)memx_runtime_context_malloc_tensor(ctx, tc->size_bytes, &tc->desc);
    if (!tc->ptr) {
        fprintf(stderr, "%s allocation failed errno=%d\n", tc->name, errno);
        return -1;
    }
    fill(tc->ptr, tc->size_bytes);
    return 0;
}

static int verify_all(const tensor_case_t *cases) {
    if (verify_weights(cases[0].ptr, cases[0].size_bytes) != 0) return -1;
    if (verify_kv(cases[1].ptr, cases[1].size_bytes) != 0) return -1;
    if (verify_prefix(cases[2].ptr, cases[2].size_bytes) != 0) return -1;
    if (verify_sparse(cases[3].ptr, cases[3].size_bytes) != 0) return -1;
    return 0;
}

static int apply_windows(memx_runtime_context_t *ctx, tensor_case_t *cases) {
    memx_runtime_kv_cache_window_t kv_window;
    memset(&kv_window, 0, sizeof(kv_window));
    kv_window.struct_size = sizeof(kv_window);
    kv_window.managed_offset = 0;
    kv_window.managed_length = cases[1].size_bytes;
    kv_window.hot_offset = cases[1].size_bytes - 4 * PAGE_SZ;
    kv_window.hot_length = 4 * PAGE_SZ;
    kv_window.prefetch_offset = cases[1].size_bytes - 8 * PAGE_SZ;
    kv_window.prefetch_length = 4 * PAGE_SZ;
    if (memx_runtime_context_update_kv_cache_window(ctx, cases[1].ptr, &kv_window) != 0) return -1;

    memx_runtime_weight_window_t weight_window;
    memset(&weight_window, 0, sizeof(weight_window));
    weight_window.struct_size = sizeof(weight_window);
    weight_window.managed_offset = 0;
    weight_window.managed_length = cases[0].size_bytes;
    weight_window.hot_offset = 0;
    weight_window.hot_length = 2 * PAGE_SZ;
    weight_window.prefetch_offset = 2 * PAGE_SZ;
    weight_window.prefetch_length = 2 * PAGE_SZ;
    if (memx_runtime_context_update_weight_window(ctx, cases[0].ptr, &weight_window) != 0) return -1;

    if (memx_runtime_context_mark_access_range(ctx, cases[1].ptr, kv_window.prefetch_offset, kv_window.prefetch_length) != 0) return -1;
    if (memx_runtime_context_mark_access_range(ctx, cases[0].ptr, weight_window.prefetch_offset, weight_window.prefetch_length) != 0) return -1;
    return 0;
}

int main(void) {
    memx_runtime_context_t *ctx = NULL;
    memx_runtime_stats_t stats_before_verify;
    memx_runtime_stats_t stats_after_verify;
    memx_runtime_pressure_t pressure;
    memx_runtime_context_stats_t ctx_stats;
    uint64_t reclaimed = 0;

    tensor_case_t cases[4] = {
        {
            .name = "cold-fp16-weight",
            .size_bytes = 96 * MB,
            .desc = {0}
        },
        {
            .name = "old-kv-cache",
            .size_bytes = 64 * MB,
            .desc = {0}
        },
        {
            .name = "repeated-prefix",
            .size_bytes = 48 * MB,
            .desc = {0}
        },
        {
            .name = "sparse-activation",
            .size_bytes = 32 * MB,
            .desc = {0}
        }
    };

    cases[0].desc = desc_for(MEMX_TENSOR_ROLE_WEIGHT, MEMX_TENSOR_DTYPE_FP16,
                             MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_SEQUENTIAL | MEMX_TENSOR_FLAG_COLD, 0);
    cases[1].desc = desc_for(MEMX_TENSOR_ROLE_KV_CACHE, MEMX_TENSOR_DTYPE_FP16,
                             MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD, 12);
    cases[2].desc = desc_for(MEMX_TENSOR_ROLE_KV_CACHE, MEMX_TENSOR_DTYPE_UINT8,
                             MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD, 99);
    cases[3].desc = desc_for(MEMX_TENSOR_ROLE_ACTIVATION, MEMX_TENSOR_DTYPE_UINT8,
                             MEMX_TENSOR_FLAG_COLD | MEMX_TENSOR_FLAG_READ_MOSTLY, 24);

    if (memx_runtime_context_create("effective-capacity", &ctx) != 0 || !ctx) {
        fprintf(stderr, "memx_runtime_context_create failed\n");
        return 1;
    }
    if (memx_runtime_context_set_quota(ctx, 512 * MB) != 0) {
        fprintf(stderr, "memx_runtime_context_set_quota failed\n");
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 2;
    }

    printf("MemX effective capacity benchmark\n");
    printf("logical_target=%.1fMB quota=512.0MB page=%lluKB\n",
           (double)logical_bytes(cases, 4) / (double)MB,
           (unsigned long long)(PAGE_SZ / 1024));

    if (allocate_case(ctx, &cases[0], fill_weights) != 0 ||
        allocate_case(ctx, &cases[1], fill_kv) != 0 ||
        allocate_case(ctx, &cases[2], fill_prefix) != 0 ||
        allocate_case(ctx, &cases[3], fill_sparse) != 0) {
        for (size_t i = 0; i < 4; i++) if (cases[i].ptr) memx_runtime_context_free(ctx, cases[i].ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 3;
    }

    if (wait_for_capacity_signals(cases, 4, 30) != 0) {
        memx_runtime_stats_t stats;
        refresh_infos(cases, 4);
        memset(&stats, 0, sizeof(stats));
        memx_runtime_get_stats(&stats);
        fprintf(stderr, "capacity signal wait failed weight=%llu kv=%llu prefix=%llu sparse=%llu tensor=%llu sparse_pages=%llu dedup=%llu\n",
                (unsigned long long)cases[0].info.compressed_pages,
                (unsigned long long)cases[1].info.compressed_pages,
                (unsigned long long)cases[2].info.compressed_pages,
                (unsigned long long)cases[3].info.compressed_pages,
                (unsigned long long)stats.tensor_codec_pages,
                (unsigned long long)stats.tensor_sparse_pages,
                (unsigned long long)stats.dedup_hits);
        for (size_t i = 0; i < 4; i++) memx_runtime_context_free(ctx, cases[i].ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 4;
    }

    if (apply_windows(ctx, cases) != 0) {
        fprintf(stderr, "window policy update failed\n");
        for (size_t i = 0; i < 4; i++) memx_runtime_context_free(ctx, cases[i].ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 5;
    }

    usleep(300000);

    if (refresh_infos(cases, 4) != 0 ||
        memx_runtime_get_stats(&stats_before_verify) != 0 ||
        memx_runtime_get_pressure(&pressure) != 0 ||
        memx_runtime_context_get_stats(ctx, &ctx_stats) != 0) {
        fprintf(stderr, "telemetry read failed\n");
        for (size_t i = 0; i < 4; i++) memx_runtime_context_free(ctx, cases[i].ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 6;
    }

    uint64_t logical = logical_bytes(cases, 4);
    uint64_t stored_before_dedup = stored_bytes_before_dedup(cases, 4);
    uint64_t physical_estimate = physical_estimate_bytes(cases, 4, &stats_before_verify);
    double logical_to_stored = stored_before_dedup ? (double)logical / (double)stored_before_dedup : 0.0;
    double logical_to_physical = physical_estimate ? (double)logical / (double)physical_estimate : 0.0;
    uint64_t mac16 = 16ULL * 1024ULL * MB;
    uint64_t mac32 = 32ULL * 1024ULL * MB;
    uint64_t projected16 = projected_logical_bytes(logical_to_physical, mac16, 1.0);
    uint64_t projected32 = projected_logical_bytes(logical_to_physical, mac32, 1.0);
    uint64_t projected16_safe = projected_logical_bytes(logical_to_physical, mac16, 0.80);
    uint64_t projected32_safe = projected_logical_bytes(logical_to_physical, mac32, 0.80);

    printf("\ncapacity-ledger:\n");
    for (size_t i = 0; i < 4; i++) print_tensor_case(&cases[i]);
    printf("  logical_total=%5.1fMB stored_before_dedup=%5.1fMB pool_used=%5.1fMB resident_est=%5.1fMB physical_est=%5.1fMB\n",
           (double)logical / (double)MB,
           (double)stored_before_dedup / (double)MB,
           (double)stats_before_verify.pool_used_bytes / (double)MB,
           (double)resident_bytes_from_infos(cases, 4) / (double)MB,
           (double)physical_estimate / (double)MB);
    printf("  effective_ratio_before_dedup=%5.2fx effective_ratio_physical_est=%5.2fx\n",
           logical_to_stored,
           logical_to_physical);
    print_capacity_projection("16gb", mac16, logical_to_physical);
    print_capacity_projection("32gb", mac32, logical_to_physical);
    printf("  telemetry: tensor_pages=%llu split=%llu delta_split=%llu bitplane=%llu sparse=%llu weight_pages=%llu kv_pages=%llu hot_resident=%llu no_compress_resident=%llu dedup_hits=%llu dedup_saved=%5.1fMB prefetch=%llu hits=%llu pressure=%u%% frag=%u%%\n",
           (unsigned long long)stats_before_verify.tensor_codec_pages,
           (unsigned long long)stats_before_verify.tensor_split_pages,
           (unsigned long long)stats_before_verify.tensor_delta_split_pages,
           (unsigned long long)stats_before_verify.tensor_bitplane_pages,
           (unsigned long long)stats_before_verify.tensor_sparse_pages,
           (unsigned long long)stats_before_verify.weight_compressed_pages,
           (unsigned long long)stats_before_verify.kv_cache_compressed_pages,
           (unsigned long long)stats_before_verify.hot_resident_pages,
           (unsigned long long)stats_before_verify.no_compress_resident_pages,
           (unsigned long long)stats_before_verify.dedup_hits,
           (double)stats_before_verify.dedup_bytes_saved / (double)MB,
           (unsigned long long)stats_before_verify.prefetch_count,
           (unsigned long long)stats_before_verify.prefetch_hits,
           pressure.pool_pressure_percent,
           pressure.pool_fragmentation_percent);
    printf("  context: tensor=%5.1fMB weight=%5.1fMB kv=%5.1fMB hot=%5.1fMB no_compress=%5.1fMB live=%llu peak=%5.1fMB\n",
           (double)ctx_stats.tensor_bytes_in_use / (double)MB,
           (double)ctx_stats.weight_bytes_in_use / (double)MB,
           (double)ctx_stats.kv_cache_bytes_in_use / (double)MB,
           (double)ctx_stats.hot_bytes_in_use / (double)MB,
           (double)ctx_stats.no_compress_bytes_in_use / (double)MB,
           (unsigned long long)ctx_stats.allocations_live,
           (double)ctx_stats.peak_bytes_in_use / (double)MB);

    uint64_t faults_before = stats_before_verify.faults;
    if (verify_all(cases) != 0) {
        fprintf(stderr, "bit-exact sampled verification failed\n");
        for (size_t i = 0; i < 4; i++) memx_runtime_context_free(ctx, cases[i].ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 7;
    }
    if (memx_runtime_get_stats(&stats_after_verify) != 0) {
        fprintf(stderr, "post-verify stats failed\n");
        for (size_t i = 0; i < 4; i++) memx_runtime_context_free(ctx, cases[i].ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 8;
    }

    printf("  bitexact_sample=pass verify_faults=%llu\n",
           (unsigned long long)(stats_after_verify.faults - faults_before));

    if (stats_before_verify.tensor_codec_pages == 0 ||
        stats_before_verify.weight_compressed_pages == 0 ||
        stats_before_verify.kv_cache_compressed_pages == 0 ||
        stats_before_verify.tensor_sparse_pages == 0 ||
        stats_before_verify.tensor_delta_split_pages == 0 ||
        stats_before_verify.dedup_hits == 0 ||
        stats_before_verify.prefetch_count == 0 ||
        stats_before_verify.prefetch_hits == 0 ||
        stats_before_verify.hot_resident_pages == 0 ||
        stats_before_verify.no_compress_resident_pages == 0 ||
        projected16 < mac16 * 4 ||
        projected32 < mac32 * 4 ||
        projected16_safe < mac16 * 4 ||
        projected32_safe < mac32 * 4 ||
        logical_to_physical < 4.0) {
        fprintf(stderr, "effective capacity gates failed ratio=%.2f projected16=%.1fGB safe16=%.1fGB projected32=%.1fGB safe32=%.1fGB target16=64.0GB target32=128.0GB tensor=%llu weight=%llu kv=%llu sparse=%llu delta_split=%llu dedup=%llu prefetch=%llu hits=%llu hot=%llu no_compress=%llu\n",
                logical_to_physical,
                (double)projected16 / 1024.0 / (double)MB,
                (double)projected16_safe / 1024.0 / (double)MB,
                (double)projected32 / 1024.0 / (double)MB,
                (double)projected32_safe / 1024.0 / (double)MB,
                (unsigned long long)stats_before_verify.tensor_codec_pages,
                (unsigned long long)stats_before_verify.weight_compressed_pages,
                (unsigned long long)stats_before_verify.kv_cache_compressed_pages,
                (unsigned long long)stats_before_verify.tensor_sparse_pages,
                (unsigned long long)stats_before_verify.tensor_delta_split_pages,
                (unsigned long long)stats_before_verify.dedup_hits,
                (unsigned long long)stats_before_verify.prefetch_count,
                (unsigned long long)stats_before_verify.prefetch_hits,
                (unsigned long long)stats_before_verify.hot_resident_pages,
                (unsigned long long)stats_before_verify.no_compress_resident_pages);
        for (size_t i = 0; i < 4; i++) memx_runtime_context_free(ctx, cases[i].ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 9;
    }

    if (memx_runtime_reclaim(&reclaimed) != 0) {
        fprintf(stderr, "reclaim failed\n");
        for (size_t i = 0; i < 4; i++) memx_runtime_context_free(ctx, cases[i].ptr);
        memx_runtime_context_destroy(ctx);
        memx_runtime_shutdown();
        return 10;
    }
    printf("  reclaim_before_free=%5.1fMB\n", (double)reclaimed / (double)MB);

    for (size_t i = 0; i < 4; i++) memx_runtime_context_free(ctx, cases[i].ptr);
    memx_runtime_context_destroy(ctx);
    memx_runtime_shutdown();
    return 0;
}
