#include <stdio.h>
#include <sys/mman.h>
#include <mach/mach.h>
#include <mach/vm_map.h>
#include <mach/vm_statistics.h>
#include <mach/mach_vm.h>

int main() {
    printf("=== macOS mmap/vm flags ===\n");

#ifdef MAP_NOCACHE
    printf("MAP_NOCACHE: %d\n", MAP_NOCACHE);
#else
    printf("MAP_NOCACHE: NOT DEFINED\n");
#endif

#ifdef VM_FLAGS_SUPERPAGE_SIZE_ANY
    printf("VM_FLAGS_SUPERPAGE_SIZE_ANY: %d\n", VM_FLAGS_SUPERPAGE_SIZE_ANY);
#else
    printf("VM_FLAGS_SUPERPAGE_SIZE_ANY: NOT DEFINED\n");
#endif

#ifdef VM_FLAGS_SUPERPAGE_SIZE_2MB
    printf("VM_FLAGS_SUPERPAGE_SIZE_2MB: %d\n", VM_FLAGS_SUPERPAGE_SIZE_2MB);
#else
    printf("VM_FLAGS_SUPERPAGE_SIZE_2MB: NOT DEFINED\n");
#endif

#ifdef VM_FLAGS_PURGABLE
    printf("VM_FLAGS_PURGABLE: %d\n", VM_FLAGS_PURGABLE);
#else
    printf("VM_FLAGS_PURGABLE: NOT DEFINED\n");
#endif

#ifdef VM_PURGABLE_VOLATILE
    printf("VM_PURGABLE_VOLATILE: %d\n", VM_PURGABLE_VOLATILE);
#else
    printf("VM_PURGABLE_VOLATILE: NOT DEFINED\n");
#endif

#ifdef VM_PURGABLE_EMPTY
    printf("VM_PURGABLE_EMPTY: %d\n", VM_PURGABLE_EMPTY);
#else
    printf("VM_PURGABLE_EMPTY: NOT DEFINED\n");
#endif

    printf("\n=== Testing mach_vm_allocate with superpage ===\n");
    mach_vm_address_t addr = 0;
    kern_return_t kr;

    kr = mach_vm_allocate(mach_task_self(), &addr, 2*1024*1024, VM_FLAGS_ANYWHERE | VM_FLAGS_SUPERPAGE_SIZE_2MB);
    if (kr == KERN_SUCCESS) {
        printf("2MB superpage allocation: SUCCESS at %p\n", (void*)addr);
        volatile char *p = (char*)addr;
        p[0] = 1;
        p[2*1024*1024-1] = 1;
        printf("Touch superpage: SUCCESS\n");
        mach_vm_deallocate(mach_task_self(), addr, 2*1024*1024);
    } else {
        printf("2MB superpage allocation: FAILED with %d\n", kr);
    }

    printf("\n=== Testing purgable memory ===\n");
    addr = 0;
    kr = mach_vm_allocate(mach_task_self(), &addr, vm_page_size * 100, VM_FLAGS_ANYWHERE | VM_FLAGS_PURGABLE);
    if (kr == KERN_SUCCESS) {
        printf("Purgable allocation: SUCCESS at %p\n", (void*)addr);
        int state = VM_PURGABLE_VOLATILE;
        kr = vm_purgable_control(mach_task_self(), (vm_offset_t)addr, VM_PURGABLE_SET_STATE, &state);
        printf("Set purgable volatile: %s\n", kr == KERN_SUCCESS ? "SUCCESS" : "FAILED");
        mach_vm_deallocate(mach_task_self(), addr, vm_page_size * 100);
    } else {
        printf("Purgable allocation: FAILED with %d\n", kr);
    }

    printf("\n=== VM Statistics ===\n");
    struct vm_statistics64 stats;
    unsigned int count = HOST_VM_INFO64_COUNT;
    kr = host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&stats, &count);
    if (kr == KERN_SUCCESS) {
        printf("Pages free: %llu\n", (unsigned long long)stats.free_count);
        printf("Pages active: %llu\n", (unsigned long long)stats.active_count);
        printf("Pages inactive: %llu\n", (unsigned long long)stats.inactive_count);
        printf("Pages wired: %llu\n", (unsigned long long)stats.wire_count);
        printf("Compressions: %llu\n", (unsigned long long)stats.compressions);
        printf("Decompressions: %llu\n", (unsigned long long)stats.decompressions);
        printf("Pages in compressor: %llu\n", (unsigned long long)stats.compressor_page_count);
    }

    return 0;
}
