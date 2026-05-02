// Diagnose: Are allocations going through our zone or system malloc?
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <malloc/malloc.h>
#include <mach/mach.h>

static long long get_fp(void) {
    task_vm_info_data_t info; mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
    return (long long)info.phys_footprint;
}

int main() {
    printf("═══ Zone Diagnosis ═══\n\n");
    
    // Check if our zone is registered
    vm_address_t *zones = NULL;
    unsigned count = 0;
    malloc_get_all_zones(mach_task_self(), NULL, &zones, &count);
    printf("Registered malloc zones: %u\n", count);
    for (unsigned i = 0; i < count; i++) {
        malloc_zone_t *z = (malloc_zone_t *)zones[i];
        printf("  Zone %u: %s\n", i, z->zone_name ? z->zone_name : "(unnamed)");
    }
    
    // Test allocation
    void *p = malloc(65536);
    malloc_zone_t *zone = malloc_zone_from_ptr(p);
    printf("\n64KB allocation zone: %s\n", zone ? (zone->zone_name ? zone->zone_name : "(unnamed)") : "UNKNOWN");
    
    void *p2 = malloc(1024);
    zone = malloc_zone_from_ptr(p2);
    printf("1KB allocation zone: %s\n", zone ? (zone->zone_name ? zone->zone_name : "(unnamed)") : "UNKNOWN");
    
    // Check env
    printf("\nDYLD_INSERT_LIBRARIES: %s\n", getenv("DYLD_INSERT_LIBRARIES") ? getenv("DYLD_INSERT_LIBRARIES") : "(not set)");
    
    // Now test with large allocation
    printf("\nAllocating 100MB...\n");
    void *ptrs[1600];
    int n = 0;
    for (int i = 0; i < 1600; i++) {
        ptrs[i] = malloc(65536);
        if (ptrs[i]) n++;
    }
    printf("Allocated %d x 64KB = %d MB\n", n, n*64/1024);
    
    long long fp = get_fp();
    printf("Footprint: %lld MB\n", fp/(1024*1024));
    
    // Check which zone the allocations are in
    int in_memx = 0, in_default = 0;
    for (int i = 0; i < n; i++) {
        zone = malloc_zone_from_ptr(ptrs[i]);
        if (zone && zone->zone_name && strstr(zone->zone_name, "MemX")) in_memx++;
        else in_default++;
    }
    printf("In MemX zone: %d, In default zone: %d\n", in_memx, in_default);
    
    for (int i = 0; i < n; i++) free(ptrs[i]);
    free(p); free(p2);
    return 0;
}
