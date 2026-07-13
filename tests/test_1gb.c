#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach.h>
#include <unistd.h>

#define MB (1024ULL*1024)

static long long get_fp(void) {
    task_vm_info_data_t info; mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
    return (long long)info.phys_footprint;
}

int main() {
    printf("=== 1GB Malloc Test ===\n");
    size_t sz = 1024*MB;
    void *p = malloc(sz);
    printf("1. malloc(1GB)... ptr=%p, footprint=%lld MB\n", p, get_fp()/MB);
    if (!p) return 1;
    memset(p, 0x42, sz);
    printf("2. Writing all 1GB... footprint=%lld MB\n", get_fp()/MB);
    printf("3. Sleeping 10s for compression...\n");
    sleep(10);
    long long fp = get_fp();
    printf("   footprint=%lld MB (saved=%lld MB)\n", fp/MB, (1125-fp/MB));
    int ok = 1;
    for (size_t i = 0; i < sz; i++) if (((char*)p)[i] != 0x42) { ok = 0; break; }
    printf("4. Integrity: %s\n", ok ? "PERFECT" : "CORRUPT");
    free(p);
    return 0;
}
