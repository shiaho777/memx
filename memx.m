// MemX - GPU Memory Expansion Launcher
// Usage: memx <any command>
//   memx python3 script.py
//   memx node server.js
//   memx ./my_app
//
// This simply sets DYLD_INSERT_LIBRARIES and exec's the command.
// No SIP changes, no system modifications, no risk.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mach/mach.h>

static long long get_footprint(void) {
    task_vm_info_data_t info; mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) return 0;
    return (long long)info.phys_footprint;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("MemX - GPU Memory Expansion\n\n");
        printf("Usage:\n");
        printf("  memx <command>          Run command with GPU memory compression\n");
        printf("  memx status             Show current memory info\n");
        printf("\nExamples:\n");
        printf("  memx python3 train.py   Run Python with memory expansion\n");
        printf("  memx node server.js     Run Node.js with memory expansion\n");
        printf("  memx ./my_app           Run any executable\n");
        printf("\nHow it works:\n");
        printf("  Allocations > 64KB are compressed by GPU (Delta+LZ77)\n");
        printf("  On access, decompressed instantly via CPU signal handler\n");
        printf("  Typical savings: 4-50x depending on data pattern\n");
        return 0;
    }

    if (strcmp(argv[1], "status") == 0) {
        long long fp = get_footprint();
        printf("Current process footprint: %lld MB\n", fp / (1024*1024));
        const char *injected = getenv("DYLD_INSERT_LIBRARIES");
        if (injected) printf("MemX: ✅ Active (%s)\n", injected);
        else printf("MemX: ❌ Not active (run: memx <command>)\n");
        return 0;
    }

    // Find our dylib next to our executable
    char dylib_path[1024];
    uint32_t size = sizeof(dylib_path);
    if (_NSGetExecutablePath(dylib_path, &size) != 0) {
        fprintf(stderr, "memx: cannot find executable path\n");
        return 1;
    }
    // Replace basename with libmemx3.dylib
    char *last_slash = strrchr(dylib_path, '/');
    if (last_slash) {
        strcpy(last_slash + 1, "libmemx3.dylib");
    } else {
        strcpy(dylib_path, "libmemx3.dylib");
    }

    if (access(dylib_path, R_OK) != 0) {
        fprintf(stderr, "memx: dylib not found at %s\n", dylib_path);
        return 1;
    }

    setenv("DYLD_INSERT_LIBRARIES", dylib_path, 1);
    execvp(argv[1], argv + 1);

    // If execvp fails
    fprintf(stderr, "memx: cannot execute '%s': ", argv[1]);
    perror(NULL);
    return 127;
}
