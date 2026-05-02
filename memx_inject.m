// MemX Global Injector - injects libmemx2.dylib into ALL user processes
// USAGE:
//   sudo ./memx_inject install   - Enable global injection
//   sudo ./memx_inject remove    - Disable global injection
//   sudo ./memx_inject status    - Check status

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/sysctl.h>

#define DYLIB_PATH "/Users/shiaho/Desktop/memx/libmemx2.dylib"
#define ENV_VAR    "DYLD_INSERT_LIBRARIES"

static int install(void) {
    if (access(DYLIB_PATH, R_OK) != 0) { printf("❌ Dylib not found: %s\n", DYLIB_PATH); return 1; }
    
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "launchctl setenv %s %s", ENV_VAR, DYLIB_PATH);
    if (system(cmd) != 0) { printf("❌ Failed (need sudo?)\n"); return 1; }
    
    int64_t ms=0; size_t len=sizeof(ms);
    sysctlbyname("hw.memsize", &ms, &len, NULL, 0);
    
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  ✅ MemX GLOBAL INJECTION ACTIVE          ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  Physical:  %4lld MB                     ║\n", ms/(1024*1024));
    printf("║  Virtual:   %4lld MB (4x physical)       ║\n", ms*4/(1024*1024));
    printf("║  New procs: auto GPU-compressed           ║\n");
    printf("║  Old procs: need restart                  ║\n");
    printf("║  Remove:    sudo memx_inject remove       ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    return 0;
}

static int remove_injection(void) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "launchctl unsetenv %s", ENV_VAR);
    system(cmd);
    printf("✅ MemX global injection removed.\n");
    printf("   Already-running processes still use it until they exit.\n");
    return 0;
}

static int show_status(void) {
    char *existing = getenv(ENV_VAR);
    int64_t ms=0; size_t len=sizeof(ms);
    sysctlbyname("hw.memsize", &ms, &len, NULL, 0);
    
    printf("MemX Status: %s\n", (existing && strstr(existing,"libmemx2")) ? "✅ ACTIVE" : "❌ INACTIVE");
    printf("Physical: %lld MB | Virtual pool: %lld MB\n", ms/(1024*1024), ms*4/(1024*1024));
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("Usage: sudo %s [install|remove|status]\n", argv[0]); return 1; }
    if (strcmp(argv[1], "install") == 0) return install();
    if (strcmp(argv[1], "remove") == 0) return remove_injection();
    if (strcmp(argv[1], "status") == 0) return show_status();
    printf("Unknown command: %s\n", argv[1]); return 1;
}
