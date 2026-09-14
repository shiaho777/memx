#include "memx_runtime.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#define PAGE_SZ 16384ull
#define MAX_SEGMENT_NAME 64
#define MAX_SEGMENTS 128
#define MAX_PAYLOAD (256ull * 1024 * 1024)
#define MAX_GENS 4096
#define SHUTDOWN_DRAIN_MS 3000

typedef struct {
    char name[MAX_SEGMENT_NAME];
    uint32_t pages;
    uint64_t nbytes;
    uint8_t *data;
    int live;
} seg_slot_t;

typedef struct {
    char name[MAX_SEGMENT_NAME];
    pid_t pid;
    int conn_fd;
    uint64_t compressed_pages;
    uint64_t resident_pages;
    uint64_t saved_bytes;
    uint64_t beats;
    time_t last_beat;
    int live;
} client_slot_t;

#define MAX_CLIENTS 64
#define PRESSURE_DECAY_S 60

static seg_slot_t g_segs[MAX_SEGMENTS];
static client_slot_t g_clients[MAX_CLIENTS];
static pthread_mutex_t g_segs_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_clients_mu = PTHREAD_MUTEX_INITIALIZER;
static volatile sig_atomic_t g_running = 1;
static volatile int g_live = 0;
static volatile unsigned g_pressure_level = 0;
static volatile time_t g_pressure_ts = 0;
static char g_store_root[1024];
static char g_sock_path[1024];

static void logf(const char *fmt, ...) {
    char ts[32];
    time_t now = time(NULL);
    struct tm tmv;
    localtime_r(&now, &tmv);
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tmv);
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[%s] ", ts);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static void on_signal(int sig) {
    (void)sig;
    g_running = 0;
}

static unsigned pressure_level_now(void);

#ifdef __APPLE__
static void on_pressure(void *ctx) {
    dispatch_source_t src = (dispatch_source_t)ctx;
    unsigned long lvl = dispatch_source_get_data(src);
    unsigned level = 1;
    if (lvl & DISPATCH_MEMORYPRESSURE_CRITICAL) level = 2;
    g_pressure_level = level;
    g_pressure_ts = time(NULL);
    logf("memory pressure event raw=0x%lx level=%u", lvl, level);
}

static void install_pressure_source(void) {
    dispatch_source_t src = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
        DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));
    if (!src) return;
    dispatch_set_context(src, src);
    dispatch_source_set_event_handler_f(src, on_pressure);
    dispatch_resume(src);
}
#else
static void install_pressure_source(void) {}
#endif

static void mkdir_p(const char *path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t len = strlen(tmp);
    if (len && tmp[len - 1] == '/') tmp[len - 1] = 0;
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

static int read_full(int fd, void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, p + got, n - got);
        if (r <= 0) return -1;
        got += (size_t)r;
    }
    return 0;
}

static int write_full(int fd, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t put = 0;
    while (put < n) {
        ssize_t w = write(fd, p + put, n - put);
        if (w <= 0) return -1;
        put += (size_t)w;
    }
    return 0;
}

static int send_frame(int fd, const void *buf, uint32_t len) {
    uint32_t net = htonl(len);
    if (write_full(fd, &net, 4) != 0) return -1;
    if (len && write_full(fd, buf, len) != 0) return -1;
    return 0;
}

static uint8_t *recv_frame(int fd, uint32_t *out_len) {
    uint32_t net;
    if (read_full(fd, &net, 4) != 0) return NULL;
    uint32_t len = ntohl(net);
    if (len > MAX_PAYLOAD) return NULL;
    uint8_t *buf = (uint8_t *)malloc(len ? len : 1);
    if (!buf) return NULL;
    if (len && read_full(fd, buf, len) != 0) {
        free(buf);
        return NULL;
    }
    *out_len = len;
    return buf;
}

static int seg_find_locked(const char *name) {
    for (int i = 0; i < MAX_SEGMENTS; i++) {
        if (g_segs[i].live && strncmp(g_segs[i].name, name, MAX_SEGMENT_NAME) == 0) return i;
    }
    return -1;
}

static void seg_release_locked(int idx) {
    free(g_segs[idx].data);
    memset(&g_segs[idx], 0, sizeof(g_segs[idx]));
    g_segs[idx].live = 0;
}

static int do_put(const uint8_t *body, uint32_t len, uint8_t **reply, uint32_t *reply_len) {
    if (len < 1 || len > MAX_PAYLOAD) {
        *reply = (uint8_t *)strdup("ERR bad_request");
        *reply_len = (uint32_t)strlen((char *)*reply);
        return 0;
    }
    uint8_t nlen = body[0];
    if (nlen == 0 || nlen >= MAX_SEGMENT_NAME || 1 + (uint32_t)nlen > len) {
        *reply = (uint8_t *)strdup("ERR bad_name");
        *reply_len = (uint32_t)strlen((char *)*reply);
        return 0;
    }
    char name[MAX_SEGMENT_NAME];
    memcpy(name, body + 1, nlen);
    name[nlen] = 0;
    uint32_t data_len = len - 1 - nlen;
    if (data_len == 0 || (data_len % PAGE_SZ) != 0) {
        *reply = (uint8_t *)strdup("ERR bad_size");
        *reply_len = (uint32_t)strlen((char *)*reply);
        return 0;
    }

    pthread_mutex_lock(&g_segs_mu);
    int idx = seg_find_locked(name);
    if (idx < 0) {
        for (int i = 0; i < MAX_SEGMENTS; i++) {
            if (!g_segs[i].live) { idx = i; break; }
        }
    } else {
        seg_release_locked(idx);
    }
    if (idx < 0) {
        pthread_mutex_unlock(&g_segs_mu);
        *reply = (uint8_t *)strdup("ERR full");
        *reply_len = (uint32_t)strlen((char *)*reply);
        return 0;
    }
    g_segs[idx].data = (uint8_t *)malloc(data_len);
    if (!g_segs[idx].data) {
        pthread_mutex_unlock(&g_segs_mu);
        *reply = (uint8_t *)strdup("ERR oom");
        *reply_len = (uint32_t)strlen((char *)*reply);
        return 0;
    }
    memcpy(g_segs[idx].data, body + 1 + nlen, data_len);
    memcpy(g_segs[idx].name, name, nlen + 1);
    g_segs[idx].nbytes = data_len;
    g_segs[idx].pages = (uint32_t)(data_len / PAGE_SZ);
    g_segs[idx].live = 1;
    pthread_mutex_unlock(&g_segs_mu);

    char msg[128];
    snprintf(msg, sizeof(msg), "OK put %s pages=%u", name, g_segs[idx].pages);
    *reply = (uint8_t *)strdup(msg);
    *reply_len = (uint32_t)strlen(msg);
    return 0;
}

static int do_get(const uint8_t *body, uint32_t len, uint8_t **reply, uint32_t *reply_len) {
    if (len < 1 || len >= MAX_SEGMENT_NAME) {
        *reply = (uint8_t *)strdup("ERR bad_request");
        *reply_len = (uint32_t)strlen((char *)*reply);
        return 0;
    }
    char name[MAX_SEGMENT_NAME];
    memcpy(name, body, len);
    name[len] = 0;

    pthread_mutex_lock(&g_segs_mu);
    int idx = seg_find_locked(name);
    if (idx < 0) {
        pthread_mutex_unlock(&g_segs_mu);
        *reply = (uint8_t *)strdup("ERR noent");
        *reply_len = (uint32_t)strlen((char *)*reply);
        return 0;
    }
    uint64_t nbytes = g_segs[idx].nbytes;
    uint8_t *out = (uint8_t *)malloc((size_t)nbytes + 8);
    if (!out) {
        pthread_mutex_unlock(&g_segs_mu);
        *reply = (uint8_t *)strdup("ERR oom");
        *reply_len = (uint32_t)strlen((char *)*reply);
        return 0;
    }
    memcpy(out, g_segs[idx].data, (size_t)nbytes);
    pthread_mutex_unlock(&g_segs_mu);
    *reply = out;
    *reply_len = (uint32_t)nbytes;
    return 0;
}

static int do_list(uint8_t **reply, uint32_t *reply_len) {
    char buf[8192];
    size_t off = 0;
    pthread_mutex_lock(&g_segs_mu);
    for (int i = 0; i < MAX_SEGMENTS; i++) {
        if (!g_segs[i].live) continue;
        off += (size_t)snprintf(buf + off, sizeof(buf) - off, "%s %u %" PRIu64 "\n",
                                g_segs[i].name, g_segs[i].pages, g_segs[i].nbytes);
        if (off >= sizeof(buf) - 96) break;
    }
    pthread_mutex_unlock(&g_segs_mu);
    if (off == 0) off += (size_t)snprintf(buf, sizeof(buf), "(empty)\n");
    *reply = (uint8_t *)malloc(off + 1);
    if (!*reply) return -1;
    memcpy(*reply, buf, off + 1);
    *reply_len = (uint32_t)off;
    return 0;
}

static int do_drop(const uint8_t *body, uint32_t len, uint8_t **reply, uint32_t *reply_len) {
    if (len < 1 || len >= MAX_SEGMENT_NAME) {
        *reply = (uint8_t *)strdup("ERR bad_request");
        *reply_len = (uint32_t)strlen((char *)*reply);
        return 0;
    }
    char name[MAX_SEGMENT_NAME];
    memcpy(name, body, len);
    name[len] = 0;
    pthread_mutex_lock(&g_segs_mu);
    int idx = seg_find_locked(name);
    if (idx >= 0) seg_release_locked(idx);
    pthread_mutex_unlock(&g_segs_mu);
    *reply = (uint8_t *)strdup(idx >= 0 ? "OK dropped" : "ERR noent");
    *reply_len = (uint32_t)strlen((char *)*reply);
    return 0;
}

static uint32_t parse_gen(const char *name) {
    if (strncmp(name, "gen-", 4) != 0) return 0;
    const char *p = name + 4;
    uint32_t v = 0;
    int digits = 0;
    while (*p >= '0' && *p <= '9') {
        v = v * 10 + (uint32_t)(*p - '0');
        p++;
        digits++;
    }
    if (*p != 0 || digits == 0) return 0;
    return v;
}

static uint32_t scan_max_gen(void) {
    uint32_t max = 0;
    DIR *d = opendir(g_store_root);
    if (!d) return 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        uint32_t g = parse_gen(e->d_name);
        if (g > max) max = g;
    }
    closedir(d);
    return max;
}

static int rm_rf(const char *path) {
    struct stat st;
    if (lstat(path, &st) != 0) return -1;
    if (S_ISDIR(st.st_mode)) {
        DIR *d = opendir(path);
        if (!d) return -1;
        struct dirent *e;
        while ((e = readdir(d)) != NULL) {
            if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
            char child[1152];
            snprintf(child, sizeof(child), "%s/%s", path, e->d_name);
            (void)rm_rf(child);
        }
        closedir(d);
        return rmdir(path);
    }
    return unlink(path);
}

static int gen_cmp_desc(const void *a, const void *b) {
    uint32_t ga = *(const uint32_t *)a, gb = *(const uint32_t *)b;
    return (ga < gb) - (ga > gb);
}

static void gc_generations(void) {
    long keep = 8;
    const char *env = getenv("MEMX_STORE_KEEP_GENS");
    if (env) {
        long v = strtol(env, NULL, 10);
        if (v >= 0) keep = v;
    }
    if (keep == 0) return;
    uint32_t gens[MAX_GENS];
    size_t n = 0;
    DIR *d = opendir(g_store_root);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL && n < MAX_GENS) {
        uint32_t g = parse_gen(e->d_name);
        if (g) gens[n++] = g;
    }
    closedir(d);
    if (n <= (size_t)keep) return;
    qsort(gens, n, sizeof(uint32_t), gen_cmp_desc);
    for (size_t i = (size_t)keep; i < n; i++) {
        char dir[1152];
        snprintf(dir, sizeof(dir), "%s/gen-%06u", g_store_root, gens[i]);
        if (rm_rf(dir) == 0) logf("gc: pruned gen-%06u", gens[i]);
    }
}

static int do_commit(uint8_t **reply, uint32_t *reply_len) {
    if (memx_runtime_init() != 0) {
        *reply = (uint8_t *)strdup("ERR init");
        *reply_len = (uint32_t)strlen((char *)*reply);
        return 0;
    }
    memx_runtime_context_t *ctx = NULL;
    if (memx_runtime_context_create("memx-stored", &ctx) != 0 || !ctx) {
        *reply = (uint8_t *)strdup("ERR ctx");
        *reply_len = (uint32_t)strlen((char *)*reply);
        return 0;
    }
    int named = 0;
    char gen_dir[1152];
    uint32_t gen = scan_max_gen() + 1;
    snprintf(gen_dir, sizeof(gen_dir), "%s/gen-%06u", g_store_root, gen);
    mkdir_p(gen_dir);

    memx_runtime_tensor_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.struct_size = sizeof(desc);
    desc.role = MEMX_TENSOR_ROLE_DATA;
    desc.dtype = MEMX_TENSOR_DTYPE_INT32;
    desc.layout = MEMX_TENSOR_LAYOUT_ROW_MAJOR;
    desc.flags = MEMX_TENSOR_FLAG_READ_MOSTLY | MEMX_TENSOR_FLAG_COLD;

    int failed = 0;
    pthread_mutex_lock(&g_segs_mu);
    for (int i = 0; i < MAX_SEGMENTS && !failed; i++) {
        if (!g_segs[i].live) continue;
        size_t nbytes = (size_t)g_segs[i].nbytes;
        desc.rank = 2;
        desc.shape[0] = g_segs[i].pages;
        desc.shape[1] = PAGE_SZ / 4;
        desc.stride[0] = PAGE_SZ / 4;
        desc.stride[1] = 1;
        void *ptr = memx_runtime_context_malloc_tensor(ctx, nbytes, &desc);
        if (!ptr) { failed = 1; break; }
        memcpy(ptr, g_segs[i].data, nbytes);
        uint64_t done = 0;
        (void)memx_runtime_context_force_compress_range(ctx, ptr, 0, nbytes, &done);
        if (memx_runtime_context_name_segment(ctx, ptr, g_segs[i].name) != 0) failed = 1;
        named++;
    }
    pthread_mutex_unlock(&g_segs_mu);

    uint64_t bytes = 0;
    int rc = failed ? -1 : memx_runtime_capsule_export(gen_dir, &bytes);
    if (rc != 0 || named == 0) {
        char msg[160];
        snprintf(msg, sizeof(msg), "ERR commit rc=%d named=%d", rc, named);
        *reply = (uint8_t *)strdup(msg);
        *reply_len = (uint32_t)strlen(msg);
        memx_runtime_context_destroy(ctx);
        (void)memx_runtime_capsule_detach();
        memx_runtime_shutdown();
        (void)rm_rf(gen_dir);
        return 0;
    }
    gc_generations();
    char msg[256];
    snprintf(msg, sizeof(msg), "OK commit gen-%06u segments=%d bytes=%" PRIu64, gen, named, bytes);
    *reply = (uint8_t *)strdup(msg);
    *reply_len = (uint32_t)strlen(msg);
    memx_runtime_context_destroy(ctx);
    (void)memx_runtime_capsule_detach();
    memx_runtime_shutdown();
    return 0;
}

static int do_verify(const uint8_t *body, uint32_t len, uint8_t **reply, uint32_t *reply_len) {
    if (len == 0 || len >= 900) {
        *reply = (uint8_t *)strdup("ERR bad_request");
        *reply_len = (uint32_t)strlen((char *)*reply);
        return 0;
    }
    char gen[960];
    memcpy(gen, body, len);
    gen[len] = 0;
    char dir[1152];
    snprintf(dir, sizeof(dir), "%s/%s", g_store_root, gen);
    int arc = memx_runtime_capsule_attach(dir);
    if (arc != 0) {
        char msg[80];
        snprintf(msg, sizeof(msg), "ERR attach rc=%d dir=%s", arc, dir);
        *reply = (uint8_t *)strdup(msg);
        *reply_len = (uint32_t)strlen(msg);
        return 0;
    }
    uint64_t bad = 0, pages = 0;
    int rc = memx_runtime_capsule_verify(&bad, &pages);
    (void)memx_runtime_capsule_detach();
    char msg[128];
    if (rc == 0) snprintf(msg, sizeof(msg), "OK verify bad=0 pages=%" PRIu64, pages);
    else if (rc == EBADMSG) snprintf(msg, sizeof(msg), "OK verify corrupt bad=%" PRIu64 " pages=%" PRIu64, bad, pages);
    else snprintf(msg, sizeof(msg), "OK verify unsupported pages=%" PRIu64, pages);
    *reply = (uint8_t *)strdup(msg);
    *reply_len = (uint32_t)strlen(msg);
    return 0;
}

static int do_stats(uint8_t **reply, uint32_t *reply_len) {
    char buf[1024];
    int n = 0;
    pthread_mutex_lock(&g_segs_mu);
    uint64_t total = 0;
    int count = 0;
    for (int i = 0; i < MAX_SEGMENTS; i++) {
        if (!g_segs[i].live) continue;
        total += g_segs[i].nbytes;
        count++;
    }
    pthread_mutex_unlock(&g_segs_mu);
    pthread_mutex_lock(&g_clients_mu);
    int nclients = 0;
    for (int i = 0; i < MAX_CLIENTS; i++) if (g_clients[i].live) nclients++;
    pthread_mutex_unlock(&g_clients_mu);
    n = snprintf(buf, sizeof(buf), "OK stats segments=%d bytes=%" PRIu64 " clients=%d level=%u root=%s",
                 count, total, nclients, pressure_level_now(), g_store_root);
    (void)n;
    *reply = (uint8_t *)strdup(buf);
    *reply_len = (uint32_t)strlen(buf);
    return 0;
}

static unsigned pressure_level_now(void) {
    unsigned lvl = g_pressure_level;
    if (lvl && time(NULL) - g_pressure_ts > PRESSURE_DECAY_S) {
        g_pressure_level = 0;
        lvl = 0;
    }
    return lvl;
}

static pid_t peer_pid(int fd) {
    pid_t pid = 0;
    socklen_t len = (socklen_t)sizeof(pid);
    if (getsockopt(fd, 0, LOCAL_PEERPID, &pid, &len) != 0) return 0;
    return pid;
}

static client_slot_t *client_for_pid_locked(pid_t pid, int create) {
    int free_idx = -1;
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (g_clients[i].live && g_clients[i].pid == pid) return &g_clients[i];
        if (!g_clients[i].live && free_idx < 0) free_idx = i;
    }
    if (!create || free_idx < 0) return NULL;
    memset(&g_clients[free_idx], 0, sizeof(g_clients[free_idx]));
    g_clients[free_idx].pid = pid;
    g_clients[free_idx].live = 1;
    return &g_clients[free_idx];
}

static int do_regs(int fd, const uint8_t *body, uint32_t len, uint8_t **reply, uint32_t *reply_len) {
    if (len < 1 || len >= MAX_SEGMENT_NAME) {
        *reply = (uint8_t *)strdup("ERR bad_request");
        *reply_len = (uint32_t)strlen((char *)*reply);
        return 0;
    }
    char name[MAX_SEGMENT_NAME];
    memcpy(name, body, len);
    name[len] = 0;
    pid_t pid = peer_pid(fd);
    pthread_mutex_lock(&g_clients_mu);
    client_slot_t *c = client_for_pid_locked(pid, 1);
    if (c) {
        memcpy(c->name, name, len + 1);
        c->conn_fd = fd;
        c->last_beat = time(NULL);
    }
    pthread_mutex_unlock(&g_clients_mu);
    if (!c) {
        *reply = (uint8_t *)strdup("ERR full");
        *reply_len = 8;
        return 0;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "OK regs pid=%d name=%s", (int)pid, name);
    *reply = (uint8_t *)strdup(msg);
    *reply_len = (uint32_t)strlen(msg);
    logf("client registered pid=%d name=%s", (int)pid, name);
    return 0;
}

static int do_beat(int fd, const uint8_t *body, uint32_t len, uint8_t **reply, uint32_t *reply_len) {
    if (len < 1 || len > 128) {
        *reply = (uint8_t *)strdup("ERR bad_request");
        *reply_len = (uint32_t)strlen((char *)*reply);
        return 0;
    }
    char buf[129];
    memcpy(buf, body, len);
    buf[len] = 0;
    unsigned long long cp = 0, rp = 0, sv = 0;
    sscanf(buf, "%llu %llu %llu", &cp, &rp, &sv);
    pid_t pid = peer_pid(fd);
    pthread_mutex_lock(&g_clients_mu);
    client_slot_t *c = client_for_pid_locked(pid, 1);
    if (c) {
        c->conn_fd = fd;
        c->compressed_pages = cp;
        c->resident_pages = rp;
        c->saved_bytes = sv;
        c->beats++;
        c->last_beat = time(NULL);
    }
    pthread_mutex_unlock(&g_clients_mu);
    *reply = (uint8_t *)strdup(c ? "OK beat" : "ERR full");
    *reply_len = (uint32_t)strlen((char *)*reply);
    return 0;
}

static int do_poli(uint8_t **reply, uint32_t *reply_len) {
    char msg[64];
    snprintf(msg, sizeof(msg), "OK POLI level=%u", pressure_level_now());
    *reply = (uint8_t *)strdup(msg);
    *reply_len = (uint32_t)strlen(msg);
    return 0;
}

static int do_clnt(uint8_t **reply, uint32_t *reply_len) {
    char buf[8192];
    size_t off = 0;
    time_t now = time(NULL);
    pthread_mutex_lock(&g_clients_mu);
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (!g_clients[i].live) continue;
        off += (size_t)snprintf(buf + off, sizeof(buf) - off,
                                "%s %d %llu %lld %llu %llu %llu\n",
                                g_clients[i].name[0] ? g_clients[i].name : "-",
                                (int)g_clients[i].pid,
                                (unsigned long long)g_clients[i].beats,
                                (long long)(now - g_clients[i].last_beat),
                                (unsigned long long)g_clients[i].compressed_pages,
                                (unsigned long long)g_clients[i].resident_pages,
                                (unsigned long long)g_clients[i].saved_bytes);
        if (off >= sizeof(buf) - 128) break;
    }
    pthread_mutex_unlock(&g_clients_mu);
    if (off == 0) off += (size_t)snprintf(buf, sizeof(buf), "(empty)\n");
    *reply = (uint8_t *)malloc(off + 1);
    if (!*reply) return -1;
    memcpy(*reply, buf, off + 1);
    *reply_len = (uint32_t)off;
    return 0;
}

static int do_pres(const uint8_t *body, uint32_t len, uint8_t **reply, uint32_t *reply_len) {
    const char *dbg = getenv("MEMX_STORE_DEBUG");
    if (!dbg || dbg[0] != '1') {
        *reply = (uint8_t *)strdup("ERR disabled");
        *reply_len = 12;
        return 0;
    }
    if (len != 1 || body[0] < '0' || body[0] > '2') {
        *reply = (uint8_t *)strdup("ERR bad_level");
        *reply_len = 13;
        return 0;
    }
    g_pressure_level = (unsigned)(body[0] - '0');
    g_pressure_ts = time(NULL);
    char msg[64];
    snprintf(msg, sizeof(msg), "OK pres level=%u", g_pressure_level);
    *reply = (uint8_t *)strdup(msg);
    *reply_len = (uint32_t)strlen(msg);
    return 0;
}

static void *conn_main(void *arg) {
    int fd = (int)(intptr_t)arg;
    while (g_running) {
        uint32_t len = 0;
        uint8_t *frame = recv_frame(fd, &len);
        if (!frame) break;
        if (len < 4) {
            free(frame);
            break;
        }
        uint8_t *reply = NULL;
        uint32_t reply_len = 0;
        int close_conn = 0;
        if (len >= 8 && memcmp(frame, "PUT ", 4) == 0) {
            (void)do_put(frame + 4, len - 4, &reply, &reply_len);
        } else if (len >= 5 && memcmp(frame, "GET ", 4) == 0) {
            (void)do_get(frame + 4, len - 4, &reply, &reply_len);
        } else if (len == 4 && memcmp(frame, "LIST", 4) == 0) {
            (void)do_list(&reply, &reply_len);
        } else if (len >= 5 && memcmp(frame, "DROP", 4) == 0) {
            (void)do_drop(frame + 4, len - 4, &reply, &reply_len);
        } else if (len == 5 && memcmp(frame, "COMIT", 5) == 0) {
            (void)do_commit(&reply, &reply_len);
        } else if (len >= 5 && memcmp(frame, "VERIY", 5) == 0) {
            (void)do_verify(frame + 5, len - 5, &reply, &reply_len);
        } else if (len >= 5 && memcmp(frame, "REGS ", 5) == 0) {
            (void)do_regs(fd, frame + 5, len - 5, &reply, &reply_len);
        } else if (len >= 5 && memcmp(frame, "BEAT ", 5) == 0) {
            (void)do_beat(fd, frame + 5, len - 5, &reply, &reply_len);
        } else if (len == 4 && memcmp(frame, "POLI", 4) == 0) {
            (void)do_poli(&reply, &reply_len);
        } else if (len == 4 && memcmp(frame, "CLNT", 4) == 0) {
            (void)do_clnt(&reply, &reply_len);
        } else if (len >= 5 && memcmp(frame, "PRES ", 5) == 0) {
            (void)do_pres(frame + 5, len - 5, &reply, &reply_len);
        } else if (len == 4 && memcmp(frame, "STAT", 4) == 0) {
            (void)do_stats(&reply, &reply_len);
        } else if (len == 4 && memcmp(frame, "QUIT", 4) == 0) {
            reply = (uint8_t *)strdup("OK bye");
            reply_len = 6;
            close_conn = 1;
        } else {
            reply = (uint8_t *)strdup("ERR unknown_command");
            reply_len = (uint32_t)strlen((char *)reply);
        }
        if (send_frame(fd, reply, reply_len) != 0) {
            free(reply);
            free(frame);
            break;
        }
        free(reply);
        free(frame);
        if (close_conn) break;
    }
    close(fd);
    pthread_mutex_lock(&g_clients_mu);
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (g_clients[i].live && g_clients[i].conn_fd == fd) {
            logf("client gone pid=%d name=%s", (int)g_clients[i].pid, g_clients[i].name);
            g_clients[i].live = 0;
        }
    }
    pthread_mutex_unlock(&g_clients_mu);
    __sync_sub_and_fetch(&g_live, 1);
    return NULL;
}

static int acquire_lock(const char *path) {
    for (int attempt = 0; attempt < 2; attempt++) {
        int fd = open(path, O_RDWR | O_CREAT | O_EXCL, 0644);
        if (fd >= 0) {
            dprintf(fd, "%d\n", (int)getpid());
            return fd;
        }
        if (errno != EEXIST) return -1;
        int rfd = open(path, O_RDONLY);
        if (rfd < 0) {
            if (unlink(path) == 0) continue;
            return -1;
        }
        char buf[64] = {0};
        (void)read(rfd, buf, sizeof(buf) - 1);
        close(rfd);
        long pid = strtol(buf, NULL, 10);
        if (pid > 0 && kill((pid_t)pid, 0) == 0) return -1;
        if (pid > 0 && errno == EPERM) return -1;
        if (unlink(path) != 0) return -1;
        logf("reclaimed stale lock pid=%ld", pid);
    }
    return -1;
}

int main(int argc, char **argv) {
    const char *sock_override = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sock") == 0 && i + 1 < argc) sock_override = argv[++i];
        else if (strcmp(argv[i], "--root") == 0 && i + 1 < argc) {
            snprintf(g_store_root, sizeof(g_store_root), "%s", argv[++i]);
        }
    }
    const char *home = getenv("HOME");
    if (!g_store_root[0]) {
        snprintf(g_store_root, sizeof(g_store_root), "%s/Library/Application Support/MemX/store", home ? home : "/tmp");
    }
    if (sock_override) {
        snprintf(g_sock_path, sizeof(g_sock_path), "%s", sock_override);
    } else {
        snprintf(g_sock_path, sizeof(g_sock_path), "%s/store.sock", g_store_root);
    }
    mkdir_p(g_store_root);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);
    if (!getenv("MEMX_NO_SELFTEST")) setenv("MEMX_NO_SELFTEST", "1", 0);

    unlink(g_sock_path);
    int lfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (lfd < 0) {
        perror("socket");
        return 1;
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, g_sock_path, sizeof(addr.sun_path) - 1);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        perror("bind");
        return 1;
    }
    chmod(g_sock_path, 0600);
    if (listen(lfd, 8) != 0) {
        perror("listen");
        return 1;
    }

    char lock_path[1100];
    snprintf(lock_path, sizeof(lock_path), "%s/store.lock", g_store_root);
    int lock_fd = acquire_lock(lock_path);
    if (lock_fd < 0) {
        logf("another instance holds %s", lock_path);
        return 1;
    }

    install_pressure_source();
    logf("listening %s root=%s", g_sock_path, g_store_root);

    while (g_running) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(lfd, &rfds);
        struct timeval tv = {1, 0};
        int rv = select(lfd + 1, &rfds, NULL, NULL, &tv);
        if (rv <= 0) continue;
        if (!FD_ISSET(lfd, &rfds)) continue;
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) continue;
        uid_t euid = (uid_t)-1;
        gid_t egid = (gid_t)-1;
        if (getpeereid(cfd, &euid, &egid) != 0 || euid != getuid()) {
            logf("refused peer uid=%d", (int)euid);
            close(cfd);
            continue;
        }
        __sync_add_and_fetch(&g_live, 1);
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 8u << 20);
        pthread_t thr;
        int prc = pthread_create(&thr, &attr, conn_main, (void *)(intptr_t)cfd);
        pthread_attr_destroy(&attr);
        if (prc != 0) {
            __sync_sub_and_fetch(&g_live, 1);
            close(cfd);
            continue;
        }
        pthread_detach(thr);
    }

    for (int i = 0; i < SHUTDOWN_DRAIN_MS / 10 && g_live > 0; i++) {
        struct timespec ts = {0, 10000000L};
        nanosleep(&ts, NULL);
    }
    close(lfd);
    close(lock_fd);
    unlink(g_sock_path);
    unlink(lock_path);
    logf("bye live=%d", g_live);
    return 0;
}
