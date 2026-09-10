#include "memx_runtime.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#define PAGE_SZ 16384ull
#define MAX_SEGMENT_NAME 64
#define MAX_SEGMENTS 128
#define MAX_PAYLOAD (256ull * 1024 * 1024)

typedef struct {
    char name[MAX_SEGMENT_NAME];
    uint32_t pages;
    uint64_t nbytes;
    uint8_t *data;
    int live;
} seg_slot_t;

static seg_slot_t g_segs[MAX_SEGMENTS];
static pthread_mutex_t g_segs_mu = PTHREAD_MUTEX_INITIALIZER;
static volatile sig_atomic_t g_running = 1;
static char g_store_root[1024];
static char g_sock_path[1024];

static void on_signal(int sig) {
    (void)sig;
    g_running = 0;
}

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
    static uint32_t gen_counter = 0;
    uint32_t gen = ++gen_counter;
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
        return 0;
    }
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
    n = snprintf(buf, sizeof(buf), "OK stats segments=%d bytes=%" PRIu64 " root=%s", count, total, g_store_root);
    (void)n;
    *reply = (uint8_t *)strdup(buf);
    *reply_len = (uint32_t)strlen(buf);
    return 0;
}

static void handle_conn(int fd) {
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
    int lock_fd = open(lock_path, O_RDWR | O_CREAT | O_EXCL, 0644);
    if (lock_fd < 0) {
        fprintf(stderr, "[memx_stored] another instance holds %s\n", lock_path);
        return 1;
    }
    dprintf(lock_fd, "%d\n", (int)getpid());

    fprintf(stderr, "[memx_stored] listening %s root=%s\n", g_sock_path, g_store_root);

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
        handle_conn(cfd);
    }

    unlink(g_sock_path);
    unlink(lock_path);
    fprintf(stderr, "[memx_stored] bye\n");
    return 0;
}
