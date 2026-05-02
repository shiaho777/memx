CC = clang
CFLAGS = -O2
FRAMEWORKS = -framework Metal -framework Foundation
LIBS = -lz

.PHONY: all clean install uninstall test

all: libmemx3.dylib memx

libmemx3.dylib: libmemx3.m
	$(CC) -dynamiclib $(CFLAGS) $(FRAMEWORKS) $(LIBS) -o $@ $<

memx: memx.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $<

# Benchmarks
bench_all: bench_all.m libmemx3.dylib memx
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $<

bench_real_apps: bench_real_apps.m libmemx3.dylib memx
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $<

bench_latency: bench_latency.m libmemx3.dylib memx
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $<

bench_dedup: bench_dedup.m libmemx3.dylib memx
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $<

bench_mt_expansion: bench_mt_expansion.m libmemx3.dylib memx
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $<

benchmarks: bench_all bench_real_apps bench_latency bench_dedup bench_mt_expansion

# Run all benchmarks
test: benchmarks
	@echo "═══ Running all benchmarks ═══"
	@./memx ./bench_all 2>&1 | grep -E "saved:|Integrity"
	@./memx ./bench_latency 2>&1 | grep -E "P50|P99|Throughput|Integrity"
	@./memx ./bench_dedup 2>&1 | grep -E "Integrity"
	@./memx ./bench_mt_expansion 2>&1 | grep -E "Integrity|expansion"

# Install/uninstall global mode
install: libmemx3.dylib memx
	@./install.sh

uninstall:
	@./uninstall.sh

clean:
	rm -f libmemx3.dylib memx bench_all bench_real_apps bench_latency \
	      bench_dedup bench_mt_expansion bench_comparison bench_cpu_overhead \
	      bench_latency_breakdown bench_gpu_throughput bench_ablation \
	      bench_prefetch bench_evaluation
