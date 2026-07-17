CC = clang
CFLAGS = -O2
CPPFLAGS = -Iinclude
FRAMEWORKS = -framework Metal -framework Foundation
LIBS = -lz
BUILD_DIR = build
RUNTIME_DYLIB = $(BUILD_DIR)/libmemx_runtime.dylib
BENCHMARK_DIR = benchmarks
EXAMPLE_DIR = examples
RUNTIME_BENCHES = benchmark_runtime_suite bench_context_stress bench_tensor_codecs bench_effective_capacity bench_hot_path_latency
RUNTIME_BENCH_BINS = $(addprefix $(BUILD_DIR)/,$(RUNTIME_BENCHES))
EXPLICIT_TEST = $(BUILD_DIR)/test_explicit_runtime
COMPRESSING_RACE_TEST = $(BUILD_DIR)/test_compressing_race
EMBEDDED_EXAMPLE = $(BUILD_DIR)/embedded_runtime_demo
CAPSULE_VESSEL = $(BUILD_DIR)/memx_capsule_vessel

.PHONY: all benchmarks examples clean test capsule-vessel explicit-runtime test-explicit test-compressing-race test-python-runtime test-python-bitexact test-weight-archive test-materialize test-python-transformer test-python-torch-transformer test-python-torch-pressure test-python example-embedded benchmark-runtime benchmark-stress benchmark-tensor-codecs benchmark-effective-capacity benchmark-hot-path-latency

all: $(RUNTIME_DYLIB) $(CAPSULE_VESSEL)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(RUNTIME_DYLIB): libmemx3.m include/memx_runtime.h | $(BUILD_DIR)
	$(CC) -dynamiclib $(CPPFLAGS) $(CFLAGS) $(FRAMEWORKS) $(LIBS) -o $@ $<

$(EXPLICIT_TEST): tests/test_explicit_runtime.c include/memx_runtime.h $(RUNTIME_DYLIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude -L$(BUILD_DIR) -Wl,-rpath,@executable_path -o $@ $< -lmemx_runtime

$(COMPRESSING_RACE_TEST): tests/test_compressing_race.c include/memx_runtime.h $(RUNTIME_DYLIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -std=c11 -Iinclude -L$(BUILD_DIR) -Wl,-rpath,@executable_path -o $@ $< -lmemx_runtime

$(EMBEDDED_EXAMPLE): $(EXAMPLE_DIR)/embedded_runtime_demo.c include/memx_runtime.h $(RUNTIME_DYLIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude -L$(BUILD_DIR) -Wl,-rpath,@executable_path -o $@ $< -lmemx_runtime

$(BUILD_DIR)/benchmark_runtime_suite: $(BENCHMARK_DIR)/benchmark_runtime_suite.c include/memx_runtime.h $(RUNTIME_DYLIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude -L$(BUILD_DIR) -Wl,-rpath,@executable_path -o $@ $< -lmemx_runtime

$(BUILD_DIR)/bench_context_stress: $(BENCHMARK_DIR)/bench_context_stress.c include/memx_runtime.h $(RUNTIME_DYLIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude -L$(BUILD_DIR) -Wl,-rpath,@executable_path -o $@ $< -lmemx_runtime

$(BUILD_DIR)/bench_tensor_codecs: $(BENCHMARK_DIR)/bench_tensor_codecs.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD_DIR)/bench_effective_capacity: $(BENCHMARK_DIR)/bench_effective_capacity.c include/memx_runtime.h $(RUNTIME_DYLIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude -L$(BUILD_DIR) -Wl,-rpath,@executable_path -o $@ $< -lmemx_runtime

$(BUILD_DIR)/bench_hot_path_latency: $(BENCHMARK_DIR)/bench_hot_path_latency.c include/memx_runtime.h $(RUNTIME_DYLIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude -L$(BUILD_DIR) -Wl,-rpath,@executable_path -o $@ $< -lmemx_runtime

benchmarks: $(RUNTIME_BENCH_BINS)

explicit-runtime: $(RUNTIME_DYLIB)

test-explicit: $(EXPLICIT_TEST)
	@$(EXPLICIT_TEST)

test-compressing-race: $(COMPRESSING_RACE_TEST)
	@$(COMPRESSING_RACE_TEST)

test-python-runtime: $(RUNTIME_DYLIB)
	@python3 tests/test_python_runtime.py

test-python-bitexact: $(RUNTIME_DYLIB)
	@python3 tests/test_python_bitexact_compute.py

test-weight-archive: $(RUNTIME_DYLIB)
	@python3 tests/test_weight_archive.py

test-materialize: $(RUNTIME_DYLIB)
	@python3 tests/test_materialize_bitexact.py

test-python-transformer: $(RUNTIME_DYLIB)
	@python3 tests/test_python_transformer_lifecycle.py

test-python-torch-transformer: $(RUNTIME_DYLIB)
	@python3 tests/test_python_torch_transformer.py

test-python-torch-pressure: $(RUNTIME_DYLIB)
	@python3 tests/test_python_torch_pressure.py

test-python: test-python-runtime test-python-bitexact test-python-transformer test-python-torch-transformer test-python-torch-pressure

examples: $(EMBEDDED_EXAMPLE)

example-embedded: $(EMBEDDED_EXAMPLE)
	@$(EMBEDDED_EXAMPLE)

benchmark-runtime: $(BUILD_DIR)/benchmark_runtime_suite
	@$(BUILD_DIR)/benchmark_runtime_suite

benchmark-stress: $(BUILD_DIR)/bench_context_stress
	@$(BUILD_DIR)/bench_context_stress

benchmark-tensor-codecs: $(BUILD_DIR)/bench_tensor_codecs
	@$(BUILD_DIR)/bench_tensor_codecs

benchmark-effective-capacity: $(BUILD_DIR)/bench_effective_capacity
	@$(BUILD_DIR)/bench_effective_capacity

benchmark-hot-path-latency: $(BUILD_DIR)/bench_hot_path_latency
	@$(BUILD_DIR)/bench_hot_path_latency

test: test-explicit test-compressing-race example-embedded

clean:
	rm -rf $(BUILD_DIR)

$(CAPSULE_VESSEL): tools/memx_capsule_vessel.c include/memx_runtime.h $(RUNTIME_DYLIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude -L$(BUILD_DIR) -Wl,-rpath,@executable_path -o $@ $< -lmemx_runtime

capsule-vessel: $(CAPSULE_VESSEL)
