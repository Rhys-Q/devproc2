// M11 Kernel C++ unit tests — CUDAKernelRegistry
// No GPU required — tests only registration/lookup.
// Build: cmake -DDEVPROC2_BUILD_TESTS=ON && make test_m11_kernel
// Run:   ./build/runtime/tests/test_m11_kernel

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#ifdef DEVPROC2_WITH_CUDA
#include "devproc2/runtime/cuda_kernel_registry.h"
#include "devproc2/runtime/kernel.h"
#endif

namespace {

int g_pass = 0;
int g_fail = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "  FAIL: " #cond "\n    at " __FILE__ ":"              \
                      << __LINE__ << "\n";                                       \
            ++g_fail;                                                            \
            return;                                                              \
        }                                                                        \
    } while (0)

#define RUN(fn)                                                                  \
    do {                                                                         \
        int prev_fail = g_fail;                                                  \
        std::cout << "[ RUN  ] " #fn "\n";                                      \
        fn();                                                                    \
        if (g_fail == prev_fail) {                                               \
            std::cout << "[ PASS ] " #fn "\n";                                  \
            ++g_pass;                                                            \
        }                                                                        \
    } while (0)

#ifdef DEVPROC2_WITH_CUDA

using namespace devproc2;

// ── Test 1: register and get kernel ─────────────────────────────────────────

void test_registry_register_and_get() {
    const std::string name = "test.m11_relu_kernel";
    std::vector<uint8_t> fake_cubin = {0x00, 0x01, 0x02, 0x03};
    std::string func_name = "relu_fp16";
    std::array<int32_t, 3> block_dims = {128, 1, 1};

    CUDAKernelRegistry::Global().Register(name, fake_cubin, func_name, block_dims);

    CHECK(CUDAKernelRegistry::Global().Has(name));

    KernelObj* k = CUDAKernelRegistry::Global().Get(name);
    CHECK(k != nullptr);
    CHECK(k->name == name);
    CHECK(k->func_name == func_name);
    CHECK(k->cubin_data == fake_cubin);
    CHECK(k->block_dims == block_dims);
}

// ── Test 2: missing key returns nullptr ──────────────────────────────────────

void test_registry_missing_returns_nullptr() {
    const std::string name = "test.m11_definitely_not_registered";
    CHECK(CUDAKernelRegistry::Global().Get(name) == nullptr);
    CHECK(!CUDAKernelRegistry::Global().Has(name));
}

// ── Test 3: overwrite registration ───────────────────────────────────────────

void test_registry_overwrite() {
    const std::string name = "test.m11_overwrite";
    CUDAKernelRegistry::Global().Register(name, {0xAA}, "fn_v1", {32, 1, 1});
    CUDAKernelRegistry::Global().Register(name, {0xBB}, "fn_v2", {64, 1, 1});

    KernelObj* k = CUDAKernelRegistry::Global().Get(name);
    CHECK(k != nullptr);
    CHECK(k->func_name == "fn_v2");
    CHECK(k->cubin_data.size() == 1);
    CHECK(k->cubin_data[0] == 0xBB);
}

// ── Test 4: multiple kernels coexist ─────────────────────────────────────────

void test_registry_multiple_kernels() {
    CUDAKernelRegistry::Global().Register("test.m11_k1", {0x01}, "fn1", {128, 1, 1});
    CUDAKernelRegistry::Global().Register("test.m11_k2", {0x02}, "fn2", {256, 1, 1});

    CHECK(CUDAKernelRegistry::Global().Has("test.m11_k1"));
    CHECK(CUDAKernelRegistry::Global().Has("test.m11_k2"));

    auto* k1 = CUDAKernelRegistry::Global().Get("test.m11_k1");
    auto* k2 = CUDAKernelRegistry::Global().Get("test.m11_k2");
    CHECK(k1 != nullptr && k1->func_name == "fn1");
    CHECK(k2 != nullptr && k2->func_name == "fn2");
}

#else  // !DEVPROC2_WITH_CUDA

void test_cuda_disabled_placeholder() {
    std::cout << "  (CUDA disabled — CUDAKernelRegistry tests skipped)\n";
    ++g_pass;
}

#endif  // DEVPROC2_WITH_CUDA

}  // namespace

int main() {
#ifdef DEVPROC2_WITH_CUDA
    RUN(test_registry_register_and_get);
    RUN(test_registry_missing_returns_nullptr);
    RUN(test_registry_overwrite);
    RUN(test_registry_multiple_kernels);
#else
    RUN(test_cuda_disabled_placeholder);
#endif

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return (g_fail == 0) ? 0 : 1;
}
