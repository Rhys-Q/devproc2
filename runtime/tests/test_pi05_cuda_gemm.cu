// Pi0.5 CUDA FP8 cuBLASLt packed GEMM smoke test.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <dlfcn.h>
#include <dlpack/dlpack.h>

#include "devproc2/runtime/device_api.h"
#include "devproc2/runtime/packed_backend.h"
#include "devproc2/runtime/packed_func.h"
#include "devproc2/runtime/tensor.h"
#include "devproc2/runtime/vm_value.h"

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

using namespace devproc2;

#ifndef DEVPROC2_PI05_CUDA_BACKEND_SO
#define DEVPROC2_PI05_CUDA_BACKEND_SO "libdevproc2_pi05_cuda_backend.so"
#endif

using BackendRegisterFn = void (*)(PackedFuncRegistry*);

void register_pi05_cuda_backend_for_test() {
    (void)CurrentCUDAPackedFuncStream();
    static void* handle = []() -> void* {
        void* h = dlopen(DEVPROC2_PI05_CUDA_BACKEND_SO, RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            const char* err = dlerror();
            throw std::runtime_error(
                std::string("failed to dlopen Pi0.5 CUDA backend: ") +
                (err ? err : "unknown error"));
        }
        return h;
    }();
    dlerror();
    auto* fn = reinterpret_cast<BackendRegisterFn>(
        dlsym(handle, "devproc2_register_pi05_cuda_backend"));
    const char* err = dlerror();
    if (err || !fn) {
        throw std::runtime_error(
            std::string("failed to resolve Pi0.5 CUDA backend register symbol: ") +
            (err ? err : "missing symbol"));
    }
    fn(&PackedFuncRegistry::Global());
}

template <typename T>
static Tensor external_cpu_tensor(std::vector<T>& data,
                                  std::vector<int64_t> shape,
                                  DLDataType dtype) {
    return Tensor::FromExternalBuffer(data.data(), DLDevice{kDLCPU, 0}, shape, dtype);
}

void test_fp8_nt_bf16_matches_small_reference() {
    register_pi05_cuda_backend_for_test();
    auto pf = PackedFuncRegistry::Global().Get("pi05.cuda.fp8_nt_bf16");
    CHECK(pf.defined());

    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;
    std::vector<__nv_fp8_e4m3> a(M * K);
    std::vector<__nv_fp8_e4m3> b_nk(N * K);
    for (int i = 0; i < M * K; ++i) {
        a[static_cast<size_t>(i)] = __nv_fp8_e4m3(static_cast<float>((i % 5) - 2));
    }
    for (int i = 0; i < N * K; ++i) {
        b_nk[static_cast<size_t>(i)] = __nv_fp8_e4m3(static_cast<float>((i % 7) - 3));
    }
    std::vector<float> scale_a = {0.25f};
    std::vector<float> scale_b = {0.5f};
    std::vector<__nv_bfloat16> out(M * N);

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    auto host_a = external_cpu_tensor(a, {M, K}, DLDataType{kDLFloat8_e4m3, 8, 1});
    auto host_b = external_cpu_tensor(b_nk, {N, K}, DLDataType{kDLFloat8_e4m3, 8, 1});
    auto host_sa = external_cpu_tensor(scale_a, {1}, DLDataType{kDLFloat, 32, 1});
    auto host_sb = external_cpu_tensor(scale_b, {1}, DLDataType{kDLFloat, 32, 1});
    auto host_out = external_cpu_tensor(out, {M, N}, DLDataType{kDLBfloat, 16, 1});

    auto dev_a = Tensor::Empty({M, K}, DLDataType{kDLFloat8_e4m3, 8, 1}, cuda_dev);
    auto dev_b = Tensor::Empty({N, K}, DLDataType{kDLFloat8_e4m3, 8, 1}, cuda_dev);
    auto dev_sa = Tensor::Empty({1}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_sb = Tensor::Empty({1}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_out = Tensor::Empty({M, N}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);

    cuda_api->CopyDataFromTo(host_a->dl(), dev_a->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_b->dl(), dev_b->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_sa->dl(), dev_sa->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_sb->dl(), dev_sb->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_a),
        VMValue::ObjRef(dev_b),
        VMValue::Int(M),
        VMValue::Int(N),
        VMValue::Int(K),
        VMValue::ObjRef(dev_sa),
        VMValue::ObjRef(dev_sb),
        VMValue::ObjRef(dev_out),
    };
    PackedArgs packed(args);
    pf->Call(packed);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_out->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float want = 0.0f;
            for (int k = 0; k < K; ++k) {
                want += static_cast<float>(a[static_cast<size_t>(m * K + k)]) * scale_a[0]
                      * static_cast<float>(b_nk[static_cast<size_t>(n * K + k)]) * scale_b[0];
            }
            float got = __bfloat162float(out[static_cast<size_t>(m * N + n)]);
            CHECK(std::fabs(got - want) < 0.5f);
        }
    }
}

void test_bf16_nn_bf16_matches_small_reference() {
    register_pi05_cuda_backend_for_test();
    auto pf = PackedFuncRegistry::Global().Get("pi05.cuda.bf16_nn_bf16");
    CHECK(pf.defined());

    constexpr int M = 8;
    constexpr int N = 12;
    constexpr int K = 16;
    std::vector<__nv_bfloat16> a(M * K);
    std::vector<__nv_bfloat16> b_kn(K * N);
    for (int i = 0; i < M * K; ++i) {
        a[static_cast<size_t>(i)] =
            __float2bfloat16(static_cast<float>((i % 9) - 4) * 0.25f);
    }
    for (int i = 0; i < K * N; ++i) {
        b_kn[static_cast<size_t>(i)] =
            __float2bfloat16(static_cast<float>((i % 7) - 3) * 0.125f);
    }
    std::vector<__nv_bfloat16> out(M * N);

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    auto host_a = external_cpu_tensor(a, {M, K}, DLDataType{kDLBfloat, 16, 1});
    auto host_b = external_cpu_tensor(b_kn, {K, N}, DLDataType{kDLBfloat, 16, 1});
    auto host_out = external_cpu_tensor(out, {M, N}, DLDataType{kDLBfloat, 16, 1});

    auto dev_a = Tensor::Empty({M, K}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_b = Tensor::Empty({K, N}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_out = Tensor::Empty({M, N}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);

    cuda_api->CopyDataFromTo(host_a->dl(), dev_a->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_b->dl(), dev_b->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_a),
        VMValue::ObjRef(dev_b),
        VMValue::Int(M),
        VMValue::Int(N),
        VMValue::Int(K),
        VMValue::ObjRef(dev_out),
    };
    PackedArgs packed(args);
    pf->Call(packed);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_out->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float want = 0.0f;
            for (int k = 0; k < K; ++k) {
                want += __bfloat162float(a[static_cast<size_t>(m * K + k)])
                      * __bfloat162float(b_kn[static_cast<size_t>(k * N + n)]);
            }
            float got = __bfloat162float(out[static_cast<size_t>(m * N + n)]);
            CHECK(std::fabs(got - want) < 0.125f);
        }
    }
}

void test_fp8_nt_bf16_accum_adds_into_output() {
    register_pi05_cuda_backend_for_test();
    auto pf = PackedFuncRegistry::Global().Get("pi05.cuda.fp8_nt_bf16_accum");
    CHECK(pf.defined());

    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;
    std::vector<__nv_fp8_e4m3> a(M * K);
    std::vector<__nv_fp8_e4m3> b_nk(N * K);
    std::vector<float> initial(M * N);
    for (int i = 0; i < M * K; ++i) {
        a[static_cast<size_t>(i)] = __nv_fp8_e4m3(static_cast<float>((i % 3) - 1));
    }
    for (int i = 0; i < N * K; ++i) {
        b_nk[static_cast<size_t>(i)] = __nv_fp8_e4m3(static_cast<float>((i % 5) - 2));
    }
    std::vector<__nv_bfloat16> out(M * N);
    for (int i = 0; i < M * N; ++i) {
        initial[static_cast<size_t>(i)] = static_cast<float>((i % 11) - 5) * 0.125f;
        out[static_cast<size_t>(i)] = __float2bfloat16(initial[static_cast<size_t>(i)]);
    }
    std::vector<float> scale_a = {0.125f};
    std::vector<float> scale_b = {0.25f};

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    auto host_a = external_cpu_tensor(a, {M, K}, DLDataType{kDLFloat8_e4m3, 8, 1});
    auto host_b = external_cpu_tensor(b_nk, {N, K}, DLDataType{kDLFloat8_e4m3, 8, 1});
    auto host_sa = external_cpu_tensor(scale_a, {1}, DLDataType{kDLFloat, 32, 1});
    auto host_sb = external_cpu_tensor(scale_b, {1}, DLDataType{kDLFloat, 32, 1});
    auto host_out = external_cpu_tensor(out, {M, N}, DLDataType{kDLBfloat, 16, 1});

    auto dev_a = Tensor::Empty({M, K}, DLDataType{kDLFloat8_e4m3, 8, 1}, cuda_dev);
    auto dev_b = Tensor::Empty({N, K}, DLDataType{kDLFloat8_e4m3, 8, 1}, cuda_dev);
    auto dev_sa = Tensor::Empty({1}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_sb = Tensor::Empty({1}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_out = Tensor::Empty({M, N}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);

    cuda_api->CopyDataFromTo(host_a->dl(), dev_a->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_b->dl(), dev_b->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_sa->dl(), dev_sa->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_sb->dl(), dev_sb->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_out->dl(), dev_out->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_a),
        VMValue::ObjRef(dev_b),
        VMValue::ObjRef(dev_out),
        VMValue::Int(M),
        VMValue::Int(N),
        VMValue::Int(K),
        VMValue::ObjRef(dev_sa),
        VMValue::ObjRef(dev_sb),
    };
    PackedArgs packed(args);
    pf->Call(packed);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_out->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float want = initial[static_cast<size_t>(m * N + n)];
            for (int k = 0; k < K; ++k) {
                want += static_cast<float>(a[static_cast<size_t>(m * K + k)]) * scale_a[0]
                      * static_cast<float>(b_nk[static_cast<size_t>(n * K + k)]) * scale_b[0];
            }
            float got = __bfloat162float(out[static_cast<size_t>(m * N + n)]);
            CHECK(std::fabs(got - want) < 0.5f);
        }
    }
}

}  // namespace

int main() {
    RUN(test_fp8_nt_bf16_matches_small_reference);
    RUN(test_bf16_nn_bf16_matches_small_reference);
    RUN(test_fp8_nt_bf16_accum_adds_into_output);

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
