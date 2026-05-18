// Optional Pi0.5 CUDA kernel launch smoke test.

#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

#include "devproc2/runtime/cuda_kernel_registry.h"
#include "devproc2/runtime/device_api.h"
#include "devproc2/runtime/kernel.h"
#include "devproc2/runtime/tensor.h"
#include "devproc2/runtime/vm_value.h"

namespace devproc2 {
void CUDAKernelLauncher_Launch(
    const KernelObj* kernel,
    std::vector<VMValue>& args,
    const std::vector<int64_t>& launch_args,
    void* stream);
}

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

static bool read_file_binary(const std::string& path, std::vector<uint8_t>* out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return false;
    auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    out->resize(size);
    f.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(size));
    return true;
}

static Tensor external_cpu_tensor(void* data, std::vector<int64_t> shape, DLDataType dtype) {
    return Tensor::FromExternalBuffer(data, DLDevice{kDLCPU, 0}, shape, dtype);
}

static float bf16_bits_to_f32(uint16_t x) {
    uint32_t bits = static_cast<uint32_t>(x) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

static uint16_t f32_to_bf16_bits(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(float));
    return static_cast<uint16_t>(bits >> 16);
}

static int64_t f32_to_i64_bits(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(float));
    return static_cast<int64_t>(bits);
}

static KernelObj* register_kernel_from_artifact(
    const std::string& name,
    const std::string& symbol) {
    const std::string cubin_path =
        std::string(DEVPROC2_SOURCE_DIR)
        + "/build/pi05_fp8_artifact/kernels/"
        + name.substr(std::string("kernel.").size()) + ".cubin";
    std::vector<uint8_t> cubin;
    if (!read_file_binary(cubin_path, &cubin)) {
        std::cout << "  SKIP: missing " << cubin_path << "\n";
        return nullptr;
    }
    CUDAKernelRegistry::Global().Register(
        name, cubin, symbol, {1, 1, 1}, {256, 1, 1}, 0);
    return CUDAKernelRegistry::Global().Get(name);
}

void test_image_u8_to_bf16_norm_launches_from_real_pi05_cubin() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_image_u8_to_bf16_norm",
        "pi05_image_u8_to_bf16_norm");
    if (kernel == nullptr) return;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<uint8_t> cpu_in = {0, 127, 255, 64};
    std::vector<uint16_t> cpu_out(4, 0);
    auto host_in = external_cpu_tensor(
        cpu_in.data(), {4}, DLDataType{kDLUInt, 8, 1});
    auto host_out = external_cpu_tensor(
        cpu_out.data(), {4}, DLDataType{kDLBfloat, 16, 1});
    auto dev_in = Tensor::Empty({4}, DLDataType{kDLUInt, 8, 1}, cuda_dev);
    auto dev_out = Tensor::Empty({4}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);

    cuda_api->CopyDataFromTo(host_in->dl(), dev_in->dl(), nullptr);
    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_in),
        VMValue::Int(4),
        VMValue::ObjRef(dev_out),
    };
    CUDAKernelLauncher_Launch(kernel, args, {}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_out->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    const float expected[4] = {
        -1.0f,
        127.0f / 127.5f - 1.0f,
        1.0f,
        64.0f / 127.5f - 1.0f,
    };
    for (int i = 0; i < 4; ++i) {
        float got = bf16_bits_to_f32(cpu_out[static_cast<size_t>(i)]);
        CHECK(std::fabs(got - expected[i]) < 0.01f);
    }
}

void test_cast_f32_to_bf16_launch_matches_reference() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_cast_f32_to_bf16",
        "pi05_cast_f32_to_bf16");
    if (kernel == nullptr) return;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<float> in = {1.0f, -2.0f, 0.5f, 3.25f};
    std::vector<uint16_t> out(4, 0);
    auto host_in = external_cpu_tensor(in.data(), {4}, DLDataType{kDLFloat, 32, 1});
    auto host_out = external_cpu_tensor(out.data(), {4}, DLDataType{kDLBfloat, 16, 1});
    auto dev_in = Tensor::Empty({4}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_out = Tensor::Empty({4}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_in->dl(), dev_in->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_in),
        VMValue::Int(4),
        VMValue::ObjRef(dev_out),
    };
    CUDAKernelLauncher_Launch(kernel, args, {}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_out->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    for (int i = 0; i < 4; ++i) {
        CHECK(std::fabs(bf16_bits_to_f32(out[static_cast<size_t>(i)]) - in[static_cast<size_t>(i)]) < 0.02f);
    }
}

void test_bias_add_bf16_launch_updates_in_place() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_bias_add_bf16",
        "pi05_bias_add_bf16");
    if (kernel == nullptr) return;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<uint16_t> x(6), bias(3), out(6, 0);
    for (int i = 0; i < 6; ++i) x[static_cast<size_t>(i)] = f32_to_bf16_bits(static_cast<float>(i));
    bias[0] = f32_to_bf16_bits(10.0f);
    bias[1] = f32_to_bf16_bits(20.0f);
    bias[2] = f32_to_bf16_bits(30.0f);
    auto host_x = external_cpu_tensor(x.data(), {2, 3}, DLDataType{kDLBfloat, 16, 1});
    auto host_bias = external_cpu_tensor(bias.data(), {3}, DLDataType{kDLBfloat, 16, 1});
    auto host_out = external_cpu_tensor(out.data(), {2, 3}, DLDataType{kDLBfloat, 16, 1});
    auto dev_x = Tensor::Empty({2, 3}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_bias = Tensor::Empty({3}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_x->dl(), dev_x->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_bias->dl(), dev_bias->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_x),
        VMValue::ObjRef(dev_bias),
        VMValue::Int(2),
        VMValue::Int(3),
    };
    CUDAKernelLauncher_Launch(kernel, args, {1, 1, 1, 256, 1, 1, 0}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_x->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    for (int i = 0; i < 6; ++i) {
        float want = static_cast<float>(i) + (i % 3 == 0 ? 10.0f : (i % 3 == 1 ? 20.0f : 30.0f));
        CHECK(std::fabs(bf16_bits_to_f32(out[static_cast<size_t>(i)]) - want) < 0.02f);
    }
}

void test_position_add_bf16_launch_updates_in_place() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_position_add_bf16",
        "pi05_position_add_bf16");
    if (kernel == nullptr) return;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    constexpr int ROWS = 4;
    constexpr int POS = 2;
    constexpr int COLS = 3;
    std::vector<uint16_t> x(ROWS * COLS), pos(POS * COLS), out(ROWS * COLS, 0);
    for (int i = 0; i < ROWS * COLS; ++i) {
        x[static_cast<size_t>(i)] = f32_to_bf16_bits(static_cast<float>(i));
    }
    for (int i = 0; i < POS * COLS; ++i) {
        pos[static_cast<size_t>(i)] = f32_to_bf16_bits(100.0f + static_cast<float>(i));
    }

    auto host_x = external_cpu_tensor(x.data(), {ROWS, COLS}, DLDataType{kDLBfloat, 16, 1});
    auto host_pos = external_cpu_tensor(pos.data(), {POS, COLS}, DLDataType{kDLBfloat, 16, 1});
    auto host_out = external_cpu_tensor(out.data(), {ROWS, COLS}, DLDataType{kDLBfloat, 16, 1});
    auto dev_x = Tensor::Empty({ROWS, COLS}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_pos = Tensor::Empty({POS, COLS}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_x->dl(), dev_x->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_pos->dl(), dev_pos->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_x),
        VMValue::ObjRef(dev_pos),
        VMValue::Int(ROWS),
        VMValue::Int(POS),
        VMValue::Int(COLS),
    };
    CUDAKernelLauncher_Launch(kernel, args, {1, 1, 1, 256, 1, 1, 0}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_x->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            int idx = r * COLS + c;
            int pidx = (r % POS) * COLS + c;
            float want = static_cast<float>(idx) + 100.0f + static_cast<float>(pidx);
            CHECK(std::fabs(bf16_bits_to_f32(out[static_cast<size_t>(idx)]) - want) < 0.02f);
        }
    }
}

void test_euler_update_bf16_launch_updates_float_actions() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_euler_update_bf16",
        "pi05_euler_update_bf16");
    if (kernel == nullptr) return;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<float> x = {1.0f, 2.0f, -1.0f};
    std::vector<uint16_t> v = {
        f32_to_bf16_bits(4.0f),
        f32_to_bf16_bits(-2.0f),
        f32_to_bf16_bits(1.5f),
    };
    std::vector<float> out(3, 0.0f);
    auto host_x = external_cpu_tensor(x.data(), {3}, DLDataType{kDLFloat, 32, 1});
    auto host_v = external_cpu_tensor(v.data(), {3}, DLDataType{kDLBfloat, 16, 1});
    auto host_out = external_cpu_tensor(out.data(), {3}, DLDataType{kDLFloat, 32, 1});
    auto dev_x = Tensor::Empty({3}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_v = Tensor::Empty({3}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_x->dl(), dev_x->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_v->dl(), dev_v->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_x),
        VMValue::ObjRef(dev_v),
        VMValue::Int(f32_to_i64_bits(-0.5f)),
        VMValue::Int(3),
    };
    CUDAKernelLauncher_Launch(kernel, args, {1, 1, 1, 256, 1, 1, 0}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_x->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    CHECK(std::fabs(out[0] - -1.0f) < 0.001f);
    CHECK(std::fabs(out[1] - 3.0f) < 0.001f);
    CHECK(std::fabs(out[2] - -1.75f) < 0.01f);
}

void test_qkv_split_rope_bf16_uses_interleaved_qk_pairs() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_qkv_split_rope_bf16",
        "pi05_qkv_split_rope_bf16");
    if (kernel == nullptr) return;

    constexpr int ROWS = 1;
    constexpr int Q_DIM = 4;
    constexpr int K_DIM = 4;
    constexpr int V_DIM = 4;
    constexpr int HD = 4;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    // Q/K are stored in the Pi0.5 interleaved RoPE layout:
    // [lo0, hi0, lo1, hi1]. V is copied without RoPE.
    std::vector<uint16_t> qkv = {
        f32_to_bf16_bits(1.0f), f32_to_bf16_bits(3.0f),
        f32_to_bf16_bits(2.0f), f32_to_bf16_bits(4.0f),
        f32_to_bf16_bits(0.5f), f32_to_bf16_bits(1.5f),
        f32_to_bf16_bits(-0.5f), f32_to_bf16_bits(-1.5f),
        f32_to_bf16_bits(10.0f), f32_to_bf16_bits(20.0f),
        f32_to_bf16_bits(30.0f), f32_to_bf16_bits(40.0f),
    };
    std::vector<uint16_t> rope = {
        f32_to_bf16_bits(0.5f), f32_to_bf16_bits(0.25f),
        f32_to_bf16_bits(1.0f), f32_to_bf16_bits(-0.5f),
    };
    std::vector<uint16_t> q(Q_DIM, 0), k(K_DIM, 0), v(V_DIM, 0);

    auto host_qkv = external_cpu_tensor(qkv.data(), {ROWS, Q_DIM + K_DIM + V_DIM}, DLDataType{kDLBfloat, 16, 1});
    auto host_rope = external_cpu_tensor(rope.data(), {ROWS, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_q = external_cpu_tensor(q.data(), {ROWS, 1, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_k = external_cpu_tensor(k.data(), {ROWS, 1, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_v = external_cpu_tensor(v.data(), {ROWS, 1, HD}, DLDataType{kDLBfloat, 16, 1});
    auto dev_qkv = Tensor::Empty({ROWS, Q_DIM + K_DIM + V_DIM}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_rope = Tensor::Empty({ROWS, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_q = Tensor::Empty({ROWS, 1, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_k = Tensor::Empty({ROWS, 1, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_v = Tensor::Empty({ROWS, 1, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_qkv->dl(), dev_qkv->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_rope->dl(), dev_rope->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_qkv),
        VMValue::ObjRef(dev_rope),
        VMValue::Int(ROWS),
        VMValue::Int(Q_DIM),
        VMValue::Int(K_DIM),
        VMValue::Int(V_DIM),
        VMValue::Int(HD),
        VMValue::ObjRef(dev_q),
        VMValue::ObjRef(dev_k),
        VMValue::ObjRef(dev_v),
    };
    CUDAKernelLauncher_Launch(kernel, args, {1, 1, 1, 256, 1, 1, 0}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_q->dl(), host_q->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_k->dl(), host_k->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_v->dl(), host_v->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    const std::vector<float> want_q = {-0.25f, 1.75f, 4.0f, 3.0f};
    const std::vector<float> want_k = {-0.125f, 0.875f, -1.25f, -1.25f};
    const std::vector<float> want_v = {10.0f, 20.0f, 30.0f, 40.0f};
    for (int i = 0; i < HD; ++i) {
        CHECK(std::fabs(bf16_bits_to_f32(q[static_cast<size_t>(i)]) - want_q[static_cast<size_t>(i)]) < 0.02f);
        CHECK(std::fabs(bf16_bits_to_f32(k[static_cast<size_t>(i)]) - want_k[static_cast<size_t>(i)]) < 0.02f);
        CHECK(std::fabs(bf16_bits_to_f32(v[static_cast<size_t>(i)]) - want_v[static_cast<size_t>(i)]) < 0.02f);
    }
}

void test_qkv_split_rope_cache_bf16_writes_cache_layer() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_qkv_split_rope_cache_bf16",
        "pi05_qkv_split_rope_cache_bf16");
    if (kernel == nullptr) return;

    constexpr int L = 3;
    constexpr int ROWS = 1;
    constexpr int CACHE_ROWS = 2;
    constexpr int Q_DIM = 4;
    constexpr int K_DIM = 4;
    constexpr int V_DIM = 4;
    constexpr int HD = 4;
    constexpr int layer = 1;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<uint16_t> qkv = {
        f32_to_bf16_bits(1.0f), f32_to_bf16_bits(3.0f),
        f32_to_bf16_bits(2.0f), f32_to_bf16_bits(4.0f),
        f32_to_bf16_bits(0.5f), f32_to_bf16_bits(1.5f),
        f32_to_bf16_bits(-0.5f), f32_to_bf16_bits(-1.5f),
        f32_to_bf16_bits(10.0f), f32_to_bf16_bits(20.0f),
        f32_to_bf16_bits(30.0f), f32_to_bf16_bits(40.0f),
    };
    std::vector<uint16_t> rope = {
        f32_to_bf16_bits(0.5f), f32_to_bf16_bits(0.25f),
        f32_to_bf16_bits(1.0f), f32_to_bf16_bits(-0.5f),
    };
    std::vector<uint16_t> q(Q_DIM, f32_to_bf16_bits(0.0f));
    std::vector<uint16_t> k_cache(L * CACHE_ROWS * K_DIM, f32_to_bf16_bits(0.0f));
    std::vector<uint16_t> v_cache(L * CACHE_ROWS * V_DIM, f32_to_bf16_bits(0.0f));

    auto host_qkv = external_cpu_tensor(qkv.data(), {ROWS, Q_DIM + K_DIM + V_DIM}, DLDataType{kDLBfloat, 16, 1});
    auto host_rope = external_cpu_tensor(rope.data(), {ROWS, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_kc = external_cpu_tensor(k_cache.data(), {L, CACHE_ROWS, 1, K_DIM}, DLDataType{kDLBfloat, 16, 1});
    auto host_vc = external_cpu_tensor(v_cache.data(), {L, CACHE_ROWS, 1, V_DIM}, DLDataType{kDLBfloat, 16, 1});
    auto host_q = external_cpu_tensor(q.data(), {ROWS, 1, HD}, DLDataType{kDLBfloat, 16, 1});
    auto dev_qkv = Tensor::Empty({ROWS, Q_DIM + K_DIM + V_DIM}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_rope = Tensor::Empty({ROWS, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_kc = Tensor::Empty({L, CACHE_ROWS, 1, K_DIM}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_vc = Tensor::Empty({L, CACHE_ROWS, 1, V_DIM}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_q = Tensor::Empty({ROWS, 1, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_qkv->dl(), dev_qkv->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_rope->dl(), dev_rope->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_kc->dl(), dev_kc->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_vc->dl(), dev_vc->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_qkv),
        VMValue::ObjRef(dev_rope),
        VMValue::ObjRef(dev_kc),
        VMValue::ObjRef(dev_vc),
        VMValue::Int(layer),
        VMValue::Int(ROWS),
        VMValue::Int(CACHE_ROWS),
        VMValue::Int(Q_DIM),
        VMValue::Int(K_DIM),
        VMValue::Int(V_DIM),
        VMValue::Int(HD),
        VMValue::ObjRef(dev_q),
    };
    CUDAKernelLauncher_Launch(kernel, args, {1, 1, 1, 256, 1, 1, 0}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_q->dl(), host_q->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_kc->dl(), host_kc->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_vc->dl(), host_vc->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    const std::vector<float> want_q = {-0.25f, 1.75f, 4.0f, 3.0f};
    const std::vector<float> want_k = {-0.125f, 0.875f, -1.25f, -1.25f};
    const std::vector<float> want_v = {10.0f, 20.0f, 30.0f, 40.0f};
    for (int i = 0; i < HD; ++i) {
        CHECK(std::fabs(bf16_bits_to_f32(q[static_cast<size_t>(i)]) - want_q[static_cast<size_t>(i)]) < 0.02f);
    }
    const int base = layer * CACHE_ROWS * K_DIM;
    for (int i = 0; i < static_cast<int>(k_cache.size()); ++i) {
        const bool in_written_row = i >= base && i < base + K_DIM;
        const int local = i - base;
        const float want_k_i = in_written_row ? want_k[static_cast<size_t>(local)] : 0.0f;
        const float want_v_i = in_written_row ? want_v[static_cast<size_t>(local)] : 0.0f;
        CHECK(std::fabs(bf16_bits_to_f32(k_cache[static_cast<size_t>(i)]) - want_k_i) < 0.02f);
        CHECK(std::fabs(bf16_bits_to_f32(v_cache[static_cast<size_t>(i)]) - want_v_i) < 0.02f);
    }
}

void test_qkv_split_rope_concat_bf16_writes_prefix_then_suffix() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_qkv_split_rope_concat_bf16",
        "pi05_qkv_split_rope_concat_bf16");
    if (kernel == nullptr) return;

    constexpr int PREFIX_ROWS = 2;
    constexpr int SUFFIX_ROWS = 1;
    constexpr int Q_DIM = 4;
    constexpr int K_DIM = 4;
    constexpr int V_DIM = 4;
    constexpr int HD = 4;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<uint16_t> qkv = {
        f32_to_bf16_bits(1.0f), f32_to_bf16_bits(3.0f),
        f32_to_bf16_bits(2.0f), f32_to_bf16_bits(4.0f),
        f32_to_bf16_bits(0.5f), f32_to_bf16_bits(1.5f),
        f32_to_bf16_bits(-0.5f), f32_to_bf16_bits(-1.5f),
        f32_to_bf16_bits(10.0f), f32_to_bf16_bits(20.0f),
        f32_to_bf16_bits(30.0f), f32_to_bf16_bits(40.0f),
    };
    std::vector<uint16_t> rope = {
        f32_to_bf16_bits(0.5f), f32_to_bf16_bits(0.25f),
        f32_to_bf16_bits(1.0f), f32_to_bf16_bits(-0.5f),
    };
    std::vector<uint16_t> prefix_k(PREFIX_ROWS * K_DIM);
    std::vector<uint16_t> prefix_v(PREFIX_ROWS * V_DIM);
    for (int i = 0; i < PREFIX_ROWS * K_DIM; ++i) {
        prefix_k[static_cast<size_t>(i)] = f32_to_bf16_bits(100.0f + i);
        prefix_v[static_cast<size_t>(i)] = f32_to_bf16_bits(200.0f + i);
    }
    std::vector<uint16_t> q(Q_DIM, f32_to_bf16_bits(0.0f));
    std::vector<uint16_t> out_k((PREFIX_ROWS + SUFFIX_ROWS) * K_DIM, f32_to_bf16_bits(0.0f));
    std::vector<uint16_t> out_v((PREFIX_ROWS + SUFFIX_ROWS) * V_DIM, f32_to_bf16_bits(0.0f));

    auto host_qkv = external_cpu_tensor(qkv.data(), {SUFFIX_ROWS, Q_DIM + K_DIM + V_DIM}, DLDataType{kDLBfloat, 16, 1});
    auto host_rope = external_cpu_tensor(rope.data(), {SUFFIX_ROWS, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_pk = external_cpu_tensor(prefix_k.data(), {PREFIX_ROWS, 1, K_DIM}, DLDataType{kDLBfloat, 16, 1});
    auto host_pv = external_cpu_tensor(prefix_v.data(), {PREFIX_ROWS, 1, V_DIM}, DLDataType{kDLBfloat, 16, 1});
    auto host_q = external_cpu_tensor(q.data(), {SUFFIX_ROWS, 1, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_ok = external_cpu_tensor(out_k.data(), {PREFIX_ROWS + SUFFIX_ROWS, 1, K_DIM}, DLDataType{kDLBfloat, 16, 1});
    auto host_ov = external_cpu_tensor(out_v.data(), {PREFIX_ROWS + SUFFIX_ROWS, 1, V_DIM}, DLDataType{kDLBfloat, 16, 1});
    auto dev_qkv = Tensor::Empty({SUFFIX_ROWS, Q_DIM + K_DIM + V_DIM}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_rope = Tensor::Empty({SUFFIX_ROWS, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_pk = Tensor::Empty({PREFIX_ROWS, 1, K_DIM}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_pv = Tensor::Empty({PREFIX_ROWS, 1, V_DIM}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_q = Tensor::Empty({SUFFIX_ROWS, 1, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_ok = Tensor::Empty({PREFIX_ROWS + SUFFIX_ROWS, 1, K_DIM}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_ov = Tensor::Empty({PREFIX_ROWS + SUFFIX_ROWS, 1, V_DIM}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_qkv->dl(), dev_qkv->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_rope->dl(), dev_rope->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_pk->dl(), dev_pk->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_pv->dl(), dev_pv->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_qkv),
        VMValue::ObjRef(dev_rope),
        VMValue::ObjRef(dev_pk),
        VMValue::ObjRef(dev_pv),
        VMValue::Int(PREFIX_ROWS),
        VMValue::Int(SUFFIX_ROWS),
        VMValue::Int(Q_DIM),
        VMValue::Int(K_DIM),
        VMValue::Int(V_DIM),
        VMValue::Int(HD),
        VMValue::ObjRef(dev_q),
        VMValue::ObjRef(dev_ok),
        VMValue::ObjRef(dev_ov),
    };
    CUDAKernelLauncher_Launch(kernel, args, {1, 1, 1, 256, 1, 1, 0}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_q->dl(), host_q->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_ok->dl(), host_ok->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_ov->dl(), host_ov->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    const std::vector<float> want_q = {-0.25f, 1.75f, 4.0f, 3.0f};
    const std::vector<float> want_k = {-0.125f, 0.875f, -1.25f, -1.25f};
    const std::vector<float> want_v = {10.0f, 20.0f, 30.0f, 40.0f};
    for (int i = 0; i < HD; ++i) {
        CHECK(std::fabs(bf16_bits_to_f32(q[static_cast<size_t>(i)]) - want_q[static_cast<size_t>(i)]) < 0.02f);
    }
    const int suffix_base = PREFIX_ROWS * K_DIM;
    for (int i = 0; i < static_cast<int>(out_k.size()); ++i) {
        const bool is_suffix = i >= suffix_base;
        const int local = i - suffix_base;
        const float want_k_i = is_suffix ? want_k[static_cast<size_t>(local)] : 100.0f + i;
        const float want_v_i = is_suffix ? want_v[static_cast<size_t>(local)] : 200.0f + i;
        CHECK(std::fabs(bf16_bits_to_f32(out_k[static_cast<size_t>(i)]) - want_k_i) < 0.02f);
        CHECK(std::fabs(bf16_bits_to_f32(out_v[static_cast<size_t>(i)]) - want_v_i) < 0.02f);
    }
}

void test_layer_norm_bf16_launch_matches_reference() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_layer_norm_bf16",
        "pi05_layer_norm_bf16");
    if (kernel == nullptr) return;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<float> x_f32 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint16_t> x(4), weight(4), bias(4), out(4, 0);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = f32_to_bf16_bits(x_f32[i]);
        weight[i] = f32_to_bf16_bits(1.0f);
        bias[i] = f32_to_bf16_bits(0.0f);
    }

    auto host_x = external_cpu_tensor(x.data(), {1, 4}, DLDataType{kDLBfloat, 16, 1});
    auto host_w = external_cpu_tensor(weight.data(), {4}, DLDataType{kDLBfloat, 16, 1});
    auto host_b = external_cpu_tensor(bias.data(), {4}, DLDataType{kDLBfloat, 16, 1});
    auto host_out = external_cpu_tensor(out.data(), {1, 4}, DLDataType{kDLBfloat, 16, 1});
    auto dev_x = Tensor::Empty({1, 4}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_w = Tensor::Empty({4}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_b = Tensor::Empty({4}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_out = Tensor::Empty({1, 4}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_x->dl(), dev_x->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_w->dl(), dev_w->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_b->dl(), dev_b->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_x),
        VMValue::ObjRef(dev_w),
        VMValue::ObjRef(dev_b),
        VMValue::Int(1),
        VMValue::Int(4),
        VMValue::Int(f32_to_i64_bits(1.0e-5f)),
        VMValue::ObjRef(dev_out),
    };
    CUDAKernelLauncher_Launch(kernel, args, {}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_out->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    float mean = 2.5f;
    float var = 1.25f;
    float inv_std = 1.0f / std::sqrt(var + 1.0e-5f);
    for (int i = 0; i < 4; ++i) {
        float want = (x_f32[static_cast<size_t>(i)] - mean) * inv_std;
        float got = bf16_bits_to_f32(out[static_cast<size_t>(i)]);
        CHECK(std::fabs(got - want) < 0.02f);
    }
}

void test_rms_norm_unit_bf16_launch_matches_reference() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_rms_norm_unit_bf16",
        "pi05_rms_norm_unit_bf16");
    if (kernel == nullptr) return;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<float> x_f32 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint16_t> x(4), out(4, 0);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = f32_to_bf16_bits(x_f32[i]);
    }

    auto host_x = external_cpu_tensor(x.data(), {1, 4}, DLDataType{kDLBfloat, 16, 1});
    auto host_out = external_cpu_tensor(out.data(), {1, 4}, DLDataType{kDLBfloat, 16, 1});
    auto dev_x = Tensor::Empty({1, 4}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_out = Tensor::Empty({1, 4}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_x->dl(), dev_x->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_x),
        VMValue::Int(1),
        VMValue::Int(4),
        VMValue::Int(f32_to_i64_bits(1.0e-6f)),
        VMValue::ObjRef(dev_out),
    };
    CUDAKernelLauncher_Launch(kernel, args, {}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_out->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    float ssq = 0.0f;
    for (float v : x_f32) ssq += v * v;
    float rstd = 1.0f / std::sqrt(ssq / 4.0f + 1.0e-6f);
    for (int i = 0; i < 4; ++i) {
        float want = x_f32[static_cast<size_t>(i)] * rstd;
        float got = bf16_bits_to_f32(out[static_cast<size_t>(i)]);
        CHECK(std::fabs(got - want) < 0.02f);
    }
}

void test_layer_norm_to_fp8_bf16_launch_matches_two_step_reference() {
    auto* norm_kernel = register_kernel_from_artifact(
        "kernel.pi05_layer_norm_bf16",
        "pi05_layer_norm_bf16");
    auto* quant_kernel = register_kernel_from_artifact(
        "kernel.pi05_quantize_fp8_static_bf16",
        "pi05_quantize_fp8_static_bf16");
    auto* fused_kernel = register_kernel_from_artifact(
        "kernel.pi05_layer_norm_to_fp8_bf16",
        "pi05_layer_norm_to_fp8_bf16");
    if (norm_kernel == nullptr || quant_kernel == nullptr || fused_kernel == nullptr) return;

    constexpr int ROWS = 2;
    constexpr int COLS = 4;
    constexpr int N = ROWS * COLS;
    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<float> x_f32 = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 2.5f, 5.0f};
    std::vector<float> weight_f32 = {1.0f, 0.75f, 1.25f, 1.5f};
    std::vector<float> bias_f32 = {0.0f, 0.125f, -0.25f, 0.5f};
    std::vector<uint16_t> x(N), weight(COLS), bias(COLS), norm_out(N, 0);
    std::vector<float> scale = {0.25f};
    std::vector<uint8_t> ref_fp8(N, 0), fused_fp8(N, 0);
    for (int i = 0; i < N; ++i) x[static_cast<size_t>(i)] = f32_to_bf16_bits(x_f32[static_cast<size_t>(i)]);
    for (int i = 0; i < COLS; ++i) {
        weight[static_cast<size_t>(i)] = f32_to_bf16_bits(weight_f32[static_cast<size_t>(i)]);
        bias[static_cast<size_t>(i)] = f32_to_bf16_bits(bias_f32[static_cast<size_t>(i)]);
    }

    auto host_x = external_cpu_tensor(x.data(), {ROWS, COLS}, DLDataType{kDLBfloat, 16, 1});
    auto host_w = external_cpu_tensor(weight.data(), {COLS}, DLDataType{kDLBfloat, 16, 1});
    auto host_b = external_cpu_tensor(bias.data(), {COLS}, DLDataType{kDLBfloat, 16, 1});
    auto host_scale = external_cpu_tensor(scale.data(), {1}, DLDataType{kDLFloat, 32, 1});
    auto host_ref = external_cpu_tensor(ref_fp8.data(), {ROWS, COLS}, DLDataType{kDLFloat8_e4m3, 8, 1});
    auto host_fused = external_cpu_tensor(fused_fp8.data(), {ROWS, COLS}, DLDataType{kDLFloat8_e4m3, 8, 1});
    auto dev_x = Tensor::Empty({ROWS, COLS}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_w = Tensor::Empty({COLS}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_b = Tensor::Empty({COLS}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_scale = Tensor::Empty({1}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_norm = Tensor::Empty({ROWS, COLS}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_ref = Tensor::Empty({ROWS, COLS}, DLDataType{kDLFloat8_e4m3, 8, 1}, cuda_dev);
    auto dev_fused = Tensor::Empty({ROWS, COLS}, DLDataType{kDLFloat8_e4m3, 8, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_x->dl(), dev_x->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_w->dl(), dev_w->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_b->dl(), dev_b->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_scale->dl(), dev_scale->dl(), nullptr);

    std::vector<VMValue> norm_args = {
        VMValue::ObjRef(dev_x),
        VMValue::ObjRef(dev_w),
        VMValue::ObjRef(dev_b),
        VMValue::Int(ROWS),
        VMValue::Int(COLS),
        VMValue::Int(f32_to_i64_bits(1.0e-5f)),
        VMValue::ObjRef(dev_norm),
    };
    CUDAKernelLauncher_Launch(norm_kernel, norm_args, {ROWS, 1, 1, 256, 1, 1, 0}, nullptr);
    std::vector<VMValue> quant_args = {
        VMValue::ObjRef(dev_norm),
        VMValue::ObjRef(dev_scale),
        VMValue::Int(N),
        VMValue::ObjRef(dev_ref),
    };
    CUDAKernelLauncher_Launch(quant_kernel, quant_args, {}, nullptr);
    std::vector<VMValue> fused_args = {
        VMValue::ObjRef(dev_x),
        VMValue::ObjRef(dev_w),
        VMValue::ObjRef(dev_b),
        VMValue::ObjRef(dev_scale),
        VMValue::Int(ROWS),
        VMValue::Int(COLS),
        VMValue::Int(f32_to_i64_bits(1.0e-5f)),
        VMValue::ObjRef(dev_fused),
    };
    CUDAKernelLauncher_Launch(fused_kernel, fused_args, {ROWS, 1, 1, 256, 1, 1, 0}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_ref->dl(), host_ref->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_fused->dl(), host_fused->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    for (int i = 0; i < N; ++i) {
        CHECK(ref_fp8[static_cast<size_t>(i)] == fused_fp8[static_cast<size_t>(i)]);
    }
}

void test_rms_norm_unit_to_fp8_bf16_launch_matches_two_step_reference() {
    auto* norm_kernel = register_kernel_from_artifact(
        "kernel.pi05_rms_norm_unit_bf16",
        "pi05_rms_norm_unit_bf16");
    auto* quant_kernel = register_kernel_from_artifact(
        "kernel.pi05_quantize_fp8_static_bf16",
        "pi05_quantize_fp8_static_bf16");
    auto* fused_kernel = register_kernel_from_artifact(
        "kernel.pi05_rms_norm_unit_to_fp8_bf16",
        "pi05_rms_norm_unit_to_fp8_bf16");
    if (norm_kernel == nullptr || quant_kernel == nullptr || fused_kernel == nullptr) return;

    constexpr int ROWS = 2;
    constexpr int COLS = 4;
    constexpr int N = ROWS * COLS;
    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<float> x_f32 = {1.0f, 2.0f, 3.0f, 4.0f, -2.0f, -0.5f, 1.5f, 3.0f};
    std::vector<uint16_t> x(N), norm_out(N, 0);
    std::vector<float> scale = {0.25f};
    std::vector<uint8_t> ref_fp8(N, 0), fused_fp8(N, 0);
    for (int i = 0; i < N; ++i) x[static_cast<size_t>(i)] = f32_to_bf16_bits(x_f32[static_cast<size_t>(i)]);

    auto host_x = external_cpu_tensor(x.data(), {ROWS, COLS}, DLDataType{kDLBfloat, 16, 1});
    auto host_scale = external_cpu_tensor(scale.data(), {1}, DLDataType{kDLFloat, 32, 1});
    auto host_ref = external_cpu_tensor(ref_fp8.data(), {ROWS, COLS}, DLDataType{kDLFloat8_e4m3, 8, 1});
    auto host_fused = external_cpu_tensor(fused_fp8.data(), {ROWS, COLS}, DLDataType{kDLFloat8_e4m3, 8, 1});
    auto dev_x = Tensor::Empty({ROWS, COLS}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_scale = Tensor::Empty({1}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_norm = Tensor::Empty({ROWS, COLS}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_ref = Tensor::Empty({ROWS, COLS}, DLDataType{kDLFloat8_e4m3, 8, 1}, cuda_dev);
    auto dev_fused = Tensor::Empty({ROWS, COLS}, DLDataType{kDLFloat8_e4m3, 8, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_x->dl(), dev_x->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_scale->dl(), dev_scale->dl(), nullptr);

    std::vector<VMValue> norm_args = {
        VMValue::ObjRef(dev_x),
        VMValue::Int(ROWS),
        VMValue::Int(COLS),
        VMValue::Int(f32_to_i64_bits(1.0e-6f)),
        VMValue::ObjRef(dev_norm),
    };
    CUDAKernelLauncher_Launch(norm_kernel, norm_args, {ROWS, 1, 1, 256, 1, 1, 0}, nullptr);
    std::vector<VMValue> quant_args = {
        VMValue::ObjRef(dev_norm),
        VMValue::ObjRef(dev_scale),
        VMValue::Int(N),
        VMValue::ObjRef(dev_ref),
    };
    CUDAKernelLauncher_Launch(quant_kernel, quant_args, {}, nullptr);
    std::vector<VMValue> fused_args = {
        VMValue::ObjRef(dev_x),
        VMValue::ObjRef(dev_scale),
        VMValue::Int(ROWS),
        VMValue::Int(COLS),
        VMValue::Int(f32_to_i64_bits(1.0e-6f)),
        VMValue::ObjRef(dev_fused),
    };
    CUDAKernelLauncher_Launch(fused_kernel, fused_args, {ROWS, 1, 1, 256, 1, 1, 0}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_ref->dl(), host_ref->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_fused->dl(), host_fused->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    for (int i = 0; i < N; ++i) {
        CHECK(ref_fp8[static_cast<size_t>(i)] == fused_fp8[static_cast<size_t>(i)]);
    }
}

void test_qkv_split_bf16_launch_writes_three_outputs() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_qkv_split_bf16",
        "pi05_qkv_split_bf16");
    if (kernel == nullptr) return;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<uint16_t> qkv(6), q(2, 0), k(2, 0), v(2, 0);
    for (int i = 0; i < 6; ++i) qkv[static_cast<size_t>(i)] = f32_to_bf16_bits(static_cast<float>(i + 1));

    auto host_qkv = external_cpu_tensor(qkv.data(), {1, 6}, DLDataType{kDLBfloat, 16, 1});
    auto host_q = external_cpu_tensor(q.data(), {1, 2}, DLDataType{kDLBfloat, 16, 1});
    auto host_k = external_cpu_tensor(k.data(), {1, 2}, DLDataType{kDLBfloat, 16, 1});
    auto host_v = external_cpu_tensor(v.data(), {1, 2}, DLDataType{kDLBfloat, 16, 1});
    auto dev_qkv = Tensor::Empty({1, 6}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_q = Tensor::Empty({1, 2}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_k = Tensor::Empty({1, 2}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_v = Tensor::Empty({1, 2}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_qkv->dl(), dev_qkv->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_qkv),
        VMValue::Int(1),
        VMValue::Int(2),
        VMValue::Int(2),
        VMValue::Int(2),
        VMValue::ObjRef(dev_q),
        VMValue::ObjRef(dev_k),
        VMValue::ObjRef(dev_v),
    };
    CUDAKernelLauncher_Launch(kernel, args, {}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_q->dl(), host_q->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_k->dl(), host_k->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_v->dl(), host_v->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    CHECK(bf16_bits_to_f32(q[0]) == 1.0f);
    CHECK(bf16_bits_to_f32(q[1]) == 2.0f);
    CHECK(bf16_bits_to_f32(k[0]) == 3.0f);
    CHECK(bf16_bits_to_f32(k[1]) == 4.0f);
    CHECK(bf16_bits_to_f32(v[0]) == 5.0f);
    CHECK(bf16_bits_to_f32(v[1]) == 6.0f);
}

void test_qkv_bias_split_bf16_launch_adds_bias_then_writes_three_outputs() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_qkv_bias_split_bf16",
        "pi05_qkv_bias_split_bf16");
    if (kernel == nullptr) return;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    constexpr int ROWS = 2;
    constexpr int Q = 2;
    constexpr int K = 2;
    constexpr int V = 2;
    constexpr int STRIDE = Q + K + V;
    std::vector<uint16_t> qkv(ROWS * STRIDE), bias(STRIDE), q(ROWS * Q, 0), k(ROWS * K, 0), v(ROWS * V, 0);
    for (int i = 0; i < ROWS * STRIDE; ++i) {
        qkv[static_cast<size_t>(i)] = f32_to_bf16_bits(static_cast<float>(i + 1));
    }
    for (int i = 0; i < STRIDE; ++i) {
        bias[static_cast<size_t>(i)] = f32_to_bf16_bits(0.5f * static_cast<float>(i + 1));
    }

    auto host_qkv = external_cpu_tensor(qkv.data(), {ROWS, STRIDE}, DLDataType{kDLBfloat, 16, 1});
    auto host_bias = external_cpu_tensor(bias.data(), {STRIDE}, DLDataType{kDLBfloat, 16, 1});
    auto host_q = external_cpu_tensor(q.data(), {ROWS, Q}, DLDataType{kDLBfloat, 16, 1});
    auto host_k = external_cpu_tensor(k.data(), {ROWS, K}, DLDataType{kDLBfloat, 16, 1});
    auto host_v = external_cpu_tensor(v.data(), {ROWS, V}, DLDataType{kDLBfloat, 16, 1});
    auto dev_qkv = Tensor::Empty({ROWS, STRIDE}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_bias = Tensor::Empty({STRIDE}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_q = Tensor::Empty({ROWS, Q}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_k = Tensor::Empty({ROWS, K}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_v = Tensor::Empty({ROWS, V}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_qkv->dl(), dev_qkv->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_bias->dl(), dev_bias->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_qkv),
        VMValue::ObjRef(dev_bias),
        VMValue::Int(ROWS),
        VMValue::Int(Q),
        VMValue::Int(K),
        VMValue::Int(V),
        VMValue::ObjRef(dev_q),
        VMValue::ObjRef(dev_k),
        VMValue::ObjRef(dev_v),
    };
    CUDAKernelLauncher_Launch(kernel, args, {1, 1, 1, 256, 1, 1, 0}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_q->dl(), host_q->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_k->dl(), host_k->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_v->dl(), host_v->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    CHECK(std::fabs(bf16_bits_to_f32(q[0]) - 1.5f) < 0.01f);
    CHECK(std::fabs(bf16_bits_to_f32(q[1]) - 3.0f) < 0.01f);
    CHECK(std::fabs(bf16_bits_to_f32(k[0]) - 4.5f) < 0.01f);
    CHECK(std::fabs(bf16_bits_to_f32(k[1]) - 6.0f) < 0.01f);
    CHECK(std::fabs(bf16_bits_to_f32(v[0]) - 7.5f) < 0.01f);
    CHECK(std::fabs(bf16_bits_to_f32(v[1]) - 9.0f) < 0.01f);
    CHECK(std::fabs(bf16_bits_to_f32(q[2]) - 7.5f) < 0.01f);
    CHECK(std::fabs(bf16_bits_to_f32(v[3]) - 15.0f) < 0.01f);
}

void test_dynamic_fp8_quant_launch_writes_scale() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_quantize_fp8_dynamic_bf16",
        "pi05_quantize_fp8_dynamic_bf16");
    if (kernel == nullptr) return;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<uint16_t> x = {
        f32_to_bf16_bits(-448.0f),
        f32_to_bf16_bits(224.0f),
        f32_to_bf16_bits(0.0f),
        f32_to_bf16_bits(112.0f),
    };
    std::vector<uint8_t> out(4, 0);
    std::vector<float> scale(1, 0.0f);
    auto host_x = external_cpu_tensor(x.data(), {4}, DLDataType{kDLBfloat, 16, 1});
    auto host_scale = external_cpu_tensor(scale.data(), {1}, DLDataType{kDLFloat, 32, 1});
    auto dev_x = Tensor::Empty({4}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_out = Tensor::Empty({4}, DLDataType{kDLFloat8_e4m3, 8, 1}, cuda_dev);
    auto dev_scale = Tensor::Empty({1}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_x->dl(), dev_x->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_x),
        VMValue::Int(4),
        VMValue::ObjRef(dev_out),
        VMValue::ObjRef(dev_scale),
    };
    CUDAKernelLauncher_Launch(kernel, args, {}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_scale->dl(), host_scale->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);
    CHECK(std::fabs(scale[0] - 1.0f) < 0.001f);
}

void test_bias_gelu_to_fp8_bf16_launch_matches_two_step_reference() {
    auto* bias_kernel = register_kernel_from_artifact(
        "kernel.pi05_bias_add_bf16",
        "pi05_bias_add_bf16");
    auto* gelu_kernel = register_kernel_from_artifact(
        "kernel.pi05_gelu_inplace_bf16",
        "pi05_gelu_inplace_bf16");
    auto* quant_kernel = register_kernel_from_artifact(
        "kernel.pi05_quantize_fp8_static_bf16",
        "pi05_quantize_fp8_static_bf16");
    auto* fused_kernel = register_kernel_from_artifact(
        "kernel.pi05_bias_gelu_to_fp8_bf16",
        "pi05_bias_gelu_to_fp8_bf16");
    if (bias_kernel == nullptr || gelu_kernel == nullptr || quant_kernel == nullptr || fused_kernel == nullptr) return;

    constexpr int ROWS = 2;
    constexpr int COLS = 4;
    constexpr int N = ROWS * COLS;
    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<float> x_f32 = {-1.5f, -0.25f, 0.25f, 1.5f, -2.0f, -0.5f, 0.75f, 2.0f};
    std::vector<float> bias_f32 = {0.25f, -0.125f, 0.5f, -0.25f};
    std::vector<uint16_t> x(N), bias(COLS);
    std::vector<float> scale = {0.25f};
    std::vector<uint8_t> ref_fp8(N, 0), fused_fp8(N, 0);
    for (int i = 0; i < N; ++i) x[static_cast<size_t>(i)] = f32_to_bf16_bits(x_f32[static_cast<size_t>(i)]);
    for (int i = 0; i < COLS; ++i) bias[static_cast<size_t>(i)] = f32_to_bf16_bits(bias_f32[static_cast<size_t>(i)]);

    auto host_x = external_cpu_tensor(x.data(), {ROWS, COLS}, DLDataType{kDLBfloat, 16, 1});
    auto host_bias = external_cpu_tensor(bias.data(), {COLS}, DLDataType{kDLBfloat, 16, 1});
    auto host_scale = external_cpu_tensor(scale.data(), {1}, DLDataType{kDLFloat, 32, 1});
    auto host_ref = external_cpu_tensor(ref_fp8.data(), {ROWS, COLS}, DLDataType{kDLFloat8_e4m3, 8, 1});
    auto host_fused = external_cpu_tensor(fused_fp8.data(), {ROWS, COLS}, DLDataType{kDLFloat8_e4m3, 8, 1});
    auto dev_x_ref = Tensor::Empty({ROWS, COLS}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_x_fused = Tensor::Empty({ROWS, COLS}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_bias = Tensor::Empty({COLS}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_scale = Tensor::Empty({1}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_ref = Tensor::Empty({ROWS, COLS}, DLDataType{kDLFloat8_e4m3, 8, 1}, cuda_dev);
    auto dev_fused = Tensor::Empty({ROWS, COLS}, DLDataType{kDLFloat8_e4m3, 8, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_x->dl(), dev_x_ref->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_x->dl(), dev_x_fused->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_bias->dl(), dev_bias->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_scale->dl(), dev_scale->dl(), nullptr);

    std::vector<VMValue> bias_args = {
        VMValue::ObjRef(dev_x_ref),
        VMValue::ObjRef(dev_bias),
        VMValue::Int(ROWS),
        VMValue::Int(COLS),
    };
    CUDAKernelLauncher_Launch(bias_kernel, bias_args, {}, nullptr);
    std::vector<VMValue> gelu_args = {
        VMValue::ObjRef(dev_x_ref),
        VMValue::Int(N),
    };
    CUDAKernelLauncher_Launch(gelu_kernel, gelu_args, {}, nullptr);
    std::vector<VMValue> quant_args = {
        VMValue::ObjRef(dev_x_ref),
        VMValue::ObjRef(dev_scale),
        VMValue::Int(N),
        VMValue::ObjRef(dev_ref),
    };
    CUDAKernelLauncher_Launch(quant_kernel, quant_args, {}, nullptr);
    std::vector<VMValue> fused_args = {
        VMValue::ObjRef(dev_x_fused),
        VMValue::ObjRef(dev_bias),
        VMValue::ObjRef(dev_scale),
        VMValue::Int(ROWS),
        VMValue::Int(COLS),
        VMValue::ObjRef(dev_fused),
    };
    CUDAKernelLauncher_Launch(fused_kernel, fused_args, {}, nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_ref->dl(), host_ref->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_fused->dl(), host_fused->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    for (int i = 0; i < N; ++i) {
        CHECK(ref_fp8[static_cast<size_t>(i)] == fused_fp8[static_cast<size_t>(i)]);
    }
}

void test_attention_bf16_launch_matches_reference() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_attention_bf16",
        "pi05_attention_bf16");
    if (kernel == nullptr) return;

    constexpr int RQ = 2;
    constexpr int RK = 3;
    constexpr int HQ = 2;
    constexpr int HKV = 1;
    constexpr int HD = 4;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<uint16_t> q(RQ * HQ * HD), k(RK * HKV * HD), v(RK * HKV * HD), out(RQ * HQ * HD, 0);
    std::vector<float> qf(q.size()), kf(k.size()), vf(v.size());
    for (size_t i = 0; i < q.size(); ++i) {
        qf[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f;
        q[i] = f32_to_bf16_bits(qf[i]);
    }
    for (size_t i = 0; i < k.size(); ++i) {
        kf[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.2f;
        vf[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.125f;
        k[i] = f32_to_bf16_bits(kf[i]);
        v[i] = f32_to_bf16_bits(vf[i]);
    }

    auto host_q = external_cpu_tensor(q.data(), {RQ, HQ, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_k = external_cpu_tensor(k.data(), {RK, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_v = external_cpu_tensor(v.data(), {RK, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_out = external_cpu_tensor(out.data(), {RQ, HQ, HD}, DLDataType{kDLBfloat, 16, 1});
    auto dev_q = Tensor::Empty({RQ, HQ, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_k = Tensor::Empty({RK, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_v = Tensor::Empty({RK, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_out = Tensor::Empty({RQ, HQ, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_q->dl(), dev_q->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_k->dl(), dev_k->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_v->dl(), dev_v->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_q),
        VMValue::ObjRef(dev_k),
        VMValue::ObjRef(dev_v),
        VMValue::Int(RQ),
        VMValue::Int(RK),
        VMValue::Int(HQ),
        VMValue::Int(HKV),
        VMValue::Int(HD),
        VMValue::Int(f32_to_i64_bits(scale)),
        VMValue::ObjRef(dev_out),
    };
    CUDAKernelLauncher_Launch(
        kernel,
        args,
        {RQ, HQ, 1, 256, 1, 1, RK * static_cast<int64_t>(sizeof(float))},
        nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_out->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    for (int rq = 0; rq < RQ; ++rq) {
        for (int hq = 0; hq < HQ; ++hq) {
            float scores[RK];
            float max_score = -1.0e30f;
            for (int rk = 0; rk < RK; ++rk) {
                float s = 0.0f;
                for (int d = 0; d < HD; ++d) {
                    s += qf[static_cast<size_t>((rq * HQ + hq) * HD + d)]
                       * kf[static_cast<size_t>(rk * HD + d)];
                }
                scores[rk] = s * scale;
                if (scores[rk] > max_score) max_score = scores[rk];
            }
            float denom = 0.0f;
            for (int rk = 0; rk < RK; ++rk) denom += std::exp(scores[rk] - max_score);
            for (int d = 0; d < HD; ++d) {
                float want = 0.0f;
                for (int rk = 0; rk < RK; ++rk) {
                    float w = std::exp(scores[rk] - max_score) / denom;
                    want += w * vf[static_cast<size_t>(rk * HD + d)];
                }
                float got = bf16_bits_to_f32(out[static_cast<size_t>((rq * HQ + hq) * HD + d)]);
                CHECK(std::fabs(got - want) < 0.03f);
            }
        }
    }
}

void test_attention_prefix_bf16_uses_valid_prefix_rows() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_attention_prefix_bf16",
        "pi05_attention_prefix_bf16");
    if (kernel == nullptr) return;

    constexpr int RQ = 1;
    constexpr int PREFIX_VALID = 2;
    constexpr int SUFFIX = 1;
    constexpr int RK = PREFIX_VALID + SUFFIX;
    constexpr int HQ = 2;
    constexpr int HKV = 1;
    constexpr int HD = 4;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<uint16_t> q(RQ * HQ * HD), k(RK * HKV * HD), v(RK * HKV * HD), out(RQ * HQ * HD, 0);
    std::vector<float> qf(q.size()), kf(k.size()), vf(v.size());
    for (size_t i = 0; i < q.size(); ++i) {
        qf[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.25f;
        q[i] = f32_to_bf16_bits(qf[i]);
    }
    for (size_t i = 0; i < k.size(); ++i) {
        kf[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
        vf[i] = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.2f;
        k[i] = f32_to_bf16_bits(kf[i]);
        v[i] = f32_to_bf16_bits(vf[i]);
    }

    auto host_q = external_cpu_tensor(q.data(), {RQ, HQ, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_k = external_cpu_tensor(k.data(), {RK, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_v = external_cpu_tensor(v.data(), {RK, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_out = external_cpu_tensor(out.data(), {RQ, HQ, HD}, DLDataType{kDLBfloat, 16, 1});
    auto dev_q = Tensor::Empty({RQ, HQ, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_k = Tensor::Empty({RK, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_v = Tensor::Empty({RK, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_out = Tensor::Empty({RQ, HQ, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_q->dl(), dev_q->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_k->dl(), dev_k->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_v->dl(), dev_v->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_q),
        VMValue::ObjRef(dev_k),
        VMValue::ObjRef(dev_v),
        VMValue::Int(RQ),
        VMValue::Int(PREFIX_VALID),
        VMValue::Int(SUFFIX),
        VMValue::Int(HQ),
        VMValue::Int(HKV),
        VMValue::Int(HD),
        VMValue::Int(f32_to_i64_bits(scale)),
        VMValue::ObjRef(dev_out),
    };
    CUDAKernelLauncher_Launch(
        kernel,
        args,
        {RQ, HQ, 1, 256, 1, 1, RK * static_cast<int64_t>(sizeof(float))},
        nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_out->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    for (int hq = 0; hq < HQ; ++hq) {
        float scores[RK];
        float max_score = -1.0e30f;
        for (int rk = 0; rk < RK; ++rk) {
            float s = 0.0f;
            for (int d = 0; d < HD; ++d) {
                s += qf[static_cast<size_t>(hq * HD + d)]
                   * kf[static_cast<size_t>(rk * HD + d)];
            }
            scores[rk] = s * scale;
            if (scores[rk] > max_score) max_score = scores[rk];
        }
        float denom = 0.0f;
        for (int rk = 0; rk < RK; ++rk) denom += std::exp(scores[rk] - max_score);
        for (int d = 0; d < HD; ++d) {
            float want = 0.0f;
            for (int rk = 0; rk < RK; ++rk) {
                float w = std::exp(scores[rk] - max_score) / denom;
                want += w * vf[static_cast<size_t>(rk * HD + d)];
            }
            float got = bf16_bits_to_f32(out[static_cast<size_t>(hq * HD + d)]);
            CHECK(std::fabs(got - want) < 0.03f);
        }
    }
}

void test_kv_concat_bf16_launch_writes_prefix_then_suffix() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_kv_concat_bf16",
        "pi05_kv_concat_bf16");
    if (kernel == nullptr) return;

    constexpr int PK = 2;
    constexpr int SK = 3;
    constexpr int HKV = 1;
    constexpr int HD = 2;
    constexpr int prefix_elems = PK * HKV * HD;
    constexpr int suffix_elems = SK * HKV * HD;
    constexpr int total_elems = prefix_elems + suffix_elems;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<uint16_t> prefix_k(prefix_elems), prefix_v(prefix_elems);
    std::vector<uint16_t> suffix_k(suffix_elems), suffix_v(suffix_elems);
    std::vector<uint16_t> out_k(total_elems, 0), out_v(total_elems, 0);
    for (int i = 0; i < prefix_elems; ++i) {
        prefix_k[static_cast<size_t>(i)] = f32_to_bf16_bits(10.0f + i);
        prefix_v[static_cast<size_t>(i)] = f32_to_bf16_bits(20.0f + i);
    }
    for (int i = 0; i < suffix_elems; ++i) {
        suffix_k[static_cast<size_t>(i)] = f32_to_bf16_bits(30.0f + i);
        suffix_v[static_cast<size_t>(i)] = f32_to_bf16_bits(40.0f + i);
    }

    auto host_pk = external_cpu_tensor(prefix_k.data(), {PK, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_pv = external_cpu_tensor(prefix_v.data(), {PK, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_sk = external_cpu_tensor(suffix_k.data(), {SK, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_sv = external_cpu_tensor(suffix_v.data(), {SK, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_ok = external_cpu_tensor(out_k.data(), {PK + SK, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_ov = external_cpu_tensor(out_v.data(), {PK + SK, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto dev_pk = Tensor::Empty({PK, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_pv = Tensor::Empty({PK, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_sk = Tensor::Empty({SK, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_sv = Tensor::Empty({SK, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_ok = Tensor::Empty({PK + SK, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_ov = Tensor::Empty({PK + SK, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_pk->dl(), dev_pk->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_pv->dl(), dev_pv->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_sk->dl(), dev_sk->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_sv->dl(), dev_sv->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_pk),
        VMValue::ObjRef(dev_pv),
        VMValue::ObjRef(dev_sk),
        VMValue::ObjRef(dev_sv),
        VMValue::Int(PK),
        VMValue::Int(SK),
        VMValue::Int(HKV),
        VMValue::Int(HD),
        VMValue::ObjRef(dev_ok),
        VMValue::ObjRef(dev_ov),
    };
    CUDAKernelLauncher_Launch(
        kernel,
        args,
        {1, 1, 1, 256, 1, 1, 0},
        nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_ok->dl(), host_ok->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_ov->dl(), host_ov->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    for (int i = 0; i < total_elems; ++i) {
        const float want_k = i < prefix_elems
            ? 10.0f + i
            : 30.0f + (i - prefix_elems);
        const float want_v = i < prefix_elems
            ? 20.0f + i
            : 40.0f + (i - prefix_elems);
        CHECK(bf16_bits_to_f32(out_k[static_cast<size_t>(i)]) == want_k);
        CHECK(bf16_bits_to_f32(out_v[static_cast<size_t>(i)]) == want_v);
    }
}

void test_copy_kv_cache_layer_bf16_writes_selected_layer() {
    auto* kernel = register_kernel_from_artifact(
        "kernel.pi05_copy_kv_cache_layer_bf16",
        "pi05_copy_kv_cache_layer_bf16");
    if (kernel == nullptr) return;

    constexpr int L = 3;
    constexpr int R = 2;
    constexpr int KV = 4;
    constexpr int layer = 1;
    constexpr int layer_elems = R * KV;
    constexpr int total_elems = L * layer_elems;

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<uint16_t> k(layer_elems), v(layer_elems);
    std::vector<uint16_t> k_cache(total_elems, f32_to_bf16_bits(0.0f));
    std::vector<uint16_t> v_cache(total_elems, f32_to_bf16_bits(0.0f));
    for (int i = 0; i < layer_elems; ++i) {
        k[static_cast<size_t>(i)] = f32_to_bf16_bits(100.0f + i);
        v[static_cast<size_t>(i)] = f32_to_bf16_bits(200.0f + i);
    }

    auto host_k = external_cpu_tensor(k.data(), {R, 1, KV}, DLDataType{kDLBfloat, 16, 1});
    auto host_v = external_cpu_tensor(v.data(), {R, 1, KV}, DLDataType{kDLBfloat, 16, 1});
    auto host_kc = external_cpu_tensor(k_cache.data(), {L, R, 1, KV}, DLDataType{kDLBfloat, 16, 1});
    auto host_vc = external_cpu_tensor(v_cache.data(), {L, R, 1, KV}, DLDataType{kDLBfloat, 16, 1});
    auto dev_k = Tensor::Empty({R, 1, KV}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_v = Tensor::Empty({R, 1, KV}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_kc = Tensor::Empty({L, R, 1, KV}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_vc = Tensor::Empty({L, R, 1, KV}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_k->dl(), dev_k->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_v->dl(), dev_v->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_kc->dl(), dev_kc->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_vc->dl(), dev_vc->dl(), nullptr);

    std::vector<VMValue> args = {
        VMValue::ObjRef(dev_k),
        VMValue::ObjRef(dev_v),
        VMValue::ObjRef(dev_kc),
        VMValue::ObjRef(dev_vc),
        VMValue::Int(layer),
        VMValue::Int(R),
        VMValue::Int(KV),
    };
    CUDAKernelLauncher_Launch(
        kernel,
        args,
        {1, 1, 1, 256, 1, 1, 0},
        nullptr);
    cuda_api->DeviceSync(cuda_dev);
    cuda_api->CopyDataFromTo(dev_kc->dl(), host_kc->dl(), nullptr);
    cuda_api->CopyDataFromTo(dev_vc->dl(), host_vc->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    for (int i = 0; i < total_elems; ++i) {
        const bool in_layer = i >= layer * layer_elems && i < (layer + 1) * layer_elems;
        const int local = i - layer * layer_elems;
        const float want_k = in_layer ? 100.0f + local : 0.0f;
        const float want_v = in_layer ? 200.0f + local : 0.0f;
        CHECK(bf16_bits_to_f32(k_cache[static_cast<size_t>(i)]) == want_k);
        CHECK(bf16_bits_to_f32(v_cache[static_cast<size_t>(i)]) == want_v);
    }
}

}  // namespace

int main() {
    RUN(test_image_u8_to_bf16_norm_launches_from_real_pi05_cubin);
    RUN(test_cast_f32_to_bf16_launch_matches_reference);
    RUN(test_bias_add_bf16_launch_updates_in_place);
    RUN(test_position_add_bf16_launch_updates_in_place);
    RUN(test_euler_update_bf16_launch_updates_float_actions);
    RUN(test_qkv_split_rope_bf16_uses_interleaved_qk_pairs);
    RUN(test_qkv_split_rope_cache_bf16_writes_cache_layer);
    RUN(test_qkv_split_rope_concat_bf16_writes_prefix_then_suffix);
    RUN(test_layer_norm_bf16_launch_matches_reference);
    RUN(test_rms_norm_unit_bf16_launch_matches_reference);
    RUN(test_layer_norm_to_fp8_bf16_launch_matches_two_step_reference);
    RUN(test_rms_norm_unit_to_fp8_bf16_launch_matches_two_step_reference);
    RUN(test_qkv_split_bf16_launch_writes_three_outputs);
    RUN(test_qkv_bias_split_bf16_launch_adds_bias_then_writes_three_outputs);
    RUN(test_dynamic_fp8_quant_launch_writes_scale);
    RUN(test_bias_gelu_to_fp8_bf16_launch_matches_two_step_reference);
    RUN(test_attention_bf16_launch_matches_reference);
    RUN(test_attention_prefix_bf16_uses_valid_prefix_rows);
    RUN(test_kv_concat_bf16_launch_writes_prefix_then_suffix);
    RUN(test_copy_kv_cache_layer_bf16_writes_selected_layer);

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
