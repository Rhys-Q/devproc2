// Pi0.5 denoise fast-path latency benchmark.

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

#include "devproc2/runtime/cuda_graph.h"
#include "devproc2/runtime/cuda_kernel_registry.h"
#include "devproc2/runtime/device_api.h"
#include "devproc2/runtime/kernel.h"
#include "devproc2/runtime/tensor.h"
#include "devproc2/runtime/vm.h"
#include "devproc2/runtime/vm_value.h"

namespace devproc2 {
void CUDAKernelLauncher_Launch(
    const KernelObj* kernel,
    std::vector<VMValue>& args,
    const std::vector<int64_t>& launch_args,
    void* stream);
}

namespace {

using namespace devproc2;

template <typename T>
bool read_raw(const std::string& path, std::vector<T>* out, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    out->resize(count);
    f.read(reinterpret_cast<char*>(out->data()),
           static_cast<std::streamsize>(count * sizeof(T)));
    return static_cast<size_t>(f.gcount()) == count * sizeof(T);
}

Tensor external_cpu_tensor(void* data, std::vector<int64_t> shape, DLDataType dtype) {
    return Tensor::FromExternalBuffer(data, DLDevice{kDLCPU, 0}, shape, dtype);
}

int64_t f32_to_i64_bits(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(float));
    return static_cast<int64_t>(bits);
}

double diff_abs_mean(const std::vector<float>& got, const std::vector<float>& want, double* abs_max) {
    double sum = 0.0;
    *abs_max = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
        sum += d;
        if (d > *abs_max) *abs_max = d;
    }
    return sum / static_cast<double>(got.size());
}

}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::max(1, std::atoi(argv[1])) : 5;
    bool use_graph = true;
    bool entry_is_loop = true;
    bool entry_is_sample_precomputed = false;
    bool entry_uses_prefix_embs = false;
    bool entry_uses_tokens = false;
    int max_prompt_len = 200;
    int prefix_valid_rows = 895;
    int num_views = 3;
    int action_horizon = 50;
    std::string oracle_dir_arg;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-graph") {
            use_graph = false;
        } else if (arg == "--max-prompt-len" && i + 1 < argc) {
            max_prompt_len = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--prefix-valid-rows" && i + 1 < argc) {
            prefix_valid_rows = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--num-views" && i + 1 < argc) {
            num_views = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--action-horizon" && i + 1 < argc) {
            action_horizon = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--oracle-dir" && i + 1 < argc) {
            oracle_dir_arg = argv[++i];
        } else if (arg == "--entry-kind" && i + 1 < argc) {
            std::string kind = argv[++i];
            if (kind == "loop") {
                entry_is_loop = true;
                entry_is_sample_precomputed = false;
                entry_uses_prefix_embs = false;
                entry_uses_tokens = false;
            } else if (kind == "step") {
                entry_is_loop = false;
                entry_is_sample_precomputed = false;
                entry_uses_prefix_embs = false;
                entry_uses_tokens = false;
            } else if (kind == "sample_precomputed_prefix") {
                entry_is_loop = true;
                entry_is_sample_precomputed = true;
                entry_uses_prefix_embs = false;
                entry_uses_tokens = false;
            } else if (kind == "sample_precomputed_prefix_embs") {
                entry_is_loop = true;
                entry_is_sample_precomputed = false;
                entry_uses_prefix_embs = true;
                entry_uses_tokens = false;
            } else if (kind == "sample_tokens") {
                entry_is_loop = true;
                entry_is_sample_precomputed = false;
                entry_uses_prefix_embs = false;
                entry_uses_tokens = true;
            } else {
                std::cerr << "unknown --entry-kind: " << kind << "\n";
                return 2;
            }
        }
    }
    const std::string root = std::string(DEVPROC2_SOURCE_DIR);
    std::string artifact_dir = root + (
        entry_uses_tokens
            ? "/build/pi05_fp8_sample_tokens_artifact"
            : (entry_uses_prefix_embs
            ? "/build/pi05_fp8_sample_precomputed_prefix_embs_artifact"
            : (entry_is_sample_precomputed
            ? "/build/pi05_fp8_sample_precomputed_prefix_artifact"
            : (entry_is_loop ? "/build/pi05_fp8_loop_artifact" : "/build/pi05_fp8_artifact"))));
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--artifact-dir" && i + 1 < argc) {
            artifact_dir = argv[++i];
        } else if (arg == "--entry-kind" && i + 1 < argc) {
            ++i;
        }
    }
    const std::string oracle_dir = oracle_dir_arg.empty()
        ? root + "/build/pi05_torch_denoise_oracle/bf16_example0/raw"
        : oracle_dir_arg;

    constexpr int L = 18;
    const int P = entry_uses_tokens ? (num_views * 256 + max_prompt_len) : 968;
    const int PV = prefix_valid_rows;
    constexpr int HKV = 1;
    constexpr int HD = 256;
    const int T = action_horizon;
    constexpr int A = 32;
    constexpr int IMG = 224;
    constexpr int C = 3;
    const int TOK = max_prompt_len;
    constexpr int num_steps = 10;
    const int64_t prefix_elems = static_cast<int64_t>(L) * P * HKV * HD;
    const int64_t rope_elems = static_cast<int64_t>(T) * HD;
    const int64_t action_elems = static_cast<int64_t>(T) * A;

    std::vector<float> init_actions;
    std::vector<float> final_target;
    std::vector<uint16_t> prefix_k;
    std::vector<uint16_t> prefix_v;
    std::vector<uint16_t> rope;
    std::vector<uint16_t> prefix_embs;
    std::vector<uint16_t> prefix_rope;
    std::vector<uint8_t> images_u8;
    std::vector<int32_t> token_ids;
    if (!read_raw(oracle_dir + "/step_000/actions_f32.bin", &init_actions, action_elems) ||
        !read_raw(oracle_dir + "/step_009/x_next_f32.bin", &final_target, action_elems) ||
        !read_raw(oracle_dir + "/prefix_k_cache_bf16.bin", &prefix_k, prefix_elems) ||
        !read_raw(oracle_dir + "/prefix_v_cache_bf16.bin", &prefix_v, prefix_elems) ||
        !read_raw(oracle_dir + "/rope_interleaved_bf16.bin", &rope, rope_elems)) {
        std::cerr << "missing pi05 artifact/oracle inputs\n";
        return 2;
    }
    if ((entry_uses_prefix_embs || entry_uses_tokens) &&
        (!read_raw(oracle_dir + "/prefix_embs_bf16.bin", &prefix_embs, P * 2048) ||
         !read_raw(oracle_dir + "/prefix_rope_interleaved_bf16.bin", &prefix_rope, P * HD))) {
        std::cerr << "missing pi05 prefix embedding oracle inputs\n";
        return 2;
    }
    if (entry_uses_tokens &&
        (!read_raw(oracle_dir + "/images_u8.bin", &images_u8, num_views * IMG * IMG * C) ||
         !read_raw(oracle_dir + "/token_ids_i32.bin", &token_ids, TOK))) {
        std::cerr << "missing pi05 image/token oracle inputs\n";
        return 2;
    }

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    std::vector<float> actions = init_actions;
    auto host_actions = external_cpu_tensor(actions.data(), {T, A}, DLDataType{kDLFloat, 32, 1});
    auto host_pk = external_cpu_tensor(prefix_k.data(), {L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_pv = external_cpu_tensor(prefix_v.data(), {L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_rope = external_cpu_tensor(rope.data(), {T, HD}, DLDataType{kDLBfloat, 16, 1});
    auto dev_actions = Tensor::Empty({T, A}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_pk = Tensor::Empty({L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_pv = Tensor::Empty({L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_rope = Tensor::Empty({T, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    Tensor host_prefix_embs;
    Tensor host_prefix_rope;
    Tensor dev_prefix_embs;
    Tensor dev_prefix_rope;
    Tensor host_images;
    Tensor host_token_ids;
    Tensor dev_images;
    Tensor dev_token_ids;
    cuda_api->CopyDataFromTo(host_pk->dl(), dev_pk->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_pv->dl(), dev_pv->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_rope->dl(), dev_rope->dl(), nullptr);
    if (entry_uses_prefix_embs || entry_uses_tokens) {
        host_prefix_embs = external_cpu_tensor(
            prefix_embs.data(), {P, 2048}, DLDataType{kDLBfloat, 16, 1});
        host_prefix_rope = external_cpu_tensor(
            prefix_rope.data(), {P, HD}, DLDataType{kDLBfloat, 16, 1});
        dev_prefix_embs = Tensor::Empty({P, 2048}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
        dev_prefix_rope = Tensor::Empty({P, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
        cuda_api->CopyDataFromTo(host_prefix_embs->dl(), dev_prefix_embs->dl(), nullptr);
        cuda_api->CopyDataFromTo(host_prefix_rope->dl(), dev_prefix_rope->dl(), nullptr);
    }
    if (entry_uses_tokens) {
        host_images = external_cpu_tensor(
            images_u8.data(), {num_views, IMG, IMG, C}, DLDataType{kDLUInt, 8, 1});
        host_token_ids = external_cpu_tensor(
            token_ids.data(), {TOK}, DLDataType{kDLInt, 32, 1});
        dev_images = Tensor::Empty({num_views, IMG, IMG, C}, DLDataType{kDLUInt, 8, 1}, cuda_dev);
        dev_token_ids = Tensor::Empty({TOK}, DLDataType{kDLInt, 32, 1}, cuda_dev);
        cuda_api->CopyDataFromTo(host_images->dl(), dev_images->dl(), nullptr);
        cuda_api->CopyDataFromTo(host_token_ids->dl(), dev_token_ids->dl(), nullptr);
    }
    cuda_api->DeviceSync(cuda_dev);

    auto session = ModelSession::LoadArtifact(artifact_dir);
    void* vm_stream = session.GetDefaultStream(cuda_dev);
    auto* update_kernel = CUDAKernelRegistry::Global().Get("kernel.pi05_euler_update_bf16");
    if (!entry_is_loop && update_kernel == nullptr) {
        std::cerr << "missing kernel.pi05_euler_update_bf16\n";
        return 3;
    }

    auto reset_actions = [&]() {
        std::copy(init_actions.begin(), init_actions.end(), actions.begin());
        cuda_api->CopyDataFromTo(host_actions->dl(), dev_actions->dl(), vm_stream);
        if (entry_uses_prefix_embs) {
            cuda_api->CopyDataFromTo(host_prefix_embs->dl(), dev_prefix_embs->dl(), vm_stream);
        }
    };

    std::vector<VMValue> live_outputs;
    std::vector<VMValue> graph_live_outputs;

    auto run_denoise_loop = [&]() {
        live_outputs.reserve(num_steps);
        if (entry_uses_tokens) {
            VMValue result = session.Invoke("main", {
                VMValue::ObjRef(dev_actions),
                VMValue::ObjRef(dev_images),
                VMValue::ObjRef(dev_token_ids),
                VMValue::Int(PV),
                VMValue::ObjRef(dev_prefix_rope),
                VMValue::ObjRef(dev_rope),
            });
            auto* out = result.AsObjectAs<TensorObj>();
            if (out == nullptr) {
                throw std::runtime_error("pi05 sample_tokens returned non-tensor");
            }
            cuda_api->CopyDataFromTo(out->dl(), dev_actions->dl(), vm_stream);
            live_outputs.push_back(result);
            return;
        }
        if (entry_uses_prefix_embs) {
            VMValue result = session.Invoke("main", {
                VMValue::ObjRef(dev_actions),
                VMValue::ObjRef(dev_prefix_embs),
                VMValue::Int(PV),
                VMValue::ObjRef(dev_prefix_rope),
                VMValue::ObjRef(dev_rope),
            });
            auto* out = result.AsObjectAs<TensorObj>();
            if (out == nullptr) {
                throw std::runtime_error("pi05 sample_prefix_embs returned non-tensor");
            }
            cuda_api->CopyDataFromTo(out->dl(), dev_actions->dl(), vm_stream);
            live_outputs.push_back(result);
            return;
        }
        if (entry_is_loop) {
            VMValue result = session.Invoke("main", {
                VMValue::ObjRef(dev_actions),
                VMValue::ObjRef(dev_pk),
                VMValue::ObjRef(dev_pv),
                VMValue::Int(PV),
                VMValue::ObjRef(dev_rope),
            });
            live_outputs.push_back(result);
            return;
        }
        for (int step = 0; step < num_steps; ++step) {
            VMValue result = session.Invoke("main", {
                VMValue::ObjRef(dev_actions),
                VMValue::ObjRef(dev_pk),
                VMValue::ObjRef(dev_pv),
                VMValue::Int(PV),
                VMValue::ObjRef(dev_rope),
                VMValue::Int(step),
            });
            auto* delta = result.AsObjectAs<TensorObj>();
            if (delta == nullptr) {
                throw std::runtime_error("pi05 denoise returned non-tensor");
            }
            std::vector<VMValue> update_args = {
                VMValue::ObjRef(dev_actions),
                VMValue::ObjRef(ObjectRef(delta)),
                VMValue::Int(f32_to_i64_bits(1.0f)),
                VMValue::Int(action_elems),
            };
            CUDAKernelLauncher_Launch(
                update_kernel,
                update_args,
                {7, 1, 1, 256, 1, 1, 0},
                vm_stream);
            live_outputs.push_back(result);
        }
    };

    auto run_once = [&]() -> double {
        live_outputs.clear();
        reset_actions();
        cuda_api->StreamSync(cuda_dev, vm_stream);
        auto t0 = std::chrono::steady_clock::now();
        run_denoise_loop();
        cuda_api->StreamSync(cuda_dev, vm_stream);
        auto t1 = std::chrono::steady_clock::now();
        live_outputs.clear();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    const double warmup_ms = run_once();
    // A second warmup populates the exact-size CUDA caching pool before stream
    // capture so Tensor::Empty calls in the VM do not perform cudaMalloc.
    (void)run_once();
    double total_ms = 0.0;
    std::string mode = "stream";
    CUDAGraphExec graph_exec;

    if (use_graph) {
        try {
            live_outputs.clear();
            reset_actions();
            cuda_api->StreamSync(cuda_dev, vm_stream);
            graph_exec = CUDAGraphExec::Capture(vm_stream, run_denoise_loop);
            graph_live_outputs = live_outputs;
            live_outputs.clear();
            mode = "cuda_graph";
        } catch (const std::exception& e) {
            std::cerr << "CUDA graph capture failed, falling back to stream mode: "
                      << e.what() << "\n";
            graph_exec.Reset();
            cuda_api->DeviceSync(cuda_dev);
        }
    }

    if (graph_exec.IsValid()) {
        // Upload once so the measured loop only includes action reset + graph launch.
        graph_exec.Upload(vm_stream);
        cuda_api->StreamSync(cuda_dev, vm_stream);
        for (int i = 0; i < iters; ++i) {
            reset_actions();
            auto t0 = std::chrono::steady_clock::now();
            graph_exec.Launch(vm_stream);
            cuda_api->StreamSync(cuda_dev, vm_stream);
            auto t1 = std::chrono::steady_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
    } else {
        for (int i = 0; i < iters; ++i) {
            total_ms += run_once();
        }
    }
    cuda_api->CopyDataFromTo(dev_actions->dl(), host_actions->dl(), vm_stream);
    cuda_api->StreamSync(cuda_dev, vm_stream);

    double final_abs_max = 0.0;
    double final_abs_mean = diff_abs_mean(actions, final_target, &final_abs_max);
    const double mean_ms = total_ms / static_cast<double>(iters);
    std::cout << std::fixed << std::setprecision(3)
              << "pi05_denoise_bench iters=" << iters
              << " entry=" << (
                    entry_uses_tokens
                        ? "sample_tokens"
                        : (entry_uses_prefix_embs
                        ? "sample_precomputed_prefix_embs"
                        : (entry_is_sample_precomputed
                        ? "sample_precomputed_prefix"
                        : (entry_is_loop ? "loop" : "step"))))
              << " mode=" << mode
              << " warmup_ms=" << warmup_ms
              << " mean_10step_ms=" << mean_ms
              << " mean_step_ms=" << (mean_ms / static_cast<double>(num_steps))
              << " final_abs_max=" << final_abs_max
              << " final_abs_mean=" << final_abs_mean << "\n";
    return 0;
}
