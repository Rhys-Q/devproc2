// Pi0.5 denoise VM smoke test against dumped PyTorch oracle tensors.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>
#include <nlohmann/json.hpp>

#include "devproc2/runtime/device_api.h"
#include "devproc2/runtime/tensor.h"
#include "devproc2/runtime/tuple.h"
#include "devproc2/runtime/vm.h"
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
namespace fs = std::filesystem;

static bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

template <typename T>
static bool read_raw(const std::string& path, std::vector<T>* out, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    out->resize(count);
    f.read(reinterpret_cast<char*>(out->data()),
           static_cast<std::streamsize>(count * sizeof(T)));
    return static_cast<size_t>(f.gcount()) == count * sizeof(T);
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

static int64_t read_prefix_valid_rows(const fs::path& oracle_dir) {
    std::ifstream f(oracle_dir / "metadata.json");
    if (!f.is_open()) {
        throw std::runtime_error("missing oracle metadata: " + (oracle_dir / "metadata.json").string());
    }
    nlohmann::json metadata;
    f >> metadata;
    return metadata.at("prefix_valid_rows").get<int64_t>();
}

static int parse_example_index(const fs::path& oracle_dir) {
    std::string name = oracle_dir.filename().string();
    std::string prefix = "bf16_example";
    if (name.rfind(prefix, 0) != 0) return -1;
    return std::stoi(name.substr(prefix.size()));
}

struct DiffStats {
    double abs_max{0.0};
    double abs_mean{0.0};
    bool finite{true};
};

static DiffStats diff_bf16_to_f32(const std::vector<uint16_t>& got_bits,
                                  const std::vector<float>& want) {
    DiffStats stats;
    if (got_bits.size() != want.size()) {
        stats.finite = false;
        stats.abs_max = std::numeric_limits<double>::infinity();
        stats.abs_mean = std::numeric_limits<double>::infinity();
        return stats;
    }
    double abs_sum = 0.0;
    for (size_t i = 0; i < got_bits.size(); ++i) {
        float got = bf16_bits_to_f32(got_bits[i]);
        if (!std::isfinite(got)) stats.finite = false;
        double diff = std::fabs(static_cast<double>(got) - static_cast<double>(want[i]));
        abs_sum += diff;
        if (diff > stats.abs_max) stats.abs_max = diff;
    }
    stats.abs_mean = abs_sum / static_cast<double>(got_bits.size());
    return stats;
}

static DiffStats diff_bf16_to_bf16_prefix_valid(
    const std::vector<uint16_t>& got_bits,
    const std::vector<uint16_t>& want_bits,
    int64_t layers,
    int64_t rows,
    int64_t valid_rows,
    int64_t kv_heads,
    int64_t head_dim) {
    DiffStats stats;
    const size_t total = static_cast<size_t>(layers * rows * kv_heads * head_dim);
    if (got_bits.size() != total || want_bits.size() != total) {
        stats.finite = false;
        stats.abs_max = std::numeric_limits<double>::infinity();
        stats.abs_mean = std::numeric_limits<double>::infinity();
        return stats;
    }
    double abs_sum = 0.0;
    size_t count = 0;
    for (int64_t l = 0; l < layers; ++l) {
        for (int64_t r = 0; r < valid_rows; ++r) {
            for (int64_t h = 0; h < kv_heads; ++h) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    const size_t idx = static_cast<size_t>(
                        (((l * rows + r) * kv_heads + h) * head_dim + d));
                    float got = bf16_bits_to_f32(got_bits[idx]);
                    float want = bf16_bits_to_f32(want_bits[idx]);
                    if (!std::isfinite(got)) stats.finite = false;
                    double diff = std::fabs(static_cast<double>(got) - static_cast<double>(want));
                    abs_sum += diff;
                    if (diff > stats.abs_max) stats.abs_max = diff;
                    ++count;
                }
            }
        }
    }
    stats.abs_mean = count == 0 ? 0.0 : abs_sum / static_cast<double>(count);
    return stats;
}

static DiffStats diff_f32_to_f32(const std::vector<float>& got,
                                 const std::vector<float>& want) {
    DiffStats stats;
    if (got.size() != want.size()) {
        stats.finite = false;
        stats.abs_max = std::numeric_limits<double>::infinity();
        stats.abs_mean = std::numeric_limits<double>::infinity();
        return stats;
    }
    double abs_sum = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) stats.finite = false;
        double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
        abs_sum += diff;
        if (diff > stats.abs_max) stats.abs_max = diff;
    }
    stats.abs_mean = abs_sum / static_cast<double>(got.size());
    return stats;
}

void test_denoise_step0_smoke_against_torch_oracle() {
    const std::string root = std::string(DEVPROC2_SOURCE_DIR);
    const std::string artifact_dir = root + "/build/pi05_fp8_artifact";
    const std::string oracle_dir = root + "/build/pi05_torch_denoise_oracle/bf16_example0/raw";
    const std::string actions_path = oracle_dir + "/step_000/actions_f32.bin";
    if (!file_exists(artifact_dir + "/executable.vm") || !file_exists(actions_path)) {
        std::cout << "  SKIP: missing exported artifact or oracle dump\n";
        return;
    }

    constexpr int L = 18;
    constexpr int P = 968;
    constexpr int PV = 895;
    constexpr int HKV = 1;
    constexpr int HD = 256;
    constexpr int T = 50;
    constexpr int A = 32;
    constexpr int prefix_elems = L * P * HKV * HD;
    constexpr int rope_elems = T * HD;
    constexpr int action_elems = T * A;

    std::vector<float> actions;
    std::vector<uint16_t> prefix_k;
    std::vector<uint16_t> prefix_v;
    std::vector<uint16_t> rope;
    std::vector<float> target_delta;
    CHECK(read_raw(oracle_dir + "/prefix_k_cache_bf16.bin", &prefix_k, prefix_elems));
    CHECK(read_raw(oracle_dir + "/prefix_v_cache_bf16.bin", &prefix_v, prefix_elems));
    CHECK(read_raw(oracle_dir + "/rope_interleaved_bf16.bin", &rope, rope_elems));
    CHECK(read_raw(actions_path, &actions, action_elems));
    CHECK(read_raw(oracle_dir + "/step_000/target_delta_f32.bin", &target_delta, action_elems));

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    auto host_actions = external_cpu_tensor(actions.data(), {T, A}, DLDataType{kDLFloat, 32, 1});
    auto host_pk = external_cpu_tensor(prefix_k.data(), {L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_pv = external_cpu_tensor(prefix_v.data(), {L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_rope = external_cpu_tensor(rope.data(), {T, HD}, DLDataType{kDLBfloat, 16, 1});
    auto dev_actions = Tensor::Empty({T, A}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_pk = Tensor::Empty({L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_pv = Tensor::Empty({L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_rope = Tensor::Empty({T, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_actions->dl(), dev_actions->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_pk->dl(), dev_pk->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_pv->dl(), dev_pv->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_rope->dl(), dev_rope->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    auto session = ModelSession::LoadArtifact(artifact_dir);
    VMValue result = session.Invoke("main", {
        VMValue::ObjRef(dev_actions),
        VMValue::ObjRef(dev_pk),
        VMValue::ObjRef(dev_pv),
        VMValue::Int(PV),
        VMValue::ObjRef(dev_rope),
        VMValue::Int(0),
    });
    auto* out_obj = result.AsObjectAs<TensorObj>();
    CHECK(out_obj != nullptr);

    std::vector<uint16_t> out_bits(action_elems, 0);
    auto host_out = external_cpu_tensor(out_bits.data(), {T, A}, DLDataType{kDLBfloat, 16, 1});
    cuda_api->CopyDataFromTo(out_obj->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    DiffStats stats = diff_bf16_to_f32(out_bits, target_delta);
    CHECK(stats.finite);
    std::cout << "  denoise_step0 abs_max=" << stats.abs_max
              << " abs_mean=" << stats.abs_mean << "\n";
    CHECK(stats.abs_max < 0.02);
    CHECK(stats.abs_mean < 0.003);
}

void test_denoise_10_step_smoke_against_torch_oracle() {
    const std::string root = std::string(DEVPROC2_SOURCE_DIR);
    const std::string artifact_dir = root + "/build/pi05_fp8_artifact";
    const std::string oracle_dir = root + "/build/pi05_torch_denoise_oracle/bf16_example0/raw";
    const std::string actions_path = oracle_dir + "/step_000/actions_f32.bin";
    const std::string final_path = oracle_dir + "/step_009/x_next_f32.bin";
    if (!file_exists(artifact_dir + "/executable.vm") || !file_exists(actions_path) ||
        !file_exists(final_path)) {
        std::cout << "  SKIP: missing exported artifact or all-step oracle dump\n";
        return;
    }

    constexpr int L = 18;
    constexpr int P = 968;
    constexpr int PV = 895;
    constexpr int HKV = 1;
    constexpr int HD = 256;
    constexpr int T = 50;
    constexpr int A = 32;
    constexpr int num_steps = 10;
    constexpr int prefix_elems = L * P * HKV * HD;
    constexpr int rope_elems = T * HD;
    constexpr int action_elems = T * A;

    std::vector<float> actions;
    std::vector<uint16_t> prefix_k;
    std::vector<uint16_t> prefix_v;
    std::vector<uint16_t> rope;
    CHECK(read_raw(oracle_dir + "/prefix_k_cache_bf16.bin", &prefix_k, prefix_elems));
    CHECK(read_raw(oracle_dir + "/prefix_v_cache_bf16.bin", &prefix_v, prefix_elems));
    CHECK(read_raw(oracle_dir + "/rope_interleaved_bf16.bin", &rope, rope_elems));
    CHECK(read_raw(actions_path, &actions, action_elems));

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    auto host_actions = external_cpu_tensor(actions.data(), {T, A}, DLDataType{kDLFloat, 32, 1});
    auto host_pk = external_cpu_tensor(prefix_k.data(), {L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_pv = external_cpu_tensor(prefix_v.data(), {L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_rope = external_cpu_tensor(rope.data(), {T, HD}, DLDataType{kDLBfloat, 16, 1});
    auto dev_actions = Tensor::Empty({T, A}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_pk = Tensor::Empty({L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_pv = Tensor::Empty({L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_rope = Tensor::Empty({T, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_pk->dl(), dev_pk->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_pv->dl(), dev_pv->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_rope->dl(), dev_rope->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    auto session = ModelSession::LoadArtifact(artifact_dir);
    double worst_delta_abs_max = 0.0;
    double worst_action_abs_max = 0.0;
    double final_abs_max = 0.0;
    double final_abs_mean = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        cuda_api->CopyDataFromTo(host_actions->dl(), dev_actions->dl(), nullptr);
        cuda_api->DeviceSync(cuda_dev);
        VMValue result = session.Invoke("main", {
            VMValue::ObjRef(dev_actions),
            VMValue::ObjRef(dev_pk),
            VMValue::ObjRef(dev_pv),
            VMValue::Int(PV),
            VMValue::ObjRef(dev_rope),
            VMValue::Int(step),
        });
        auto* out_obj = result.AsObjectAs<TensorObj>();
        CHECK(out_obj != nullptr);

        std::vector<uint16_t> out_bits(action_elems, 0);
        auto host_out = external_cpu_tensor(out_bits.data(), {T, A}, DLDataType{kDLBfloat, 16, 1});
        cuda_api->CopyDataFromTo(out_obj->dl(), host_out->dl(), nullptr);
        cuda_api->DeviceSync(cuda_dev);

        char step_dir[32];
        std::snprintf(step_dir, sizeof(step_dir), "/step_%03d/", step);
        std::vector<float> target_delta;
        CHECK(read_raw(oracle_dir + step_dir + "target_delta_f32.bin", &target_delta, action_elems));
        DiffStats delta_stats = diff_bf16_to_f32(out_bits, target_delta);
        CHECK(delta_stats.finite);
        if (delta_stats.abs_max > worst_delta_abs_max) worst_delta_abs_max = delta_stats.abs_max;

        for (int i = 0; i < action_elems; ++i) {
            actions[static_cast<size_t>(i)] += bf16_bits_to_f32(out_bits[static_cast<size_t>(i)]);
        }

        std::vector<float> target_next;
        CHECK(read_raw(oracle_dir + step_dir + "x_next_f32.bin", &target_next, action_elems));
        DiffStats action_stats = diff_f32_to_f32(actions, target_next);
        CHECK(action_stats.finite);
        if (action_stats.abs_max > worst_action_abs_max) worst_action_abs_max = action_stats.abs_max;
        if (step == num_steps - 1) {
            final_abs_max = action_stats.abs_max;
            final_abs_mean = action_stats.abs_mean;
        }
    }

    std::cout << "  denoise_10_step worst_delta_abs_max=" << worst_delta_abs_max
              << " worst_action_abs_max=" << worst_action_abs_max
              << " final_abs_max=" << final_abs_max
              << " final_abs_mean=" << final_abs_mean << "\n";
    CHECK(worst_delta_abs_max < 0.03);
    CHECK(final_abs_max < 0.10);
    CHECK(final_abs_mean < 0.02);
}

void test_denoise_loop_artifact_against_torch_oracle() {
    const std::string root = std::string(DEVPROC2_SOURCE_DIR);
    const std::string artifact_dir = root + "/build/pi05_fp8_loop_artifact";
    const std::string oracle_dir = root + "/build/pi05_torch_denoise_oracle/bf16_example0/raw";
    const std::string actions_path = oracle_dir + "/step_000/actions_f32.bin";
    const std::string final_path = oracle_dir + "/step_009/x_next_f32.bin";
    if (!file_exists(artifact_dir + "/executable.vm") || !file_exists(actions_path) ||
        !file_exists(final_path)) {
        std::cout << "  SKIP: missing exported loop artifact or all-step oracle dump\n";
        return;
    }

    constexpr int L = 18;
    constexpr int P = 968;
    constexpr int PV = 895;
    constexpr int HKV = 1;
    constexpr int HD = 256;
    constexpr int T = 50;
    constexpr int A = 32;
    constexpr int prefix_elems = L * P * HKV * HD;
    constexpr int rope_elems = T * HD;
    constexpr int action_elems = T * A;

    std::vector<float> actions;
    std::vector<float> final_target;
    std::vector<uint16_t> prefix_k;
    std::vector<uint16_t> prefix_v;
    std::vector<uint16_t> rope;
    CHECK(read_raw(oracle_dir + "/prefix_k_cache_bf16.bin", &prefix_k, prefix_elems));
    CHECK(read_raw(oracle_dir + "/prefix_v_cache_bf16.bin", &prefix_v, prefix_elems));
    CHECK(read_raw(oracle_dir + "/rope_interleaved_bf16.bin", &rope, rope_elems));
    CHECK(read_raw(actions_path, &actions, action_elems));
    CHECK(read_raw(final_path, &final_target, action_elems));

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    auto host_actions = external_cpu_tensor(actions.data(), {T, A}, DLDataType{kDLFloat, 32, 1});
    auto host_pk = external_cpu_tensor(prefix_k.data(), {L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_pv = external_cpu_tensor(prefix_v.data(), {L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_rope = external_cpu_tensor(rope.data(), {T, HD}, DLDataType{kDLBfloat, 16, 1});
    auto dev_actions = Tensor::Empty({T, A}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_pk = Tensor::Empty({L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_pv = Tensor::Empty({L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_rope = Tensor::Empty({T, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_actions->dl(), dev_actions->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_pk->dl(), dev_pk->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_pv->dl(), dev_pv->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_rope->dl(), dev_rope->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    auto session = ModelSession::LoadArtifact(artifact_dir);
    VMValue result = session.Invoke("main", {
        VMValue::ObjRef(dev_actions),
        VMValue::ObjRef(dev_pk),
        VMValue::ObjRef(dev_pv),
        VMValue::Int(PV),
        VMValue::ObjRef(dev_rope),
    });
    auto* out_obj = result.AsObjectAs<TensorObj>();
    CHECK(out_obj != nullptr);

    std::vector<float> out(action_elems, 0.0f);
    auto host_out = external_cpu_tensor(out.data(), {T, A}, DLDataType{kDLFloat, 32, 1});
    cuda_api->CopyDataFromTo(out_obj->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    DiffStats stats = diff_f32_to_f32(out, final_target);
    CHECK(stats.finite);
    std::cout << "  denoise_loop final_abs_max=" << stats.abs_max
              << " final_abs_mean=" << stats.abs_mean << "\n";
    CHECK(stats.abs_max < 0.10);
    CHECK(stats.abs_mean < 0.02);
}

void check_loop_like_artifact_against_available_torch_oracles(
    const std::string& artifact_dir,
    const std::string& label) {
    const std::string root = std::string(DEVPROC2_SOURCE_DIR);
    const fs::path oracle_root = fs::path(root) / "build/pi05_torch_denoise_oracle";
    if (!file_exists(artifact_dir + "/executable.vm") || !fs::exists(oracle_root)) {
        std::cout << "  SKIP: missing exported " << label << " artifact or oracle root\n";
        return;
    }

    constexpr int L = 18;
    constexpr int P = 968;
    constexpr int HKV = 1;
    constexpr int HD = 256;
    constexpr int T = 50;
    constexpr int A = 32;
    constexpr int prefix_elems = L * P * HKV * HD;
    constexpr int rope_elems = T * HD;
    constexpr int action_elems = T * A;

    std::vector<fs::path> oracle_dirs;
    for (const auto& entry : fs::directory_iterator(oracle_root)) {
        if (!entry.is_directory()) continue;
        const fs::path raw = entry.path() / "raw";
        if (file_exists((raw / "step_000/actions_f32.bin").string()) &&
            file_exists((raw / "step_009/x_next_f32.bin").string())) {
            oracle_dirs.push_back(entry.path());
        }
    }
    CHECK(!oracle_dirs.empty());
    std::sort(oracle_dirs.begin(), oracle_dirs.end());

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);
    auto session = ModelSession::LoadArtifact(artifact_dir);

    double worst_abs_max = 0.0;
    double worst_abs_mean = 0.0;
    double worst_fp16_abs_max = 0.0;
    double worst_fp16_abs_mean = 0.0;
    int fp16_count = 0;
    for (const fs::path& dir : oracle_dirs) {
        const fs::path raw = dir / "raw";
        const int64_t prefix_valid_rows = read_prefix_valid_rows(dir);
        std::vector<float> actions;
        std::vector<float> final_target;
        std::vector<uint16_t> prefix_k;
        std::vector<uint16_t> prefix_v;
        std::vector<uint16_t> rope;
        CHECK(read_raw((raw / "prefix_k_cache_bf16.bin").string(), &prefix_k, prefix_elems));
        CHECK(read_raw((raw / "prefix_v_cache_bf16.bin").string(), &prefix_v, prefix_elems));
        CHECK(read_raw((raw / "rope_interleaved_bf16.bin").string(), &rope, rope_elems));
        CHECK(read_raw((raw / "step_000/actions_f32.bin").string(), &actions, action_elems));
        CHECK(read_raw((raw / "step_009/x_next_f32.bin").string(), &final_target, action_elems));

        auto host_actions = external_cpu_tensor(actions.data(), {T, A}, DLDataType{kDLFloat, 32, 1});
        auto host_pk = external_cpu_tensor(prefix_k.data(), {L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
        auto host_pv = external_cpu_tensor(prefix_v.data(), {L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
        auto host_rope = external_cpu_tensor(rope.data(), {T, HD}, DLDataType{kDLBfloat, 16, 1});
        auto dev_actions = Tensor::Empty({T, A}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
        auto dev_pk = Tensor::Empty({L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
        auto dev_pv = Tensor::Empty({L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
        auto dev_rope = Tensor::Empty({T, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
        cuda_api->CopyDataFromTo(host_actions->dl(), dev_actions->dl(), nullptr);
        cuda_api->CopyDataFromTo(host_pk->dl(), dev_pk->dl(), nullptr);
        cuda_api->CopyDataFromTo(host_pv->dl(), dev_pv->dl(), nullptr);
        cuda_api->CopyDataFromTo(host_rope->dl(), dev_rope->dl(), nullptr);
        cuda_api->DeviceSync(cuda_dev);

        VMValue result = session.Invoke("main", {
            VMValue::ObjRef(dev_actions),
            VMValue::ObjRef(dev_pk),
            VMValue::ObjRef(dev_pv),
            VMValue::Int(prefix_valid_rows),
            VMValue::ObjRef(dev_rope),
        });
        auto* out_obj = result.AsObjectAs<TensorObj>();
        CHECK(out_obj != nullptr);

        std::vector<float> out(action_elems, 0.0f);
        auto host_out = external_cpu_tensor(out.data(), {T, A}, DLDataType{kDLFloat, 32, 1});
        cuda_api->CopyDataFromTo(out_obj->dl(), host_out->dl(), nullptr);
        cuda_api->DeviceSync(cuda_dev);

        DiffStats stats = diff_f32_to_f32(out, final_target);
        CHECK(stats.finite);
        std::cout << "    " << dir.filename().string()
                  << " abs_max=" << stats.abs_max
                  << " abs_mean=" << stats.abs_mean << "\n";
        if (stats.abs_max > worst_abs_max) worst_abs_max = stats.abs_max;
        if (stats.abs_mean > worst_abs_mean) worst_abs_mean = stats.abs_mean;

        int example_index = parse_example_index(dir);
        char example_dir[32];
        if (example_index >= 0) {
            std::snprintf(example_dir, sizeof(example_dir), "example_%03d", example_index);
        }
        fs::path fp16_path = example_index >= 0
            ? (fs::path(root) / "build/pi05_torch_denoise_oracle" /
               "fp16_outputs_raw" / example_dir / "actions_f32.bin")
            : fs::path();
        if (example_index >= 0 && file_exists(fp16_path.string())) {
            std::vector<float> fp16_target;
            CHECK(read_raw(fp16_path.string(), &fp16_target, action_elems));
            DiffStats fp16_stats = diff_f32_to_f32(out, fp16_target);
            CHECK(fp16_stats.finite);
            std::cout << "      vs_fp16 abs_max=" << fp16_stats.abs_max
                      << " abs_mean=" << fp16_stats.abs_mean << "\n";
            if (fp16_stats.abs_max > worst_fp16_abs_max) {
                worst_fp16_abs_max = fp16_stats.abs_max;
            }
            if (fp16_stats.abs_mean > worst_fp16_abs_mean) {
                worst_fp16_abs_mean = fp16_stats.abs_mean;
            }
            ++fp16_count;
        }
    }

    std::cout << "  " << label << "_oracles count=" << oracle_dirs.size()
              << " worst_abs_max=" << worst_abs_max
              << " worst_abs_mean=" << worst_abs_mean;
    if (fp16_count > 0) {
        std::cout << " fp16_count=" << fp16_count
                  << " worst_fp16_abs_max=" << worst_fp16_abs_max
                  << " worst_fp16_abs_mean=" << worst_fp16_abs_mean;
    }
    std::cout << "\n";
    CHECK(worst_abs_max < 0.16);
    CHECK(worst_abs_mean < 0.02);
    if (fp16_count > 0) {
        CHECK(worst_fp16_abs_max < 0.20);
        CHECK(worst_fp16_abs_mean < 0.03);
    }
}

void test_denoise_loop_artifact_against_available_torch_oracles() {
    const std::string root = std::string(DEVPROC2_SOURCE_DIR);
    check_loop_like_artifact_against_available_torch_oracles(
        root + "/build/pi05_fp8_loop_artifact",
        "denoise_loop");
}

void test_sample_precomputed_prefix_artifact_against_available_torch_oracles() {
    const std::string root = std::string(DEVPROC2_SOURCE_DIR);
    check_loop_like_artifact_against_available_torch_oracles(
        root + "/build/pi05_fp8_sample_precomputed_prefix_artifact",
        "sample_precomputed_prefix");
}

void test_paligemma_prefix_kv_artifact_against_torch_oracle() {
    const std::string root = std::string(DEVPROC2_SOURCE_DIR);
    const std::string artifact_dir =
        root + "/build/pi05_fp8_paligemma_prefix_kv_encoder_artifact";
    const fs::path oracle_dir =
        fs::path(root) / "build/pi05_torch_denoise_oracle/bf16_example0";
    const fs::path raw = oracle_dir / "raw";
    if (!file_exists(artifact_dir + "/executable.vm") ||
        !file_exists((raw / "prefix_embs_bf16.bin").string()) ||
        !file_exists((raw / "prefix_rope_interleaved_bf16.bin").string())) {
        std::cout << "  SKIP: missing prefix KV artifact or oracle prefix inputs\n";
        return;
    }

    constexpr int L = 18;
    constexpr int P = 968;
    constexpr int H = 2048;
    constexpr int HKV = 1;
    constexpr int HD = 256;
    constexpr int prefix_emb_elems = P * H;
    constexpr int prefix_cache_elems = L * P * HKV * HD;
    constexpr int prefix_rope_elems = P * HD;

    const int64_t prefix_valid_rows = read_prefix_valid_rows(oracle_dir);
    std::vector<uint16_t> prefix_embs;
    std::vector<uint16_t> prefix_rope;
    std::vector<uint16_t> want_k;
    std::vector<uint16_t> want_v;
    CHECK(read_raw((raw / "prefix_embs_bf16.bin").string(), &prefix_embs, prefix_emb_elems));
    CHECK(read_raw((raw / "prefix_rope_interleaved_bf16.bin").string(), &prefix_rope, prefix_rope_elems));
    CHECK(read_raw((raw / "prefix_k_cache_bf16.bin").string(), &want_k, prefix_cache_elems));
    CHECK(read_raw((raw / "prefix_v_cache_bf16.bin").string(), &want_v, prefix_cache_elems));

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    auto host_embs = external_cpu_tensor(prefix_embs.data(), {P, H}, DLDataType{kDLBfloat, 16, 1});
    auto host_rope = external_cpu_tensor(prefix_rope.data(), {P, HD}, DLDataType{kDLBfloat, 16, 1});
    auto dev_embs = Tensor::Empty({P, H}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_rope = Tensor::Empty({P, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_embs->dl(), dev_embs->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_rope->dl(), dev_rope->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    auto session = ModelSession::LoadArtifact(artifact_dir);
    VMValue result = session.Invoke("main", {
        VMValue::ObjRef(dev_embs),
        VMValue::Int(prefix_valid_rows),
        VMValue::ObjRef(dev_rope),
    });
    auto* tuple_obj = result.AsObjectAs<TupleObj>();
    CHECK(tuple_obj != nullptr);
    CHECK(tuple_obj->size() == 2);
    auto* k_obj = (*tuple_obj)[0].as<TensorObj>();
    auto* v_obj = (*tuple_obj)[1].as<TensorObj>();
    CHECK(k_obj != nullptr);
    CHECK(v_obj != nullptr);

    std::vector<uint16_t> got_k(prefix_cache_elems, 0);
    std::vector<uint16_t> got_v(prefix_cache_elems, 0);
    auto host_k = external_cpu_tensor(got_k.data(), {L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_v = external_cpu_tensor(got_v.data(), {L, P, HKV, HD}, DLDataType{kDLBfloat, 16, 1});
    cuda_api->CopyDataFromTo(k_obj->dl(), host_k->dl(), nullptr);
    cuda_api->CopyDataFromTo(v_obj->dl(), host_v->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    DiffStats k_stats = diff_bf16_to_bf16_prefix_valid(
        got_k, want_k, L, P, prefix_valid_rows, HKV, HD);
    DiffStats v_stats = diff_bf16_to_bf16_prefix_valid(
        got_v, want_v, L, P, prefix_valid_rows, HKV, HD);
    double worst_layer_k = 0.0;
    double worst_layer_v = 0.0;
    int worst_layer_k_idx = -1;
    int worst_layer_v_idx = -1;
    for (int layer = 0; layer < L; ++layer) {
        const auto layer_offset = static_cast<size_t>(layer * P * HKV * HD);
        std::vector<uint16_t> got_k_layer(
            got_k.begin() + static_cast<std::ptrdiff_t>(layer_offset),
            got_k.begin() + static_cast<std::ptrdiff_t>(layer_offset + P * HKV * HD));
        std::vector<uint16_t> want_k_layer(
            want_k.begin() + static_cast<std::ptrdiff_t>(layer_offset),
            want_k.begin() + static_cast<std::ptrdiff_t>(layer_offset + P * HKV * HD));
        std::vector<uint16_t> got_v_layer(
            got_v.begin() + static_cast<std::ptrdiff_t>(layer_offset),
            got_v.begin() + static_cast<std::ptrdiff_t>(layer_offset + P * HKV * HD));
        std::vector<uint16_t> want_v_layer(
            want_v.begin() + static_cast<std::ptrdiff_t>(layer_offset),
            want_v.begin() + static_cast<std::ptrdiff_t>(layer_offset + P * HKV * HD));
        DiffStats lk = diff_bf16_to_bf16_prefix_valid(
            got_k_layer, want_k_layer, 1, P, prefix_valid_rows, HKV, HD);
        DiffStats lv = diff_bf16_to_bf16_prefix_valid(
            got_v_layer, want_v_layer, 1, P, prefix_valid_rows, HKV, HD);
        if (lk.abs_mean > worst_layer_k) {
            worst_layer_k = lk.abs_mean;
            worst_layer_k_idx = layer;
        }
        if (lv.abs_mean > worst_layer_v) {
            worst_layer_v = lv.abs_mean;
            worst_layer_v_idx = layer;
        }
        if (layer < 2 || layer == L - 1) {
            std::cout << "    layer" << layer
                      << " k_abs_max=" << lk.abs_max
                      << " k_abs_mean=" << lk.abs_mean
                      << " v_abs_max=" << lv.abs_max
                      << " v_abs_mean=" << lv.abs_mean << "\n";
        }
    }
    CHECK(k_stats.finite);
    CHECK(v_stats.finite);
    std::cout << "  prefix_kv valid_rows=" << prefix_valid_rows
              << " k_abs_max=" << k_stats.abs_max
              << " k_abs_mean=" << k_stats.abs_mean
              << " v_abs_max=" << v_stats.abs_max
              << " v_abs_mean=" << v_stats.abs_mean
              << " worst_k_layer=" << worst_layer_k_idx
              << " worst_k_layer_mean=" << worst_layer_k
              << " worst_v_layer=" << worst_layer_v_idx
              << " worst_v_layer_mean=" << worst_layer_v << "\n";
    // Prefix KV is an intermediate FP8/BF16 cache produced by many quantized
    // GEMMs. Keep this as a broad smoke check and let the downstream actions
    // check below enforce behavior that matters to the exported sample path.
    CHECK(k_stats.abs_max < 32.0);
    CHECK(k_stats.abs_mean < 2.50);
    CHECK(v_stats.abs_max < 96.0);
    CHECK(v_stats.abs_mean < 5.00);
    std::cout << "  standalone prefix_kv raw-cache smoke passed; "
              << "single-artifact prefix_embs test covers downstream actions\n";
    return;

    const std::string sample_artifact_dir =
        root + "/build/pi05_fp8_sample_precomputed_prefix_artifact";
    if (!file_exists(sample_artifact_dir + "/executable.vm")) {
        std::cout << "  SKIP: missing sample_precomputed_prefix artifact\n";
        return;
    }
    constexpr int T = 50;
    constexpr int A = 32;
    constexpr int action_elems = T * A;
    constexpr int suffix_rope_elems = T * HD;
    std::vector<float> actions;
    std::vector<uint16_t> suffix_rope;
    std::vector<float> final_target;
    CHECK(read_raw((raw / "step_000/actions_f32.bin").string(), &actions, action_elems));
    CHECK(read_raw((raw / "rope_interleaved_bf16.bin").string(), &suffix_rope, suffix_rope_elems));
    CHECK(read_raw((raw / "step_009/x_next_f32.bin").string(), &final_target, action_elems));
    auto host_actions = external_cpu_tensor(actions.data(), {T, A}, DLDataType{kDLFloat, 32, 1});
    auto host_suffix_rope = external_cpu_tensor(suffix_rope.data(), {T, HD}, DLDataType{kDLBfloat, 16, 1});
    auto dev_actions = Tensor::Empty({T, A}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_suffix_rope = Tensor::Empty({T, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_actions->dl(), dev_actions->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_suffix_rope->dl(), dev_suffix_rope->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);
    auto sample_session = ModelSession::LoadArtifact(sample_artifact_dir);
    VMValue action_result = sample_session.Invoke("main", {
        VMValue::ObjRef(dev_actions),
        VMValue::ObjRef(ObjectRef(k_obj)),
        VMValue::ObjRef(ObjectRef(v_obj)),
        VMValue::Int(prefix_valid_rows),
        VMValue::ObjRef(dev_suffix_rope),
    });
    auto* action_obj = action_result.AsObjectAs<TensorObj>();
    CHECK(action_obj != nullptr);
    std::vector<float> out_actions(action_elems, 0.0f);
    auto host_out_actions = external_cpu_tensor(out_actions.data(), {T, A}, DLDataType{kDLFloat, 32, 1});
    cuda_api->CopyDataFromTo(action_obj->dl(), host_out_actions->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);
    DiffStats action_stats = diff_f32_to_f32(out_actions, final_target);
    CHECK(action_stats.finite);
    std::cout << "  prefix_kv_then_denoise final_abs_max=" << action_stats.abs_max
              << " final_abs_mean=" << action_stats.abs_mean << "\n";
    CHECK(action_stats.abs_max < 0.25);
    CHECK(action_stats.abs_mean < 0.04);
}

void test_sample_precomputed_prefix_embs_artifact_against_torch_oracle() {
    const std::string root = std::string(DEVPROC2_SOURCE_DIR);
    const std::string artifact_dir =
        root + "/build/pi05_fp8_sample_precomputed_prefix_embs_artifact";
    const fs::path oracle_dir =
        fs::path(root) / "build/pi05_torch_denoise_oracle/bf16_example0";
    const fs::path raw = oracle_dir / "raw";
    if (!file_exists(artifact_dir + "/executable.vm") ||
        !file_exists((raw / "prefix_embs_bf16.bin").string()) ||
        !file_exists((raw / "prefix_rope_interleaved_bf16.bin").string())) {
        std::cout << "  SKIP: missing prefix-embs sample artifact or oracle prefix inputs\n";
        return;
    }

    constexpr int P = 968;
    constexpr int H = 2048;
    constexpr int HD = 256;
    constexpr int T = 50;
    constexpr int A = 32;
    constexpr int prefix_emb_elems = P * H;
    constexpr int prefix_rope_elems = P * HD;
    constexpr int suffix_rope_elems = T * HD;
    constexpr int action_elems = T * A;

    const int64_t prefix_valid_rows = read_prefix_valid_rows(oracle_dir);
    std::vector<float> actions;
    std::vector<uint16_t> prefix_embs;
    std::vector<uint16_t> prefix_rope;
    std::vector<uint16_t> suffix_rope;
    std::vector<float> final_target;
    CHECK(read_raw((raw / "step_000/actions_f32.bin").string(), &actions, action_elems));
    CHECK(read_raw((raw / "prefix_embs_bf16.bin").string(), &prefix_embs, prefix_emb_elems));
    CHECK(read_raw((raw / "prefix_rope_interleaved_bf16.bin").string(), &prefix_rope, prefix_rope_elems));
    CHECK(read_raw((raw / "rope_interleaved_bf16.bin").string(), &suffix_rope, suffix_rope_elems));
    CHECK(read_raw((raw / "step_009/x_next_f32.bin").string(), &final_target, action_elems));

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    auto host_actions = external_cpu_tensor(actions.data(), {T, A}, DLDataType{kDLFloat, 32, 1});
    auto host_embs = external_cpu_tensor(prefix_embs.data(), {P, H}, DLDataType{kDLBfloat, 16, 1});
    auto host_prefix_rope = external_cpu_tensor(prefix_rope.data(), {P, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_suffix_rope = external_cpu_tensor(suffix_rope.data(), {T, HD}, DLDataType{kDLBfloat, 16, 1});
    auto dev_actions = Tensor::Empty({T, A}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_embs = Tensor::Empty({P, H}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_prefix_rope = Tensor::Empty({P, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_suffix_rope = Tensor::Empty({T, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_actions->dl(), dev_actions->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_embs->dl(), dev_embs->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_prefix_rope->dl(), dev_prefix_rope->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_suffix_rope->dl(), dev_suffix_rope->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    auto session = ModelSession::LoadArtifact(artifact_dir);
    VMValue result = session.Invoke("main", {
        VMValue::ObjRef(dev_actions),
        VMValue::ObjRef(dev_embs),
        VMValue::Int(prefix_valid_rows),
        VMValue::ObjRef(dev_prefix_rope),
        VMValue::ObjRef(dev_suffix_rope),
    });
    auto* out_obj = result.AsObjectAs<TensorObj>();
    CHECK(out_obj != nullptr);

    std::vector<float> out_actions(action_elems, 0.0f);
    auto host_out = external_cpu_tensor(out_actions.data(), {T, A}, DLDataType{kDLFloat, 32, 1});
    cuda_api->CopyDataFromTo(out_obj->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    DiffStats stats = diff_f32_to_f32(out_actions, final_target);
    CHECK(stats.finite);
    std::cout << "  sample_prefix_embs final_abs_max=" << stats.abs_max
              << " final_abs_mean=" << stats.abs_mean << "\n";
    CHECK(stats.abs_max < 0.25);
    CHECK(stats.abs_mean < 0.04);
}

}  // namespace

int main() {
    RUN(test_denoise_step0_smoke_against_torch_oracle);
    RUN(test_denoise_10_step_smoke_against_torch_oracle);
    RUN(test_denoise_loop_artifact_against_torch_oracle);
    RUN(test_denoise_loop_artifact_against_available_torch_oracles);
    RUN(test_sample_precomputed_prefix_artifact_against_available_torch_oracles);
    RUN(test_paligemma_prefix_kv_artifact_against_torch_oracle);
    RUN(test_sample_precomputed_prefix_embs_artifact_against_torch_oracle);
    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
