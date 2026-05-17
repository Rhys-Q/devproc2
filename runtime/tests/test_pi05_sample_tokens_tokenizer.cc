// Pi0.5 sample_tokens smoke test with tokenizers-cpp generated token ids.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

#include "devproc2/runtime/device_api.h"
#include "devproc2/runtime/packed_func.h"
#include "devproc2/runtime/string.h"
#include "devproc2/runtime/tensor.h"
#include "devproc2/runtime/tokenizer.h"
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

template <typename T>
bool read_raw(const std::string& path, std::vector<T>* out, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    out->resize(count);
    f.read(reinterpret_cast<char*>(out->data()),
           static_cast<std::streamsize>(count * sizeof(T)));
    return static_cast<size_t>(f.gcount()) == count * sizeof(T);
}

std::string read_text_trimmed(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string out = ss.str();
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return out;
}

Tensor external_cpu_tensor(void* data, std::vector<int64_t> shape, DLDataType dtype) {
    return Tensor::FromExternalBuffer(data, DLDevice{kDLCPU, 0}, shape, dtype);
}

double diff_abs_mean(const std::vector<float>& got,
                     const std::vector<float>& want,
                     double* abs_max) {
    double sum = 0.0;
    *abs_max = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
        sum += d;
        if (d > *abs_max) *abs_max = d;
    }
    return sum / static_cast<double>(got.size());
}

void encode_pi05_tokens(const std::string& prompt,
                        std::vector<float>& state,
                        std::vector<int32_t>& token_ids,
                        std::vector<int32_t>& token_mask) {
    auto pf = PackedFuncRegistry::Global().Get("runtime.tokenizer.paligemma_pi05_encode");
    if (!pf.defined()) {
        throw std::runtime_error("missing runtime.tokenizer.paligemma_pi05_encode");
    }
    auto prompt_obj = String::Make(prompt);
    auto state_tensor = external_cpu_tensor(
        state.data(), {static_cast<int64_t>(state.size())}, DLDataType{kDLFloat, 32, 1});
    auto token_tensor = external_cpu_tensor(
        token_ids.data(), {static_cast<int64_t>(token_ids.size())}, DLDataType{kDLInt, 32, 1});
    auto mask_tensor = external_cpu_tensor(
        token_mask.data(), {static_cast<int64_t>(token_mask.size())}, DLDataType{kDLInt, 32, 1});
    std::vector<VMValue> args = {
        VMValue::ObjRef(prompt_obj),
        VMValue::ObjRef(state_tensor),
        VMValue::ObjRef(token_tensor),
        VMValue::ObjRef(mask_tensor),
    };
    PackedArgs packed(args);
    pf->Call(packed);
}

void test_sample_tokens_consumes_runtime_tokenizer_output() {
    const std::string root = std::string(DEVPROC2_SOURCE_DIR);
    const std::string artifact_dir = root + "/build/pi05_fp8_sample_tokens_127_artifact";
    const std::string oracle_dir = root + "/build/pi05_torch_denoise_oracle/bf16_example0/raw";

    constexpr int V = 3;
    constexpr int IMG = 224;
    constexpr int C = 3;
    constexpr int TOK = 127;
    constexpr int P = 895;
    constexpr int T = 50;
    constexpr int A = 32;
    constexpr int HD = 256;

    std::vector<float> state;
    std::vector<float> actions;
    std::vector<float> final_target;
    std::vector<uint16_t> prefix_rope;
    std::vector<uint16_t> suffix_rope;
    std::vector<uint8_t> images_u8;
    std::vector<int32_t> oracle_tokens;
    CHECK(read_raw(oracle_dir + "/state_f32.bin", &state, 32));
    CHECK(read_raw(oracle_dir + "/step_000/actions_f32.bin", &actions, T * A));
    CHECK(read_raw(oracle_dir + "/step_009/x_next_f32.bin", &final_target, T * A));
    CHECK(read_raw(oracle_dir + "/prefix_rope_interleaved_bf16.bin", &prefix_rope, P * HD));
    CHECK(read_raw(oracle_dir + "/rope_interleaved_bf16.bin", &suffix_rope, T * HD));
    CHECK(read_raw(oracle_dir + "/images_u8.bin", &images_u8, V * IMG * IMG * C));
    CHECK(read_raw(oracle_dir + "/token_ids_i32.bin", &oracle_tokens, TOK));
    const std::string prompt = read_text_trimmed(oracle_dir + "/prompt.txt");
    CHECK(!prompt.empty());

    SetPaligemmaTokenizerModelPath(artifact_dir + "/resources/tokenizer.model");
    RegisterTokenizerPackedFuncs();
    std::vector<int32_t> token_ids(TOK, 0);
    std::vector<int32_t> token_mask(TOK, 0);
    encode_pi05_tokens(prompt, state, token_ids, token_mask);
    for (int i = 0; i < TOK; ++i) {
        CHECK(token_ids[static_cast<size_t>(i)] == oracle_tokens[static_cast<size_t>(i)]);
    }

    Device cuda_dev{kDLCUDA, 0};
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->SetDevice(cuda_dev);

    auto host_actions = external_cpu_tensor(actions.data(), {T, A}, DLDataType{kDLFloat, 32, 1});
    auto host_images = external_cpu_tensor(images_u8.data(), {V, IMG, IMG, C}, DLDataType{kDLUInt, 8, 1});
    auto host_tokens = external_cpu_tensor(token_ids.data(), {TOK}, DLDataType{kDLInt, 32, 1});
    auto host_prefix_rope = external_cpu_tensor(prefix_rope.data(), {P, HD}, DLDataType{kDLBfloat, 16, 1});
    auto host_suffix_rope = external_cpu_tensor(suffix_rope.data(), {T, HD}, DLDataType{kDLBfloat, 16, 1});
    auto dev_actions = Tensor::Empty({T, A}, DLDataType{kDLFloat, 32, 1}, cuda_dev);
    auto dev_images = Tensor::Empty({V, IMG, IMG, C}, DLDataType{kDLUInt, 8, 1}, cuda_dev);
    auto dev_tokens = Tensor::Empty({TOK}, DLDataType{kDLInt, 32, 1}, cuda_dev);
    auto dev_prefix_rope = Tensor::Empty({P, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    auto dev_suffix_rope = Tensor::Empty({T, HD}, DLDataType{kDLBfloat, 16, 1}, cuda_dev);
    cuda_api->CopyDataFromTo(host_actions->dl(), dev_actions->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_images->dl(), dev_images->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_tokens->dl(), dev_tokens->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_prefix_rope->dl(), dev_prefix_rope->dl(), nullptr);
    cuda_api->CopyDataFromTo(host_suffix_rope->dl(), dev_suffix_rope->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    auto session = ModelSession::LoadArtifact(artifact_dir);
    void* vm_stream = session.GetDefaultStream(cuda_dev);
    VMValue result = session.Invoke("main", {
        VMValue::ObjRef(dev_actions),
        VMValue::ObjRef(dev_images),
        VMValue::ObjRef(dev_tokens),
        VMValue::Int(P),
        VMValue::ObjRef(dev_prefix_rope),
        VMValue::ObjRef(dev_suffix_rope),
    });
    auto* out = result.AsObjectAs<TensorObj>();
    CHECK(out != nullptr);
    cuda_api->CopyDataFromTo(out->dl(), host_actions->dl(), vm_stream);
    cuda_api->StreamSync(cuda_dev, vm_stream);

    double abs_max = 0.0;
    double abs_mean = diff_abs_mean(actions, final_target, &abs_max);
    std::cout << "tokenizer_sample_tokens final_abs_max=" << abs_max
              << " final_abs_mean=" << abs_mean << "\n";
    CHECK(abs_max < 0.25);
    CHECK(abs_mean < 0.04);
}

}  // namespace

int main() {
    RUN(test_sample_tokens_consumes_runtime_tokenizer_output);
    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
