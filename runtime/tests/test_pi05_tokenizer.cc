// Pi0.5 tokenizer-cpp runtime unit tests.

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

#include "devproc2/runtime/packed_func.h"
#include "devproc2/runtime/string.h"
#include "devproc2/runtime/tensor.h"
#include "devproc2/runtime/tokenizer.h"
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

static Tensor make_cpu_i32_tensor(std::vector<int32_t>& storage) {
    return Tensor::FromExternalBuffer(
        storage.data(),
        DLDevice{kDLCPU, 0},
        {static_cast<int64_t>(storage.size())},
        DLDataType{kDLInt, 32, 1});
}

static Tensor make_cpu_f32_tensor(std::vector<float>& storage) {
    return Tensor::FromExternalBuffer(
        storage.data(),
        DLDevice{kDLCPU, 0},
        {static_cast<int64_t>(storage.size())},
        DLDataType{kDLFloat, 32, 1});
}

static void call_encode(const std::string& func_name,
                        const std::string& prompt,
                        std::vector<int32_t>& tokens,
                        std::vector<int32_t>& mask) {
    auto pf = PackedFuncRegistry::Global().Get(func_name);
    CHECK(pf.defined());

    auto prompt_obj = String::Make(prompt);
    auto tokens_obj = make_cpu_i32_tensor(tokens);
    auto mask_obj = make_cpu_i32_tensor(mask);

    std::vector<VMValue> args;
    args.push_back(VMValue::ObjRef(prompt_obj));
    args.push_back(VMValue::ObjRef(tokens_obj));
    args.push_back(VMValue::ObjRef(mask_obj));
    PackedArgs packed(args);
    pf->Call(packed);
}

static void call_pi05_encode(const std::string& prompt,
                             std::vector<float>& state,
                             std::vector<int32_t>& tokens,
                             std::vector<int32_t>& mask) {
    auto pf = PackedFuncRegistry::Global().Get("runtime.tokenizer.paligemma_pi05_encode");
    CHECK(pf.defined());

    auto prompt_obj = String::Make(prompt);
    auto state_obj = make_cpu_f32_tensor(state);
    auto tokens_obj = make_cpu_i32_tensor(tokens);
    auto mask_obj = make_cpu_i32_tensor(mask);

    std::vector<VMValue> args;
    args.push_back(VMValue::ObjRef(prompt_obj));
    args.push_back(VMValue::ObjRef(state_obj));
    args.push_back(VMValue::ObjRef(tokens_obj));
    args.push_back(VMValue::ObjRef(mask_obj));
    PackedArgs packed(args);
    pf->Call(packed);
}

void test_paligemma_encode_matches_openpi_prompt_tokens() {
    SetPaligemmaTokenizerModelPath(
        "/root/autodl-tmp/openpi/outputs/pi05_torch_infer/tokenizer.model");
    RegisterTokenizerPackedFuncs();

    std::vector<int32_t> tokens(16, -1);
    std::vector<int32_t> mask(16, -1);
    call_encode("runtime.tokenizer.paligemma_encode",
                "put the object into the container 0",
                tokens,
                mask);

    const std::vector<int32_t> expected = {
        2, 1065, 573, 4018, 1280, 573, 11254, 235248, 235276, 108};
    for (size_t i = 0; i < tokens.size(); ++i) {
        const int32_t want_token = i < expected.size() ? expected[i] : 0;
        const int32_t want_mask = i < expected.size() ? 1 : 0;
        CHECK(tokens[i] == want_token);
        CHECK(mask[i] == want_mask);
    }
}

void test_paligemma_pi05_encode_matches_openpi_state_prompt_tokens() {
    SetPaligemmaTokenizerModelPath(
        "/root/autodl-tmp/openpi/outputs/pi05_torch_infer/tokenizer.model");
    RegisterTokenizerPackedFuncs();

    std::vector<float> state = {
        -0.7178650498390198f, -0.9790545701980591f, 0.34271565079689026f,
        0.932330310344696f, 0.48586392402648926f, -0.5836233496665955f,
        0.1935804933309555f, -0.18342138826847076f, -0.4397892355918884f,
        -0.6204165816307068f, -0.6424250602722168f, 0.33827054500579834f,
        0.6963333487510681f, -0.6869465708732605f, -0.13210360705852509f,
        0.4198257327079773f, -0.9982275366783142f, -0.986490786075592f,
        0.2509860694408417f, -0.29686012864112854f, -0.09493275731801987f,
        -0.4587836265563965f, 0.19726908206939697f, 0.8774058222770691f,
        -0.8926092982292175f, -0.6363763809204102f, -0.7286711931228638f,
        -0.8714727163314819f, 0.1391676515340805f, -0.6444072723388672f,
        0.4783073961734772f, -0.5607708692550659f,
    };
    std::vector<int32_t> tokens(200, -1);
    std::vector<int32_t> mask(200, -1);
    call_pi05_encode("put the object into the container 0", state, tokens, mask);

    const std::vector<int32_t> expected = {
        2, 7071, 235292, 2507, 573, 4018, 1280, 573, 11254, 235248,
        235276, 235269, 3040, 235292, 235248, 235304, 235318, 235248,
        235284, 235248, 235274, 235324, 235274, 235248, 235284, 235310,
        235324, 235248, 235274, 235315, 235276, 235248, 235308, 235304,
        235248, 235274, 235308, 235284, 235248, 235274, 235276, 235310,
        235248, 235324, 235274, 235248, 235310, 235321, 235248, 235310,
        235308, 235248, 235274, 235324, 235274, 235248, 235284, 235274,
        235324, 235248, 235310, 235276, 235248, 235274, 235274, 235274,
        235248, 235274, 235321, 235274, 235248, 235276, 235248, 235274,
        235248, 235274, 235318, 235276, 235248, 235315, 235276, 235248,
        235274, 235274, 235308, 235248, 235318, 235315, 235248, 235274,
        235308, 235304, 235248, 235284, 235310, 235276, 235248, 235274,
        235304, 235248, 235310, 235318, 235248, 235304, 235310, 235248,
        235274, 235318, 235248, 235274, 235310, 235308, 235248, 235310,
        235308, 235248, 235274, 235321, 235315, 235248, 235308, 235318,
        235289, 108, 4022, 235292, 235248,
    };
    for (size_t i = 0; i < tokens.size(); ++i) {
        const int32_t want_token = i < expected.size() ? expected[i] : 0;
        const int32_t want_mask = i < expected.size() ? 1 : 0;
        CHECK(tokens[i] == want_token);
        CHECK(mask[i] == want_mask);
    }
}

void test_tokenizer_encode_alias_is_registered() {
    RegisterTokenizerPackedFuncs();
    CHECK(PackedFuncRegistry::Global().Has("runtime.tokenizer.encode"));
    CHECK(PackedFuncRegistry::Global().Has("runtime.tokenizer.paligemma_encode"));
    CHECK(PackedFuncRegistry::Global().Has("runtime.tokenizer.paligemma_pi05_encode"));
}

}  // namespace

int main() {
    RUN(test_tokenizer_encode_alias_is_registered);
    RUN(test_paligemma_encode_matches_openpi_prompt_tokens);
    RUN(test_paligemma_pi05_encode_matches_openpi_state_prompt_tokens);

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
