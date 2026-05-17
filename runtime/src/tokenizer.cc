#include "devproc2/runtime/tokenizer.h"

#ifdef DEVPROC2_WITH_TOKENIZERS

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#include "devproc2/runtime/packed_func.h"
#include "devproc2/runtime/string.h"
#include "devproc2/runtime/tensor.h"
#include "tokenizers_cpp.h"

namespace devproc2 {
namespace {

std::mutex g_mu;
std::string g_paligemma_model_path =
    "/root/autodl-tmp/openpi/outputs/pi05_torch_infer/tokenizer.model";
std::unique_ptr<tokenizers::Tokenizer> g_paligemma;

std::string ReadFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open tokenizer model: " + path);
    }
    return std::string(
        std::istreambuf_iterator<char>(ifs),
        std::istreambuf_iterator<char>());
}

tokenizers::Tokenizer* GetPaligemmaTokenizer() {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_paligemma) {
        const char* env = std::getenv("DEVPROC2_PALIGEMMA_TOKENIZER_MODEL");
        if (env && env[0]) {
            g_paligemma_model_path = env;
        }
        g_paligemma = tokenizers::Tokenizer::FromBlobSentencePiece(
            ReadFile(g_paligemma_model_path));
        if (!g_paligemma) {
            throw std::runtime_error(
                "Failed to create PaliGemma tokenizer from " + g_paligemma_model_path);
        }
    }
    return g_paligemma.get();
}

TensorObj* RequireTensor(const VMValue& value, const char* name) {
    if (!value.IsObjectRef()) {
        throw std::runtime_error(std::string(name) + " must be a Tensor");
    }
    auto* tensor = value.AsObjectAs<TensorObj>();
    if (!tensor) {
        throw std::runtime_error(std::string(name) + " must be a Tensor");
    }
    return tensor;
}

StringObj* RequireString(const VMValue& value, const char* name) {
    if (!value.IsObjectRef()) {
        throw std::runtime_error(std::string(name) + " must be a String");
    }
    auto* str = value.AsObjectAs<StringObj>();
    if (!str) {
        throw std::runtime_error(std::string(name) + " must be a String");
    }
    return str;
}

int64_t Numel(const TensorObj* tensor) {
    int64_t n = 1;
    for (int i = 0; i < tensor->ndim(); ++i) {
        n *= tensor->shape()[i];
    }
    return n;
}

void RequireCpuInt32(const TensorObj* tensor, const char* name) {
    if (tensor->device().device_type != kDLCPU) {
        throw std::runtime_error(std::string(name) + " must be a CPU tensor");
    }
    DLDataType dtype = tensor->dtype();
    if (dtype.code != kDLInt || dtype.bits != 32 || dtype.lanes != 1) {
        throw std::runtime_error(std::string(name) + " must be int32");
    }
}

void RequireCpuFloat32(const TensorObj* tensor, const char* name) {
    if (tensor->device().device_type != kDLCPU) {
        throw std::runtime_error(std::string(name) + " must be a CPU tensor");
    }
    DLDataType dtype = tensor->dtype();
    if (dtype.code != kDLFloat || dtype.bits != 32 || dtype.lanes != 1) {
        throw std::runtime_error(std::string(name) + " must be float32");
    }
}

std::string CleanPrompt(std::string prompt) {
    const auto first = prompt.find_first_not_of(" \t\n\r\f\v");
    if (first == std::string::npos) return "";
    const auto last = prompt.find_last_not_of(" \t\n\r\f\v");
    prompt = prompt.substr(first, last - first + 1);
    for (char& ch : prompt) {
        if (ch == '_' || ch == '\n') ch = ' ';
    }
    return prompt;
}

int32_t DiscretizePi05State(float value) {
    int32_t upper = 0;
    for (; upper < 256; ++upper) {
        const float bin = -1.0f + static_cast<float>(upper) * (2.0f / 256.0f);
        if (value < bin) break;
    }
    return upper - 1;
}

void FillTokenOutputs(const std::vector<int32_t>& ids,
                      TensorObj* tokens_out,
                      TensorObj* mask_out) {
    int32_t* token_ptr = static_cast<int32_t*>(tokens_out->data());
    int32_t* mask_ptr = mask_out ? static_cast<int32_t*>(mask_out->data()) : nullptr;
    int64_t cap = Numel(tokens_out);
    for (int64_t i = 0; i < cap; ++i) {
        bool valid = i < static_cast<int64_t>(ids.size());
        token_ptr[i] = valid ? ids[static_cast<size_t>(i)] : 0;
        if (mask_ptr) {
            mask_ptr[i] = valid ? 1 : 0;
        }
    }
}

void PaligemmaEncode(PackedArgs args) {
    if (args.size() < 2 || args.size() > 3) {
        throw std::runtime_error(
            "runtime.tokenizer.paligemma_encode expects prompt, tokens_out[, mask_out]");
    }

    auto* prompt = RequireString(args[0], "prompt");
    auto* tokens_out = RequireTensor(args[1], "tokens_out");
    TensorObj* mask_out = args.size() == 3 ? RequireTensor(args[2], "mask_out") : nullptr;
    RequireCpuInt32(tokens_out, "tokens_out");
    if (mask_out) {
        RequireCpuInt32(mask_out, "mask_out");
        if (Numel(mask_out) != Numel(tokens_out)) {
            throw std::runtime_error("mask_out shape must match tokens_out shape");
        }
    }

    std::vector<int32_t> ids;
    ids.reserve(256);
    ids.push_back(2);  // PaliGemma SentencePiece BOS.
    auto encoded = GetPaligemmaTokenizer()->Encode(prompt->data);
    ids.insert(ids.end(), encoded.begin(), encoded.end());
    ids.push_back(108);  // PaliGemma newline token used by openpi prompt path.

    FillTokenOutputs(ids, tokens_out, mask_out);
}

void PaligemmaPi05Encode(PackedArgs args) {
    if (args.size() < 3 || args.size() > 4) {
        throw std::runtime_error(
            "runtime.tokenizer.paligemma_pi05_encode expects prompt, state, tokens_out[, mask_out]");
    }

    auto* prompt = RequireString(args[0], "prompt");
    auto* state = RequireTensor(args[1], "state");
    auto* tokens_out = RequireTensor(args[2], "tokens_out");
    TensorObj* mask_out = args.size() == 4 ? RequireTensor(args[3], "mask_out") : nullptr;
    RequireCpuFloat32(state, "state");
    RequireCpuInt32(tokens_out, "tokens_out");
    if (mask_out) {
        RequireCpuInt32(mask_out, "mask_out");
        if (Numel(mask_out) != Numel(tokens_out)) {
            throw std::runtime_error("mask_out shape must match tokens_out shape");
        }
    }

    const float* state_ptr = static_cast<const float*>(state->data());
    const int64_t state_elems = Numel(state);
    std::ostringstream prompt_builder;
    prompt_builder << "Task: " << CleanPrompt(prompt->data) << ", State: ";
    for (int64_t i = 0; i < state_elems; ++i) {
        if (i) prompt_builder << ' ';
        prompt_builder << DiscretizePi05State(state_ptr[i]);
    }
    prompt_builder << ";\nAction: ";

    std::vector<int32_t> ids;
    ids.reserve(256);
    ids.push_back(2);  // PaliGemma SentencePiece BOS.
    auto encoded = GetPaligemmaTokenizer()->Encode(prompt_builder.str());
    ids.insert(ids.end(), encoded.begin(), encoded.end());

    FillTokenOutputs(ids, tokens_out, mask_out);
}

}  // namespace

void SetPaligemmaTokenizerModelPath(std::string path) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_paligemma_model_path = std::move(path);
    g_paligemma.reset();
}

void RegisterTokenizerPackedFuncs() {
    auto make = []() {
        auto* obj = new PackedFuncObj();
        obj->body = [](PackedArgs args) { PaligemmaEncode(args); };
        return PackedFunc(obj);
    };
    auto make_pi05 = []() {
        auto* obj = new PackedFuncObj();
        obj->body = [](PackedArgs args) { PaligemmaPi05Encode(args); };
        return PackedFunc(obj);
    };
    PackedFuncRegistry::Global().Register(
        "runtime.tokenizer.paligemma_encode", make());
    PackedFuncRegistry::Global().Register(
        "runtime.tokenizer.paligemma_pi05_encode", make_pi05());
    PackedFuncRegistry::Global().Register(
        "runtime.tokenizer.encode", make());
}

}  // namespace devproc2

#else

namespace devproc2 {

void SetPaligemmaTokenizerModelPath(std::string) {}

void RegisterTokenizerPackedFuncs() {}

}  // namespace devproc2

#endif  // DEVPROC2_WITH_TOKENIZERS
