// Pi0.5 WeightStore C++ unit tests.

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include "devproc2/runtime/device_api.h"
#include "devproc2/runtime/weight_store.h"

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

#define CHECK_THROWS_MSG(expr, substr)                                           \
    do {                                                                         \
        bool caught = false;                                                     \
        std::string msg;                                                         \
        try { (expr); }                                                          \
        catch (const std::exception& _e) { caught = true; msg = _e.what(); }    \
        if (!caught) {                                                           \
            std::cerr << "  FAIL: expected exception not thrown\n";             \
            ++g_fail; return;                                                    \
        }                                                                        \
        if (msg.find(substr) == std::string::npos) {                            \
            std::cerr << "  FAIL: exception msg '" << msg                       \
                      << "' does not contain '" << substr << "'\n";             \
            ++g_fail; return;                                                    \
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

static void write_text(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}

static void write_bytes(const std::string& path, size_t nbytes) {
    std::ofstream f(path, std::ios::binary);
    for (size_t i = 0; i < nbytes; ++i) {
        char c = static_cast<char>(i & 0xff);
        f.write(&c, 1);
    }
}

static std::string make_dir(const std::string& suffix) {
    std::string dir = "/tmp/devproc2_weight_store_" + suffix + "_" + std::to_string(getpid());
    std::filesystem::create_directories(dir);
    return dir;
}

void test_load_weight_store() {
    std::string dir = make_dir("valid");
    write_bytes(dir + "/weights.bin", 1024);
    write_text(dir + "/weights.index.json",
        "{\n"
        "  \"format_version\": 1,\n"
        "  \"data_file\": \"weights.bin\",\n"
        "  \"entries\": [\n"
        "    {\"name\":\"fp8.w.scale\",\"offset\":0,\"nbytes\":4,\"shape\":[1],\"dtype\":\"float32\",\"alignment\":256},\n"
        "    {\"name\":\"fp8.w.weight\",\"offset\":256,\"nbytes\":16,\"shape\":[4,4],\"dtype\":\"fp8_e4m3\",\"alignment\":256}\n"
        "  ]\n"
        "}\n");

    auto store = WeightStore::Load(dir);
    CHECK(store->Count() == 2);
    CHECK(store->DataBytes() == 1024);
    CHECK(store->Has("fp8.w.weight"));
    const auto& info = store->Info("fp8.w.weight");
    CHECK(info.shape.size() == 2);
    CHECK(info.shape[0] == 4);
    auto tensor = store->GetTensor("fp8.w.weight");
    CHECK(tensor.defined());
    CHECK(tensor->dtype().code == kDLFloat8_e4m3);
    CHECK(tensor->shape()[1] == 4);

    auto cpu_tensor = store->GetTensorOnDevice("fp8.w.weight", DLDevice{kDLCPU, 0});
    CHECK(cpu_tensor.defined());
    CHECK(cpu_tensor->device().device_type == kDLCPU);
    CHECK(cpu_tensor->data() == tensor->data());
}

void test_reject_out_of_range() {
    std::string dir = make_dir("bad_range");
    write_bytes(dir + "/weights.bin", 128);
    write_text(dir + "/weights.index.json",
        "{\n"
        "  \"format_version\": 1,\n"
        "  \"data_file\": \"weights.bin\",\n"
        "  \"entries\": [\n"
        "    {\"name\":\"bad\",\"offset\":256,\"nbytes\":4,\"shape\":[1],\"dtype\":\"float32\",\"alignment\":256}\n"
        "  ]\n"
        "}\n");
    CHECK_THROWS_MSG(WeightStore::Load(dir), "out of range");
}

#ifdef DEVPROC2_WITH_CUDA
void test_cuda_device_cache_roundtrip() {
    std::string dir = make_dir("cuda_cache");
    write_bytes(dir + "/weights.bin", 1024);
    write_text(dir + "/weights.index.json",
        "{\n"
        "  \"format_version\": 1,\n"
        "  \"data_file\": \"weights.bin\",\n"
        "  \"entries\": [\n"
        "    {\"name\":\"fp8.w.weight\",\"offset\":256,\"nbytes\":16,\"shape\":[4,4],\"dtype\":\"fp8_e4m3\",\"alignment\":256}\n"
        "  ]\n"
        "}\n");

    auto store = WeightStore::Load(dir);
    DLDevice cuda_dev{kDLCUDA, 0};
    auto dev_tensor = store->GetTensorOnDevice("fp8.w.weight", cuda_dev);
    CHECK(dev_tensor.defined());
    CHECK(dev_tensor->device().device_type == kDLCUDA);

    auto cached_tensor = store->GetTensorOnDevice("fp8.w.weight", cuda_dev);
    CHECK(cached_tensor->data() == dev_tensor->data());

    std::vector<uint8_t> copied(16, 0);
    auto host_out = Tensor::FromExternalBuffer(
        copied.data(), DLDevice{kDLCPU, 0}, {4, 4}, DLDataType{kDLFloat8_e4m3, 8, 1});
    DeviceAPI* cuda_api = DeviceAPIRegistry::Get(kDLCUDA);
    cuda_api->CopyDataFromTo(dev_tensor->dl(), host_out->dl(), nullptr);
    cuda_api->DeviceSync(cuda_dev);

    for (size_t i = 0; i < copied.size(); ++i) {
        CHECK(copied[i] == static_cast<uint8_t>(i));
    }
}
#endif

}  // namespace

int main() {
    RUN(test_load_weight_store);
    RUN(test_reject_out_of_range);
#ifdef DEVPROC2_WITH_CUDA
    RUN(test_cuda_device_cache_roundtrip);
#endif
    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
