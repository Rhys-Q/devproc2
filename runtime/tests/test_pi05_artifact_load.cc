// Pi0.5 artifact runtime-load tests.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "devproc2/runtime/cuda_kernel_registry.h"
#include "devproc2/runtime/vm.h"

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

static std::vector<uint8_t> make_minimal_vm_bytes() {
    std::vector<uint8_t> buf;
    buf.insert(buf.end(), {'D', 'V', '2', 'E'});
    uint32_t values[4] = {3, 0, 0, 0};
    for (uint32_t x : values) {
        buf.push_back(static_cast<uint8_t>(x & 0xFF));
        buf.push_back(static_cast<uint8_t>((x >> 8) & 0xFF));
        buf.push_back(static_cast<uint8_t>((x >> 16) & 0xFF));
        buf.push_back(static_cast<uint8_t>((x >> 24) & 0xFF));
    }
    return buf;
}

static void write_text(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}

static void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

static std::string make_artifact_dir() {
    std::string dir = "/tmp/devproc2_pi05_artifact_load_"
                      + std::to_string(getpid());
    std::filesystem::create_directories(dir + "/metadata");
    std::filesystem::create_directories(dir + "/kernels");
    std::filesystem::create_directories(dir + "/weights");
    std::filesystem::create_directories(dir + "/resources");

    write_bytes(dir + "/executable.vm", make_minimal_vm_bytes());
    write_text(dir + "/abi.json",
        "{\n"
        "  \"devproc_abi_version\": \"0.1\",\n"
        "  \"required_packed_funcs\": []\n"
        "}\n");
    write_bytes(dir + "/weights/weights.bin", {1, 2, 3, 4});
    write_text(dir + "/weights/weights.index.json",
        "{\n"
        "  \"format_version\": 1,\n"
        "  \"data_file\": \"weights.bin\",\n"
        "  \"entries\": [\n"
        "    {\"name\": \"tiny.weight\", \"offset\": 0, \"nbytes\": 4,\n"
        "     \"shape\": [4], \"dtype\": \"uint8\", \"alignment\": 1}\n"
        "  ]\n"
        "}\n");
    write_bytes(dir + "/kernels/fake.cubin", {0x7f, 'E', 'L', 'F'});
    write_text(dir + "/metadata/kernel_table.json",
        "[\n"
        "  {\n"
        "    \"name\": \"kernel.pi05_fake\",\n"
        "    \"backend\": \"cuda\",\n"
        "    \"symbol\": \"pi05_fake\",\n"
        "    \"cubin\": \"kernels/fake.cubin\",\n"
        "    \"launch\": {\"grid\": [1, 1, 1], \"block\": [32, 1, 1],\n"
        "                \"shared_memory_bytes\": 0}\n"
        "  }\n"
        "]\n");
    write_bytes(dir + "/resources/tokenizer.model", {0, 1, 2, 3});
    return dir;
}

void test_load_registers_weights_and_cuda_kernel_table() {
    std::string dir = make_artifact_dir();
    auto exe = Executable::Load(dir);

    CHECK(exe != nullptr);
    CHECK(exe->weights != nullptr);
    CHECK(exe->weights->Count() == 1);
    CHECK(exe->weights->Has("tiny.weight"));
    CHECK(CUDAKernelRegistry::Global().Has("kernel.pi05_fake"));
}

void test_model_session_loads_artifact_resources() {
    std::string dir = make_artifact_dir();
    auto session = ModelSession::LoadArtifact(dir);

    CHECK(session.executable() != nullptr);
    CHECK(session.weights() != nullptr);
    CHECK(session.weights()->Has("tiny.weight"));
    CHECK(CUDAKernelRegistry::Global().Has("kernel.pi05_fake"));
}

}  // namespace

int main() {
    RUN(test_load_registers_weights_and_cuda_kernel_table);
    RUN(test_model_session_loads_artifact_resources);

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
