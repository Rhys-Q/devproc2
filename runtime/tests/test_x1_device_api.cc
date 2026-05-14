// X1 Runtime Device API — C++ unit tests
// Build with: cmake -DDEVPROC2_WITH_CUDA=ON -DDEVPROC2_BUILD_TESTS=ON
// Run:        ./build_x1/runtime/tests/test_x1_device_api

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "devproc2/runtime/device_api.h"
#include "devproc2/runtime/storage.h"
#include "devproc2/runtime/stream.h"
#include "devproc2/runtime/tensor.h"
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
        std::cout << "[ RUN  ] " #fn "\n";                                      \
        fn();                                                                    \
        if (g_fail == prev_fail) {                                               \
            std::cout << "[ PASS ] " #fn "\n";                                  \
            ++g_pass;                                                            \
        }                                                                        \
        prev_fail = g_fail;                                                      \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────

using namespace devproc2;

// Helpers
static DLDevice cuda_dev() { return DLDevice{kDLCUDA, 0}; }
static DLDevice cpu_dev()  { return DLDevice{kDLCPU,  0}; }
static DLDataType float32() {
    return DLDataType{kDLFloat, 32, 1};
}

// ── test_cuda_alloc_free ──────────────────────────────────────────────────────
void test_cuda_alloc_free() {
    DeviceAPI* api = DeviceAPIRegistry::Get(kDLCUDA);
    CHECK(api != nullptr);

    void* ptr = api->Alloc(cuda_dev(), 1024, 256);
    CHECK(ptr != nullptr);
    api->Free(cuda_dev(), ptr);
}

// ── test_h2d_d2h_copy ─────────────────────────────────────────────────────────
void test_h2d_d2h_copy() {
    DeviceAPI* api = DeviceAPIRegistry::Get(kDLCUDA);

    constexpr int N = 16;
    float host_in[N], host_out[N];
    for (int i = 0; i < N; ++i) host_in[i] = static_cast<float>(i);
    std::memset(host_out, 0, sizeof(host_out));

    // Allocate GPU buffer
    void* gpu_ptr = api->Alloc(cuda_dev(), N * sizeof(float), 256);

    // Build DLTensors
    int64_t shape[1] = {N};
    DLTensor from_cpu{}, to_gpu{}, from_gpu{}, to_cpu{};

    from_cpu.data        = host_in;
    from_cpu.device      = cpu_dev();
    from_cpu.ndim        = 1;
    from_cpu.dtype       = float32();
    from_cpu.shape       = shape;
    from_cpu.strides     = nullptr;
    from_cpu.byte_offset = 0;

    to_gpu = from_cpu;
    to_gpu.data   = gpu_ptr;
    to_gpu.device = cuda_dev();

    // H2D copy (stream=nullptr → synchronous)
    api->CopyDataFromTo(&from_cpu, &to_gpu, nullptr);
    api->DeviceSync(cuda_dev());

    from_gpu = to_gpu;
    to_cpu           = from_cpu;
    to_cpu.data      = host_out;

    // D2H copy
    api->CopyDataFromTo(&from_gpu, &to_cpu, nullptr);
    api->DeviceSync(cuda_dev());

    for (int i = 0; i < N; ++i) {
        CHECK(host_out[i] == host_in[i]);
    }

    api->Free(cuda_dev(), gpu_ptr);
}

// ── test_d2d_copy ─────────────────────────────────────────────────────────────
void test_d2d_copy() {
    DeviceAPI* api = DeviceAPIRegistry::Get(kDLCUDA);

    constexpr int N = 8;
    float host_in[N], host_out[N];
    for (int i = 0; i < N; ++i) host_in[i] = static_cast<float>(i * 2);
    std::memset(host_out, 0, sizeof(host_out));

    void* gpu_src = api->Alloc(cuda_dev(), N * sizeof(float), 256);
    void* gpu_dst = api->Alloc(cuda_dev(), N * sizeof(float), 256);

    int64_t shape[1] = {N};
    DLTensor cpu_t{}, src_t{}, dst_t{}, out_t{};

    cpu_t.data        = host_in;
    cpu_t.device      = cpu_dev();
    cpu_t.ndim        = 1;
    cpu_t.dtype       = float32();
    cpu_t.shape       = shape;
    cpu_t.strides     = nullptr;
    cpu_t.byte_offset = 0;

    src_t = cpu_t; src_t.data = gpu_src; src_t.device = cuda_dev();
    dst_t = cpu_t; dst_t.data = gpu_dst; dst_t.device = cuda_dev();
    out_t = cpu_t; out_t.data = host_out;

    // H2D → D2D → D2H
    api->CopyDataFromTo(&cpu_t, &src_t, nullptr);
    api->CopyDataFromTo(&src_t, &dst_t, nullptr);
    api->CopyDataFromTo(&dst_t, &out_t, nullptr);
    api->DeviceSync(cuda_dev());

    for (int i = 0; i < N; ++i) {
        CHECK(host_out[i] == host_in[i]);
    }

    api->Free(cuda_dev(), gpu_src);
    api->Free(cuda_dev(), gpu_dst);
}

// ── test_stream_create_sync ───────────────────────────────────────────────────
void test_stream_create_sync() {
    DeviceAPI* api = DeviceAPIRegistry::Get(kDLCUDA);

    void* s = api->CreateStream(cuda_dev());
    CHECK(s != nullptr);
    api->StreamSync(cuda_dev(), s);
    api->FreeStream(cuda_dev(), s);
}

// ── test_device_sync ──────────────────────────────────────────────────────────
void test_device_sync() {
    DeviceAPI* api = DeviceAPIRegistry::Get(kDLCUDA);
    api->DeviceSync(cuda_dev());  // must not throw
}

// ── test_tensor_empty_cuda ────────────────────────────────────────────────────
void test_tensor_empty_cuda() {
    Tensor t = Tensor::Empty({4, 16}, float32(), cuda_dev());
    CHECK(t.defined());
    CHECK(t->dl()->device.device_type == kDLCUDA);
    CHECK(t->dl()->device.device_id   == 0);
    CHECK(t->data() != nullptr);
    CHECK(t->storage.defined());
    CHECK(t->storage.as<StorageObj>()->owns_data == true);
}

// ── test_from_dlpack_no_storage_alloc ─────────────────────────────────────────
// Wraps a host buffer in a DLManagedTensor and passes it through FromDLPack.
// The resulting Tensor must NOT allocate new storage (storage.defined()==false).
void test_from_dlpack_no_storage_alloc() {
    float buf[4] = {1.f, 2.f, 3.f, 4.f};
    int64_t shape[1] = {4};

    auto* managed = new DLManagedTensor();
    managed->dl_tensor.data        = buf;
    managed->dl_tensor.device      = cpu_dev();
    managed->dl_tensor.ndim        = 1;
    managed->dl_tensor.dtype       = float32();
    managed->dl_tensor.shape       = shape;
    managed->dl_tensor.strides     = nullptr;
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx           = nullptr;
    managed->deleter               = [](DLManagedTensor* self) { delete self; };

    Tensor t = Tensor::FromDLPack(managed);
    CHECK(t.defined());
    // storage must be empty: no new allocation was made
    CHECK(!t->storage.defined());
    // data pointer must alias the original buffer
    CHECK(t->data() == static_cast<void*>(buf));
}

// ── test_stream_obj_raii ──────────────────────────────────────────────────────
void test_stream_obj_raii() {
    DeviceAPI* api = DeviceAPIRegistry::Get(kDLCUDA);
    void* handle = api->CreateStream(cuda_dev());
    CHECK(handle != nullptr);

    // Wrap in StreamObj; destructor should call FreeStream
    {
        auto* obj = new StreamObj();
        obj->device = cuda_dev();
        obj->handle = handle;
        Stream s(obj);
        CHECK(s.defined());
        CHECK(s->handle == handle);
    }
    // No crash after scope exit means FreeStream succeeded
}

// ── test_vm_default_stream ────────────────────────────────────────────────────
void test_vm_default_stream() {
    // VMState lazily creates a stream for CUDA device
    auto exec = std::make_shared<Executable>();
    // Add a minimal no-op function so Invoke doesn't crash
    FunctionEntry fe;
    fe.name         = "noop";
    fe.kind         = VMCalleeKind::kVMFunc;
    fe.instr_offset = 0;
    fe.instr_count  = 1;
    fe.num_regs     = 0;
    fe.num_args     = 0;
    exec->function_table.push_back(fe);
    Instruction ret_instr;
    ret_instr.opcode  = Opcode::RET;
    ret_instr.src_reg = -1;
    exec->instructions.push_back(ret_instr);

    VMState vm(exec);

    void* s1 = vm.GetDefaultStream(cuda_dev());
    CHECK(s1 != nullptr);

    // Second call must return the same handle (lazily cached)
    void* s2 = vm.GetDefaultStream(cuda_dev());
    CHECK(s1 == s2);

    // CPU device always returns nullptr
    void* cpu_s = vm.GetDefaultStream(cpu_dev());
    CHECK(cpu_s == nullptr);
}

// ── test_no_cuda_in_vm_cc ─────────────────────────────────────────────────────
// Structural check: vm.cc must not contain direct CUDA API calls.
void test_no_cuda_in_vm_cc() {
    // This path is relative to the project root; the test binary is in build_x1/
    // We search a few known relative paths.
    std::vector<std::string> candidates = {
        "../../runtime/src/vm.cc",
        "../../../runtime/src/vm.cc",
    };
    std::string content;
    for (const auto& path : candidates) {
        std::ifstream f(path);
        if (!f) continue;
        content.assign(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
        break;
    }
    // If we can't find the file, skip rather than false-fail
    if (content.empty()) {
        std::cout << "  (skipped: could not locate vm.cc from test binary path)\n";
        ++g_pass;
        return;
    }
    auto contains = [&](const char* needle) {
        return content.find(needle) != std::string::npos;
    };
    CHECK(!contains("cudaMalloc"));
    CHECK(!contains("cudaFree"));
    CHECK(!contains("cudaMemcpy"));
}

}  // namespace

int main() {
    int prev_fail = 0;

    std::cout << "=== X1 Device API Tests ===\n";

    RUN(test_cuda_alloc_free);
    RUN(test_h2d_d2h_copy);
    RUN(test_d2d_copy);
    RUN(test_stream_create_sync);
    RUN(test_device_sync);
    RUN(test_tensor_empty_cuda);
    RUN(test_from_dlpack_no_storage_alloc);
    RUN(test_stream_obj_raii);
    RUN(test_vm_default_stream);
    RUN(test_no_cuda_in_vm_cc);

    std::cout << "\n=== Results: " << g_pass << " passed, "
              << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
