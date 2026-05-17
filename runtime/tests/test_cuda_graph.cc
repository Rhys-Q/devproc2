#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "devproc2/runtime/cuda_graph.h"

namespace {

void check_cuda(cudaError_t err, const char* expr) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(err));
    }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr)

void check_bytes(const std::array<uint8_t, 64>& bytes, uint8_t expected) {
    for (uint8_t value : bytes) {
        if (value != expected) {
            throw std::runtime_error("unexpected CUDA graph replay output");
        }
    }
}

}  // namespace

int main() {
    cudaStream_t stream = nullptr;
    void* device = nullptr;
    try {
        CHECK_CUDA(cudaStreamCreate(&stream));
        CHECK_CUDA(cudaMalloc(&device, 64));
        CHECK_CUDA(cudaMemsetAsync(device, 0, 64, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        auto graph = devproc2::CUDAGraphExec::Capture(
            static_cast<void*>(stream),
            [&]() {
                CHECK_CUDA(cudaMemsetAsync(device, 0x5a, 64, stream));
            });
        graph.Upload(static_cast<void*>(stream));

        std::array<uint8_t, 64> host{};
        for (int i = 0; i < 2; ++i) {
            CHECK_CUDA(cudaMemsetAsync(device, 0, 64, stream));
            graph.Launch(static_cast<void*>(stream));
            CHECK_CUDA(cudaMemcpyAsync(
                host.data(), device, host.size(), cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            check_bytes(host, 0x5a);
        }

        graph.Reset();
        CHECK_CUDA(cudaFree(device));
        device = nullptr;
        CHECK_CUDA(cudaStreamDestroy(stream));
        stream = nullptr;
    } catch (const std::exception& e) {
        if (device != nullptr) cudaFree(device);
        if (stream != nullptr) cudaStreamDestroy(stream);
        std::cerr << e.what() << "\n";
        return 1;
    }

    std::cout << "test_cuda_graph passed\n";
    return 0;
}
