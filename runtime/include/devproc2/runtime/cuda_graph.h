#pragma once

#include <functional>

namespace devproc2 {

// Lightweight RAII wrapper for a captured CUDA graph. The public surface uses
// void* streams to match DeviceAPI/VMState and keep cuda_runtime.h out of the
// common runtime headers.
class CUDAGraphExec {
public:
    CUDAGraphExec() = default;
    ~CUDAGraphExec();

    CUDAGraphExec(const CUDAGraphExec&) = delete;
    CUDAGraphExec& operator=(const CUDAGraphExec&) = delete;

    CUDAGraphExec(CUDAGraphExec&& other) noexcept;
    CUDAGraphExec& operator=(CUDAGraphExec&& other) noexcept;

    static CUDAGraphExec Capture(void* stream, const std::function<void()>& enqueue);

    bool IsValid() const { return graph_exec_ != nullptr; }
    void Upload(void* stream) const;
    void Launch(void* stream) const;
    void Reset();

private:
    CUDAGraphExec(void* graph, void* graph_exec)
        : graph_(graph), graph_exec_(graph_exec) {}

    void* graph_ = nullptr;
    void* graph_exec_ = nullptr;
};

}  // namespace devproc2
