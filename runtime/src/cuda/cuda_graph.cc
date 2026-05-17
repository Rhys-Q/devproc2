#include "devproc2/runtime/cuda_graph.h"

#include <stdexcept>
#include <string>

#ifdef DEVPROC2_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace devproc2 {

#ifdef DEVPROC2_WITH_CUDA
namespace {

void CheckCuda(cudaError_t err, const char* expr) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(err));
    }
}

void DestroyHandles(void* graph, void* graph_exec) {
    if (graph_exec != nullptr) {
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec));
    }
    if (graph != nullptr) {
        cudaGraphDestroy(static_cast<cudaGraph_t>(graph));
    }
}

}  // namespace
#endif

CUDAGraphExec::~CUDAGraphExec() {
    Reset();
}

CUDAGraphExec::CUDAGraphExec(CUDAGraphExec&& other) noexcept
    : graph_(other.graph_), graph_exec_(other.graph_exec_) {
    other.graph_ = nullptr;
    other.graph_exec_ = nullptr;
}

CUDAGraphExec& CUDAGraphExec::operator=(CUDAGraphExec&& other) noexcept {
    if (this == &other) return *this;
    Reset();
    graph_ = other.graph_;
    graph_exec_ = other.graph_exec_;
    other.graph_ = nullptr;
    other.graph_exec_ = nullptr;
    return *this;
}

CUDAGraphExec CUDAGraphExec::Capture(void* stream, const std::function<void()>& enqueue) {
#ifdef DEVPROC2_WITH_CUDA
    if (stream == nullptr) {
        throw std::runtime_error("CUDAGraphExec::Capture requires a non-null CUDA stream");
    }
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    CheckCuda(cudaStreamBeginCapture(cuda_stream, cudaStreamCaptureModeRelaxed),
              "cudaStreamBeginCapture");
    try {
        enqueue();
    } catch (...) {
        cudaGraph_t abandoned = nullptr;
        cudaStreamEndCapture(cuda_stream, &abandoned);
        if (abandoned != nullptr) cudaGraphDestroy(abandoned);
        throw;
    }

    try {
        CheckCuda(cudaStreamEndCapture(cuda_stream, &graph), "cudaStreamEndCapture");
        if (graph == nullptr) {
            throw std::runtime_error("CUDAGraphExec::Capture produced a null graph");
        }
        CheckCuda(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
                  "cudaGraphInstantiate");
        return CUDAGraphExec(static_cast<void*>(graph), static_cast<void*>(graph_exec));
    } catch (...) {
        DestroyHandles(static_cast<void*>(graph), static_cast<void*>(graph_exec));
        throw;
    }
#else
    (void)stream;
    (void)enqueue;
    throw std::runtime_error("CUDAGraphExec requires DEVPROC2_WITH_CUDA");
#endif
}

void CUDAGraphExec::Upload(void* stream) const {
#ifdef DEVPROC2_WITH_CUDA
    if (!IsValid()) {
        throw std::runtime_error("CUDAGraphExec::Upload called on an empty graph");
    }
    CheckCuda(cudaGraphUpload(static_cast<cudaGraphExec_t>(graph_exec_),
                              static_cast<cudaStream_t>(stream)),
              "cudaGraphUpload");
#else
    (void)stream;
    throw std::runtime_error("CUDAGraphExec requires DEVPROC2_WITH_CUDA");
#endif
}

void CUDAGraphExec::Launch(void* stream) const {
#ifdef DEVPROC2_WITH_CUDA
    if (!IsValid()) {
        throw std::runtime_error("CUDAGraphExec::Launch called on an empty graph");
    }
    CheckCuda(cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph_exec_),
                              static_cast<cudaStream_t>(stream)),
              "cudaGraphLaunch");
#else
    (void)stream;
    throw std::runtime_error("CUDAGraphExec requires DEVPROC2_WITH_CUDA");
#endif
}

void CUDAGraphExec::Reset() {
#ifdef DEVPROC2_WITH_CUDA
    DestroyHandles(graph_, graph_exec_);
#endif
    graph_ = nullptr;
    graph_exec_ = nullptr;
}

}  // namespace devproc2
