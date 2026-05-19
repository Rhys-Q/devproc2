#include <devproc2/runtime/packed_backend.h>

namespace devproc2 {

namespace {
thread_local void* g_current_cuda_packed_stream = nullptr;
}  // namespace

void* CurrentCUDAPackedFuncStream() {
    return g_current_cuda_packed_stream;
}

void SetCUDAPackedFuncStream(void* stream) {
    g_current_cuda_packed_stream = stream;
}

}  // namespace devproc2
