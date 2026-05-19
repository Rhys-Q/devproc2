#pragma once

#ifdef DEVPROC2_WITH_CUDA

#include <cuda_runtime.h>

#include <stdexcept>

namespace devproc2 {

#ifdef DEVPROC2_WITH_CUTLASS
bool CutlassFP8NTBF16CanRun(int m, int n, int k, float beta);
void CutlassFP8NTBF16Run(void* a,
                         void* b,
                         void* d,
                         int m,
                         int n,
                         int k,
                         float* a_scale,
                         float* b_scale,
                         float beta,
                         cudaStream_t stream);
#else
inline bool CutlassFP8NTBF16CanRun(int, int, int, float) {
    return false;
}
inline void CutlassFP8NTBF16Run(void*,
                                void*,
                                void*,
                                int,
                                int,
                                int,
                                float*,
                                float*,
                                float,
                                cudaStream_t) {
    throw std::runtime_error("Pi0.5 CUTLASS FP8 backend was not built");
}
#endif  // DEVPROC2_WITH_CUTLASS

}  // namespace devproc2

#endif  // DEVPROC2_WITH_CUDA
