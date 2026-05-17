#pragma once

namespace devproc2 {

void RegisterCUDAPackedFuncs();
void* CurrentCUDAPackedFuncStream();
void SetCUDAPackedFuncStream(void* stream);

}  // namespace devproc2
