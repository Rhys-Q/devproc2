#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "object.h"
#include "object_ref.h"

namespace devproc2 {

class KernelObj : public Object {
public:
    static constexpr const char* _type_key = "runtime.Kernel";
    const char* type_key() const override { return _type_key; }

    std::string name;
    std::string func_name;
    std::vector<uint8_t>       cubin_data;
    std::vector<uint8_t>       ptx_data;
    std::array<int32_t, 3>     block_dims{1, 1, 1};
    int32_t                    smem_bytes{0};   // dynamic shared memory for cuLaunchKernel
    int32_t                    num_warps{4};
    int32_t                    num_stages{3};
};

class Kernel : public ObjectRef {
public:
    DEVPROC2_DEFINE_OBJECT_REF_METHODS(Kernel, KernelObj)
};

}  // namespace devproc2
