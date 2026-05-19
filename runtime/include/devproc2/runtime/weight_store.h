#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"

namespace devproc2 {

struct WeightInfo {
    std::string name;
    size_t offset{0};
    size_t nbytes{0};
    std::vector<int64_t> shape;
    std::string dtype;
    size_t alignment{256};
};

class WeightStore {
public:
    static std::shared_ptr<WeightStore> Load(const std::string& directory);

    bool Has(const std::string& name) const;
    const WeightInfo& Info(const std::string& name) const;
    Tensor GetTensor(const std::string& name) const;
    Tensor GetTensorOnDevice(const std::string& name, DLDevice device) const;
    size_t Count() const { return infos_.size(); }
    size_t DataBytes() const { return data_.size(); }

private:
    std::vector<uint8_t> data_;
    std::vector<WeightInfo> infos_;
    std::unordered_map<std::string, size_t> index_;
    std::unordered_map<std::string, Tensor> tensors_;
    mutable std::mutex device_cache_mu_;
    mutable std::unordered_map<std::string, Tensor> device_tensors_;
};

}  // namespace devproc2
