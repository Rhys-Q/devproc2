#include "devproc2/runtime/weight_store.h"

#include "devproc2/runtime/device_api.h"

#include <fstream>
#include <iterator>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace devproc2 {
namespace {

std::string ReadText(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

std::vector<uint8_t> ReadBinary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    auto size = f.tellg();
    if (size < 0) {
        throw std::runtime_error("Cannot determine file size: " + path);
    }
    std::vector<uint8_t> data(static_cast<size_t>(size));
    f.seekg(0);
    if (!data.empty()) {
        f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!f) {
            throw std::runtime_error("Failed to read file: " + path);
        }
    }
    return data;
}

bool Exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

DLDataType ParseWeightDType(const std::string& dtype) {
    if (dtype == "float16") return DLDataType{kDLFloat, 16, 1};
    if (dtype == "float32") return DLDataType{kDLFloat, 32, 1};
    if (dtype == "bfloat16") return DLDataType{kDLBfloat, 16, 1};
    if (dtype == "fp8_e4m3") return DLDataType{kDLFloat8_e4m3, 8, 1};
    if (dtype == "uint8") return DLDataType{kDLUInt, 8, 1};
    if (dtype == "int32") return DLDataType{kDLInt, 32, 1};
    throw std::runtime_error("Unsupported weight dtype: " + dtype);
}

std::string ResolveWeightDir(const std::string& directory) {
    if (Exists(directory + "/weights.index.json")) {
        return directory;
    }
    if (Exists(directory + "/weights/weights.index.json")) {
        return directory + "/weights";
    }
    throw std::runtime_error("weights.index.json not found under " + directory);
}

std::string DeviceCacheKey(const std::string& name, DLDevice device) {
    return name + "@" + std::to_string(device.device_type) + ":" +
           std::to_string(device.device_id);
}

}  // namespace

std::shared_ptr<WeightStore> WeightStore::Load(const std::string& directory) {
    std::string weight_dir = ResolveWeightDir(directory);
    auto index_json = nlohmann::json::parse(ReadText(weight_dir + "/weights.index.json"));
    std::string data_file = index_json.value("data_file", std::string("weights.bin"));

    auto store = std::shared_ptr<WeightStore>(new WeightStore());
    store->data_ = ReadBinary(weight_dir + "/" + data_file);

    const auto& entries = index_json.at("entries");
    if (!entries.is_array()) {
        throw std::runtime_error("weights.index.json: entries must be an array");
    }

    for (const auto& item : entries) {
        WeightInfo info;
        info.name = item.at("name").get<std::string>();
        info.offset = item.at("offset").get<size_t>();
        info.nbytes = item.at("nbytes").get<size_t>();
        info.dtype = item.at("dtype").get<std::string>();
        info.alignment = item.value("alignment", static_cast<size_t>(256));
        for (const auto& dim : item.at("shape")) {
            info.shape.push_back(dim.get<int64_t>());
        }
        if (info.offset % info.alignment != 0) {
            throw std::runtime_error("weight '" + info.name + "' offset is not aligned");
        }
        if (info.offset + info.nbytes > store->data_.size()) {
            throw std::runtime_error("weight '" + info.name + "' offset/nbytes out of range");
        }
        if (store->index_.count(info.name)) {
            throw std::runtime_error("duplicate weight entry: " + info.name);
        }

        void* ptr = store->data_.data() + info.offset;
        DLDevice cpu{kDLCPU, 0};
        DLDataType dtype = ParseWeightDType(info.dtype);
        Tensor tensor = Tensor::FromExternalBuffer(ptr, cpu, info.shape, dtype);
        store->index_[info.name] = store->infos_.size();
        store->infos_.push_back(std::move(info));
        store->tensors_[store->infos_.back().name] = tensor;
    }
    return store;
}

bool WeightStore::Has(const std::string& name) const {
    return index_.count(name) != 0;
}

const WeightInfo& WeightStore::Info(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) {
        throw std::runtime_error("weight not found: " + name);
    }
    return infos_[it->second];
}

Tensor WeightStore::GetTensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("weight tensor not found: " + name);
    }
    return it->second;
}

Tensor WeightStore::GetTensorOnDevice(const std::string& name, DLDevice device) const {
    Tensor cpu_tensor = GetTensor(name);
    if (device.device_type == kDLCPU) {
        return cpu_tensor;
    }

    std::lock_guard<std::mutex> lock(device_cache_mu_);
    std::string key = DeviceCacheKey(name, device);
    auto cached = device_tensors_.find(key);
    if (cached != device_tensors_.end()) {
        return cached->second;
    }

    const WeightInfo& info = Info(name);
    DLDataType dtype = ParseWeightDType(info.dtype);
    Tensor device_tensor = Tensor::Empty(info.shape, dtype, device);
    DeviceAPI* api = DeviceAPIRegistry::Get(device.device_type);
    api->SetDevice(device);
    api->CopyDataFromTo(cpu_tensor->dl(), device_tensor->dl(), nullptr);
    api->DeviceSync(device);
    device_tensors_[key] = device_tensor;
    return device_tensor;
}

}  // namespace devproc2
