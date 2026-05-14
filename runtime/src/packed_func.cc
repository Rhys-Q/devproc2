#include <devproc2/runtime/packed_func.h>
#include <mutex>

namespace devproc2 {

PackedFuncRegistry& PackedFuncRegistry::Global() {
    static PackedFuncRegistry instance;
    return instance;
}

void PackedFuncRegistry::Register(const std::string& name, PackedFunc func) {
    std::lock_guard<std::mutex> lock(mu_);
    registry_[name] = std::move(func);
}

PackedFunc PackedFuncRegistry::Get(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = registry_.find(name);
    return (it != registry_.end()) ? it->second : PackedFunc{};
}

bool PackedFuncRegistry::Has(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mu_);
    return registry_.count(name) > 0;
}

}  // namespace devproc2
