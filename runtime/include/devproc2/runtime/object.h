#pragma once

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>

#define DEVPROC2_DCHECK(cond)                                                    \
    do {                                                                         \
        if (!(cond)) {                                                           \
            throw std::runtime_error(                                            \
                std::string("DCHECK failed: " #cond " [") +                     \
                __FILE__ + ":" + std::to_string(__LINE__) + "]");               \
        }                                                                        \
    } while (0)

namespace devproc2 {

class Object {
public:
    virtual ~Object() = default;
    virtual const char* type_key() const = 0;

    void IncRef() {
        ref_count_.fetch_add(1, std::memory_order_relaxed);
    }

    void DecRef() {
        if (ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }

    int32_t use_count() const {
        return ref_count_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<int32_t> ref_count_{0};
};

}  // namespace devproc2
