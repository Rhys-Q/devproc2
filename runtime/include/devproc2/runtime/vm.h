#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "device_api.h"
#include "stream.h"
#include "vm_value.h"

namespace devproc2 {

// ── Opcode ────────────────────────────────────────────────────────────────────

enum class Opcode : uint8_t { CALL = 0, RET = 1, IF = 2, GOTO = 3 };

enum class VMCalleeKind : uint8_t {
    kVMFunc     = 0,
    kBuiltin    = 1,
    kPackedFunc = 2,
    kKernel     = 3,
};

// ── Instruction ───────────────────────────────────────────────────────────────

struct Instruction {
    Opcode   opcode;

    // CALL
    int32_t dst_reg  = -1;
    int32_t func_idx = 0;
    // RET
    int32_t src_reg  = -1;
    // IF
    int32_t cond_reg     = 0;
    int32_t true_offset  = 0;
    int32_t false_offset = 0;
    // GOTO
    int32_t offset = 0;
    // CALL arg registers
    std::vector<int32_t> arg_regs;
};

// ── FunctionEntry ─────────────────────────────────────────────────────────────

struct ConstInit {
    int32_t reg_idx;
    int32_t const_idx;
};

struct FunctionEntry {
    std::string     name;
    VMCalleeKind    kind;
    int32_t         instr_offset;   // -1 for external callees
    int32_t         instr_count;
    int32_t         num_regs;
    int32_t         num_args;
    std::vector<ConstInit> const_inits;
};

// ── Executable ────────────────────────────────────────────────────────────────

class Executable {
public:
    std::vector<FunctionEntry> function_table;
    std::vector<Instruction>   instructions;
    std::vector<VMValue>       constants;

    int32_t GetFuncIndex(const std::string& name) const {
        for (int32_t i = 0; i < static_cast<int32_t>(function_table.size()); ++i) {
            if (function_table[i].name == name) return i;
        }
        throw std::runtime_error("Function '" + name + "' not found in Executable");
    }

    // Deserialize from raw binary bytes (produced by Python serializer.serialize()).
    static std::shared_ptr<Executable> Deserialize(const uint8_t* data, size_t size);

    // Load artifact from directory: deserializes executable.vm, validates abi.json,
    // and checks that all required packed funcs are registered.
    static std::shared_ptr<Executable> Load(const std::string& artifact_dir);
};

// ── Builtin registry ──────────────────────────────────────────────────────────

using BuiltinFn = std::function<VMValue(std::vector<VMValue>&)>;

class BuiltinRegistry;

// Represents a single registered builtin; supports chained .set_body().
class BuiltinEntry {
public:
    BuiltinEntry& set_body(BuiltinFn fn) {
        fn_ = std::move(fn);
        return *this;
    }
    const BuiltinFn& body() const { return fn_; }

private:
    friend class BuiltinRegistry;
    explicit BuiltinEntry(std::string name) : name_(std::move(name)) {}
    std::string name_;
    BuiltinFn   fn_;
};

class BuiltinRegistry {
public:
    static BuiltinRegistry& Global() {
        static BuiltinRegistry instance;
        return instance;
    }

    // Insert-or-get entry by name; returns a stable reference.
    static BuiltinEntry& Register(const std::string& name) {
        auto& self = Global();
        std::lock_guard<std::mutex> lock(self.mu_);
        auto& slot = self.entries_[name];
        if (!slot) slot.reset(new BuiltinEntry(name));
        return *slot;
    }

    BuiltinFn Get(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = entries_.find(name);
        return (it != entries_.end()) ? it->second->body() : BuiltinFn{};
    }

    bool Has(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mu_);
        return entries_.count(name) > 0;
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, std::unique_ptr<BuiltinEntry>> entries_;
};

// Self-registration macro (static-initializer pattern, like TVM_REGISTER_GLOBAL).
// Usage:
//   DEVPROC2_REGISTER_BUILTIN("vm.builtin.foo")
//       .set_body([](std::vector<VMValue>& args) -> VMValue { ... });
#define DEVPROC2_CONCAT_(x, y) x##y
#define DEVPROC2_CONCAT(x, y)  DEVPROC2_CONCAT_(x, y)
#define DEVPROC2_REGISTER_BUILTIN(name)                                 \
    [[maybe_unused]] static ::devproc2::BuiltinEntry&                   \
        DEVPROC2_CONCAT(__devproc2_builtin_, __COUNTER__) =             \
            ::devproc2::BuiltinRegistry::Register(name)

// ── VMFrame ───────────────────────────────────────────────────────────────────

struct VMFrame {
    int32_t func_idx;
    int32_t pc;           // instruction offset within the function
    int32_t reg_base;     // start of this frame's registers in global file
    int32_t caller_dst_reg;   // where to write the return value in caller (-1 = none)
    int32_t caller_reg_base;  // caller's reg_base (-1 = top-level call)
};

// ── VMState ───────────────────────────────────────────────────────────────────

class VMState {
public:
    explicit VMState(std::shared_ptr<Executable> exec);

    // Execute func_name with args; return the result VMValue.
    VMValue Invoke(const std::string& func_name, std::vector<VMValue> args);

    // Return the default stream for the given device (lazily created).
    // Returns nullptr for CPU devices.
    void* GetDefaultStream(const Device& dev);

private:
    std::shared_ptr<Executable> exec_;
    std::vector<VMFrame>        frames_;
    std::vector<VMValue>        regs_;

    // Per-device default streams; created on first use.
    std::unordered_map<Device, Stream, DeviceHash, DeviceEqual> default_streams_;

    void    PushFrame(int32_t func_idx, std::vector<VMValue>& call_args,
                      int32_t caller_dst_reg, int32_t caller_reg_base);
    VMValue ExecuteLoop();
    VMValue DispatchExternal(const FunctionEntry& callee,
                             std::vector<VMValue>& args);
};


}  // namespace devproc2
