#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "device_api.h"
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
};

// ── Builtin registry ──────────────────────────────────────────────────────────

using BuiltinFn = std::function<VMValue(std::vector<VMValue>&)>;

class BuiltinRegistry {
public:
    static BuiltinRegistry& Global() {
        static BuiltinRegistry instance;
        return instance;
    }

    void Register(const std::string& name, BuiltinFn fn) {
        fns_[name] = std::move(fn);
    }

    BuiltinFn Get(const std::string& name) const {
        auto it = fns_.find(name);
        return (it != fns_.end()) ? it->second : BuiltinFn{};
    }

    bool Has(const std::string& name) const {
        return fns_.count(name) > 0;
    }

private:
    std::unordered_map<std::string, BuiltinFn> fns_;
};

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

private:
    std::shared_ptr<Executable> exec_;
    std::vector<VMFrame>        frames_;
    std::vector<VMValue>        regs_;

    void    PushFrame(int32_t func_idx, std::vector<VMValue>& call_args,
                      int32_t caller_dst_reg, int32_t caller_reg_base);
    VMValue ExecuteLoop();
    VMValue DispatchExternal(const FunctionEntry& callee,
                             std::vector<VMValue>& args);
};

// Registers all vm.builtin.* functions (idempotent after first call).
void RegisterVMBuiltins();

}  // namespace devproc2
