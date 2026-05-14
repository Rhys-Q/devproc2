#include <devproc2/runtime/vm.h>
#include <devproc2/runtime/packed_func.h>
#include <stdexcept>

namespace devproc2 {

VMState::VMState(std::shared_ptr<Executable> exec)
    : exec_(std::move(exec)) {}

VMValue VMState::Invoke(const std::string& func_name, std::vector<VMValue> args) {
    int32_t func_idx = exec_->GetFuncIndex(func_name);
    regs_.clear();
    frames_.clear();
    PushFrame(func_idx, args, /*caller_dst_reg=*/-1, /*caller_reg_base=*/-1);
    return ExecuteLoop();
}

void VMState::PushFrame(int32_t func_idx,
                        std::vector<VMValue>& call_args,
                        int32_t caller_dst_reg,
                        int32_t caller_reg_base) {
    const FunctionEntry& fe = exec_->function_table[func_idx];
    int32_t new_base = static_cast<int32_t>(regs_.size());
    regs_.resize(regs_.size() + static_cast<size_t>(fe.num_regs));

    // Copy call args into first num_args registers
    for (int32_t i = 0; i < static_cast<int32_t>(call_args.size()); ++i) {
        regs_[new_base + i] = std::move(call_args[i]);
    }
    // Apply const_inits
    for (const auto& ci : fe.const_inits) {
        regs_[new_base + ci.reg_idx] = exec_->constants[ci.const_idx];
    }
    frames_.push_back({func_idx, 0, new_base, caller_dst_reg, caller_reg_base});
}

VMValue VMState::ExecuteLoop() {
    while (!frames_.empty()) {
        VMFrame& frame = frames_.back();
        const FunctionEntry& fe = exec_->function_table[frame.func_idx];
        const Instruction& instr =
            exec_->instructions[static_cast<size_t>(fe.instr_offset + frame.pc)];

        switch (instr.opcode) {
        case Opcode::CALL: {
            const FunctionEntry& callee =
                exec_->function_table[static_cast<size_t>(instr.func_idx)];

            // Collect args from current frame's register file
            std::vector<VMValue> call_args;
            call_args.reserve(instr.arg_regs.size());
            for (int32_t r : instr.arg_regs) {
                call_args.push_back(regs_[static_cast<size_t>(frame.reg_base + r)]);
            }

            if (callee.kind == VMCalleeKind::kVMFunc) {
                // Advance pc before pushing new frame (caller resumes at pc+1)
                frame.pc++;
                PushFrame(instr.func_idx, call_args,
                          instr.dst_reg, frame.reg_base);
                continue;  // no extra ++pc
            } else {
                VMValue result = DispatchExternal(callee, call_args);
                if (instr.dst_reg >= 0) {
                    regs_[static_cast<size_t>(frame.reg_base + instr.dst_reg)] =
                        std::move(result);
                }
            }
            break;  // fall through to ++frame.pc
        }

        case Opcode::RET: {
            VMValue result;
            if (instr.src_reg >= 0) {
                result = regs_[static_cast<size_t>(frame.reg_base + instr.src_reg)];
            }
            int32_t caller_dst  = frame.caller_dst_reg;
            int32_t caller_base = frame.caller_reg_base;

            // Shrink register file back to caller's extent
            regs_.resize(static_cast<size_t>(frame.reg_base));
            frames_.pop_back();

            if (frames_.empty()) {
                // Top-level return
                return result;
            }
            // Write return value into caller's register
            if (caller_dst >= 0) {
                regs_[static_cast<size_t>(caller_base + caller_dst)] =
                    std::move(result);
            }
            continue;  // pc was already advanced before pushing the callee frame
        }

        case Opcode::IF: {
            bool cond = regs_[static_cast<size_t>(
                frame.reg_base + instr.cond_reg)].AsBool();
            frame.pc += (cond ? instr.true_offset : instr.false_offset);
            continue;  // no ++pc
        }

        case Opcode::GOTO: {
            frame.pc += instr.offset;
            continue;  // no ++pc
        }
        }  // switch

        ++frame.pc;
    }
    return VMValue{};
}

VMValue VMState::DispatchExternal(const FunctionEntry& callee,
                                  std::vector<VMValue>& args) {
    switch (callee.kind) {
    case VMCalleeKind::kBuiltin: {
        auto fn = BuiltinRegistry::Global().Get(callee.name);
        if (!fn) {
            throw std::runtime_error("Unknown builtin: " + callee.name);
        }
        return fn(args);
    }
    case VMCalleeKind::kPackedFunc: {
        auto pf = PackedFuncRegistry::Global().Get(callee.name);
        if (!pf.defined()) {
            throw std::runtime_error(
                "PackedFunc '" + callee.name + "' not registered");
        }
        PackedArgs pa(args);
        pf->Call(pa);
        // Return convention: PackedFunc body writes its result into args[0].
        // A void PackedFunc (no dst_reg caller) is called with an empty args
        // vector, so guard before accessing.
        return args.empty() ? VMValue{} : args[0];
    }
    case VMCalleeKind::kKernel: {
        // M11: KernelRegistry lookup + launch
        // M8 stub: no-op
        return VMValue{};
    }
    default:
        throw std::runtime_error(
            "DispatchExternal: unexpected callee kind for " + callee.name);
    }
}

}  // namespace devproc2
