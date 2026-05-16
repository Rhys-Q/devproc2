#include <devproc2/runtime/vm.h>
#include <devproc2/runtime/packed_func.h>
#include <devproc2/runtime/stream.h>
#ifdef DEVPROC2_WITH_CUDA
#include <devproc2/runtime/cuda_kernel_registry.h>
// Forward declaration for CUDAKernelLauncher_Launch (defined in cuda_kernel.cc)
namespace devproc2 {
    class KernelObj;
    void CUDAKernelLauncher_Launch(
        const KernelObj*,
        std::vector<VMValue>&,
        const std::vector<int64_t>&,
        void*);
}
#endif
#include <nlohmann/json.hpp>
#include <array>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace devproc2 {

// ── Binary deserialization helpers ────────────────────────────────────────────

namespace {

struct ByteReader {
    const uint8_t* data;
    size_t         size;
    size_t         pos = 0;

    template <typename T>
    T read() {
        if (pos + sizeof(T) > size)
            throw std::runtime_error("Deserialize: unexpected end of data");
        T val{};
        std::memcpy(&val, data + pos, sizeof(T));
        pos += sizeof(T);
        return val;
    }

    std::string read_string() {
        uint32_t len = read<uint32_t>();
        if (pos + len > size)
            throw std::runtime_error("Deserialize: unexpected end of string");
        std::string s(reinterpret_cast<const char*>(data + pos), len);
        pos += len;
        return s;
    }

    void skip(size_t n) {
        if (pos + n > size)
            throw std::runtime_error("Deserialize: unexpected end of data (skip)");
        pos += n;
    }
};

static constexpr uint8_t _MAGIC[4] = {'D', 'V', '2', 'E'};
static constexpr uint32_t _VERSION  = 2;
static constexpr uint8_t _TAG_NULL  = 0;
static constexpr uint8_t _TAG_INT   = 1;
static constexpr uint8_t _TAG_FLOAT = 2;
static constexpr uint8_t _TAG_BOOL  = 3;
static constexpr uint8_t _TAG_STR   = 4;

static std::string read_file_text(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::vector<uint8_t> read_file_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);
    auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> buf(size);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
    return buf;
}

static bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

#ifdef DEVPROC2_WITH_CUDA
static std::string artifact_path_join(
    const std::string& artifact_dir,
    const std::string& rel
) {
    if (!rel.empty() && rel[0] == '/') return rel;
    return artifact_dir + "/" + rel;
}

static std::string require_string_field(
    const nlohmann::json& obj,
    const std::string& field,
    const std::string& kernel_name
) {
    if (!obj.contains(field) || !obj[field].is_string()) {
        throw std::runtime_error(
            "kernel_table entry for '" + kernel_name
            + "' missing string field '" + field + "'");
    }
    return obj[field].get<std::string>();
}

static std::array<int32_t, 3> read_i32_triple(
    const nlohmann::json& obj,
    const std::string& field,
    std::array<int32_t, 3> defaults,
    bool allow_dynamic
) {
    if (!obj.contains(field)) return defaults;
    const auto& arr = obj[field];
    if (!arr.is_array() || arr.size() != 3) {
        throw std::runtime_error("kernel_table field '" + field + "' must be a 3-element array");
    }
    std::array<int32_t, 3> out = defaults;
    for (size_t i = 0; i < 3; ++i) {
        if (arr[i].is_number_integer()) {
            out[i] = arr[i].get<int32_t>();
        } else if (!allow_dynamic) {
            throw std::runtime_error(
                "kernel_table field '" + field + "' must contain integer values");
        }
    }
    return out;
}

static void load_kernel_table_into_cuda_registry(const std::string& artifact_dir) {
    std::string path = artifact_dir + "/metadata/kernel_table.json";
    if (!file_exists(path)) return;

    auto table = nlohmann::json::parse(read_file_text(path));
    if (!table.is_array()) {
        throw std::runtime_error("metadata/kernel_table.json must be an array");
    }

    for (const auto& entry : table) {
        if (!entry.is_object()) {
            throw std::runtime_error("kernel_table entries must be objects");
        }
        std::string name = require_string_field(entry, "name", "<unknown>");
        std::string cubin = require_string_field(entry, "cubin", name);
        std::string symbol = require_string_field(entry, "symbol", name);
        std::string cubin_path = artifact_path_join(artifact_dir, cubin);
        if (!file_exists(cubin_path)) {
            throw std::runtime_error(
                "Kernel '" + name + "' cubin not found: " + cubin_path);
        }

        const nlohmann::json& launch =
            entry.contains("launch") && entry["launch"].is_object()
                ? entry["launch"]
                : entry;
        auto grid = read_i32_triple(launch, "grid", {1, 1, 1}, /*allow_dynamic=*/true);
        auto block = read_i32_triple(launch, "block", {256, 1, 1}, /*allow_dynamic=*/false);
        int32_t smem = launch.value("shared_memory_bytes", 0);

        CUDAKernelRegistry::Global().Register(
            name, read_file_binary(cubin_path), symbol, grid, block, smem);
    }
}
#endif

}  // namespace

// ── Executable::Deserialize ───────────────────────────────────────────────────

std::shared_ptr<Executable> Executable::Deserialize(const uint8_t* data, size_t size) {
    ByteReader r{data, size};

    // Magic
    if (size < 4 || std::memcmp(data, _MAGIC, 4) != 0)
        throw std::runtime_error("Deserialize: invalid magic bytes");
    r.pos = 4;

    uint32_t version    = r.read<uint32_t>();
    uint32_t num_funcs  = r.read<uint32_t>();
    uint32_t num_instrs = r.read<uint32_t>();
    uint32_t num_consts = r.read<uint32_t>();

    if (version != _VERSION)
        throw std::runtime_error("Deserialize: bytecode version mismatch: expected "
                                 + std::to_string(_VERSION) + ", got "
                                 + std::to_string(version));

    auto exe = std::make_shared<Executable>();

    // Function table
    exe->function_table.resize(num_funcs);
    for (uint32_t i = 0; i < num_funcs; ++i) {
        FunctionEntry& fe = exe->function_table[i];
        fe.name         = r.read_string();
        fe.kind         = static_cast<VMCalleeKind>(r.read<uint8_t>());
        fe.instr_offset = r.read<int32_t>();
        fe.instr_count  = r.read<int32_t>();
        fe.num_regs     = r.read<int32_t>();
        fe.num_args     = r.read<int32_t>();
        uint32_t n_ci   = static_cast<uint32_t>(r.read<int32_t>());
        fe.const_inits.resize(n_ci);
        for (uint32_t j = 0; j < n_ci; ++j) {
            fe.const_inits[j].reg_idx   = r.read<int32_t>();
            fe.const_inits[j].const_idx = r.read<int32_t>();
        }
    }

    // Instructions
    exe->instructions.resize(num_instrs);
    for (uint32_t i = 0; i < num_instrs; ++i) {
        Instruction& ins = exe->instructions[i];
        ins.opcode       = static_cast<Opcode>(r.read<uint8_t>());
        ins.dst_reg      = r.read<int32_t>();
        ins.func_idx     = r.read<int32_t>();
        ins.src_reg      = r.read<int32_t>();
        ins.cond_reg     = r.read<int32_t>();
        ins.true_offset  = r.read<int32_t>();
        ins.false_offset = r.read<int32_t>();
        ins.offset       = r.read<int32_t>();
        uint32_t nargs   = r.read<uint32_t>();
        ins.arg_regs.resize(nargs);
        for (uint32_t j = 0; j < nargs; ++j)
            ins.arg_regs[j] = r.read<int32_t>();
        uint32_t nlaunch = r.read<uint32_t>();
        ins.launch_regs.resize(nlaunch);
        for (uint32_t j = 0; j < nlaunch; ++j)
            ins.launch_regs[j] = r.read<int32_t>();
    }

    // Constants: each is 1 tag byte + 8 value bytes
    exe->constants.resize(num_consts);
    for (uint32_t i = 0; i < num_consts; ++i) {
        uint8_t tag = r.read<uint8_t>();
        if (tag == _TAG_NULL) {
            r.skip(8);
            exe->constants[i] = VMValue{};
        } else if (tag == _TAG_INT) {
            int64_t v = r.read<int64_t>();
            exe->constants[i] = VMValue::Int(v);
        } else if (tag == _TAG_FLOAT) {
            double v = r.read<double>();
            exe->constants[i] = VMValue::Float(v);
        } else if (tag == _TAG_BOOL) {
            int64_t v = r.read<int64_t>();
            exe->constants[i] = VMValue::Bool(static_cast<bool>(v));
        } else if (tag == _TAG_STR) {
            // String constant: uint32 length + bytes.
            // VMValue has no string type; store as null (string is used for
            // assert_le_i64 message and the C++ builtin reads it from a
            // different mechanism — see builtins.cc).
            uint32_t slen = r.read<uint32_t>();
            r.skip(slen);
            exe->constants[i] = VMValue{};
        } else {
            throw std::runtime_error("Deserialize: unknown constant tag "
                                     + std::to_string(tag));
        }
    }

    return exe;
}

// ── Executable::Load ──────────────────────────────────────────────────────────

std::shared_ptr<Executable> Executable::Load(const std::string& artifact_dir) {
    // 1. Deserialize executable.vm
    std::string vm_path = artifact_dir + "/executable.vm";
    auto vm_bytes = read_file_binary(vm_path);
    auto exe = Executable::Deserialize(vm_bytes.data(), vm_bytes.size());

    // 2. Parse abi.json
    std::string abi_path = artifact_dir + "/abi.json";
    auto abi = nlohmann::json::parse(read_file_text(abi_path));

    // 3. ABI version check (major component must match)
    std::string abi_version = abi.value("devproc_abi_version", std::string{});
    if (!abi_version.empty()) {
        std::string expected_major = "0";
        std::string actual_major   = abi_version.substr(0, abi_version.find('.'));
        if (actual_major != expected_major) {
            throw std::runtime_error(
                "ABI version mismatch: expected major " + expected_major
                + ", got " + actual_major + " (full version: " + abi_version + ")");
        }
    }

    // 4. PackedFunc dependency check
    for (const auto& name : abi.value("required_packed_funcs",
                                       nlohmann::json::array())) {
        std::string fn = name.get<std::string>();
        if (!PackedFuncRegistry::Global().Has(fn)) {
            throw std::runtime_error(
                "PackedFunc '" + fn + "' is required but not registered.");
        }
    }

#ifdef DEVPROC2_WITH_CUDA
    load_kernel_table_into_cuda_registry(artifact_dir);
#else
    std::string kernel_table_path = artifact_dir + "/metadata/kernel_table.json";
    if (file_exists(kernel_table_path)) {
        auto table = nlohmann::json::parse(read_file_text(kernel_table_path));
        if (table.is_array() && !table.empty()) {
            throw std::runtime_error(
                "Artifact contains CUDA kernels but runtime was built without DEVPROC2_WITH_CUDA");
        }
    }
#endif

    return exe;
}

VMState::VMState(std::shared_ptr<Executable> exec)
    : exec_(std::move(exec)) {}

void* VMState::GetDefaultStream(const Device& dev) {
    if (dev.device_type == kDLCPU) return nullptr;

    auto it = default_streams_.find(dev);
    if (it != default_streams_.end()) {
        return it->second->handle;
    }

    DeviceAPI* api = DeviceAPIRegistry::Get(dev.device_type);
    void* handle = api->CreateStream(dev);

    auto* obj = new StreamObj();
    obj->device = dev;
    obj->handle = handle;
    Stream s(obj);
    default_streams_[dev] = s;
    return handle;
}

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
            std::vector<int64_t> launch_args;
            launch_args.reserve(instr.launch_regs.size());
            for (int32_t r : instr.launch_regs) {
                launch_args.push_back(
                    regs_[static_cast<size_t>(frame.reg_base + r)].AsInt());
            }

            if (callee.kind == VMCalleeKind::kVMFunc) {
                // Advance pc before pushing new frame (caller resumes at pc+1)
                frame.pc++;
                PushFrame(instr.func_idx, call_args,
                          instr.dst_reg, frame.reg_base);
                continue;  // no extra ++pc
            } else {
                VMValue result = DispatchExternal(callee, call_args, launch_args);
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

VMValue VMState::DispatchExternal(
    const FunctionEntry& callee,
    std::vector<VMValue>& args,
    const std::vector<int64_t>& launch_args
) {
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
#ifdef DEVPROC2_WITH_CUDA
        auto* k = CUDAKernelRegistry::Global().Get(callee.name);
        if (!k) {
            throw std::runtime_error(
                "Kernel '" + callee.name + "' not registered in CUDAKernelRegistry");
        }
        Device cuda_dev{kDLCUDA, 0};
        void* stream = GetDefaultStream(cuda_dev);
        CUDAKernelLauncher_Launch(k, args, launch_args, stream);
        return VMValue{};
#else
        throw std::runtime_error(
            "kKernel dispatch requires DEVPROC2_WITH_CUDA (kernel: " + callee.name + ")");
#endif
    }
    default:
        throw std::runtime_error(
            "DispatchExternal: unexpected callee kind for " + callee.name);
    }
}

}  // namespace devproc2
