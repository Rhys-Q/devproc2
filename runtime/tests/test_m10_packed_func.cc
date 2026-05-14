// M10 PackedFunc C++ unit tests — VMState dispatch to PackedFunc
// Build: cmake -DDEVPROC2_BUILD_TESTS=ON && make test_m10_packed_func
// Run:   ./build/runtime/tests/test_m10_packed_func

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "devproc2/runtime/packed_func.h"
#include "devproc2/runtime/vm.h"

namespace {

int g_pass = 0;
int g_fail = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "  FAIL: " #cond "\n    at " __FILE__ ":"              \
                      << __LINE__ << "\n";                                       \
            ++g_fail;                                                            \
            return;                                                              \
        }                                                                        \
    } while (0)

#define CHECK_THROWS_MSG(expr, substr)                                           \
    do {                                                                         \
        bool caught = false;                                                     \
        std::string msg;                                                         \
        try { (expr); }                                                          \
        catch (const std::exception& _e) { caught = true; msg = _e.what(); }    \
        if (!caught) {                                                           \
            std::cerr << "  FAIL: expected exception not thrown\n";             \
            ++g_fail; return;                                                    \
        }                                                                        \
        if (msg.find(substr) == std::string::npos) {                            \
            std::cerr << "  FAIL: exception msg '" << msg                       \
                      << "' does not contain '" << substr << "'\n";             \
            ++g_fail; return;                                                    \
        }                                                                        \
    } while (0)

#define RUN(fn)                                                                  \
    do {                                                                         \
        int prev_fail = g_fail;                                                  \
        std::cout << "[ RUN  ] " #fn "\n";                                      \
        fn();                                                                    \
        if (g_fail == prev_fail) {                                               \
            std::cout << "[ PASS ] " #fn "\n";                                  \
            ++g_pass;                                                            \
        }                                                                        \
    } while (0)

using namespace devproc2;

// Helpers to make PackedFunc from a lambda.
static PackedFunc make_pf(std::function<void(PackedArgs)> body) {
    auto* obj = new PackedFuncObj();
    obj->body  = std::move(body);
    return PackedFunc(obj);
}

// Build a minimal Executable:
//   func[0] "main" (vm_func): CALL dst=-1 func=1 arg_regs=[0], RET src=-1
//   func[1] pf_name (packed_func): external
static std::shared_ptr<Executable> make_packed_func_exe(const std::string& pf_name) {
    auto exec = std::make_shared<Executable>();

    // Instruction 0: CALL dst=-1 func=1 arg_regs=[0]
    Instruction call_instr;
    call_instr.opcode   = Opcode::CALL;
    call_instr.dst_reg  = -1;
    call_instr.func_idx = 1;
    call_instr.arg_regs = {0};

    // Instruction 1: RET src=-1
    Instruction ret_instr;
    ret_instr.opcode  = Opcode::RET;
    ret_instr.src_reg = -1;

    exec->instructions.push_back(call_instr);
    exec->instructions.push_back(ret_instr);

    // func[0] "main" — vm_func with 2 instructions, 1 register
    FunctionEntry main_fe;
    main_fe.name         = "main";
    main_fe.kind         = VMCalleeKind::kVMFunc;
    main_fe.instr_offset = 0;
    main_fe.instr_count  = 2;
    main_fe.num_regs     = 1;
    main_fe.num_args     = 1;
    exec->function_table.push_back(main_fe);

    // func[1] pf_name — packed_func (no instructions)
    FunctionEntry pf_fe;
    pf_fe.name         = pf_name;
    pf_fe.kind         = VMCalleeKind::kPackedFunc;
    pf_fe.instr_offset = -1;
    pf_fe.instr_count  = 0;
    pf_fe.num_regs     = 0;
    pf_fe.num_args     = 0;
    exec->function_table.push_back(pf_fe);

    return exec;
}

static VMValue make_int(int64_t v) {
    return VMValue::Int(v);
}

// ── Test 1: registered PackedFunc is called ──────────────────────────────────

void test_dispatch_packed_func_called() {
    const std::string pf_name = "test.m10_pf_called";
    auto exec = make_packed_func_exe(pf_name);

    int call_count = 0;
    PackedFuncRegistry::Global().Register(
        pf_name, make_pf([&call_count](PackedArgs) { ++call_count; }));

    VMState vm(exec);
    vm.Invoke("main", {make_int(42)});

    CHECK(call_count == 1);
}

// ── Test 2: unregistered PackedFunc throws ────────────────────────────────────

void test_dispatch_packed_func_unregistered_throws() {
    const std::string pf_name = "test.m10_pf_not_registered_xyz";
    auto exec = make_packed_func_exe(pf_name);
    // Do NOT register pf_name.

    VMState vm(exec);
    CHECK_THROWS_MSG(vm.Invoke("main", {make_int(0)}), pf_name);
}

// ── Test 3: args are passed through VM to PackedFunc ─────────────────────────

void test_dispatch_packed_func_args_passed() {
    const std::string pf_name = "test.m10_pf_args";
    auto exec = make_packed_func_exe(pf_name);

    int64_t received_arg = -1;
    PackedFuncRegistry::Global().Register(
        pf_name, make_pf([&received_arg](PackedArgs args) {
            if (args.size() >= 1 && args[0].IsInt()) {
                received_arg = args[0].AsInt();
            }
        }));

    VMState vm(exec);
    vm.Invoke("main", {make_int(99)});

    CHECK(received_arg == 99);
}

// ── Test 4: PackedFuncRegistry basic API ─────────────────────────────────────

void test_packed_func_registry_basic() {
    const std::string name = "test.m10_registry_basic";
    bool called = false;
    PackedFuncRegistry::Global().Register(
        name, make_pf([&called](PackedArgs) { called = true; }));

    CHECK(PackedFuncRegistry::Global().Has(name));

    auto pf = PackedFuncRegistry::Global().Get(name);
    CHECK(pf.defined());
    std::vector<VMValue> empty_args;
    PackedArgs pa(empty_args);
    pf->Call(pa);
    CHECK(called);
}

// ── Test 5: missing PackedFunc returns undefined (not throws at Get) ──────────

void test_packed_func_registry_missing_returns_undefined() {
    auto pf = PackedFuncRegistry::Global().Get("test.m10_definitely_not_registered");
    CHECK(!pf.defined());
    CHECK(!PackedFuncRegistry::Global().Has("test.m10_definitely_not_registered"));
}

} // namespace

int main() {
    RUN(test_dispatch_packed_func_called);
    RUN(test_dispatch_packed_func_unregistered_throws);
    RUN(test_dispatch_packed_func_args_passed);
    RUN(test_packed_func_registry_basic);
    RUN(test_packed_func_registry_missing_returns_undefined);

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return (g_fail == 0) ? 0 : 1;
}
