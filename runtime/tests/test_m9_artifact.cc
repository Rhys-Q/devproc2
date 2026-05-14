// M9 Artifact C++ unit tests — Executable::Load ABI validation
// Build: cmake -DDEVPROC2_BUILD_TESTS=ON && make test_m9_artifact
// Run:   ./build/runtime/tests/test_m9_artifact

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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

// Build a minimal valid executable.vm binary (empty function table, 0 instrs)
static std::vector<uint8_t> make_minimal_vm_bytes() {
    std::vector<uint8_t> buf;
    // Magic
    buf.insert(buf.end(), {'D', 'V', '2', 'E'});
    // version=1, num_funcs=0, num_instrs=0 (3 x uint32 LE)
    uint32_t v[4] = {1, 0, 0, 0};  // version, funcs, instrs, consts
    for (uint32_t x : v) {
        buf.push_back(x & 0xFF);
        buf.push_back((x >> 8) & 0xFF);
        buf.push_back((x >> 16) & 0xFF);
        buf.push_back((x >> 24) & 0xFF);
    }
    return buf;
}

static void write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}

static void write_binary(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
}

// Create a temp artifact dir with valid executable.vm and given abi.json content
static std::string make_artifact_dir(const std::string& suffix,
                                      const std::string& abi_content) {
    std::string dir = "/tmp/devproc2_m9_test_" + suffix;
    std::filesystem::create_directories(dir);
    write_binary(dir + "/executable.vm", make_minimal_vm_bytes());
    write_file(dir + "/abi.json", abi_content);
    return dir;
}

// ── test_load_abi_version_mismatch ────────────────────────────────────────────

void test_load_abi_version_mismatch() {
    std::string dir = make_artifact_dir("ver_mismatch",
        "{\n  \"devproc_abi_version\": \"9.0\",\n  \"required_packed_funcs\": []\n}\n");
    CHECK_THROWS_MSG(Executable::Load(dir), "ABI version mismatch");
}

// ── test_load_missing_packed_func ─────────────────────────────────────────────

void test_load_missing_packed_func() {
    std::string dir = make_artifact_dir("missing_pf",
        "{\n  \"devproc_abi_version\": \"0.1\",\n"
        "  \"required_packed_funcs\": [\n    \"runtime.tokenizer.encode\"\n  ]\n}\n");
    CHECK_THROWS_MSG(Executable::Load(dir),
                     "PackedFunc 'runtime.tokenizer.encode' is required but not registered.");
}

// ── test_load_valid_no_packed_funcs ───────────────────────────────────────────

void test_load_valid_no_packed_funcs() {
    std::string dir = make_artifact_dir("valid_empty",
        "{\n  \"devproc_abi_version\": \"0.1\",\n  \"required_packed_funcs\": []\n}\n");
    auto exe = Executable::Load(dir);
    CHECK(exe != nullptr);
    CHECK(exe->function_table.empty());
    CHECK(exe->instructions.empty());
}

// ── test_deserialize_minimal ──────────────────────────────────────────────────

void test_deserialize_minimal() {
    auto bytes = make_minimal_vm_bytes();
    auto exe = Executable::Deserialize(bytes.data(), bytes.size());
    CHECK(exe != nullptr);
    CHECK(exe->function_table.empty());
    CHECK(exe->instructions.empty());
    CHECK(exe->constants.empty());
}

// ── test_deserialize_bad_magic ────────────────────────────────────────────────

void test_deserialize_bad_magic() {
    std::vector<uint8_t> bad = {0x00, 0x01, 0x02, 0x03};
    CHECK_THROWS_MSG(Executable::Deserialize(bad.data(), bad.size()), "invalid magic");
}

// ── test_load_missing_executable_vm ──────────────────────────────────────────

void test_load_missing_executable_vm() {
    std::string dir = "/tmp/devproc2_m9_test_no_vm";
    std::filesystem::create_directories(dir);
    write_file(dir + "/abi.json",
               "{\"devproc_abi_version\": \"0.1\", \"required_packed_funcs\": []}");
    // No executable.vm
    CHECK_THROWS_MSG(Executable::Load(dir), "Cannot open file");
}

}  // namespace

int main() {
    int prev_fail = g_fail;
    (void)prev_fail;

    RUN(test_deserialize_minimal);
    RUN(test_deserialize_bad_magic);
    RUN(test_load_valid_no_packed_funcs);
    RUN(test_load_abi_version_mismatch);
    RUN(test_load_missing_packed_func);
    RUN(test_load_missing_executable_vm);

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
