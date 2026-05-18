#ifdef DEVPROC2_WITH_CUDA

#include "devproc2/runtime/cuda_gemm.h"

#include "cutlass_fp8_gemm_sm89.h"

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "devproc2/runtime/packed_func.h"
#include "devproc2/runtime/tensor.h"
#include "devproc2/runtime/vm_value.h"

#ifdef DEVPROC2_WITH_PI05_FA2
extern "C" void devproc2_pi05_attention_fa2_fwd_bf16(
    const void*, const void*, const void*,
    void*, void*, void*, void*,
    int, int, int,
    int, int, int,
    int, int, int,
    int, int, int,
    int, int, int,
    int, int, int,
    float, int, cudaStream_t);
#else
extern "C" void devproc2_pi05_attention_fa2_fwd_bf16(
    const void*, const void*, const void*,
    void*, void*, void*, void*,
    int, int, int,
    int, int, int,
    int, int, int,
    int, int, int,
    int, int, int,
    int, int, int,
    float, int, cudaStream_t) {
    throw std::runtime_error("Pi0.5 FA2 backend was not built into this runtime");
}
#endif

namespace devproc2 {
namespace {

#define CUDA_RT_CHECK(expr)                                                     \
    do {                                                                       \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            throw std::runtime_error(                                           \
                std::string("CUDA runtime error in " #expr ": ") +            \
                cudaGetErrorString(_err));                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_LT_CHECK(expr)                                                   \
    do {                                                                       \
        cublasStatus_t _st = (expr);                                            \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                     \
            throw std::runtime_error(                                           \
                std::string("cuBLASLt error in " #expr ": ") +                \
                std::to_string(static_cast<int>(_st)));                         \
        }                                                                      \
    } while (0)

int EnvInt(const char* name, int default_value, int min_value, int max_value) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') return default_value;
    int value = std::atoi(raw);
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

bool EnvFlag(const char* name, bool default_value) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') return default_value;
    return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 &&
           std::strcmp(raw, "FALSE") != 0;
}

bool StreamIsCapturing(cudaStream_t stream) {
    if (!stream) return false;
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(stream, &status);
    if (err != cudaSuccess) return false;
    return status != cudaStreamCaptureStatusNone;
}

enum class FP8Layout : int {
    kNN = 0,  // B is row-major [K, N].
    kNT = 1,  // B is row-major [N, K], consumed as B^T.
};

enum class BF16Layout : int {
    kNN = 0,  // B is row-major [K, N].
    kNT = 1,  // B is row-major [N, K], consumed as B^T.
};

struct GemmKey {
    int kind;
    int layout;
    int m;
    int n;
    int k;

    bool operator==(const GemmKey& other) const {
        return kind == other.kind && layout == other.layout && m == other.m &&
               n == other.n && k == other.k;
    }
};

struct GemmKeyHash {
    size_t operator()(const GemmKey& key) const {
        size_t h = std::hash<int>()(key.kind);
        h ^= std::hash<int>()(key.layout) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(key.m) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(key.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(key.k) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct CachedGemm {
    cublasLtMatmulDesc_t matmul_desc{nullptr};
    cublasLtMatrixLayout_t a_desc{nullptr};
    cublasLtMatrixLayout_t b_desc{nullptr};
    cublasLtMatrixLayout_t d_desc{nullptr};
    cublasLtMatmulAlgo_t algo{};
    std::vector<cublasLtMatmulAlgo_t> candidate_algos;
    bool tuned{false};
};

class LtFP8Runner {
public:
    LtFP8Runner() {
        CUBLAS_LT_CHECK(cublasLtCreate(&handle_));
        workspace_size_ = 256 * 1024 * 1024;
        CUDA_RT_CHECK(cudaMalloc(&workspace_, workspace_size_));
        tune_algos_ = EnvInt("DEVPROC2_CUBLASLT_FP8_TUNE_ALGOS", 8, 1, 64);
        tune_repeats_ = EnvInt("DEVPROC2_CUBLASLT_FP8_TUNE_REPEATS", 2, 1, 16);
        tune_enabled_ = EnvFlag("DEVPROC2_CUBLASLT_FP8_TUNE", true);
        tune_log_ = EnvFlag("DEVPROC2_CUBLASLT_TUNE_LOG", false);
        fast_accum_ = EnvFlag("DEVPROC2_CUBLASLT_FP8_FAST_ACCUM", true);
        cutlass_nt_enabled_ = EnvFlag("DEVPROC2_CUTLASS_FP8_NT", true);
    }

    ~LtFP8Runner() {
        for (auto& item : cache_) {
            auto& entry = item.second;
            if (entry.a_desc) cublasLtMatrixLayoutDestroy(entry.a_desc);
            if (entry.b_desc) cublasLtMatrixLayoutDestroy(entry.b_desc);
            if (entry.d_desc) cublasLtMatrixLayoutDestroy(entry.d_desc);
            if (entry.matmul_desc) cublasLtMatmulDescDestroy(entry.matmul_desc);
        }
        if (workspace_) cudaFree(workspace_);
        if (handle_) cublasLtDestroy(handle_);
    }

    void Run(FP8Layout layout,
             void* a,
             void* b,
             void* d,
             int m,
             int n,
             int k,
             float* a_scale,
             float* b_scale,
             float beta_value,
             cudaStream_t stream) {
        if (cutlass_nt_enabled_ && layout == FP8Layout::kNT &&
            CutlassFP8NTBF16CanRun(m, n, k, beta_value)) {
            CutlassFP8NTBF16Run(
                a, b, d, m, n, k, a_scale, b_scale, beta_value, stream);
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        auto& entry = GetOrCreate(layout, m, n, k);
        CUBLAS_LT_CHECK(cublasLtMatmulDescSetAttribute(
            entry.matmul_desc,
            CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &a_scale,
            sizeof(a_scale)));
        CUBLAS_LT_CHECK(cublasLtMatmulDescSetAttribute(
            entry.matmul_desc,
            CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &b_scale,
            sizeof(b_scale)));
        MaybeTune(entry, layout, a, b, d, m, n, k, beta_value, stream);
        float alpha = 1.0f;
        CUBLAS_LT_CHECK(cublasLtMatmul(
            handle_,
            entry.matmul_desc,
            &alpha,
            a,
            entry.a_desc,
            b,
            entry.b_desc,
            &beta_value,
            d,
            entry.d_desc,
            d,
            entry.d_desc,
            &entry.algo,
            workspace_,
            workspace_size_,
            stream));
    }

private:
    void MaybeTune(CachedGemm& entry,
                   FP8Layout layout,
                   void* a,
                   void* b,
                   void* d,
                   int m,
                   int n,
                   int k,
                   float beta_value,
                   cudaStream_t stream) {
        if (entry.tuned || !tune_enabled_ || entry.candidate_algos.size() <= 1 ||
            StreamIsCapturing(stream)) {
            return;
        }

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_RT_CHECK(cudaEventCreateWithFlags(&start, cudaEventDefault));
        CUDA_RT_CHECK(cudaEventCreateWithFlags(&stop, cudaEventDefault));

        float alpha = 1.0f;
        float beta = 0.0f;
        float best_ms = std::numeric_limits<float>::infinity();
        int best_index = -1;
        void* d_backup = nullptr;
        if (beta_value != 0.0f) {
            const size_t d_bytes = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(uint16_t);
            CUDA_RT_CHECK(cudaMalloc(&d_backup, d_bytes));
            CUDA_RT_CHECK(cudaMemcpyAsync(d_backup, d, d_bytes, cudaMemcpyDeviceToDevice, stream));
        }

        for (int i = 0; i < static_cast<int>(entry.candidate_algos.size()); ++i) {
            auto& algo = entry.candidate_algos[static_cast<size_t>(i)];
            cublasStatus_t st = cublasLtMatmul(
                handle_,
                entry.matmul_desc,
                &alpha,
                a,
                entry.a_desc,
                b,
                entry.b_desc,
                &beta,
                d,
                entry.d_desc,
                d,
                entry.d_desc,
                &algo,
                workspace_,
                workspace_size_,
                stream);
            if (st != CUBLAS_STATUS_SUCCESS) continue;
            cudaError_t sync_st = cudaStreamSynchronize(stream);
            if (sync_st != cudaSuccess) continue;

            CUDA_RT_CHECK(cudaEventRecord(start, stream));
            bool ok = true;
            for (int repeat = 0; repeat < tune_repeats_; ++repeat) {
                st = cublasLtMatmul(
                    handle_,
                    entry.matmul_desc,
                    &alpha,
                    a,
                    entry.a_desc,
                    b,
                    entry.b_desc,
                    &beta,
                    d,
                    entry.d_desc,
                    d,
                    entry.d_desc,
                    &algo,
                    workspace_,
                    workspace_size_,
                    stream);
                if (st != CUBLAS_STATUS_SUCCESS) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            CUDA_RT_CHECK(cudaEventRecord(stop, stream));
            sync_st = cudaEventSynchronize(stop);
            if (sync_st != cudaSuccess) continue;

            float elapsed_ms = 0.0f;
            CUDA_RT_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            float per_run_ms = elapsed_ms / static_cast<float>(tune_repeats_);
            if (per_run_ms < best_ms) {
                best_ms = per_run_ms;
                best_index = i;
            }
        }

        if (best_index >= 0) {
            entry.algo = entry.candidate_algos[static_cast<size_t>(best_index)];
            if (tune_log_) {
                std::fprintf(
                    stderr,
                    "devproc2 cublasLt fp8 tune layout=%d m=%d n=%d k=%d algos=%zu best=%d %.4fms\n",
                    static_cast<int>(layout),
                    m,
                    n,
                    k,
                    entry.candidate_algos.size(),
                    best_index,
                    best_ms);
            }
        }
        entry.tuned = true;

        if (d_backup != nullptr) {
            const size_t d_bytes = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(uint16_t);
            CUDA_RT_CHECK(cudaMemcpyAsync(d, d_backup, d_bytes, cudaMemcpyDeviceToDevice, stream));
            CUDA_RT_CHECK(cudaStreamSynchronize(stream));
            CUDA_RT_CHECK(cudaFree(d_backup));
        }
        CUDA_RT_CHECK(cudaEventDestroy(start));
        CUDA_RT_CHECK(cudaEventDestroy(stop));
    }

    CachedGemm& GetOrCreate(FP8Layout layout, int m, int n, int k) {
        GemmKey key{0, static_cast<int>(layout), m, n, k};
        auto found = cache_.find(key);
        if (found != cache_.end()) return found->second;

        CachedGemm entry;
        cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
        cublasOperation_t op_n = CUBLAS_OP_N;
        cublasOperation_t op_t = CUBLAS_OP_T;

        CUBLAS_LT_CHECK(cublasLtMatmulDescCreate(
            &entry.matmul_desc,
            CUBLAS_COMPUTE_32F,
            CUDA_R_32F));
        if (fast_accum_) {
            int8_t fast_accum = 1;
            CUBLAS_LT_CHECK(cublasLtMatmulDescSetAttribute(
                entry.matmul_desc,
                CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                &fast_accum,
                sizeof(fast_accum)));
        }
        CUBLAS_LT_CHECK(cublasLtMatmulDescSetAttribute(
            entry.matmul_desc,
            CUBLASLT_MATMUL_DESC_TRANSA,
            &op_n,
            sizeof(op_n)));
        cublasOperation_t trans_b =
            layout == FP8Layout::kNT ? op_t : op_n;
        CUBLAS_LT_CHECK(cublasLtMatmulDescSetAttribute(
            entry.matmul_desc,
            CUBLASLT_MATMUL_DESC_TRANSB,
            &trans_b,
            sizeof(trans_b)));

        CUBLAS_LT_CHECK(cublasLtMatrixLayoutCreate(
            &entry.a_desc, CUDA_R_8F_E4M3, m, k, k));
        CUBLAS_LT_CHECK(cublasLtMatrixLayoutSetAttribute(
            entry.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
            &row_order, sizeof(row_order)));

        if (layout == FP8Layout::kNT) {
            CUBLAS_LT_CHECK(cublasLtMatrixLayoutCreate(
                &entry.b_desc, CUDA_R_8F_E4M3, n, k, k));
        } else {
            CUBLAS_LT_CHECK(cublasLtMatrixLayoutCreate(
                &entry.b_desc, CUDA_R_8F_E4M3, k, n, n));
        }
        CUBLAS_LT_CHECK(cublasLtMatrixLayoutSetAttribute(
            entry.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
            &row_order, sizeof(row_order)));

        CUBLAS_LT_CHECK(cublasLtMatrixLayoutCreate(
            &entry.d_desc, CUDA_R_16BF, m, n, n));
        CUBLAS_LT_CHECK(cublasLtMatrixLayoutSetAttribute(
            entry.d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
            &row_order, sizeof(row_order)));

        cublasLtMatmulPreference_t preference;
        CUBLAS_LT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size_,
            sizeof(workspace_size_)));
        std::vector<cublasLtMatmulHeuristicResult_t> heuristics(
            static_cast<size_t>(tune_algos_));
        int returned = 0;
        CUBLAS_LT_CHECK(cublasLtMatmulAlgoGetHeuristic(
            handle_,
            entry.matmul_desc,
            entry.a_desc,
            entry.b_desc,
            entry.d_desc,
            entry.d_desc,
            preference,
            tune_algos_,
            heuristics.data(),
            &returned));
        cublasLtMatmulPreferenceDestroy(preference);
        if (returned == 0) {
            throw std::runtime_error("cuBLASLt found no FP8 GEMM algorithm");
        }
        entry.algo = heuristics[0].algo;
        entry.candidate_algos.reserve(static_cast<size_t>(returned));
        for (int i = 0; i < returned; ++i) {
            entry.candidate_algos.push_back(heuristics[static_cast<size_t>(i)].algo);
        }
        if (entry.candidate_algos.size() <= 1) entry.tuned = true;

        auto inserted = cache_.emplace(key, entry);
        return inserted.first->second;
    }

    std::mutex mu_;
    cublasLtHandle_t handle_{nullptr};
    void* workspace_{nullptr};
    size_t workspace_size_{0};
    int tune_algos_{8};
    int tune_repeats_{2};
    bool tune_enabled_{true};
    bool tune_log_{false};
    bool fast_accum_{false};
    bool cutlass_nt_enabled_{true};
    std::unordered_map<GemmKey, CachedGemm, GemmKeyHash> cache_;
};

LtFP8Runner& GlobalRunner() {
    static LtFP8Runner runner;
    return runner;
}

class LtBF16Runner {
public:
    LtBF16Runner() {
        CUBLAS_LT_CHECK(cublasLtCreate(&handle_));
        workspace_size_ = 64 * 1024 * 1024;
        CUDA_RT_CHECK(cudaMalloc(&workspace_, workspace_size_));
    }

    ~LtBF16Runner() {
        for (auto& item : cache_) {
            auto& entry = item.second;
            if (entry.a_desc) cublasLtMatrixLayoutDestroy(entry.a_desc);
            if (entry.b_desc) cublasLtMatrixLayoutDestroy(entry.b_desc);
            if (entry.d_desc) cublasLtMatrixLayoutDestroy(entry.d_desc);
            if (entry.matmul_desc) cublasLtMatmulDescDestroy(entry.matmul_desc);
        }
        if (workspace_) cudaFree(workspace_);
        if (handle_) cublasLtDestroy(handle_);
    }

    void Run(BF16Layout layout,
             void* a,
             void* b,
             void* d,
             int m,
             int n,
             int k,
             cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mu_);
        auto& entry = GetOrCreate(layout, m, n, k);
        float alpha = 1.0f;
        float beta = 0.0f;
        CUBLAS_LT_CHECK(cublasLtMatmul(
            handle_,
            entry.matmul_desc,
            &alpha,
            a,
            entry.a_desc,
            b,
            entry.b_desc,
            &beta,
            d,
            entry.d_desc,
            d,
            entry.d_desc,
            &entry.algo,
            workspace_,
            workspace_size_,
            stream));
    }

private:
    CachedGemm& GetOrCreate(BF16Layout layout, int m, int n, int k) {
        GemmKey key{1, static_cast<int>(layout), m, n, k};
        auto found = cache_.find(key);
        if (found != cache_.end()) return found->second;

        CachedGemm entry;
        cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
        cublasOperation_t op_n = CUBLAS_OP_N;
        cublasOperation_t op_t = CUBLAS_OP_T;

        CUBLAS_LT_CHECK(cublasLtMatmulDescCreate(
            &entry.matmul_desc,
            CUBLAS_COMPUTE_32F,
            CUDA_R_32F));
        CUBLAS_LT_CHECK(cublasLtMatmulDescSetAttribute(
            entry.matmul_desc,
            CUBLASLT_MATMUL_DESC_TRANSA,
            &op_n,
            sizeof(op_n)));
        cublasOperation_t trans_b =
            layout == BF16Layout::kNT ? op_t : op_n;
        CUBLAS_LT_CHECK(cublasLtMatmulDescSetAttribute(
            entry.matmul_desc,
            CUBLASLT_MATMUL_DESC_TRANSB,
            &trans_b,
            sizeof(trans_b)));

        CUBLAS_LT_CHECK(cublasLtMatrixLayoutCreate(
            &entry.a_desc, CUDA_R_16BF, m, k, k));
        CUBLAS_LT_CHECK(cublasLtMatrixLayoutSetAttribute(
            entry.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
            &row_order, sizeof(row_order)));

        if (layout == BF16Layout::kNT) {
            CUBLAS_LT_CHECK(cublasLtMatrixLayoutCreate(
                &entry.b_desc, CUDA_R_16BF, n, k, k));
        } else {
            CUBLAS_LT_CHECK(cublasLtMatrixLayoutCreate(
                &entry.b_desc, CUDA_R_16BF, k, n, n));
        }
        CUBLAS_LT_CHECK(cublasLtMatrixLayoutSetAttribute(
            entry.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
            &row_order, sizeof(row_order)));

        CUBLAS_LT_CHECK(cublasLtMatrixLayoutCreate(
            &entry.d_desc, CUDA_R_16BF, m, n, n));
        CUBLAS_LT_CHECK(cublasLtMatrixLayoutSetAttribute(
            entry.d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
            &row_order, sizeof(row_order)));

        cublasLtMatmulPreference_t preference;
        CUBLAS_LT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size_,
            sizeof(workspace_size_)));
        cublasLtMatmulHeuristicResult_t heuristic;
        int returned = 0;
        CUBLAS_LT_CHECK(cublasLtMatmulAlgoGetHeuristic(
            handle_,
            entry.matmul_desc,
            entry.a_desc,
            entry.b_desc,
            entry.d_desc,
            entry.d_desc,
            preference,
            1,
            &heuristic,
            &returned));
        cublasLtMatmulPreferenceDestroy(preference);
        if (returned == 0) {
            throw std::runtime_error("cuBLASLt found no BF16 GEMM algorithm");
        }
        entry.algo = heuristic.algo;

        auto inserted = cache_.emplace(key, entry);
        return inserted.first->second;
    }

    std::mutex mu_;
    cublasLtHandle_t handle_{nullptr};
    void* workspace_{nullptr};
    size_t workspace_size_{0};
    std::unordered_map<GemmKey, CachedGemm, GemmKeyHash> cache_;
};

LtBF16Runner& GlobalBF16Runner() {
    static LtBF16Runner runner;
    return runner;
}

float HostF32FromBitsI64(int64_t bits) {
    uint32_t lo = static_cast<uint32_t>(bits & 0xffffffffull);
    float out = 0.0f;
    std::memcpy(&out, &lo, sizeof(float));
    return out;
}

class Pi05FA2Runner {
public:
    Pi05FA2Runner() {
        int dev = 0;
        CUDA_RT_CHECK(cudaGetDevice(&dev));
        CUDA_RT_CHECK(cudaDeviceGetAttribute(
            &num_sms_, cudaDevAttrMultiProcessorCount, dev));
        split_q_threshold_ = EnvInt("DEVPROC2_FA2_SPLIT_Q_THRESHOLD", 4096, 0, 4096);
        num_sms_override_ = EnvInt("DEVPROC2_FA2_NUM_SMS", -1, -1, 4096);
    }

    ~Pi05FA2Runner() {
        if (softmax_lse_) cudaFree(softmax_lse_);
        if (softmax_lse_accum_) cudaFree(softmax_lse_accum_);
        if (o_accum_) cudaFree(o_accum_);
    }

    void Run(void* q,
             void* k,
             void* v,
             void* out,
             int rows_q,
             int rows_k,
             int num_heads_q,
             int num_heads_kv,
             int head_dim,
             float scale,
             cudaStream_t stream) {
        RunBatched(
            q,
            k,
            v,
            out,
            1,
            rows_q,
            rows_k,
            num_heads_q,
            num_heads_kv,
            head_dim,
            scale,
            stream);
    }

    void RunBatched(void* q,
                    void* k,
                    void* v,
                    void* out,
                    int batch,
                    int rows_q,
                    int rows_k,
                    int num_heads_q,
                    int num_heads_kv,
                    int head_dim,
                    float scale,
                    cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mu_);
        int effective_num_sms =
            num_sms_override_ >= 0 ? num_sms_override_ : num_sms_;
        bool use_splitkv = effective_num_sms > 0 && rows_q <= split_q_threshold_;
        EnsureScratch(batch, rows_q, rows_k, num_heads_q, head_dim, use_splitkv);
        int q_batch_stride = rows_q * num_heads_q * head_dim;
        int q_row_stride = num_heads_q * head_dim;
        int q_head_stride = head_dim;
        int kv_batch_stride = rows_k * num_heads_kv * head_dim;
        int kv_row_stride = num_heads_kv * head_dim;
        int kv_head_stride = head_dim;
        devproc2_pi05_attention_fa2_fwd_bf16(
            q, k, v,
            out,
            softmax_lse_,
            use_splitkv ? softmax_lse_accum_ : nullptr,
            use_splitkv ? o_accum_ : nullptr,
            batch,
            rows_q,
            rows_k,
            num_heads_q,
            num_heads_kv,
            head_dim,
            q_batch_stride,
            q_row_stride,
            q_head_stride,
            kv_batch_stride,
            kv_row_stride,
            kv_head_stride,
            kv_batch_stride,
            kv_row_stride,
            kv_head_stride,
            q_batch_stride,
            q_row_stride,
            q_head_stride,
            scale,
            use_splitkv ? effective_num_sms : 0,
            stream);
    }

private:
    static int RoundUp128(int x) {
        return ((x + 127) / 128) * 128;
    }

    static int RoundHeadDim(int x) {
        if (x <= 64) return 64;
        if (x <= 96) return 96;
        if (x <= 128) return 128;
        return 256;
    }

    void EnsureScratch(int batch,
                       int rows_q,
                       int rows_k,
                       int num_heads_q,
                       int head_dim,
                       bool use_splitkv) {
        int head_dim_rounded = RoundHeadDim(head_dim);
        size_t lse = static_cast<size_t>(batch) * num_heads_q * RoundUp128(rows_q) * sizeof(float);
        if (lse > lse_bytes_) {
            if (softmax_lse_) cudaFree(softmax_lse_);
            CUDA_RT_CHECK(cudaMalloc(&softmax_lse_, lse));
            lse_bytes_ = lse;
        }
        if (!use_splitkv) return;

        int splits = std::min(128, (rows_k + 63) / 64);
        if (splits < 1) splits = 1;
        size_t lse_accum = static_cast<size_t>(splits) * batch * num_heads_q * rows_q * sizeof(float);
        size_t o_accum = static_cast<size_t>(splits) * batch * num_heads_q * rows_q * head_dim_rounded * sizeof(float);
        if (lse_accum > lse_accum_bytes_) {
            if (softmax_lse_accum_) cudaFree(softmax_lse_accum_);
            CUDA_RT_CHECK(cudaMalloc(&softmax_lse_accum_, lse_accum));
            lse_accum_bytes_ = lse_accum;
        }
        if (o_accum > o_accum_bytes_) {
            if (o_accum_) cudaFree(o_accum_);
            CUDA_RT_CHECK(cudaMalloc(&o_accum_, o_accum));
            o_accum_bytes_ = o_accum;
        }
    }

    std::mutex mu_;
    int num_sms_{0};
    int split_q_threshold_{4096};
    int num_sms_override_{-1};
    void* softmax_lse_{nullptr};
    void* softmax_lse_accum_{nullptr};
    void* o_accum_{nullptr};
    size_t lse_bytes_{0};
    size_t lse_accum_bytes_{0};
    size_t o_accum_bytes_{0};
};

Pi05FA2Runner& GlobalFA2Runner() {
    static Pi05FA2Runner runner;
    return runner;
}

thread_local cudaStream_t g_current_packed_stream = nullptr;

TensorObj* RequireTensor(const VMValue& value, const char* name) {
    if (!value.IsObjectRef()) {
        throw std::runtime_error(std::string(name) + " must be a Tensor");
    }
    auto* tensor = value.AsObjectAs<TensorObj>();
    if (!tensor) {
        throw std::runtime_error(std::string(name) + " must be a Tensor");
    }
    return tensor;
}

int64_t RequireInt(const VMValue& value, const char* name) {
    if (!value.IsInt()) {
        throw std::runtime_error(std::string(name) + " must be an int");
    }
    return value.AsInt();
}

void RequireCudaTensor(const TensorObj* tensor, const char* name) {
    if (tensor->device().device_type != kDLCUDA) {
        throw std::runtime_error(std::string(name) + " must be a CUDA tensor");
    }
}

void FP8Gemm(PackedArgs args, FP8Layout layout) {
    if (args.size() < 8 || args.size() > 9) {
        throw std::runtime_error(
            "runtime.cuda.fp8_*_bf16 expects A, B, M, N, K, A_scale, B_scale[, stream], D_out");
    }
    auto* a = RequireTensor(args[0], "A");
    auto* b = RequireTensor(args[1], "B");
    auto m = static_cast<int>(RequireInt(args[2], "M"));
    auto n = static_cast<int>(RequireInt(args[3], "N"));
    auto k = static_cast<int>(RequireInt(args[4], "K"));
    auto* a_scale = RequireTensor(args[5], "A_scale");
    auto* b_scale = RequireTensor(args[6], "B_scale");
    int output_index = args.size() == 9 ? 8 : 7;
    auto* d = RequireTensor(args[output_index], "D_out");
    RequireCudaTensor(a, "A");
    RequireCudaTensor(b, "B");
    RequireCudaTensor(d, "D");
    RequireCudaTensor(a_scale, "A_scale");
    RequireCudaTensor(b_scale, "B_scale");
    auto stream = args.size() == 9
        ? reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(RequireInt(args[7], "stream")))
        : g_current_packed_stream;
    GlobalRunner().Run(
        layout,
        a->data(),
        b->data(),
        d->data(),
        m,
        n,
        k,
        static_cast<float*>(a_scale->data()),
        static_cast<float*>(b_scale->data()),
        0.0f,
        stream);
}

void FP8GemmAccum(PackedArgs args, FP8Layout layout) {
    if (args.size() < 8 || args.size() > 9) {
        throw std::runtime_error(
            "runtime.cuda.fp8_*_bf16_accum expects A, B, D_inout, M, N, K, A_scale, B_scale[, stream]");
    }
    auto* a = RequireTensor(args[0], "A");
    auto* b = RequireTensor(args[1], "B");
    auto* d = RequireTensor(args[2], "D_inout");
    auto m = static_cast<int>(RequireInt(args[3], "M"));
    auto n = static_cast<int>(RequireInt(args[4], "N"));
    auto k = static_cast<int>(RequireInt(args[5], "K"));
    auto* a_scale = RequireTensor(args[6], "A_scale");
    auto* b_scale = RequireTensor(args[7], "B_scale");
    cudaStream_t stream = args.size() == 9
        ? reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(RequireInt(args[8], "stream")))
        : g_current_packed_stream;
    RequireCudaTensor(a, "A");
    RequireCudaTensor(b, "B");
    RequireCudaTensor(d, "D_inout");
    RequireCudaTensor(a_scale, "A_scale");
    RequireCudaTensor(b_scale, "B_scale");
    GlobalRunner().Run(
        layout,
        a->data(),
        b->data(),
        d->data(),
        m,
        n,
        k,
        static_cast<float*>(a_scale->data()),
        static_cast<float*>(b_scale->data()),
        1.0f,
        stream);
}

void BF16Gemm(PackedArgs args, BF16Layout layout) {
    if (args.size() < 6 || args.size() > 7) {
        throw std::runtime_error(
            "runtime.cuda.bf16_*_bf16 expects A, B, M, N, K[, stream], D_out");
    }
    auto* a = RequireTensor(args[0], "A");
    auto* b = RequireTensor(args[1], "B");
    auto m = static_cast<int>(RequireInt(args[2], "M"));
    auto n = static_cast<int>(RequireInt(args[3], "N"));
    auto k = static_cast<int>(RequireInt(args[4], "K"));
    int output_index = args.size() == 7 ? 6 : 5;
    auto* d = RequireTensor(args[output_index], "D_out");
    RequireCudaTensor(a, "A");
    RequireCudaTensor(b, "B");
    RequireCudaTensor(d, "D");
    auto stream = args.size() == 7
        ? reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(RequireInt(args[5], "stream")))
        : g_current_packed_stream;
    GlobalBF16Runner().Run(
        layout,
        a->data(),
        b->data(),
        d->data(),
        m,
        n,
        k,
        stream);
}

void Pi05FA2BF16(PackedArgs args) {
    if (args.size() < 11 || args.size() > 12) {
        throw std::runtime_error(
            "runtime.cuda.pi05_fa2_bf16 expects Q, K, V, rows_q, prefix_rows, suffix_rows, Hq, Hkv, D, scale_bits[, stream], O_out");
    }
    auto* q = RequireTensor(args[0], "Q");
    auto* k = RequireTensor(args[1], "K");
    auto* v = RequireTensor(args[2], "V");
    int rows_q = static_cast<int>(RequireInt(args[3], "rows_q"));
    int prefix_rows = static_cast<int>(RequireInt(args[4], "prefix_rows"));
    int suffix_rows = static_cast<int>(RequireInt(args[5], "suffix_rows"));
    int rows_k = prefix_rows + suffix_rows;
    int num_heads_q = static_cast<int>(RequireInt(args[6], "num_heads_q"));
    int num_heads_kv = static_cast<int>(RequireInt(args[7], "num_heads_kv"));
    int head_dim = static_cast<int>(RequireInt(args[8], "head_dim"));
    float scale = HostF32FromBitsI64(RequireInt(args[9], "scale_bits"));
    int output_index = args.size() == 12 ? 11 : 10;
    auto* out = RequireTensor(args[output_index], "O_out");
    RequireCudaTensor(q, "Q");
    RequireCudaTensor(k, "K");
    RequireCudaTensor(v, "V");
    RequireCudaTensor(out, "O_out");
    auto stream = args.size() == 12
        ? reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(RequireInt(args[10], "stream")))
        : g_current_packed_stream;
    GlobalFA2Runner().Run(
        q->data(),
        k->data(),
        v->data(),
        out->data(),
        rows_q,
        rows_k,
        num_heads_q,
        num_heads_kv,
        head_dim,
        scale,
        stream);
}

void Pi05FA2BF16Batched(PackedArgs args) {
    if (args.size() < 11 || args.size() > 12) {
        throw std::runtime_error(
            "runtime.cuda.pi05_fa2_bf16_batched expects Q, K, V, batch, rows_q, rows_k, Hq, Hkv, D, scale_bits[, stream], O_out");
    }
    auto* q = RequireTensor(args[0], "Q");
    auto* k = RequireTensor(args[1], "K");
    auto* v = RequireTensor(args[2], "V");
    int batch = static_cast<int>(RequireInt(args[3], "batch"));
    int rows_q = static_cast<int>(RequireInt(args[4], "rows_q"));
    int rows_k = static_cast<int>(RequireInt(args[5], "rows_k"));
    int num_heads_q = static_cast<int>(RequireInt(args[6], "num_heads_q"));
    int num_heads_kv = static_cast<int>(RequireInt(args[7], "num_heads_kv"));
    int head_dim = static_cast<int>(RequireInt(args[8], "head_dim"));
    float scale = HostF32FromBitsI64(RequireInt(args[9], "scale_bits"));
    int output_index = args.size() == 12 ? 11 : 10;
    auto* out = RequireTensor(args[output_index], "O_out");
    RequireCudaTensor(q, "Q");
    RequireCudaTensor(k, "K");
    RequireCudaTensor(v, "V");
    RequireCudaTensor(out, "O_out");
    auto stream = args.size() == 12
        ? reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(RequireInt(args[10], "stream")))
        : g_current_packed_stream;
    GlobalFA2Runner().RunBatched(
        q->data(),
        k->data(),
        v->data(),
        out->data(),
        batch,
        rows_q,
        rows_k,
        num_heads_q,
        num_heads_kv,
        head_dim,
        scale,
        stream);
}

PackedFunc MakeFP8Packed(FP8Layout layout) {
    auto* obj = new PackedFuncObj();
    obj->body = [layout](PackedArgs args) { FP8Gemm(args, layout); };
    return PackedFunc(obj);
}

PackedFunc MakeFP8AccumPacked(FP8Layout layout) {
    auto* obj = new PackedFuncObj();
    obj->body = [layout](PackedArgs args) { FP8GemmAccum(args, layout); };
    return PackedFunc(obj);
}

PackedFunc MakeBF16Packed(BF16Layout layout) {
    auto* obj = new PackedFuncObj();
    obj->body = [layout](PackedArgs args) { BF16Gemm(args, layout); };
    return PackedFunc(obj);
}

}  // namespace

void* CurrentCUDAPackedFuncStream() {
    return static_cast<void*>(g_current_packed_stream);
}

void SetCUDAPackedFuncStream(void* stream) {
    g_current_packed_stream = static_cast<cudaStream_t>(stream);
}

void RegisterCUDAPackedFuncs() {
    PackedFuncRegistry::Global().Register(
        "runtime.cuda.fp8_nn_bf16", MakeFP8Packed(FP8Layout::kNN));
    PackedFuncRegistry::Global().Register(
        "runtime.cuda.fp8_nt_bf16", MakeFP8Packed(FP8Layout::kNT));
    PackedFuncRegistry::Global().Register(
        "runtime.cuda.fp8_nn_bf16_accum", MakeFP8AccumPacked(FP8Layout::kNN));
    PackedFuncRegistry::Global().Register(
        "runtime.cuda.fp8_nt_bf16_accum", MakeFP8AccumPacked(FP8Layout::kNT));
    PackedFuncRegistry::Global().Register(
        "runtime.cuda.bf16_nn_bf16", MakeBF16Packed(BF16Layout::kNN));
    PackedFuncRegistry::Global().Register(
        "runtime.cuda.bf16_nt_bf16", MakeBF16Packed(BF16Layout::kNT));
    auto* fa2 = new PackedFuncObj();
    fa2->body = [](PackedArgs args) { Pi05FA2BF16(args); };
    PackedFuncRegistry::Global().Register(
        "runtime.cuda.pi05_fa2_bf16", PackedFunc(fa2));
    auto* fa2_batched = new PackedFuncObj();
    fa2_batched->body = [](PackedArgs args) { Pi05FA2BF16Batched(args); };
    PackedFuncRegistry::Global().Register(
        "runtime.cuda.pi05_fa2_bf16_batched", PackedFunc(fa2_batched));
}

}  // namespace devproc2

#else

#include "devproc2/runtime/cuda_gemm.h"

namespace devproc2 {

void RegisterCUDAPackedFuncs() {}

void* CurrentCUDAPackedFuncStream() {
    return nullptr;
}

void SetCUDAPackedFuncStream(void*) {}

}  // namespace devproc2

#endif  // DEVPROC2_WITH_CUDA
