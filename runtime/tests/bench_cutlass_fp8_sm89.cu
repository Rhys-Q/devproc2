// Standalone CUTLASS SM89 FP8 GEMM benchmark for Pi0.5 candidate shapes.
//
// This intentionally stays outside the default test list.  Build manually with
// nvcc when evaluating whether a shape-specialized CUTLASS path is worth
// wiring into runtime.cuda.fp8_nt_bf16.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

namespace {

#define CUDA_CHECK(expr)                                                         \
    do {                                                                        \
        cudaError_t _err = (expr);                                               \
        if (_err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " #expr ": "                            \
                      << cudaGetErrorString(_err) << "\n";                      \
            std::exit(1);                                                        \
        }                                                                       \
    } while (0)

#define CUTLASS_CHECK(expr)                                                      \
    do {                                                                        \
        cutlass::Status _st = (expr);                                            \
        if (_st != cutlass::Status::kSuccess) {                                  \
            std::cerr << "CUTLASS error in " #expr ": "                         \
                      << cutlassGetStatusString(_st) << "\n";                   \
            std::exit(1);                                                        \
        }                                                                       \
    } while (0)

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

template <
    typename ElementOutput_,
    int Count,
    typename ElementAccumulator_ = ElementOutput_,
    typename ElementCompute_ = ElementOutput_,
    cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest,
    typename ElementSource_ = ElementOutput_>
class LinearCombinationTwoScale {
public:
    using ElementOutput = ElementOutput_;
    using ElementSource = ElementSource_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    using ElementScalar = ElementCompute;
    using ElementC = ElementSource_;
    using ElementD = ElementOutput_;

    static int const kCount = Count;
    using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
    using FragmentSource = cutlass::Array<ElementSource, kCount>;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
    using FragmentCompute = cutlass::Array<ElementCompute, kCount>;
    static cutlass::FloatRoundStyle const kRound = Round;

    struct Params {
        ElementCompute const* a_scale;
        ElementCompute const* b_scale;
        ElementCompute beta;

        CUTLASS_HOST_DEVICE
        Params() : a_scale(nullptr), b_scale(nullptr), beta(ElementCompute(0)) {}

        CUTLASS_HOST_DEVICE
        Params(ElementCompute const* a_scale,
               ElementCompute const* b_scale,
               ElementCompute beta = ElementCompute(0))
            : a_scale(a_scale), b_scale(b_scale), beta(beta) {}
    };

private:
    ElementCompute alpha_;
    ElementCompute beta_;

public:
    CUTLASS_HOST_DEVICE
    explicit LinearCombinationTwoScale(Params const& params, int) {
        ElementCompute a = params.a_scale ? *params.a_scale : ElementCompute(1);
        ElementCompute b = params.b_scale ? *params.b_scale : ElementCompute(1);
        alpha_ = a * b;
        beta_ = params.beta;
    }

    CUTLASS_HOST_DEVICE
    explicit LinearCombinationTwoScale(Params const& params)
        : LinearCombinationTwoScale(params, 0) {}

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const {
        return beta_ != ElementCompute(0);
    }

    CUTLASS_HOST_DEVICE
    void set_k_partition(int k_partition, int) {
        if (k_partition) {
            beta_ = ElementCompute(1);
        }
    }

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& accumulator,
                              FragmentSource const& source) const {
        cutlass::NumericArrayConverter<ElementCompute, ElementSource, kCount, Round>
            source_converter;
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
            accumulator_converter;
        cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
            destination_converter;
        FragmentCompute converted_source = source_converter(source);
        FragmentCompute converted_accumulator = accumulator_converter(accumulator);
        FragmentCompute intermediate;
        cutlass::multiplies<FragmentCompute> mul_source;
        cutlass::multiply_add<FragmentCompute> madd_accumulator;
        intermediate = mul_source(beta_, converted_source);
        intermediate = madd_accumulator(alpha_, converted_accumulator, intermediate);
        return destination_converter(intermediate);
    }

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
            accumulator_converter;
        cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
            destination_converter;
        FragmentCompute converted_accumulator = accumulator_converter(accumulator);
        cutlass::multiplies<FragmentCompute> mul_accumulator;
        return destination_converter(mul_accumulator(alpha_, converted_accumulator));
    }

    CUTLASS_HOST_DEVICE
    ElementD operator()(ElementAccumulator const accumulator, ElementC const source) const {
        cutlass::NumericConverter<ElementCompute, ElementAccumulator, Round>
            accumulator_converter;
        cutlass::NumericConverter<ElementCompute, ElementC, Round> source_converter;
        cutlass::NumericConverter<ElementD, ElementCompute, Round> destination_converter;
        cutlass::multiply_add<ElementCompute> madd;
        return destination_converter(
            madd(alpha_, accumulator_converter(accumulator), beta_ * source_converter(source)));
    }

    CUTLASS_HOST_DEVICE
    ElementD operator()(ElementAccumulator const accumulator) const {
        cutlass::NumericConverter<ElementCompute, ElementAccumulator, Round>
            accumulator_converter;
        cutlass::NumericConverter<ElementD, ElementCompute, Round> destination_converter;
        return destination_converter(alpha_ * accumulator_converter(accumulator));
    }
};

using EpilogueOutputOp =
    LinearCombinationTwoScale<ElementOutput, 8, ElementAccumulator, ElementAccumulator>;

template <typename ThreadblockShape, typename WarpShape>
using Sm89FP8Gemm =
    cutlass::gemm::device::GemmUniversal<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementOutput,
        LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm89,
        ThreadblockShape,
        WarpShape,
        cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3,
        16,
        16,
        cutlass::arch::OpMultiplyAddFastAccum>;

using Gemm128x64 =
    Sm89FP8Gemm<
        cutlass::gemm::GemmShape<128, 64, 128>,
        cutlass::gemm::GemmShape<64, 32, 128>>;

using Gemm64x128 =
    Sm89FP8Gemm<
        cutlass::gemm::GemmShape<64, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 128>>;

using Gemm128x128 =
    Sm89FP8Gemm<
        cutlass::gemm::GemmShape<128, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 128>>;

struct Shape {
    const char* name;
    int m;
    int n;
    int k;
};

template <typename Gemm>
float run_gemm(const Shape& shape, int warmup, int iters, cudaStream_t stream) {
    void* a = nullptr;
    void* b = nullptr;
    void* d = nullptr;
    float* a_scale = nullptr;
    float* b_scale = nullptr;
    const size_t a_bytes =
        static_cast<size_t>(shape.m) * static_cast<size_t>(shape.k) * sizeof(ElementA);
    const size_t b_bytes =
        static_cast<size_t>(shape.n) * static_cast<size_t>(shape.k) * sizeof(ElementB);
    const size_t d_bytes =
        static_cast<size_t>(shape.m) * static_cast<size_t>(shape.n) * sizeof(ElementOutput);
    CUDA_CHECK(cudaMalloc(&a, a_bytes));
    CUDA_CHECK(cudaMalloc(&b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d, d_bytes));
    CUDA_CHECK(cudaMalloc(&a_scale, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_scale, sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(a, 0x38, a_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(b, 0x38, b_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(d, 0, d_bytes, stream));
    float one = 1.0f;
    CUDA_CHECK(cudaMemcpyAsync(a_scale, &one, sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(b_scale, &one, sizeof(float), cudaMemcpyHostToDevice, stream));

    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        cutlass::gemm::GemmCoord(shape.m, shape.n, shape.k),
        1,
        typename Gemm::EpilogueOutputOp::Params(a_scale, b_scale, 0.0f),
        a,
        b,
        d,
        d,
        static_cast<int64_t>(shape.m) * shape.k,
        static_cast<int64_t>(shape.n) * shape.k,
        static_cast<int64_t>(shape.m) * shape.n,
        static_cast<int64_t>(shape.m) * shape.n,
        shape.k,
        shape.k,
        shape.n,
        shape.n);
    cutlass::Status status = Gemm::can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        CUDA_CHECK(cudaFree(a));
        CUDA_CHECK(cudaFree(b));
        CUDA_CHECK(cudaFree(d));
        CUDA_CHECK(cudaFree(a_scale));
        CUDA_CHECK(cudaFree(b_scale));
        return std::numeric_limits<float>::quiet_NaN();
    }
    size_t workspace_size = Gemm::get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size != 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    Gemm gemm;
    status = gemm.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) CUDA_CHECK(cudaFree(workspace));
        CUDA_CHECK(cudaFree(a));
        CUDA_CHECK(cudaFree(b));
        CUDA_CHECK(cudaFree(d));
        CUDA_CHECK(cudaFree(a_scale));
        CUDA_CHECK(cudaFree(b_scale));
        return std::numeric_limits<float>::quiet_NaN();
    }
    for (int i = 0; i < warmup; ++i) {
        status = gemm.run(stream);
        if (status != cutlass::Status::kSuccess) {
            if (workspace) CUDA_CHECK(cudaFree(workspace));
            CUDA_CHECK(cudaFree(a));
            CUDA_CHECK(cudaFree(b));
            CUDA_CHECK(cudaFree(d));
            CUDA_CHECK(cudaFree(a_scale));
            CUDA_CHECK(cudaFree(b_scale));
            return std::numeric_limits<float>::quiet_NaN();
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        status = gemm.run(stream);
        if (status != cutlass::Status::kSuccess) {
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            if (workspace) CUDA_CHECK(cudaFree(workspace));
            CUDA_CHECK(cudaFree(a));
            CUDA_CHECK(cudaFree(b));
            CUDA_CHECK(cudaFree(d));
            CUDA_CHECK(cudaFree(a_scale));
            CUDA_CHECK(cudaFree(b_scale));
            return std::numeric_limits<float>::quiet_NaN();
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    if (workspace) CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFree(a_scale));
    CUDA_CHECK(cudaFree(b_scale));
    return elapsed_ms / static_cast<float>(iters);
}

template <typename Gemm>
void print_result(const char* kernel, const Shape& shape, int warmup, int iters, cudaStream_t stream) {
    float ms = run_gemm<Gemm>(shape, warmup, iters, stream);
    if (ms != ms) {
        std::cout << std::left << std::setw(18) << shape.name
                  << std::setw(12) << kernel
                  << " unsupported\n";
        return;
    }
    double flops = 2.0 * static_cast<double>(shape.m) * shape.n * shape.k;
    double tflops = flops / (static_cast<double>(ms) / 1000.0) / 1.0e12;
    std::cout << std::left << std::setw(18) << shape.name
              << std::setw(12) << kernel
              << " m=" << std::setw(4) << shape.m
              << " n=" << std::setw(6) << shape.n
              << " k=" << std::setw(6) << shape.k
              << "  " << std::fixed << std::setprecision(4) << ms << " ms"
              << "  " << std::setprecision(1) << tflops << " TFLOP/s\n";
}

}  // namespace

int main(int argc, char** argv) {
    int warmup = 20;
    int iters = 100;
    if (argc > 1) iters = std::max(1, std::atoi(argv[1]));
    if (argc > 2) warmup = std::max(0, std::atoi(argv[2]));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::vector<Shape> shapes = {
        {"vision_qkv", 512, 3456, 1152},
        {"vision_ffn_up", 512, 4304, 1152},
        {"vision_ffn_down", 512, 1152, 4304},
        {"vision_qkv_3v", 768, 3456, 1152},
        {"vision_ffn_up_3v", 768, 4304, 1152},
        {"vision_ffn_down_3v", 768, 1152, 4304},
        {"prefix_qkv_2v", 562, 2560, 2048},
        {"prefix_ffn_up_2v", 562, 32768, 2048},
        {"prefix_ffn_down_2v", 562, 2048, 16384},
        {"prefix_ffn_up_3v", 895, 32768, 2048},
        {"prefix_ffn_down_3v", 895, 2048, 16384},
        {"decoder_ffn_up", 50, 8192, 1024},
        {"decoder_ffn_down", 50, 1024, 4096},
    };

    for (const Shape& shape : shapes) {
        print_result<Gemm128x64>("tb128x64", shape, warmup, iters, stream);
        print_result<Gemm64x128>("tb64x128", shape, warmup, iters, stream);
        print_result<Gemm128x128>("tb128x128", shape, warmup, iters, stream);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
