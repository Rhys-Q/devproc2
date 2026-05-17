#ifdef DEVPROC2_WITH_CUDA
#ifdef DEVPROC2_WITH_CUTLASS

#include "cutlass_fp8_gemm_sm89.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

namespace devproc2 {
namespace {

#define DEVPROC2_CUTLASS_CHECK(expr)                                             \
    do {                                                                        \
        cutlass::Status _st = (expr);                                            \
        if (_st != cutlass::Status::kSuccess) {                                  \
            throw std::runtime_error(                                            \
                std::string("CUTLASS error in " #expr ": ") +                  \
                cutlassGetStatusString(_st));                                    \
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
        cutlass::multiplies<FragmentCompute> mul_source;
        cutlass::multiply_add<FragmentCompute> madd_accumulator;
        FragmentCompute intermediate = mul_source(beta_, converted_source);
        intermediate = madd_accumulator(alpha_, converted_accumulator, intermediate);
        return destination_converter(intermediate);
    }

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
            accumulator_converter;
        cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
            destination_converter;
        cutlass::multiplies<FragmentCompute> mul_accumulator;
        return destination_converter(mul_accumulator(alpha_, accumulator_converter(accumulator)));
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

using Gemm128x128 =
    Sm89FP8Gemm<
        cutlass::gemm::GemmShape<128, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 128>>;

template <typename Gemm>
void RunGemm(void* a,
             void* b,
             void* d,
             int m,
             int n,
             int k,
             float* a_scale,
             float* b_scale,
             float beta,
             cudaStream_t stream) {
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        cutlass::gemm::GemmCoord(m, n, k),
        1,
        typename Gemm::EpilogueOutputOp::Params(a_scale, b_scale, beta),
        a,
        b,
        d,
        d,
        static_cast<int64_t>(m) * k,
        static_cast<int64_t>(n) * k,
        static_cast<int64_t>(m) * n,
        static_cast<int64_t>(m) * n,
        k,
        k,
        n,
        n);
    DEVPROC2_CUTLASS_CHECK(Gemm::can_implement(args));
    Gemm gemm;
    DEVPROC2_CUTLASS_CHECK(gemm(args, nullptr, stream));
}

}  // namespace

bool CutlassFP8NTBF16CanRun(int m, int n, int k, float beta) {
    if (beta != 0.0f) return false;
    if (m == 512 && n == 1152 && k == 4304) return true;
    if (m == 768 && n == 1152 && k == 4304) return true;
    return false;
}

void CutlassFP8NTBF16Run(void* a,
                         void* b,
                         void* d,
                         int m,
                         int n,
                         int k,
                         float* a_scale,
                         float* b_scale,
                         float beta,
                         cudaStream_t stream) {
    if (m == 512 && n == 1152 && k == 4304) {
        RunGemm<Gemm128x64>(a, b, d, m, n, k, a_scale, b_scale, beta, stream);
        return;
    }
    if (m == 768 && n == 1152 && k == 4304) {
        RunGemm<Gemm128x64>(a, b, d, m, n, k, a_scale, b_scale, beta, stream);
        return;
    }
    throw std::runtime_error("unsupported CUTLASS FP8 NT BF16 shape");
}

}  // namespace devproc2

#endif  // DEVPROC2_WITH_CUTLASS
#endif  // DEVPROC2_WITH_CUDA
