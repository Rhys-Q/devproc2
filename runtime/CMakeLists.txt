set(DEVPROC2_RUNTIME_SOURCES
    src/object_ref.cc
    src/device_api.cc
    src/storage.cc
    src/stream.cc
    src/tensor.cc
    src/vm.cc
    src/builtins.cc
    src/packed_func.cc
    src/tokenizer.cc
    src/weight_store.cc
    src/cuda/cuda_gemm.cc
)

add_library(devproc2_runtime STATIC ${DEVPROC2_RUNTIME_SOURCES})

target_include_directories(devproc2_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(devproc2_runtime PUBLIC dlpack nlohmann_json ${CMAKE_DL_LIBS})

if(DEVPROC2_WITH_TOKENIZERS)
    target_link_libraries(devproc2_runtime PRIVATE tokenizers_cpp)
    target_compile_definitions(devproc2_runtime PUBLIC DEVPROC2_WITH_TOKENIZERS)
endif()

target_compile_options(devproc2_runtime PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)

if(DEVPROC2_WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(devproc2_runtime PRIVATE
        src/cuda/cuda_device_api.cc
        src/cuda/cuda_graph.cc
        src/cuda/cuda_module.cc
        src/cuda/cuda_kernel.cc
        src/cuda/cuda_kernel_registry.cc
        src/cuda/cuda_memory_pool.cc
    )
    target_link_libraries(devproc2_runtime
        PRIVATE
            CUDA::cuda_driver
            CUDA::cudart
            CUDA::cublasLt
    )
    target_compile_definitions(devproc2_runtime PUBLIC DEVPROC2_WITH_CUDA)
    if(DEVPROC2_WITH_CUTLASS)
        set(DEVPROC2_CUTLASS_ROOT "" CACHE PATH "Path to a CUTLASS checkout")
        if(NOT DEVPROC2_CUTLASS_ROOT AND EXISTS "/root/.agents/skills/cutlass-skill/repos/cutlass/include/cutlass/cutlass.h")
            set(DEVPROC2_CUTLASS_ROOT "/root/.agents/skills/cutlass-skill/repos/cutlass" CACHE PATH "Path to a CUTLASS checkout" FORCE)
        endif()
        if(NOT EXISTS "${DEVPROC2_CUTLASS_ROOT}/include/cutlass/cutlass.h")
            message(FATAL_ERROR "DEVPROC2_WITH_CUTLASS=ON requires DEVPROC2_CUTLASS_ROOT to point to a CUTLASS checkout")
        endif()
        target_sources(devproc2_runtime PRIVATE
            src/cuda/cutlass_fp8_gemm_sm89.cu
        )
        target_include_directories(devproc2_runtime PRIVATE
            ${DEVPROC2_CUTLASS_ROOT}/include
            ${DEVPROC2_CUTLASS_ROOT}/tools/util/include
        )
        target_compile_definitions(devproc2_runtime PUBLIC DEVPROC2_WITH_CUTLASS)
        target_compile_options(devproc2_runtime PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
    endif()
    option(DEVPROC2_WITH_PI05_FA2 "Build the Pi0.5-owned FA2 attention backend" ON)
    if(DEVPROC2_WITH_PI05_FA2)
        set(DEVPROC2_PI05_FA2_ROOT
            "${CMAKE_CURRENT_SOURCE_DIR}/../python/devproc2/models/pi05/cuda/fa2"
            CACHE PATH "Path to the Pi0.5 FA2 source subset")
        set(DEVPROC2_PI05_FA2_SRC_ROOT
            "${DEVPROC2_PI05_FA2_ROOT}/flash_attn_2_src/flash_attn")
        set(DEVPROC2_PI05_FA2_SRCS
            "${DEVPROC2_PI05_FA2_SRC_ROOT}/flash_fwd_hdim96_bf16_sm80.cu"
            "${DEVPROC2_PI05_FA2_SRC_ROOT}/flash_fwd_split_hdim96_bf16_sm80.cu"
            "${DEVPROC2_PI05_FA2_SRC_ROOT}/flash_fwd_hdim256_bf16_sm80.cu"
            "${DEVPROC2_PI05_FA2_SRC_ROOT}/flash_fwd_split_hdim256_bf16_sm80.cu"
            "${DEVPROC2_PI05_FA2_ROOT}/fa2_wrapper.cu"
        )
        foreach(_pi05_fa2_src IN LISTS DEVPROC2_PI05_FA2_SRCS)
            if(NOT EXISTS "${_pi05_fa2_src}")
                message(FATAL_ERROR "Missing Pi0.5 FA2 source: ${_pi05_fa2_src}")
            endif()
        endforeach()
        add_library(devproc2_pi05_fa2_obj OBJECT ${DEVPROC2_PI05_FA2_SRCS})
        set_target_properties(devproc2_pi05_fa2_obj PROPERTIES
            CUDA_STANDARD 17
            POSITION_INDEPENDENT_CODE ON
        )
        target_include_directories(devproc2_pi05_fa2_obj PRIVATE
            "${DEVPROC2_PI05_FA2_ROOT}"
            "${DEVPROC2_PI05_FA2_ROOT}/flash_attn_2_src"
            "${DEVPROC2_PI05_FA2_SRC_ROOT}"
            "${DEVPROC2_PI05_FA2_ROOT}/flash_attn_2_src/cutlass/include"
        )
        target_compile_definitions(devproc2_pi05_fa2_obj PRIVATE
            FA2_HAS_HDIM_96=1
            FA2_HAS_HDIM_256=1
            FA2_HAS_BF16=1
            FLASH_NAMESPACE=devproc2_pi05_fa2
        )
        target_compile_options(devproc2_pi05_fa2_obj PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                --expt-relaxed-constexpr --expt-extended-lambda
                -O3 --use_fast_math
                -U__CUDA_NO_HALF_OPERATORS__
                -U__CUDA_NO_HALF_CONVERSIONS__
                -U__CUDA_NO_HALF2_OPERATORS__
                -U__CUDA_NO_BFLOAT16_CONVERSIONS__
                $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:SHELL:-Xcompiler -Wno-deprecated-declarations>
                $<$<CXX_COMPILER_ID:MSVC>:SHELL:-Xcompiler /wd4996>
                "SHELL:-t 4"
            >
        )
        target_sources(devproc2_runtime PRIVATE $<TARGET_OBJECTS:devproc2_pi05_fa2_obj>)
        target_compile_definitions(devproc2_runtime PUBLIC DEVPROC2_WITH_PI05_FA2)
    endif()
endif()

option(DEVPROC2_BUILD_TESTS "Build C++ unit tests" OFF)
enable_testing()
if(DEVPROC2_BUILD_TESTS)
    add_subdirectory(tests)
endif()
