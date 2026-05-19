cmake_minimum_required(VERSION 3.24)
project(devproc2_pi05_cuda_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(_DEVPROC2_REPO_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../../../"
    ABSOLUTE)
set(DEVPROC2_REPO_ROOT "${_DEVPROC2_REPO_ROOT}" CACHE PATH "Path to the devproc2 repository root")

option(DEVPROC2_WITH_CUDA "Build the Pi0.5 CUDA backend extension" ON)
option(DEVPROC2_WITH_CUTLASS "Build the Pi0.5 CUTLASS FP8 path" ON)
option(DEVPROC2_WITH_PI05_FA2 "Build the Pi0.5-owned FA2 attention backend" ON)
option(DEVPROC2_BUILD_TESTS "Build Pi0.5 CUDA backend tests" OFF)

if(NOT DEVPROC2_WITH_CUDA)
    message(FATAL_ERROR "Pi0.5 CUDA backend build requires DEVPROC2_WITH_CUDA=ON")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures to build" FORCE)
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

if(NOT TARGET dlpack)
    if(NOT EXISTS "${DEVPROC2_REPO_ROOT}/3rdparty/dlpack/include/dlpack/dlpack.h")
        message(FATAL_ERROR "dlpack submodule not found under ${DEVPROC2_REPO_ROOT}/3rdparty/dlpack")
    endif()
    add_library(dlpack INTERFACE)
    target_include_directories(dlpack INTERFACE
        "${DEVPROC2_REPO_ROOT}/3rdparty/dlpack/include")
endif()

if(DEVPROC2_BUILD_TESTS AND NOT TARGET nlohmann_json)
    if(NOT EXISTS "${DEVPROC2_REPO_ROOT}/3rdparty/json/single_include/nlohmann/json.hpp")
        message(FATAL_ERROR "nlohmann/json submodule not found under ${DEVPROC2_REPO_ROOT}/3rdparty/json")
    endif()
    add_library(nlohmann_json INTERFACE)
    target_include_directories(nlohmann_json INTERFACE
        "${DEVPROC2_REPO_ROOT}/3rdparty/json/single_include")
endif()

set(DEVPROC2_PI05_CUDA_ROOT "${CMAKE_CURRENT_LIST_DIR}")
set(DEVPROC2_PI05_FP8_BACKEND_ROOT
    "${DEVPROC2_PI05_CUDA_ROOT}/backends/fp8_gemm")

add_library(devproc2_pi05_cuda_backend SHARED
    "${DEVPROC2_PI05_FP8_BACKEND_ROOT}/cublaslt_runner.cc"
)
set_target_properties(devproc2_pi05_cuda_backend PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "devproc2_pi05_cuda_backend"
)
target_include_directories(devproc2_pi05_cuda_backend PRIVATE
    "${DEVPROC2_REPO_ROOT}/runtime/include"
    "${DEVPROC2_PI05_FP8_BACKEND_ROOT}"
)
target_link_libraries(devproc2_pi05_cuda_backend
    PRIVATE
        dlpack
        CUDA::cudart
        CUDA::cublasLt
)
target_compile_definitions(devproc2_pi05_cuda_backend PRIVATE DEVPROC2_WITH_CUDA)

if(DEVPROC2_WITH_CUTLASS)
    set(DEVPROC2_CUTLASS_ROOT "" CACHE PATH "Path to a CUTLASS checkout")
    if(NOT DEVPROC2_CUTLASS_ROOT AND EXISTS "/root/.agents/skills/cutlass-skill/repos/cutlass/include/cutlass/cutlass.h")
        set(DEVPROC2_CUTLASS_ROOT "/root/.agents/skills/cutlass-skill/repos/cutlass" CACHE PATH "Path to a CUTLASS checkout" FORCE)
    endif()
    if(NOT EXISTS "${DEVPROC2_CUTLASS_ROOT}/include/cutlass/cutlass.h")
        message(FATAL_ERROR "DEVPROC2_WITH_CUTLASS=ON requires DEVPROC2_CUTLASS_ROOT to point to a CUTLASS checkout")
    endif()
    target_sources(devproc2_pi05_cuda_backend PRIVATE
        "${DEVPROC2_PI05_FP8_BACKEND_ROOT}/cutlass_fp8_gemm_sm89.cu"
    )
    target_include_directories(devproc2_pi05_cuda_backend PRIVATE
        "${DEVPROC2_CUTLASS_ROOT}/include"
        "${DEVPROC2_CUTLASS_ROOT}/tools/util/include"
    )
    target_compile_definitions(devproc2_pi05_cuda_backend PRIVATE DEVPROC2_WITH_CUTLASS)
    target_compile_options(devproc2_pi05_cuda_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

if(DEVPROC2_WITH_PI05_FA2)
    set(DEVPROC2_PI05_FA2_ROOT
        "${DEVPROC2_PI05_CUDA_ROOT}/fa2"
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
    target_sources(devproc2_pi05_cuda_backend PRIVATE $<TARGET_OBJECTS:devproc2_pi05_fa2_obj>)
    target_compile_definitions(devproc2_pi05_cuda_backend PRIVATE DEVPROC2_WITH_PI05_FA2)
endif()

if(DEVPROC2_BUILD_TESTS)
    enable_testing()
    if(NOT TARGET devproc2_runtime)
        set(_DEVPROC2_SAVED_BUILD_TESTS "${DEVPROC2_BUILD_TESTS}")
        set(DEVPROC2_BUILD_TESTS OFF CACHE BOOL "Build C++ unit tests" FORCE)
        add_subdirectory("${DEVPROC2_REPO_ROOT}/runtime" "${CMAKE_BINARY_DIR}/devproc2_runtime")
        set(DEVPROC2_BUILD_TESTS "${_DEVPROC2_SAVED_BUILD_TESTS}" CACHE BOOL "Build C++ unit tests" FORCE)
    endif()
    add_executable(test_pi05_cuda_gemm tests/test_pi05_cuda_gemm.cu)
    target_link_libraries(test_pi05_cuda_gemm PRIVATE devproc2_runtime CUDA::cudart)
    add_dependencies(test_pi05_cuda_gemm devproc2_pi05_cuda_backend)
    target_compile_definitions(test_pi05_cuda_gemm PRIVATE
        DEVPROC2_PI05_CUDA_BACKEND_SO="$<TARGET_FILE:devproc2_pi05_cuda_backend>"
    )
    target_compile_options(test_pi05_cuda_gemm PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>>
    )
    add_test(NAME test_pi05_cuda_gemm COMMAND test_pi05_cuda_gemm)
endif()
