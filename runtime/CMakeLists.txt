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
endif()

option(DEVPROC2_BUILD_TESTS "Build C++ unit tests" OFF)
enable_testing()
if(DEVPROC2_BUILD_TESTS)
    add_subdirectory(tests)
endif()
