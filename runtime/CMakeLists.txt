set(DEVPROC2_RUNTIME_SOURCES
    src/object_ref.cc
    src/device_api.cc
    src/storage.cc
    src/stream.cc
    src/tensor.cc
    src/vm.cc
    src/builtins.cc
    src/packed_func.cc
    src/packed_backend.cc
    src/tokenizer.cc
    src/weight_store.cc
)

add_library(devproc2_runtime STATIC ${DEVPROC2_RUNTIME_SOURCES})

target_include_directories(devproc2_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(devproc2_runtime PUBLIC dlpack nlohmann_json ${CMAKE_DL_LIBS})
target_link_options(devproc2_runtime PUBLIC
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wl,--export-dynamic>
)

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
    )
    target_compile_definitions(devproc2_runtime PUBLIC DEVPROC2_WITH_CUDA)
endif()

option(DEVPROC2_BUILD_TESTS "Build C++ unit tests" OFF)
enable_testing()
if(DEVPROC2_BUILD_TESTS)
    add_subdirectory(tests)
endif()
