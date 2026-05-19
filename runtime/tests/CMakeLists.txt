add_executable(test_x1_device_api test_x1_device_api.cc)
target_link_libraries(test_x1_device_api PRIVATE devproc2_runtime)
target_compile_options(test_x1_device_api PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)
add_test(NAME test_x1_device_api COMMAND test_x1_device_api)

add_executable(test_m9_artifact test_m9_artifact.cc)
target_link_libraries(test_m9_artifact PRIVATE devproc2_runtime)
target_compile_options(test_m9_artifact PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)
add_test(NAME test_m9_artifact COMMAND test_m9_artifact)

add_executable(test_m10_packed_func test_m10_packed_func.cc)
target_link_libraries(test_m10_packed_func PRIVATE devproc2_runtime)
target_compile_options(test_m10_packed_func PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)
add_test(NAME test_m10_packed_func COMMAND test_m10_packed_func)

add_executable(test_m11_kernel test_m11_kernel.cc)
target_link_libraries(test_m11_kernel PRIVATE devproc2_runtime)
target_compile_options(test_m11_kernel PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)
add_test(NAME test_m11_kernel COMMAND test_m11_kernel)

add_executable(test_pi05_weight_store test_pi05_weight_store.cc)
target_link_libraries(test_pi05_weight_store PRIVATE devproc2_runtime)
target_compile_options(test_pi05_weight_store PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)
add_test(NAME test_pi05_weight_store COMMAND test_pi05_weight_store)

if(DEVPROC2_WITH_TOKENIZERS)
add_executable(test_pi05_tokenizer test_pi05_tokenizer.cc)
target_link_libraries(test_pi05_tokenizer PRIVATE devproc2_runtime)
target_compile_options(test_pi05_tokenizer PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)
add_test(NAME test_pi05_tokenizer COMMAND test_pi05_tokenizer)
endif()

if(DEVPROC2_WITH_CUDA)
add_executable(test_cuda_graph test_cuda_graph.cc)
target_link_libraries(test_cuda_graph PRIVATE devproc2_runtime CUDA::cudart)
target_compile_options(test_cuda_graph PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)
add_test(NAME test_cuda_graph COMMAND test_cuda_graph)

add_executable(test_pi05_artifact_load test_pi05_artifact_load.cc)
target_link_libraries(test_pi05_artifact_load PRIVATE devproc2_runtime)
target_compile_options(test_pi05_artifact_load PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)
add_test(NAME test_pi05_artifact_load COMMAND test_pi05_artifact_load)

add_executable(test_pi05_kernel_launch test_pi05_kernel_launch.cc)
target_link_libraries(test_pi05_kernel_launch PRIVATE devproc2_runtime)
target_compile_definitions(test_pi05_kernel_launch PRIVATE
    DEVPROC2_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
target_compile_options(test_pi05_kernel_launch PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)
add_test(NAME test_pi05_kernel_launch COMMAND test_pi05_kernel_launch)

add_executable(test_pi05_cuda_gemm test_pi05_cuda_gemm.cu)
target_link_libraries(test_pi05_cuda_gemm PRIVATE devproc2_runtime)
add_dependencies(test_pi05_cuda_gemm devproc2_pi05_cuda_backend)
target_compile_definitions(test_pi05_cuda_gemm PRIVATE
    DEVPROC2_PI05_CUDA_BACKEND_SO="$<TARGET_FILE:devproc2_pi05_cuda_backend>"
)
target_compile_options(test_pi05_cuda_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>>
)
add_test(NAME test_pi05_cuda_gemm COMMAND test_pi05_cuda_gemm)

add_executable(test_pi05_denoise_oracle test_pi05_denoise_oracle.cc)
target_link_libraries(test_pi05_denoise_oracle PRIVATE devproc2_runtime)
target_compile_definitions(test_pi05_denoise_oracle PRIVATE
    DEVPROC2_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
target_compile_options(test_pi05_denoise_oracle PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)
add_test(NAME test_pi05_denoise_oracle COMMAND test_pi05_denoise_oracle)
set_tests_properties(test_pi05_denoise_oracle PROPERTIES
    ENVIRONMENT "DEVPROC2_CUBLASLT_FP8_FAST_ACCUM=0"
)

add_executable(bench_pi05_denoise bench_pi05_denoise.cc)
target_link_libraries(bench_pi05_denoise PRIVATE devproc2_runtime CUDA::cudart)
target_compile_definitions(bench_pi05_denoise PRIVATE
    DEVPROC2_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
target_compile_options(bench_pi05_denoise PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)
endif()

if(DEVPROC2_WITH_CUDA AND DEVPROC2_WITH_TOKENIZERS)
add_executable(test_pi05_sample_tokens_tokenizer test_pi05_sample_tokens_tokenizer.cc)
target_link_libraries(test_pi05_sample_tokens_tokenizer PRIVATE devproc2_runtime CUDA::cudart)
target_compile_definitions(test_pi05_sample_tokens_tokenizer PRIVATE
    DEVPROC2_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
target_compile_options(test_pi05_sample_tokens_tokenizer PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)
add_test(NAME test_pi05_sample_tokens_tokenizer COMMAND test_pi05_sample_tokens_tokenizer)
endif()
