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
