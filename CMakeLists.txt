cmake_minimum_required(VERSION 3.24)
project(devproc2 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DEVPROC2_WITH_CUDA          "Build CUDA backend"             OFF)
option(DEVPROC2_BUILD_PYTHON_BINDING "Build Python pybind11 binding" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# ── dlpack (git submodule, header-only) ───────────────────────────────────────
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include/dlpack/dlpack.h")
  message(FATAL_ERROR
    "dlpack submodule not found. Run:\n"
    "  git submodule update --init --recursive")
endif()
add_library(dlpack INTERFACE)
target_include_directories(dlpack INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include)

add_subdirectory(runtime)
