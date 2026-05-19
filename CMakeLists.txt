cmake_minimum_required(VERSION 3.24)
project(devproc2 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DEVPROC2_WITH_CUDA          "Build CUDA backend"             OFF)
option(DEVPROC2_WITH_CUTLASS       "Build optional CUTLASS CUDA kernels" OFF)
option(DEVPROC2_WITH_TOKENIZERS    "Build tokenizers-cpp runtime hooks" OFF)
option(DEVPROC2_BUILD_PYTHON_BINDING "Build Python pybind11 binding" OFF)

if(DEVPROC2_WITH_CUDA AND NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;89" CACHE STRING "CUDA architectures to build" FORCE)
endif()

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

# ── nlohmann/json (git submodule, header-only) ────────────────────────────────
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/json/single_include/nlohmann/json.hpp")
  message(FATAL_ERROR
    "nlohmann/json submodule not found. Run:\n"
    "  git submodule update --init --recursive")
endif()
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/json/single_include)

if(DEVPROC2_WITH_TOKENIZERS)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tokenizers-cpp/include/tokenizers_cpp.h")
    message(FATAL_ERROR
      "tokenizers-cpp submodule not found. Run:\n"
      "  git submodule update --init --recursive")
  endif()
  add_subdirectory(3rdparty/tokenizers-cpp EXCLUDE_FROM_ALL)
endif()

add_subdirectory(runtime)
