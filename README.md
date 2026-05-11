# devproc2

A compiler and runtime for LLM inference — Python DSL frontend, VM bytecode execution, AOT Triton kernel compilation.

## 构建 runtime

### 克隆（含子模块）

```bash
git clone --recurse-submodules https://github.com/your-org/devproc2.git
# 已有仓库补初始化：
git submodule update --init --recursive
```

### 依赖

- CMake >= 3.24
- GCC / Clang，支持 C++17
- dlpack（已作为 git submodule 包含于 `3rdparty/dlpack`，**无需单独安装**）

可选：
- CUDA Toolkit（启用 GPU 后端）
- pybind11（构建 Python binding）

### 快速构建（CPU only）

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DDEVPROC2_WITH_CUDA=OFF \
    -DDEVPROC2_BUILD_PYTHON_BINDING=OFF

cmake --build build -j$(nproc)
```

### 构建选项

| 选项 | 默认值 | 说明 |
|---|---|---|
| `DEVPROC2_WITH_CUDA` | `OFF` | 启用 CUDA 后端（需要 CUDA Toolkit） |
| `DEVPROC2_BUILD_PYTHON_BINDING` | `OFF` | 构建 pybind11 Python binding |
| `CMAKE_BUILD_TYPE` | `Debug` | `Debug` / `RelWithDebInfo` / `Release` |

### 启用 CUDA 后端

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DDEVPROC2_WITH_CUDA=ON
cmake --build build -j$(nproc)
```

## 目录结构

```
devproc2/
├── CMakeLists.txt          根构建入口
├── 3rdparty/
│   └── dlpack/             git submodule（header-only）
├── runtime/
│   ├── CMakeLists.txt
│   ├── include/devproc2/runtime/   C++ 头文件
│   └── src/                        C++ 实现
├── docs/
│   └── mvp_impl/           里程碑规划与设计文档
└── tests/
    └── runtime/            C++ 单元测试
```
