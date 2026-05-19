# devproc2

A compiler and runtime for LLM inference — Python DSL frontend, VM bytecode execution, AOT Triton kernel compilation.

## 文档入口

- [docs/index.md](docs/index.md)：总文档入口，按框架开发优先组织。
- [docs/design/index.md](docs/design/index.md)：IR、pass pipeline、runtime、artifact、kernel 和 packed func 设计索引。
- [docs/pi05/index.md](docs/pi05/index.md)：OpenPI0.5 当前实现、构建、验证和性能文档入口。
- [docs/archive/index.md](docs/archive/index.md)：历史任务、旧 MVP 实施计划和已执行的 Pi0.5 重构方案。

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
- git submodules：`dlpack`、`json`、`tokenizers-cpp`

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

Pi0.5 的产品化构建流程见 [docs/pi05/13_product_build_quickstart.md](docs/pi05/13_product_build_quickstart.md)。

## 目录结构

```
devproc2/
├── CMakeLists.txt          根构建入口
├── 3rdparty/
│   ├── dlpack/             DLPack headers
│   ├── json/               nlohmann/json
│   └── tokenizers-cpp/     runtime tokenizer dependency
├── python/devproc2/
│   ├── compiler/           IR passes and compiler pipeline pieces
│   ├── frontend/           Python DSL frontend
│   ├── ir/                 IR nodes, attrs, printer, verifier
│   ├── export/             generic export pipeline
│   ├── artifact/           generic artifact builder and manifest support
│   └── models/pi05/        Pi0.5 model graph, ops, weights, CUDA backend
├── runtime/
│   ├── CMakeLists.txt
│   ├── include/devproc2/runtime/   C++ 头文件
│   └── src/                        C++ 实现
├── tools/pi05/             OpenPI/Pi0.5 producer-side utilities
├── docs/
│   ├── index.md            文档总入口
│   ├── design/             框架设计文档
│   ├── pi05/               Pi0.5 当前流程和状态文档
│   └── archive/            历史任务和旧计划
└── tests/
    ├── compiler/           Python compiler and Pi0.5 build tests
    └── ir/                 IR unit tests
```
