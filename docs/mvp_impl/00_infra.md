# devproc2 项目基建

本文档描述从零搭建 devproc2 项目所需的全部基础设施：目录骨架、Python uv 环境、C++ CMake 构建系统、工具链配置。

---

## 1. 完整目录结构

```
devproc2/
│
├── python/
│   └── devproc2/
│       ├── __init__.py
│       │
│       ├── ir/
│       │   ├── __init__.py
│       │   ├── module.py          # IRModule
│       │   ├── function.py        # Function, Param
│       │   ├── block.py           # Block（BasicBlock / BindingBlock）
│       │   ├── expr.py            # Expr 基类, Var, Constant
│       │   ├── call.py            # Call, CallDPS
│       │   ├── control_flow.py    # If, For, Range
│       │   ├── tuple.py           # Tuple, TupleGetItem
│       │   ├── tensor_create.py   # TensorCreateOp（empty/zeros/full/empty_like）
│       │   ├── struct_info.py     # TensorStructInfo, TupleStructInfo
│       │   ├── shape_expr.py      # SymbolicDim, UpperBound, ShapeConstraint
│       │   ├── effect.py          # EffectInfo（pure/read_only/write/opaque）
│       │   ├── printer.py         # IR 文本打印器
│       │   └── verifier.py        # IR 不变量检查器
│       │
│       ├── frontend/
│       │   ├── __init__.py
│       │   ├── dsl.py             # @dp.function, @dp.kernel 装饰器
│       │   └── builder.py         # IRBuilder：从 Python AST 生成 IR 节点
│       │
│       ├── ops/
│       │   ├── __init__.py
│       │   ├── tensor.py          # matmul, add, relu, silu, gelu, softmax, ...
│       │   ├── nn.py              # qkv_proj, attention, embedding, layernorm, rmsnorm, ...
│       │   ├── fused.py           # matmul_add_silu, fused_rmsnorm, ...
│       │   └── stateful.py        # update_kvcache, attention_with_cache, ...
│       │
│       ├── kernel/
│       │   ├── __init__.py
│       │   ├── register.py        # KernelRegistry, kernel 匹配逻辑
│       │   ├── kernel_spec.py     # KernelSpec, KernelMatchKey, KernelABI
│       │   └── triton_kernel.py   # Triton AOT compile 包装器
│       │
│       ├── compiler/
│       │   ├── __init__.py
│       │   ├── build.py           # 顶层 dp.build(module, target) 入口
│       │   ├── pipeline.py        # PassManager, PassPipeline
│       │   └── passes/
│       │       ├── __init__.py
│       │       ├── normalize.py
│       │       ├── control_flow_normalize.py
│       │       ├── infer_struct_info.py
│       │       ├── dynamic_shape_analyze.py
│       │       ├── shape_constraint_verify.py
│       │       ├── effect_analyze.py
│       │       ├── kernel_select.py
│       │       ├── dps_lowering.py
│       │       ├── tensor_create_analyze.py
│       │       ├── lifetime_analyze.py
│       │       ├── storage_size_analyze.py
│       │       ├── storage_plan.py
│       │       ├── lower_tensor_create_to_alloc.py
│       │       ├── shape_expr_lowering.py
│       │       ├── kernel_launch_expr_lowering.py
│       │       ├── vm_codegen.py
│       │       ├── triton_aot_compile.py
│       │       ├── emit_executable.py
│       │       └── emit_abi.py
│       │
│       ├── vm/
│       │   ├── __init__.py
│       │   ├── bytecode.py        # Bytecode 常量, Opcode 枚举
│       │   ├── instruction.py     # Instruction dataclass（opcode + 操作数）
│       │   ├── executable.py      # Executable：function_table + instructions + constants
│       │   └── serializer.py      # 序列化/反序列化到字节流
│       │
│       ├── runtime/
│       │   ├── __init__.py
│       │   └── binding.py         # pybind11 桥接到 C++ runtime
│       │
│       └── testing/
│           ├── __init__.py
│           ├── verify.py          # 数值正确性校验（对比 PyTorch）
│           └── benchmark.py       # latency / throughput 测量
│
├── runtime/                        # C++ runtime
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── devproc2/
│   │       └── runtime/
│   │           ├── object.h
│   │           ├── object_ref.h
│   │           ├── vm_value.h
│   │           ├── tensor.h
│   │           ├── storage.h
│   │           ├── shape_tuple.h
│   │           ├── tuple.h
│   │           ├── string.h
│   │           ├── packed_func.h
│   │           ├── kernel.h
│   │           ├── executable.h
│   │           ├── vm.h
│   │           ├── state.h
│   │           ├── memory_pool.h
│   │           ├── device_api.h   # DeviceType, Device, DeviceAPI, DeviceAPIRegistry
│   │           └── stream.h       # StreamObj, Stream
│   ├── src/
│   │   ├── object.cc
│   │   ├── object_ref.cc
│   │   ├── vm_value.cc
│   │   ├── tensor.cc
│   │   ├── storage.cc
│   │   ├── shape_tuple.cc
│   │   ├── tuple.cc
│   │   ├── string.cc
│   │   ├── packed_func.cc
│   │   ├── kernel.cc
│   │   ├── executable.cc
│   │   ├── vm.cc
│   │   ├── state.cc
│   │   ├── memory_pool.cc
│   │   ├── device_api.cc
│   │   ├── builtins.cc            # vm.builtin.* 函数实现
│   │   └── cuda/
│   │       ├── cuda_device_api.cc
│   │       ├── cuda_module.cc     # cubin load + cuLaunchKernel 包装
│   │       ├── cuda_kernel.cc
│   │       └── cuda_memory_pool.cc
│   └── binding/
│       └── python_binding.cc      # pybind11 模块入口
│
├── kernels/
│   └── triton/
│       ├── add.py
│       ├── relu.py
│       ├── matmul.py
│       ├── layernorm.py
│       ├── matmul_add_silu.py
│       └── update_kvcache.py
│
├── examples/
│   ├── static_graph_mvp/
│   ├── control_flow_mvp/
│   ├── dynamic_shape_mvp/
│   ├── tokenizer_mvp/
│   ├── kv_cache_mvp/
│   └── fused_kernel_mvp/
│
├── tests/
│   ├── ir/
│   ├── compiler/
│   ├── runtime/
│   └── integration/
│
├── tools/
│   └── devproc_cli.py             # devproc compile / run / inspect / bench / dump-ir
│
├── scripts/
│   ├── build.sh
│   ├── test.sh
│   └── lint.sh
│
├── CMakeLists.txt                 # 根目录 CMake
├── pyproject.toml
├── .python-version
├── .gitignore
└── README.md
```

---

## 2. Python uv 环境

### pyproject.toml

放在项目根目录：

```toml
[project]
name = "devproc2"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy==1.26.4",
    "triton==3.6.0",
    "torch==2.10.0",
]

[project.optional-dependencies]
dev = [
    "pytest==8.3.5",
    "pytest-xdist==3.6.1",
    "ruff==0.9.10",
    "mypy==1.15.0",
]
binding = [
    "pybind11==2.13.6",
]

[build-system]
requires = ["setuptools>=70", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[tool.setuptools.packages.find]
where = ["python"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
ignore = ["E501"]  # 长行由格式化器处理

[tool.ruff.lint.isort]
known-first-party = ["devproc2"]

[tool.mypy]
python_version = "3.11"
strict = false
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

### .python-version

```
3.11
```

### uv cache 配置

**`.uv/cache-config.toml`**（可选，用于 CI 配置本地 cache 路径）：

```toml
[cache]
dir = ".uv-cache"
```

### 常用 uv 命令

```bash
# 初始化（首次）
uv sync                          # 安装所有基础依赖
uv sync --extra dev              # 安装开发依赖（pytest/ruff/mypy）
uv sync --extra dev --extra binding  # 安装全部依赖

# 日常开发
uv run pytest tests/             # 运行全量测试
uv run pytest tests/ir/          # 只跑 IR 层测试
uv run ruff check python/        # 代码风格检查
uv run ruff format python/       # 自动格式化
uv run mypy python/devproc2/     # 类型检查
uv run python -m devproc2 ...    # 运行 devproc2 CLI

# 锁文件管理
uv lock                          # 生成/更新 uv.lock
uv lock --upgrade                # 升级所有依赖到最新兼容版本
```

### GitHub Actions CI cache

```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v4

- name: Cache uv
  uses: actions/cache@v4
  with:
    path: |
      ~/.cache/uv
      .uv-cache
    key: uv-${{ runner.os }}-${{ hashFiles('uv.lock') }}
    restore-keys: |
      uv-${{ runner.os }}-

- name: Install dependencies
  run: uv sync --extra dev
```

---

## 3. C++ CMake 构建系统

### 根目录 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.24)
project(devproc2 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 构建选项
option(DEVPROC2_WITH_CUDA "Build CUDA backend" ON)
option(DEVPROC2_BUILD_PYTHON_BINDING "Build Python pybind11 binding" ON)
option(DEVPROC2_BUILD_TESTS "Build C++ unit tests" ON)

# 编译优化
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# 依赖
if(DEVPROC2_WITH_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  find_package(CUDAToolkit REQUIRED)
endif()

if(DEVPROC2_BUILD_PYTHON_BINDING)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
endif()

# 子目录
add_subdirectory(runtime)

if(DEVPROC2_BUILD_TESTS)
  enable_testing()
  # 如果需要 GTest，可在这里添加
  # find_package(GTest CONFIG REQUIRED)
endif()
```

### runtime/CMakeLists.txt

```cmake
# ─── 核心 C++ runtime（CPU only）───────────────────────────────────────────
set(DEVPROC2_RUNTIME_SOURCES
    src/object.cc
    src/object_ref.cc
    src/vm_value.cc
    src/tensor.cc
    src/storage.cc
    src/shape_tuple.cc
    src/tuple.cc
    src/string.cc
    src/packed_func.cc
    src/kernel.cc
    src/executable.cc
    src/vm.cc
    src/state.cc
    src/memory_pool.cc
    src/device_api.cc
    src/builtins.cc
)

add_library(devproc2_runtime SHARED ${DEVPROC2_RUNTIME_SOURCES})

target_include_directories(devproc2_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(devproc2_runtime PUBLIC cxx_std_17)

target_compile_options(devproc2_runtime PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)

# ─── CUDA 后端（可选）──────────────────────────────────────────────────────
if(DEVPROC2_WITH_CUDA)
    target_sources(devproc2_runtime PRIVATE
        src/cuda/cuda_device_api.cc
        src/cuda/cuda_module.cc
        src/cuda/cuda_kernel.cc
        src/cuda/cuda_memory_pool.cc
    )
    target_link_libraries(devproc2_runtime PRIVATE CUDA::cuda_driver)
    target_compile_definitions(devproc2_runtime PUBLIC DEVPROC2_WITH_CUDA)
endif()

# ─── Python binding（可选）─────────────────────────────────────────────────
if(DEVPROC2_BUILD_PYTHON_BINDING)
    pybind11_add_module(devproc2_cpp binding/python_binding.cc)
    target_link_libraries(devproc2_cpp PRIVATE devproc2_runtime)
    # 输出到 python/devproc2/ 目录，便于直接 import
    set_target_properties(devproc2_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/devproc2/runtime
    )
endif()
```

### 构建命令

```bash
# 标准构建（CUDA + Python binding）
cmake -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DDEVPROC2_WITH_CUDA=ON \
    -DDEVPROC2_BUILD_PYTHON_BINDING=ON
cmake --build build -j$(nproc)

# CPU-only 构建（不需要 GPU 的环境）
cmake -B build_cpu \
    -DCMAKE_BUILD_TYPE=Debug \
    -DDEVPROC2_WITH_CUDA=OFF \
    -DDEVPROC2_BUILD_PYTHON_BINDING=ON
cmake --build build_cpu -j$(nproc)

# 清理
rm -rf build build_cpu
```

### scripts/build.sh

```bash
#!/bin/bash
set -euo pipefail

BUILD_TYPE=${BUILD_TYPE:-RelWithDebInfo}
WITH_CUDA=${WITH_CUDA:-ON}

echo "[devproc2] Building with BUILD_TYPE=${BUILD_TYPE} WITH_CUDA=${WITH_CUDA}"

cmake -B build \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DDEVPROC2_WITH_CUDA="${WITH_CUDA}" \
    -DDEVPROC2_BUILD_PYTHON_BINDING=ON

cmake --build build -j"$(nproc)"
echo "[devproc2] Build done."
```

---

## 4. 工具链配置

### .gitignore

```gitignore
# Build artifacts
build/
build_cpu/
*.so
*.dylib
*.dll
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.mypy_cache/
.ruff_cache/

# uv
.venv/
.uv-cache/
uv.lock   # 如果不想提交 lockfile，可以取消注释

# CMake
CMakeFiles/
CMakeCache.txt
cmake_install.cmake
Makefile

# Compiled artifacts
python/devproc2/runtime/*.so
python/devproc2/runtime/*.pyd

# Editor
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

### .clang-format

```yaml
BasedOnStyle: Google
IndentWidth: 2
ColumnLimit: 100
BreakBeforeBraces: Attach
AllowShortFunctionsOnASingleLine: Empty
SortIncludes: CaseSensitive
```

### .pre-commit-config.yaml

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.7
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-clang-format
    rev: v18.1.5
    hooks:
      - id: clang-format
        types_or: [c, c++, cuda]
        args: [--style=file]
```

安装：`uv run pre-commit install`

### scripts/test.sh

```bash
#!/bin/bash
set -euo pipefail

echo "[devproc2] Running Python tests..."
uv run pytest tests/ -v --tb=short "$@"

echo "[devproc2] All tests passed."
```

### scripts/lint.sh

```bash
#!/bin/bash
set -euo pipefail

echo "[devproc2] Running ruff..."
uv run ruff check python/
uv run ruff format --check python/

echo "[devproc2] Running mypy..."
uv run mypy python/devproc2/

echo "[devproc2] Lint passed."
```

---

## 5. 第一步：创建目录骨架

在实施开始前，先执行以下命令创建完整目录骨架：

```bash
# Python 包目录
mkdir -p python/devproc2/{ir,frontend,ops,kernel,compiler/passes,vm,runtime,testing}
touch python/devproc2/__init__.py
touch python/devproc2/{ir,frontend,ops,kernel,compiler,compiler/passes,vm,runtime,testing}/__init__.py

# C++ runtime
mkdir -p runtime/include/devproc2/runtime
mkdir -p runtime/src/cuda
mkdir -p runtime/binding

# Triton kernels
mkdir -p kernels/triton

# Examples
mkdir -p examples/{static_graph_mvp,control_flow_mvp,dynamic_shape_mvp,tokenizer_mvp,kv_cache_mvp,fused_kernel_mvp}

# Tests
mkdir -p tests/{ir,compiler,runtime,integration}
touch tests/{ir,compiler,runtime,integration}/__init__.py

# Tools & scripts
mkdir -p tools scripts

# Build output placeholder（不提交）
echo "build/" >> .gitignore
```
