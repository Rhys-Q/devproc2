# DevProc2 Kernel Registry & Attr System (MVP Design)

## 1. Overview

DevProc2 uses a **two-stage kernel dispatch system** built on:

- Structured IR attributes (Attr)
- Hierarchical kernel registry
- Two-level matching (index + predicate)

目标是实现：

> 在保持 IR 通用性的前提下，高效将 op lowering 到具体 kernel（如 Triton）

---

## 2. Attr System

### 2.1 定义

Attr 是 op 的语义参数集合（semantic parameters）：

- compile-time 可见
- runtime 可传递
- 仅包含可序列化的 primitive values

### 2.2 支持类型

Attr 只允许：

- int
- float
- bool
- string
- tuple/list of above

❌ 不允许：

- Tensor
- PrimExpr
- Symbolic graph
- Runtime handle

### 2.3 IR 表示

Call(
    op="matmul",
    inputs=[a, b],
    attrs={
        "transpose_a": False,
        "transpose_b": False,
    }
)

### 2.4 生命周期

- IR：✔
- match：✔
- runtime：✔（作为 kernel 参数）

---

## 3. Kernel Registry Design

### 3.1 三层 dispatch

1. op_name
2. input dtype signature
3. match function

---

### 3.2 Registry 结构

registry = {
    op_name: {
        dtype_signature: [KernelSpec]
    }
}

---

### 3.3 dtype_signature

("fp16","fp16")
("int8","int8")

---

## 4. KernelSpec

@dataclass(frozen=True)
class KernelSpec:
    # ── 查表 key ──
    op_name:      str
    device:       str
    input_dtypes: tuple[str, ...]

    # ── 实现标识 ──
    kernel_name:  str                  # CallDPSOp 中用的名字，如 "kernel.relu_fp16"
    backend:      str = "triton"       # 编译后端：triton | cuda_c | python | llvm

    # ── 调度过滤 ──
    sm_arches:    tuple[int, ...] = () # () = 不限 SM
    priority:     int = 0
    match:        callable | None = None

    # ── launch 配置 ──
    grid_fn:      callable | None = None  # 返回 (grid_x, grid_y, grid_z)
                                           #   静态 shape → grid_fn(shapes: list[tuple])
                                           #   动态 shape → grid_fn() 无参回退
    num_warps:    int = 4
    num_stages:   int = 3
    block_size:   int = 256
    smem_bytes:   int = 0

    # ── 编译器透传 ──
    launch_kwargs: dict = {}

---

## 5. Matching Pipeline

1. op filter
2. dtype filter
3. match()
4. priority select

---

## 6. match function

def match(call):
    return call.M % 16 == 0

---

## 7. Launch Rule

grid = ceildiv(M,128)

---

## 8. Principles

- IR 不包含 kernel 信息
- attr = semantic parameter
- launch = runtime concern
- match = refinement only

---

## 9. Summary

Hierarchical dispatch + semantic attr + separated launch system