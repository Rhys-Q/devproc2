# Memory Planning 设计文档

## 1. 为什么需要 Memory Planning？

### 1.1 问题背景

在神经网络推理中，每一个算子（relu、layernorm、matmul…）都需要一块内存来存放它的输出。朴素做法是每个张量各自独立申请内存，用完之后释放。

问题是：**GPU 上反复 malloc/free 代价很高**，因为 CUDA 的设备内存分配是全局串行的，在推理的关键路径上执行 `cudaMalloc` 会严重拖慢延迟。

标准做法是：在模型启动时**一次性把所有内存都申请好**，推理时不再做任何动态分配。这要求编译器提前规划好每块内存的来源和归属。

### 1.2 核心观察：张量的生命周期往往不重叠

考虑下面这条计算链：

```
x  →  relu  →  a  →  layernorm  →  b  →  relu  →  c  →  layernorm  →  out
```

- `a` 在 relu 之后被写入，只被下一步 layernorm 读取一次，然后就再也没用了
- `c` 在第二个 relu 之后才被写入，此时 `a` 已经完全用完了

**`a` 的生命周期和 `c` 的生命周期没有重叠**，因此它们可以复用同一块物理内存。

这就是 Memory Planning 的核心目标：**分析所有中间张量的生命周期（Live Interval），把不重叠的张量分配到同一个物理 storage，从而最大化内存复用。**

---

## 2. 架构概览：在编译 pipeline 中的位置

```
Python DSL
   │
   ▼  @dp.function 装饰器捕获
High-level IR（CallOp）
   │
   ▼  InferStructInfoPass   — 推导 shape/dtype/device
   ▼  DPSLoweringPass       — CallOp → TensorCreateOp + CallDPSOp
   │
   ▼  【Memory Planning】
   │     MemoryPlanningPass           ← 分析生命周期，生成 StoragePlan（不改 IR）
   │     LowerTensorCreateToAllocPass ← 将 TensorCreateOp 替换为 alloc_storage + alloc_tensor
   │
   ▼  Memory-explicit IR（alloc_storage / alloc_tensor）
   │
   ▼  后续 pass（ShapeExprLowering、VMCodegen…）
```

Memory Planning 由**两个 pass** 组成，职责严格分离：

| Pass | 职责 | 是否修改 IR |
|---|---|---|
| `MemoryPlanningPass` | 纯分析：计算生命周期、分配 storage | **不修改** |
| `LowerTensorCreateToAllocPass` | 纯改写：用分析结果重写 IR | 修改 |

分离的好处：分析结果可以被调试工具直接打印，不需要解析 IR 才能知道规划结果。

---

## 3. 关键数据结构

### 3.1 LiveInterval（生命区间）

```python
@dataclass
class LiveInterval:
    first_def: int   # 张量被定义（TensorCreateOp）时的指令编号
    last_use:  int   # 张量最后一次被读或写时的指令编号

    def overlaps(self, other: LiveInterval) -> bool:
        # 两个区间有任何重叠？
        return self.first_def <= other.last_use and other.first_def <= self.last_use
```

"指令编号"是把整个函数的所有 Op（包括嵌套在 if/for 里的）按程序顺序线性排列后的序号（0, 1, 2, …）。

两个区间**不重叠**的条件：一个区间的结束点严格早于另一个的开始点。

```
不重叠：  [──a──]          [──c──]     ← a 结束后 c 才开始，可以复用内存
重叠：    [──a──]
               [──b──]               ← a 还没死，b 就开始了，不能复用
```

### 3.2 StorageEntry（物理存储块）

```python
@dataclass
class StorageEntry:
    id:         int           # 存储块唯一编号
    device:     str           # 所在设备，如 "cuda"
    size_bytes: Optional[int] # 静态时为按 256 字节对齐后的最大容量；动态时为 None
    size_expr:  PrimExpr      # 传给 AllocStorageOp 的大小表达式
                              #   静态：IntImm(aligned_nbytes)
                              #   动态：shape 各维连乘的符号表达式（运行时求值）
    alignment:  int = 256     # 内存对齐要求
    reused_by:  list[str]     # 共享这块内存的所有张量名称
```

一个 `StorageEntry` 可以被多个张量按时间顺序复用——只要它们的生命区间不重叠。`size_bytes` 仅用于贪心分配时的大小比较，`size_expr` 是写入 IR 的实际表达式。

### 3.3 StoragePlan（规划结果）

```python
@dataclass
class StoragePlan:
    entries:           list[StorageEntry]   # 所有物理存储块
    tensor_to_storage: dict[str, int]       # 张量名 → 存储块 id
```

这是 `MemoryPlanningPass` 写入 `PassContext` 的最终结果。

---

## 4. MemoryPlanningPass 四个阶段详解

### 阶段 A：收集 TensorCreateOp，标记不可复用张量

首先把函数体内所有指令线性化（DFS 顺序，包括 if/for 内部），然后找出所有 `TensorCreateOp`。

**不可复用（`is_reusable = False`）的张量：**
- 函数最终 return 的张量——它必须保持有效直到调用方读取

函数参数（输入张量）本身不是 TensorCreateOp，不在规划范围内，但它们的生命周期也不能被规划器干预。

### 阶段 B：计算 LiveInterval（两步法）

这是 Memory Planning 最核心的逻辑，分两步走：

**第一步：纯数据流分析（explicit last_use）**

遍历所有指令，对每条指令的所有"操作数引用"，更新对应张量的 `last_use`：

```
对每条指令 op（序号 idx）:
    对 op 引用的每个 OpResult ref:
        如果 ref 来自某个 TensorCreateOp，则：
            explicit_last_use[tensor_name] = max(explicit_last_use[tensor_name], idx)
```

"引用"包括：CallDPSOp 的 inputs（读）、output（写）；ReturnOp 的 values；IfOp 的 cond；等等。

**第二步：根据副作用（Effect）保守扩展**

- `WriteEffect(vars=[v1, v2])` → 明确告知编译器"这条指令会写 v1, v2"，将它们的 `last_use` 延长到此指令
- `OpaqueEffect` → 编译器不了解具体写了什么，但**只保守扩展那些在此时刻明确还活着的张量**（即 `first_def <= idx` 且 `explicit_last_use >= idx`），避免把已经死掉的张量也无故延长

> **为什么 OpaqueEffect 不能无脑扩展所有张量？**
>
> 如果把每个算子都标注为 OpaqueEffect，再无脑延长所有曾经定义过的张量，就会导致所有张量的生命周期都延伸到函数末尾，彻底失去复用机会。
>
> devproc2 的 DPS Lowering（M6）目前把所有内核调用暂时标注为 OpaqueEffect，等 Effect System（M5）完善后会精化为 WriteEffect(output)。两步法让 M7 在 M5 尚未精化之前也能正确复用内存。

### 阶段 C：计算存储大小（StorageSizeAnalyze）

每个张量所需的字节数：

```
nbytes = (∏ dim) × dtype_itemsize(dtype)
```

**静态 shape（所有维度均为 `IntImm` 或带 `upper` 的 `PrimVar`）：**

用 UpperBound 代入各维度，得到"最坏情况下的最大字节数"，再向上对齐到 **256 字节**：

```python
size_bytes = align256(∏ dim.upper × itemsize)
size_expr  = IntImm(size_bytes)
```

这是参与贪心复用比较的数值基础：entry 的 `size_bytes` 必须 ≥ 待分配 tensor 的 `size_bytes` 才能接受复用。

**动态 shape（某维度 `PrimVar` 无 `upper`）：**

编译期无法确定最大字节数，`size_bytes = None`。`size_expr` 由 `_compute_size_expr` 直接构造符号乘法链，不做 upper bound 替换：

```python
size_bytes = None
size_expr  = dim₀ * dim₁ * … * IntImm(itemsize)   # PrimExpr，运行时求值
```

`AllocStorageOp.size_bytes` 是 `PrimExpr` 类型，静态时为 `IntImm`，动态时为符号表达式，由后续 `ShapeExprLoweringPass` 展开为 VM builtin 调用序列，最终在 runtime 通过 `DeviceAPI::Alloc` 分配（`alignment` 字段通知 runtime 对齐要求）。

> **动态 tensor 是否可以复用？可以。**
>
> 只要两个动态 tensor 的 `size_expr` 结构相等，且生命区间不重叠，就能共享同一个 storage。
> `size_expr` 的相等性依赖于 `PrimExpr` 的结构等式规则：`PrimVar` 使用对象同一性（`eq=False`），因此同一函数里引用同一个 symbolic dim 对象的两个 tensor，其 `size_expr` 天然相等。

### 阶段 D：贪心存储分配（Greedy StoragePlan）

按 `first_def`（定义时间）升序排列所有张量，然后贪心分配：

```python
for tensor in sorted(tensors, by=first_def):
    if not tensor.is_reusable:
        # 必须有独立 storage（返回值）
        new_entry(tensor)
    else:
        # 找第一个满足条件的已有 storage
        candidate = find_first(entries where entry.accepts(tensor))
        if candidate:
            candidate.reused_by.append(tensor)
        else:
            new_entry(tensor)  # 没有可复用的，新建
```

`entry.accepts(tensor)` 的三种情况：

| entry 类型 | tensor 类型 | 接受条件 |
|---|---|---|
| 静态（`size_bytes` 非 None）| 静态 | `device` 相同 且 `entry.size_bytes >= tensor.size_bytes` 且区间不重叠 |
| 动态（`size_bytes` 为 None）| 动态 | `device` 相同 且 `entry.size_expr == tensor.size_expr`（结构等式）且区间不重叠 |
| 静态 / 动态 | 动态 / 静态（混合）| 直接拒绝 |

这是经典的**区间调度/寄存器分配**算法的变体：按开始时间扫描，贪心选择第一个可用的"寄存器"（storage block）。

---

## 5. LowerTensorCreateToAllocPass

读取 `PassContext` 中的 `StoragePlan`，对每个函数：

1. **为每个 StorageEntry 创建一个 `AllocStorageOp`**，放在函数入口块的最前面（session 级别，只分配一次）
2. **把每个 `TensorCreateOp` 替换为 `AllocTensorOp`**，指向对应的 storage

```
改写前：
    %tmp = dp.empty(shape=(B, 512), dtype=float16, device=cuda)
    call_dps kernel.relu(inputs=[%x], output=%tmp, ...)

改写后：
    %s0 = alloc_storage(size=8192, alignment=256, device=cuda)   ← 函数顶部，只一次
    ...
    %tmp = alloc_tensor(%s0, offset=0, shape=(B, 512), dtype=float16)
    call_dps kernel.relu(inputs=[%x], output=%tmp, ...)
```

多个共享同一个 `StorageEntry` 的张量，都指向同一个 `%s0`，在时间上轮流使用那块内存。

---

## 6. 完整 Demo：从 DSL 到 Memory-explicit IR

下面用一个四算子 decode-step 函数走完整个流程，展示 Memory Planning 的所有 feature。

### 6.1 Demo 模型

```python
B = dp.symbolic_dim("B", upper=8)    # batch size，最大 8
S = dp.symbolic_dim("S", upper=2048) # sequence length，最大 2048

@dp.function
def decode_step(
    x:       dp.Tensor[(B, S, 4096), "float16", "cuda"],  # 输入（不可复用）
    k_cache: dp.Tensor[(B, S, 4096), "float16", "cuda"],  # KV cache（有副作用，不可复用）
):
    # 第 1 步：layernorm，中间结果，可复用
    a = dp.ops.layernorm(x)

    # 第 2 步：relu，中间结果，可复用
    b = dp.ops.relu(a)

    # 第 3 步：update_kvcache，no-output DPS 调用，有 write 副作用
    #          写入 k_cache——k_cache 的生命周期必须覆盖此调用
    dp.ops.update_kvcache(k_cache, b)   # effect=write(k_cache)

    # 第 4 步：silu，中间结果，可复用
    c = dp.ops.silu(b)

    # 第 5 步：layernorm，输出，不可复用
    out = dp.ops.layernorm(c)

    return out
```

这个 demo 涵盖了 Memory Planning 的所有 feature：

| Feature | 体现在哪里 |
|---|---|
| symbolic shape + UpperBound 计算大小 | `B`、`S` 均有 upper |
| 不可复用张量（返回值）| `out` |
| 不可复用张量（副作用 mutable state）| `k_cache` 参数 |
| no-output stateful CallDPS | `update_kvcache`（output=None，effect=write） |
| WriteEffect 延长生命周期 | `k_cache` 在 update_kvcache 之后仍要活着 |
| OpaqueEffect 保守扩展 | 其余算子当前是 OpaqueEffect |
| 非相邻中间张量复用 | `a` 和 `c` 形状相同，生命周期不重叠 |
| 不同大小张量独立 storage | `b`（同 `a` 大小）vs `out` |

### 6.2 第一阶段：High-level IR（DPS Lowering 之前）

```
@decode_step(
    %x:       Tensor[(B, S, 4096), float16, cuda],
    %k_cache: Tensor[(B, S, 4096), float16, cuda]
) {
  %a   = @layernorm(%x)
  %b   = @relu(%a)
  @update_kvcache(%k_cache, %b)       ← no-output call，纯 CallOp
  %c   = @silu(%b)
  %out = @layernorm(%c)
  return %out
}
```

这一层是函数式 IR：每个算子是一个 `CallOp`，没有显式的内存分配。

### 6.3 第二阶段：DPS Lowering 之后（Memory Planning 输入）

DPS Lowering 为每个有输出的 CallOp 插入 `TensorCreateOp`（`dp.empty`），并将 CallOp 替换为 `CallDPSOp`：

```
@decode_step(
    %x:       Tensor[(B, S, 4096), float16, cuda],
    %k_cache: Tensor[(B, S, 4096), float16, cuda]
) {
  ──────── op 0 ────────
  %a = dp.empty(shape=(B, S, 4096), dtype=float16, device=cuda)

  ──────── op 1 ────────
  call_dps kernel.layernorm_fp16(
    inputs=[%x],
    output=%a,
    callee_kind=kernel,
    effect=opaque
  )

  ──────── op 2 ────────
  %b = dp.empty(shape=(B, S, 4096), dtype=float16, device=cuda)

  ──────── op 3 ────────
  call_dps kernel.relu_fp16(
    inputs=[%a],
    output=%b,
    callee_kind=kernel,
    effect=opaque
  )

  ──────── op 4 ────────
  call_dps kernel.update_kvcache(
    inputs=[%k_cache, %b],
    output=None,                  ← no-output
    callee_kind=kernel,
    effect=write(%k_cache)        ← 明确写入 k_cache
  )

  ──────── op 5 ────────
  %c = dp.empty(shape=(B, S, 4096), dtype=float16, device=cuda)

  ──────── op 6 ────────
  call_dps kernel.silu_fp16(
    inputs=[%b],
    output=%c,
    callee_kind=kernel,
    effect=opaque
  )

  ──────── op 7 ────────
  %out = dp.empty(shape=(B, S, 4096), dtype=float16, device=cuda)

  ──────── op 8 ────────
  call_dps kernel.layernorm_fp16(
    inputs=[%c],
    output=%out,
    callee_kind=kernel,
    effect=opaque
  )

  ──────── op 9 ────────
  return %out
}
```

> **注意**：`k_cache` 是函数参数（Var），不是 TensorCreateOp 的结果，不在 Memory Planning 的规划范围内。Memory Planning 只规划由 `TensorCreateOp` 创建的中间张量。

### 6.4 MemoryPlanningPass 内部推导过程

#### 阶段 A：收集 TensorCreateOp，标记返回值

| 张量 | TensorCreateOp 序号 | 是否不可复用 | 原因 |
|---|---|---|---|
| `a` | op 0 | 否 | 中间值 |
| `b` | op 2 | 否 | 中间值 |
| `c` | op 5 | 否 | 中间值 |
| `out` | op 7 | **是** | `return %out` |

#### 阶段 B：计算 LiveInterval

**第一步：纯数据流（explicit last_use）**

遍历所有 op，找到哪条指令最后一次引用了哪个张量：

| 引用张量的指令 | 序号 | 被引用张量 |
|---|---|---|
| `call_dps layernorm(inputs=[%x], output=%a)` | op 1 | `a`（作为 output） |
| `call_dps relu(inputs=[%a], output=%b)` | op 3 | `a`（作为 input），`b`（作为 output） |
| `call_dps update_kvcache(inputs=[%k_cache, %b], output=None)` | op 4 | `b`（作为 input） |
| `call_dps silu(inputs=[%b], output=%c)` | op 6 | `b`（作为 input），`c`（作为 output） |
| `call_dps layernorm(inputs=[%c], output=%out)` | op 8 | `c`（作为 input），`out`（作为 output） |
| `return %out` | op 9 | `out` |

因此 `explicit_last_use`：

| 张量 | first_def | explicit_last_use |
|---|---|---|
| `a` | 0 | 3（relu 的 input） |
| `b` | 2 | 6（silu 的 input） |
| `c` | 5 | 8（第二个 layernorm 的 input） |
| `out` | 7 | 9（return） |

**第二步：Effect 扩展**

扫描所有 CallDPSOp 的 effect：

- **op 1（OpaqueEffect）**：检查在 op 1 时刻明确活跃的张量（`first_def <= 1` 且 `explicit_last_use >= 1`）：只有 `a`（0 到 3）。`a.last_use = max(3, 1) = 3`，无变化。
- **op 3（OpaqueEffect）**：活跃张量：`a`（0~3，3>=3 ✓），`b`（2~6，6>=3 ✓）。`a.last_use = max(3,3)=3`，`b.last_use = max(6,3)=6`，均无变化。
- **op 4（WriteEffect(k_cache)）**：`k_cache` 是函数参数 Var，不在规划范围，跳过。
- **op 6（OpaqueEffect）**：活跃张量：`b`（2~6，6>=6 ✓），`c`（5~8，8>=6 ✓）。无变化。
- **op 8（OpaqueEffect）**：活跃张量：`c`（5~8，8>=8 ✓），`out`（7~9，9>=8 ✓）。无变化。

Effect 扩展后，`last_use` 与 `explicit_last_use` 相同（在此 demo 中 OpaqueEffect 没有进一步扩展任何区间）。

**最终 LiveInterval：**

| 张量 | first_def | last_use | 是否可复用 |
|---|---|---|---|
| `a` | 0 | 3 | ✅ 是 |
| `b` | 2 | 6 | ✅ 是 |
| `c` | 5 | 8 | ✅ 是 |
| `out` | 7 | 9 | ❌ 否（返回值） |

**生命周期时间轴（各 op 编号 0~9）：**

```
op:   0  1  2  3  4  5  6  7  8  9
      ┌──────────────────────────────
a     [def─────────use]
b              [def────────────use]
c                       [def──────────use]
out                              [def──use]
```

**重叠检查：**

| 张量对 | 区间 | 是否重叠 |
|---|---|---|
| `a` vs `b` | [0,3] vs [2,6] | **重叠**（2 ≤ 3） |
| `a` vs `c` | [0,3] vs [5,8] | **不重叠**（3 < 5） ✅ 可复用 |
| `a` vs `out` | [0,3] vs [7,9] | 不重叠，但 `out` 不可复用 |
| `b` vs `c` | [2,6] vs [5,8] | **重叠**（5 ≤ 6） |
| `b` vs `out` | [2,6] vs [7,9] | 不重叠，但 `out` 不可复用 |
| `c` vs `out` | [5,8] vs [7,9] | **重叠**（7 ≤ 8） |

#### 阶段 C：计算存储大小

形状 `(B, S, 4096)`，dtype `float16`（2 字节）：

```
nbytes = B.upper × S.upper × 4096 × 2
       = 8 × 2048 × 4096 × 2
       = 134,217,728 字节（128 MiB）

对齐到 256：134,217,728 已经是 256 的倍数 → size_bytes = 134,217,728
```

所有 4 个张量形状相同，`size_bytes` 均为 **134,217,728**。

#### 阶段 D：贪心存储分配

按 `first_def` 升序处理：`a`(0) → `b`(2) → `c`(5) → `out`(7)

```
处理 a（first_def=0，可复用）：
    现有 entries: 空
    → 新建 entry #0，size=134217728，device=cuda
    entry #0: reused_by=["a"], intervals=[[0,3]]

处理 b（first_def=2，可复用）：
    检查 entry #0: device=cuda ✓，size 足够 ✓，区间 [0,3] vs [2,6] → 重叠 ✗
    → 新建 entry #1，size=134217728，device=cuda
    entry #1: reused_by=["b"], intervals=[[2,6]]

处理 c（first_def=5，可复用）：
    检查 entry #0: device=cuda ✓，size 足够 ✓，区间 [0,3] vs [5,8] → 不重叠 ✓
    → c 复用 entry #0 ✅
    entry #0: reused_by=["a","c"], intervals=[[0,3],[5,8]]

处理 out（first_def=7，不可复用）：
    → 直接新建 entry #2
    entry #2: reused_by=["out"], intervals=[[7,9]]
```

**最终 StoragePlan：**

```json
{
  "storage_plan": [
    {
      "id": 0,
      "device": "cuda",
      "size_bytes": 134217728,
      "alignment": 256,
      "reused_by": ["a", "c"]         ← a 和 c 共享 128 MiB
    },
    {
      "id": 1,
      "device": "cuda",
      "size_bytes": 134217728,
      "alignment": 256,
      "reused_by": ["b"]
    },
    {
      "id": 2,
      "device": "cuda",
      "size_bytes": 134217728,
      "alignment": 256,
      "reused_by": ["out"]
    }
  ],
  "tensor_to_storage": {
    "a":   0,
    "b":   1,
    "c":   0,
    "out": 2
  }
}
```

**内存节省：**
- 朴素方案：4 个张量 × 128 MiB = **512 MiB**
- Memory Planning 后：3 个 storage × 128 MiB = **384 MiB**（节省 25%）

> 在实际 LLM decode 中，中间激活数量更多，节省比例更大。

### 6.5 LowerTensorCreateToAllocPass 之后（Memory-explicit IR）

`LowerTensorCreateToAllocPass` 把 `TensorCreateOp` 替换为 `alloc_storage + alloc_tensor`，并将所有 `alloc_storage` 提升到函数入口：

```
@decode_step(
    %x:       Tensor[(B, S, 4096), float16, cuda],
    %k_cache: Tensor[(B, S, 4096), float16, cuda]
) {
  ──── 函数入口：一次性分配所有 storage ────────────────────────────────
  %s0 = alloc_storage(size=134217728, alignment=256, device=cuda)  ← a 和 c 共用
  %s1 = alloc_storage(size=134217728, alignment=256, device=cuda)  ← b 独用
  %s2 = alloc_storage(size=134217728, alignment=256, device=cuda)  ← out 独用

  ──── 计算体 ──────────────────────────────────────────────────────────
  %a = alloc_tensor(%s0, offset=0, shape=(B, S, 4096), dtype=float16)
  call_dps kernel.layernorm_fp16(
    inputs=[%x], output=%a, callee_kind=kernel, effect=opaque
  )

  %b = alloc_tensor(%s1, offset=0, shape=(B, S, 4096), dtype=float16)
  call_dps kernel.relu_fp16(
    inputs=[%a], output=%b, callee_kind=kernel, effect=opaque
  )

  call_dps kernel.update_kvcache(
    inputs=[%k_cache, %b], output=None, callee_kind=kernel, effect=write(%k_cache)
  )

  %c = alloc_tensor(%s0, offset=0, shape=(B, S, 4096), dtype=float16)  ← 复用 %s0！
  call_dps kernel.silu_fp16(
    inputs=[%b], output=%c, callee_kind=kernel, effect=opaque
  )

  %out = alloc_tensor(%s2, offset=0, shape=(B, S, 4096), dtype=float16)
  call_dps kernel.layernorm_fp16(
    inputs=[%c], output=%out, callee_kind=kernel, effect=opaque
  )

  return %out
}
```

注意 `%a` 和 `%c` 都指向 `%s0`——它们在时间上先后使用同一块 128 MiB 内存，互不干扰。

---

## 7. 设计约束与 MVP 限制

### 当前支持

- ✅ 静态 shape 和带 UpperBound 的 symbolic shape
- ✅ 无 UpperBound 的动态 shape（`size_expr` 为符号表达式，runtime 求值）
- ✅ 动态 shape tensor 之间的存储复用（相同 `size_expr` + 区间不重叠）
- ✅ 单函数模块和多函数模块（按函数名分别规划）
- ✅ WriteEffect / OpaqueEffect 感知的生命周期计算
- ✅ 贪心存储复用（同设备、大小足够或表达式相同、区间不重叠）
- ✅ 返回值张量自动标记为不可复用
- ✅ no-output DPS 调用（`output=None`）正确处理
- ✅ 256 字节对齐（静态：编译期对齐；动态：通过 `AllocStorageOp.alignment` 字段告知 runtime）

### MVP 不做

- ❌ 跨 device 复用（CPU tensor 和 CUDA tensor 不能混用同一 storage）
- ❌ View / alias 支持（tensor 的 slice 视图暂不规划）
- ❌ Inplace 算子（如 dropout inplace）
- ❌ 激进复用（storage escape 分析）
- ❌ 对齐大小不同的 tensor 紧凑打包（当前每个 tensor 独占 storage 的全部大小，不做 offset 复用）

---

## 8. PassContext 与 Pass 间通信

`MemoryPlanningPass` 和 `LowerTensorCreateToAllocPass` 通过 `PassContext` 传递数据：

```python
ctx = PassContext()

# Pass 1：分析（不改 IR）
module = MemoryPlanningPass().run(module, ctx)

# 可以在这里打印 / 检查规划结果
plan = ctx.get("storage_plan")
for entry in plan.entries:
    size_desc = str(entry.size_bytes) if entry.size_bytes is not None else f"dynamic({entry.size_expr})"
    print(f"storage {entry.id}: {size_desc} bytes, shared by {entry.reused_by}")

# Pass 2：改写 IR
module = LowerTensorCreateToAllocPass(ctx).run(module)
```

多函数模块时，规划结果存储在 `ctx["storage_plan:<fn_name>"]` 键下；单函数模块时还额外有 `ctx["storage_plan"]` 快捷键。

---

## 9. 与后续 pass 的接口

`LowerTensorCreateToAllocPass` 执行后，IR 中的变化：

| 原 IR | 新 IR |
|---|---|
| `TensorCreateOp` | `AllocTensorOp`（引用某个 AllocStorageOp 的 result） |
| —（无显式 storage）| `AllocStorageOp`（提升到函数入口） |

下游 pass（如 `ShapeExprLoweringPass`、`VMCodegenPass`）看到的是 `AllocStorageOp + AllocTensorOp` 对，而不再有 `TensorCreateOp`。VM 在执行 `alloc_storage` 指令时调用 `DeviceAPI::Alloc`，执行 `alloc_tensor` 时建立 `TensorObj` 的视图。
