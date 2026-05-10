# 从 llama.cpp 学习什么：devproc2 设计借鉴文档

## 1. 背景

`llama.cpp` 不是一个普通的 LLM 推理 demo，而是一个已经形成完整工程闭环的本地/端侧 LLM runtime 系统。它围绕 GGUF 模型格式、ggml tensor graph、多后端执行、量化权重、CLI/server/benchmark/quantize 工具链，构建出了一套低依赖、可移植、可长期维护的推理生态。

对 devproc2 来说，llama.cpp 最值得学习的不是“怎么实现一个 LLM runtime”，也不是“怎么设计 IR”，而是：

> 一个端侧 runtime 系统如何通过清晰边界、稳定 artifact、可靠 fallback、多后端抽象、状态化执行和工具链闭环真正活下来。

因此，本文重点总结 devproc2 应该从 llama.cpp 学习的设计原则和工程方法。

---

## 2. 总体结论

devproc2 不应该照搬 llama.cpp 的 graph/IR 设计，因为 llama.cpp 的核心场景是 LLM 推理，模型结构高度固定；而 devproc2 要表达的是更复杂的端侧端到端 pipeline，包括前处理、模型主体、后处理、外部资源、动态 shape、设备切分和 zero-copy 执行。

但是，devproc2 应该重点学习 llama.cpp 的以下能力：

1. **稳定的部署 artifact 设计**：类似 GGUF，把部署所需的信息收敛到一个稳定格式中。
2. **简单清晰的 runtime 对象模型**：区分 model/module、context/session、invocation。
3. **CPU fallback-first 的工程策略**：任何系统要先保证能跑，再逐步优化。
4. **多后端不是 kernel registry，而是 device/buffer/copy/sync/execute 的完整抽象。**
5. **状态化执行与内存复用**：避免每次 invoke 都重新分配、重新初始化、重新 copy。
6. **完整工具链闭环**：compile、run、inspect、bench、profile、verify 缺一不可。
7. **场景边界克制**：第一版不要试图成为 TVM + TensorRT + llama.cpp + Triton 的总和。

一句话：

> devproc2 不要学 llama.cpp 做 LLM runtime，要学 llama.cpp 做一个真正可用、可维护、可扩展的端侧 runtime 系统。

---

## 3. 学习点一：Artifact / Package 是 runtime 的核心 ABI

### 3.1 llama.cpp 的启发

llama.cpp 的核心资产之一是 GGUF。GGUF 不只是权重文件，而是把模型部署所需的大量信息集中在一个格式中：

- model metadata
- architecture 信息
- tokenizer 信息
- tensor name
- shape
- dtype
- quantization format
- weights

这带来的好处是：用户拿到一个 GGUF 文件，基本就能用 llama.cpp 加载和运行。模型格式成为 runtime 与模型生态之间的稳定 ABI。

### 3.2 devproc2 应该怎么学

devproc2 不能只有 IR，也不能只有 runtime。它必须尽早定义自己的 package 格式。

建议 devproc2 package 至少包含：

```text
devproc2_package/
├── manifest.json
├── graph.vm 或 graph.ir
├── weights.bin
├── constants.bin
├── kernels/
│   ├── cuda/
│   │   ├── xxx.ptx
│   │   └── kernel_meta.json
│   ├── cpu/
│   └── triton/
├── tokenizer/
├── preprocess/
├── memory_plan.json
├── io_schema.json
├── shape_constraints.json
├── device_plan.json
└── version.json
```

其中关键文件包括：

| 文件                     | 作用                                                 |
| ------------------------ | ---------------------------------------------------- |
| `manifest.json`          | package 总入口，记录版本、target、依赖、ABI 信息     |
| `graph.vm / graph.ir`    | devproc2 编译后的图或 VM bytecode                    |
| `weights.bin`            | 权重数据                                             |
| `constants.bin`          | 编译期常量                                           |
| `kernels/`               | AOT 编译出的 PTX、CPU kernel、Triton kernel metadata |
| `memory_plan.json`       | 中间 tensor buffer 规划                              |
| `io_schema.json`         | 输入输出名称、shape、dtype、device、zero-copy 约束   |
| `shape_constraints.json` | 动态 shape 约束                                      |
| `device_plan.json`       | op / tensor 的设备放置计划                           |

devproc2 应该把 package 视为一等公民。没有稳定 package，runtime 后面会越来越难维护，benchmark、profile、debug、部署也都会变得混乱。

### 3.3 设计原则

建议 devproc2 遵循：

```text
IR 是编译器内部表示；
Package 是部署 ABI；
Runtime 只应该依赖稳定 package，不应该依赖 Python 编译期状态。
```

---

## 4. 学习点二：Runtime 对象模型要简单清晰

### 4.1 llama.cpp 的启发

llama.cpp 有非常清晰的长期状态对象，例如 model、context、sampler、KV cache、backend buffer 等。

它不是每次推理都从零开始，而是先加载模型，创建上下文，然后在上下文中持续执行 decode。长期状态被 runtime 明确持有。

### 4.2 devproc2 应该怎么学

devproc2 可以借鉴这种对象分层：

```text
Runtime
  └── Module
        └── Session
              └── Invocation
```

建议职责划分：

| 对象         | 职责                                                         |
| ------------ | ------------------------------------------------------------ |
| `Runtime`    | 全局 runtime 环境，注册 backend、device、allocator、op library |
| `Module`     | 一个已加载 package，持有 graph、weights、kernel metadata、schema |
| `Session`    | 一个可复用执行上下文，持有 memory plan、backend buffers、runtime state |
| `Invocation` | 一次具体调用，绑定输入输出并执行                             |

推荐 API 形态：

```cpp
Runtime runtime;
Module module = runtime.LoadPackage("model.dpkg");
Session session = module.CreateSession();

session.BindInput("image", image_buffer);
session.BindOutput("result", output_buffer);
session.Invoke();
```

或者更偏 VM：

```cpp
auto sess = devproc::LoadPackage("model.dpkg").CreateSession();

sess.SetInput("input_ids", input_ids);
sess.SetOutput("logits", logits);
sess.InvokeStateful();
```

### 4.3 设计原则

不要把所有状态塞进一个巨大 Runtime 对象。合理的分层可以让：

- package 加载和 session 执行解耦；
- 多 session 并发更容易；
- memory plan 可以挂在 session 上复用；
- zero-copy 绑定可以在 session/invocation 层管理；
- Python binding / C API 更稳定。

---

## 5. 学习点三：CPU fallback-first 是系统可用性的根

### 5.1 llama.cpp 的启发

llama.cpp 的一个重要特点是 CPU backend 很强。即使没有 GPU，用户也可以跑模型；即使 GPU 显存不够，也可以 CPU+GPU 混合执行。

这让 llama.cpp 有很强的可用性和可移植性。

### 5.2 devproc2 应该怎么学

devproc2 第一版不要一开始就追求所有 op 都有 CUDA/Triton 最优实现。应该先建立这个原则：

> 每个合法 op 至少有 CPU reference 实现。

然后再分层优化：

```text
CPU reference op
    ↓
CPU optimized op
    ↓
CUDA default op
    ↓
cuBLAS / cuDNN vendor op
    ↓
Triton AOT kernel
    ↓
CUTE DSL kernel
    ↓
agent optimized custom kernel
```

这样设计的好处是：

- 系统永远有 fallback；
- 新 op 可以先快速接入；
- CUDA/Triton 优化可以逐步推进；
- debug 时可以用 CPU reference 做 correctness oracle；
- profile 后只优化热点 op，不会被全 op 最优实现拖死。

### 5.3 设计原则

```text
第一目标：能跑通。
第二目标：结果正确。
第三目标：热点 op 性能好。
第四目标：全局性能好。
```

不要反过来。否则 devproc2 很容易一开始就陷入 kernel 优化细节，导致系统主链路迟迟跑不通。

---

## 6. 学习点四：Backend 不是简单的 op_name -> function pointer

### 6.1 llama.cpp 的启发

llama.cpp/ggml 的 backend 思路不是简单 kernel registry。一个 backend 需要处理：

- device
- buffer
- tensor allocation
- copy
- sync
- op support checking
- graph execution
- CPU/GPU offload
- 多设备协作

也就是说，backend 是 runtime 执行能力的抽象，而不是函数表。

### 6.2 devproc2 应该怎么学

devproc2 的 backend 也应该围绕真实执行问题设计。

推荐抽象：

```cpp
class Backend {
public:
    virtual Device device() const = 0;

    virtual bool Supports(const Op& op,
                          const TensorDesc& input_desc,
                          const Attrs& attrs) const = 0;

    virtual Kernel SelectKernel(const Op& op,
                                const TensorDesc& input_desc,
                                const Attrs& attrs) = 0;

    virtual Buffer Alloc(const BufferDesc& desc) = 0;
    virtual void Copy(Buffer src, Buffer dst) = 0;

    virtual void Execute(const ExecutableSegment& segment,
                         Stream stream) = 0;

    virtual void Sync(Stream stream) = 0;
};
```

kernel 匹配不能只看 name，而要看：

```text
op name
+ dtype
+ shape pattern
+ attrs
+ layout
+ device
+ memory type
+ dynamic shape support
+ workspace requirement
+ priority
```

示例：

```text
matmul
  dtype: fp16
  shape: M=1, N=4096, K=4096
  device: cuda
  layout: row-major

候选实现：
  1. custom_triton_matmul_1x4096x4096, priority=100
  2. cublas_matmul, priority=80
  3. default_cuda_matmul, priority=50
  4. cpu_reference_matmul, priority=0
```

### 6.3 设计原则

backend 至少要回答四个问题：

```text
这个 op 我能不能跑？
我用哪个 kernel 跑？
输入输出 buffer 在哪里？
执行和同步怎么管理？
```

如果 backend 只做 name dispatch，后面支持 zero-copy、多设备、Triton PTX、cuBLAS、CUTE DSL 时一定会返工。

---

## 7. 学习点五：状态化执行与内存复用非常重要

### 7.1 llama.cpp 的启发

llama.cpp 的执行不是每次请求都重新创建所有东西。模型权重、context、KV cache、backend buffer 等都是长期持有和复用的。

这背后的原则是：

> 推理性能不只是 kernel 性能，还包括 allocation、copy、dispatch、状态维护的开销。

### 7.2 devproc2 应该怎么学

devproc2 的 VM runtime 应该天然支持 stateful invoke 和 zero-copy。

建议明确支持：

```text
owned buffer
external buffer
view buffer
host buffer
device buffer
pinned host buffer
managed buffer
```

推荐 API：

```cpp
session.BindInput("input", ExternalBuffer{ptr, shape, dtype, device});
session.BindOutput("output", ExternalBuffer{ptr, shape, dtype, device});
session.InvokeStateful();
```

Session 内部应该长期持有：

```text
中间 tensor buffer
workspace buffer
backend handle
stream
kernel handle
resource handle
dynamic shape cache
```

第一版 memory plan 可以简单，不一定要做复杂 reuse，但必须做到：

1. 不要每次 invoke 都重新 malloc 大量中间 tensor；
2. 输入输出可以绑定外部 buffer；
3. 中间 buffer 可以由 session 复用；
4. dynamic shape 输出有明确分配策略；
5. CPU/GPU copy 是显式、可追踪、可 profile 的。

### 7.3 设计原则

```text
一次 compile，多次 invoke；
一次 load，多次 session；
一次 session，多次 stateful invoke。
```

这是端侧 runtime 的关键性能基础。

---

## 8. 学习点六：工具链闭环比高级抽象更重要

### 8.1 llama.cpp 的启发

llama.cpp 不只有 runtime，还有完整工具链：

- convert
- quantize
- CLI
- server
- benchmark
- examples
- tests
- docs

这让它不是一个库，而是一个可使用、可验证、可调试、可部署的系统。

### 8.2 devproc2 应该怎么学

devproc2 第一版必须尽早设计命令行工具，而不是等 runtime 完成后再补。

建议至少提供：

```text
devproc compile
devproc run
devproc inspect
devproc bench
devproc profile
devproc verify
devproc dump-ir
devproc dump-package
```

其中最重要的是 `inspect` 和 `profile`。

建议支持：

```bash
devproc inspect model.dpkg --summary
devproc inspect model.dpkg --io
devproc inspect model.dpkg --kernels
devproc inspect model.dpkg --memory
devproc inspect model.dpkg --devices
devproc inspect model.dpkg --shape-constraints
```

`inspect` 应该能回答：

- package 里有哪些 graph / kernel / weights？
- 每个 op 选择了哪个 kernel？
- 每个 tensor 放在哪个 device？
- 哪些输入输出支持 zero-copy？
- 哪些 op fallback 到 CPU？
- memory plan 是什么？
- dynamic shape 约束是什么？

`profile` 应该能回答：

- 每个 op 耗时多少？
- kernel dispatch 开销多少？
- device copy 开销多少？
- allocation 开销多少？
- fallback op 是否成为瓶颈？
- zero-copy 是否真的生效？

### 8.3 设计原则

没有工具链，runtime 就会变成黑盒。devproc2 要想长期迭代，必须让内部状态可观测。

---

## 9. 学习点七：场景边界必须克制

### 9.1 llama.cpp 的启发

llama.cpp 成功的一个重要原因是边界清晰：它就是 LLM inference runtime。它没有试图成为通用 AI 编译器，也没有试图覆盖所有模型、所有前处理、所有部署形态。

它把 LLM 本地推理这个场景打穿，才形成了强生态。

### 9.2 devproc2 应该怎么学

devproc2 第一版一定要克制。

第一版目标建议是：

```text
端侧端到端 pipeline runtime：
支持 Python DSL 表达前处理 + 模型 + 后处理，
编译为 package，
由 C++ VM runtime 加载执行，
支持 CPU fallback、CUDA/Triton 加速和 zero-copy invoke。
```

第一版应该做：

```text
Python DSL
DevProc IR
Package
C++ VM Runtime
CPU backend
CUDA backend
Triton PTX AOT
cuBLAS adapter
zero-copy binding
inspect / bench / verify
```

第一版暂时不要做：

```text
复杂 autotune
全自动图融合
完整 TVM-style pass pipeline
所有硬件后端
完整量化生态
高级 serving 系统
分布式推理
复杂内存复用优化
```

### 9.3 设计原则

```text
不要一开始做“通用 AI 编译器”。
先做一个能稳定解决端侧端到端部署问题的 runtime。
```

---

## 10. devproc2 应该借鉴的设计清单

### 10.1 必须借鉴

| 学习点                              | 对 devproc2 的价值       |
| ----------------------------------- | ------------------------ |
| GGUF-like package                   | 建立稳定部署 ABI         |
| model/context/session 分层          | 保证 runtime 状态清晰    |
| CPU fallback-first                  | 保证系统可用性           |
| backend = device + buffer + execute | 避免后端抽象过窄         |
| stateful invoke                     | 减少重复初始化和内存分配 |
| zero-copy buffer binding            | 满足端侧性能要求         |
| inspect/profile 工具                | 保证系统可调试           |
| 场景边界克制                        | 避免复杂度爆炸           |

### 10.2 可以参考，但不要照搬

| llama.cpp 设计         | devproc2 应该怎么处理                           |
| ---------------------- | ----------------------------------------------- |
| ggml graph             | 参考执行模型，不照搬为 IR                       |
| 手写模型 graph builder | devproc2 应该坚持 Python DSL / frontend capture |
| LLM 专用 op 体系       | devproc2 需要更通用的 op/resource 抽象          |
| 大量低 bit 量化格式    | 第一版只保留 quant metadata，不作为主目标       |
| 多后端全覆盖           | 第一版只做 CPU + CUDA/Triton                    |

### 10.3 不应该学习

| 不建议学习的点                  | 原因                                                    |
| ------------------------------- | ------------------------------------------------------- |
| 过度 LLM 专用化                 | devproc2 的目标是端到端 pipeline，不只是 LLM            |
| 手工适配每个模型结构            | 会削弱 DSL/frontend 的价值                              |
| 一开始铺很多硬件后端            | 维护成本过高                                            |
| 用轻量 tensor graph 替代程序 IR | 无法表达前后处理、resource、control flow、dynamic shape |

---

## 11. devproc2 推荐落地路线

### Phase 1：最小闭环

目标：先让系统完整跑起来。

范围：

```text
@devproc.function
  -> IR
  -> package
  -> C++ VM runtime
  -> CPU backend
  -> run / verify / inspect
```

重点：

- 定义 package 格式；
- 定义 Runtime / Module / Session；
- 所有 op 有 CPU reference；
- 支持基本输入输出 schema；
- 支持 inspect package。

---

### Phase 2：CUDA/Triton 加速

目标：让热点 tensor op 有 GPU 加速路径。

范围：

```text
CUDA backend
Triton PTX AOT
cuBLAS adapter
kernel registry
kernel selection
```

重点：

- kernel 匹配支持 shape/dtype/attrs/device；
- PTX kernel metadata 进入 package；
- 支持 CPU fallback；
- 支持 profile op/kernel 耗时。

---

### Phase 3：zero-copy / stateful invoke

目标：让 runtime 真正适合端侧部署。

范围：

```text
external buffer
owned buffer
view buffer
device buffer
session memory reuse
stateful invoke
```

重点：

- 输入输出可绑定外部 buffer；
- 中间 buffer session 级复用；
- 显式跟踪 copy；
- profile zero-copy 是否生效。

---

### Phase 4：优化与生态

目标：提升性能和可用性。

范围：

```text
图融合
agent kernel 优化
更多默认 op
benchmark suite
文档与 examples
```

重点：

- 只优化 profile 证明的热点；
- 不急于做复杂 autotune；
- 建立 benchmark 与 regression test；
- 形成稳定 examples。

---

## 12. 最终建议

从 llama.cpp 身上，devproc2 最应该学习的是工程成熟度，而不是具体 IR 形态。

最重要的五条建议是：

1. **尽早定义 package ABI。** 不要只有 IR 和 runtime。
2. **runtime 分层要清楚。** Runtime / Module / Session / Invocation 不要混在一起。
3. **CPU fallback 是根。** 没有 fallback，系统很难稳定迭代。
4. **backend 抽象要完整。** 不要只做 op name 到 function pointer 的映射。
5. **工具链要先行。** inspect、bench、profile、verify 是 runtime 能持续优化的基础。

最终判断：

> llama.cpp 证明了，一个端侧 runtime 的成功不只靠 kernel 性能，而是靠稳定格式、清晰状态、可靠 fallback、多后端执行、内存复用和工具链闭环。devproc2 如果能吸收这些工程原则，同时保留自己在端到端 DSL 表达上的优势，就有机会形成区别于 TVM、TensorRT、llama.cpp 的独特价值。

