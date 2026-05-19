# Pi0.5 文档入口

本目录记录 OpenPI0.5 在 devproc2 中的当前实现、构建流程、验证流程和性能快照。已执行过的阶段计划和重构方案已经移到 [archive/pi05](../archive/pi05/)。

## 当前实现边界

- Pi0.5 模型实现归属 `python/devproc2/models/pi05/`，模型 graph、ops、weights 和 model-owned CUDA backend 都在该包内。
- OpenPI checkpoint conversion 和 oracle producer 工具归属 `tools/pi05/`。
- 推荐构建路径已经收敛为 `runtime build -> weight convert -> devproc2 build -> runtime infer`。
- 产品化 artifact 通过 `python -m devproc2.build --model pi05 --entry sample_tokens ...` 构建；Pi0.5 CUDA packed backend 由 model build 显式构建或复用缓存。
- 当前性能口径见 [06_performance_snapshot.md](06_performance_snapshot.md)，以 RTX 4090 / SM89、batch size 1、`num_steps=10`、CUDA Graph replay 为主。

## 推荐阅读路径

1. [13_product_build_quickstart.md](13_product_build_quickstart.md)：产品化主流程，覆盖 runtime build、weight convert、model artifact build 和 runtime benchmark。
2. [08_build_run_profile.md](08_build_run_profile.md)：完整验证手册，覆盖 PyTorch dump、raw oracle、actions 级对点和 Nsight profile。
3. [06_performance_snapshot.md](06_performance_snapshot.md)：当前性能基线、已采用优化、未采用结论和下一阶段设计债务。
4. [14_openpi05_devproc_compile_run_flow.md](14_openpi05_devproc_compile_run_flow.md)：面向分享的完整编译运行流程说明。

## 设计与状态参考

- [01_frontend_nn_module.md](01_frontend_nn_module.md)：类 torch `nn.Module` 前端、参数/子模块注册、模型结构表达。
- [02_ir_attr_ops.md](02_ir_attr_ops.md)：IR attr 系统、基础 op 和 OpenPI0.5 所需 op 语义。
- [03_kernels_lowering.md](03_kernels_lowering.md)：kernel selection、DPS lowering、CUDA source/cubin artifact。
- [04_weight_artifact_quant.md](04_weight_artifact_quant.md)：权重包、artifact 扩展、FP8 权重量化状态。
- [05_runtime_accuracy_milestones.md](05_runtime_accuracy_milestones.md)：runtime、精度对齐流程和阶段性验证记录。
- [07_frontend_dsl_refactor.md](07_frontend_dsl_refactor.md)：`forward_fast()` 一等接口和 CUDA 自定义算子接入结果。

## 本机资产约定

当前文档里的命令默认使用本机已有资产，不安装、不更新、不修改 OpenPI 的 uv 环境：

```bash
export DEVPROC2_ROOT=/root/tw/devproc2
export OPENPI_ROOT=/root/tw/openpi
export PI05_CKPT=/root/tools/pi05_libero_base
export PI05_TORCH_DUMP=/root/tw/openpi/outputs/pi05_torch_infer
export PI05_TOKENIZER=$PI05_TORCH_DUMP/tokenizer.model
export PI05_RAW_ORACLE=$DEVPROC2_ROOT/build/pi05_torch_dump_oracle
export PYTHONPATH=$DEVPROC2_ROOT/python:${PYTHONPATH:-}
```

需要重新生成 PyTorch oracle 时，只使用 `/root/tw/openpi/.venv/bin/python`；不要在 `/root/tw/openpi` 中执行 `uv sync`、`uv run` 或重装依赖。

