# devproc2 设计文档入口

## 架构总览

- [overview.md](overview.md)：devproc2 的整体架构、模块边界和长期路线。
- [mvp.md](mvp.md)：早期 MVP 的完整设计草案，适合作为历史上下文和设计来源。
- [从llama.cpp中学到了什么.md](从llama.cpp中学到了什么.md)：runtime、artifact、backend 和工具链设计借鉴。

## IR 与编译器

- [value_system.md](value_system.md)：Value、StructInfo、PrimExpr 和 effect 的设计边界。
- [IR_Value_design.md](IR_Value_design.md)：IR Value 系统的细化设计。
- [control_flow.md](control_flow.md)：Effect-aware structured control flow 设计。
- [dynamic_shape.md](dynamic_shape.md)：动态 shape、upper bound、shape lowering 和 runtime shape builtin。
- [memory_planning.md](memory_planning.md)：DPS 之后的 storage reuse 和 memory-explicit IR。
- [elegant_ir_design.md](elegant_ir_design.md)：长期可优化 IR 的目标形态。
- [ir_refactor_plan.md](ir_refactor_plan.md)：从当前 MVP IR 迁移到目标 IR 的阶段计划。

## Runtime、Artifact 与 Backend

- [vm.md](vm.md)：VM bytecode、VMCodegenPass 和执行引擎。
- [abi_artifact.md](abi_artifact.md)：`executable.vm`、`abi.json`、manifest 和 artifact 打包。
- [packed_func.md](packed_func.md)：PackedFunc 与 `call_dps_packed`。
- [kernel_register.md](kernel_register.md)：Kernel registry 与 attr system。
- [kernel_launch.md](kernel_launch.md)：`@dp.kernel`、Triton AOT cubin 和 CUDA launch。
- [e2e_demo.md](e2e_demo.md)：M12 端到端 demo 的历史设计。

