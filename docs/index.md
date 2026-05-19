# devproc2 文档入口

本目录按框架开发优先组织。Pi0.5 是当前最完整的模型落地路径，但不是文档树的唯一主线。

## 框架开发

- [Design Index](design/index.md)：编译器、IR、runtime、artifact、kernel 和 packed func 的设计入口。
- [Design Overview](design/overview.md)：系统整体定位、模块边界和长期路线。
- [IR Refactor Plan](design/ir_refactor_plan.md)：从 MVP IR 走向更可优化 IR 的阶段计划。
- [Elegant IR Design](design/elegant_ir_design.md)：长期 IR 设计目标和 pass contract。

## 当前 Pi0.5 流程

- [Pi0.5 Index](pi05/index.md)：Pi0.5 当前文档入口。
- [Product Build Quickstart](pi05/13_product_build_quickstart.md)：推荐的 runtime build、weight convert、model build、runtime inference 闭环。
- [Build, Run And Profile](pi05/08_build_run_profile.md)：更完整的编译、运行、精度对点和 Nsight profile 手册。
- [Performance Snapshot](pi05/06_performance_snapshot.md)：当前性能快照、已采用优化和未采用结论。
- [Compile Run Flow](pi05/14_openpi05_devproc_compile_run_flow.md)：OpenPI0.5 到 devproc2 C++ runtime actions 的分享稿。

## 历史归档

- [Archive](archive/index.md)：临时任务、旧 MVP 实施计划、已执行过的 Pi0.5 重构方案和阶段基线。

