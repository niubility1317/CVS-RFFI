# BiNOVA-D92阶段A最小实验报告

本文件镜像`E:\type10-7\automation_reports\CV-SincNet\binova_d92_stagea_minimal_20260829_r1\report.md`，用于Git版本管理。

## 预登记摘要

- 状态：`LOCAL_VERIFIED`
- run ID：`binova_d92_stagea_minimal_rx20_1_s713101_20260829_r1`
- 实现提交：`6a10fb85673f5caa8b5968b7901ac10f5dd8654c`
- 阶段A：A0、A2、A3、A4；A1为既有非门槛参考，不重复训练。
- 自动阶段B：A3相对A2同时满足伪注册H提升≥0.5个百分点、forgetting不增加、旧类floor不下降、非仿射残差≥20%时，自动运行B2、B3、B4、B5；否则记录`NOT_RUN_GATE_NOT_MET`。
- 协议：`p2_min_v1`、`VALIDATED_ONCE`、固定capsule/split、适应阶段query/truth均不打开。
- 本地验证：27项聚焦测试、45项联合回归、真实checkpoint无query smoke、Python编译检查和`git diff --check`均通过。
- P0/P1审查：3项P1已定点修复，复审后无未解决P0/P1。
- GPU：`cuda:0`
- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/binova_d92_6a10fb85/repo`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/binova_d92_stagea_minimal_rx20_1_s713101_20260829_r1`
- 技术停止规则：只允许协议、安全、路径、输出碰撞、启动、确定性异常、artifact闭合或进程归属问题停止；低性能不停止。

## 运行结果

- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- r1在阶段A A2前因N607 PyTorch2.1无法推断`numpy.bool`的dtype而自然退出；未产生性能结果、未打开query/truth、未进入阶段B。
- r1输出与日志保留；定点修复显式`torch.bool`后改用不可覆盖r2。
