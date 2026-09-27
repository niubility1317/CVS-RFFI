# IR-EG / BR-IR-EG研究准备记录

状态：`PREPARED_NOT_LAUNCHED`。本轮完成设计的代码实施与本地合成验收，不启动真实数据或远端训练，不读取目标评分，不继承历史checkpoint。

设计与实现位于[独立执行包](../../../experiments/ir_eg_v1/README.md)。矩阵、六类seed、配置和输出位置的唯一登记入口是同目录`experiment.json`；开发验证见[需求追溯](../../../experiments/ir_eg_v1/docs/traceability.md)及`acceptance/`机器可读证据。

正式性能、目标评分及同源侧收敛判据下的效率结论尚未产生。实际source物理ID manifest尚未绑定；`convergence_confirmation.json`尚未冻结后续学习率和安全上限。这些是未来实验的输入状态，不是本轮额外审批，也不改变现有运行。

本轮发现并保留的CUDA验收失败见`acceptance/cuda_initial_failure/`；最终状态只以`acceptance/cuda/result.json`和交付报告为准，不以早期失败或早期成功推断当前代码状态。
