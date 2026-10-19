# 冻结确定性残差恢复器的完整正式观测确认

状态：RUNNING / VERIFIED（完整正式观测预测）。方法代码 `9522a4edb63f30293542de7326ce1e53aa816ed5`。新增源冻结/训练契约负测及 28 项回归检查、compile/help 通过；复用 R03 不变候选原独立 P0/P1 审查，不增加重复 runner review。GPU0 空闲约 24 GB，固定输入与父权重可用，新 root 不存在。

候选仅为 R03 源波形 A 与源身份 B 共同选中的确定性修正器。原 R03 源资格为 low urban 相对粗恢复 +1.5556 个百分点、clean 下降 0.1111 个百分点且平均 NMSE 改善，选择在目标访问前已冻结。本轮准备期间没有回收 R04 正式目标指标，不依目标分数改变候选或阈值。

本轮训练、适配、F 及分类头拟合均为 0。累计恢复器成本为粗恢复 E20＋修正 E20，原 cleanF E200 与 AsF E20＋自身 E10 另计。统一恢复 IQ 派生原 FFT/PA/时频分支，始终使用原默认头；clean 同样经过恢复器，没有场景真值旁路。

矩阵、命令、环境、路径、GPU 和停止规则见 [experiment.json](experiment.json)，既有实现见 [R03 追踪](../../../docs/PAMF_RESIDUAL_REFINEMENT_R03_20261019.md)与 [正式适配追踪](../../../docs/PAMF_FORMAL_RESTORE_R04_20261019.md)。复用 R04 同一套原 168000 条 opaque IDs 与 clean＋六 practical 输入，先全部七视图预测固定，再独立 CPU truth-last 评分。辅助配对特征 margin 仅用固定 512 参考记录，显式报告局限；不用于推理或选模。

R04 粗恢复保留为预登记同输入控制，不据结果择优改写既有矩阵。本轮 source qualification 允许的正式扩展仍只有单 R seed；同原生引擎、不同固定接收会话不能冒充独立物理族或在轨证明。技术异常保留产物、不自动重试；性能不足如实记录，最终身份目标保持 active。

发布证据：一次归档比较一致、远端 compile/import 通过；独立 PID/CWD/完整 argv/GPU0 匹配注册计划。真实源 2 样本复合 checkpoint 与原默认头 smoke PASS，TX/目标读取为 0；日志由 0 增至 111 字节，clean 168000 条预测已完成。七视图未全部完成前不接 truth；R04 和其他 owner 健康任务保持原运行。
