# 原始与 DURA 恢复 IQ 的源域配对诊断

状态：PLANNED；仅冻结源诊断，不启动 DURA-aware F/G 优化，不访问目标 IQ。用户要求先分析恢复 IQ 增强为何弱于无增强方案。本诊断落实 [既定计划](../../../docs/goals/PAMF_RESTORED_IQ_SSL_PLAN_20261025.md) 的第一部分。

## 固定矩阵与输入

- F：原 F0 ordinary PL 的 E30，model seeds 2026092701/02/03；各自从已核实同契约 scratch E20 接自身 10 轮。三份固定权重全部使用，不按新目标结果挑选。
- G：原 F0 同契约 DURA v1.0 G-LU，266951 参数。既有 G 不读取 TX/F/target 的来源结论沿用，加载时检查同契约。FP32/TF32off，reuse=True、graph=True、native=False。
- L_s：原 L6300，仅取 2 条用于真实冻结权重无 query smoke。验证 G eager/部署路径一致、同输入重复输出一致、原默认头和实际 160 维特征接口。
- V：固定原 source V27000、source RX/day、原 TX 映射和物理顺序。U IQ 与 U 真值均不打开。复用 VALIDATED_ONCE，不重新哈希数据。
- 每条 V 比较 raw、G(raw)、三种 H(raw) 及三种 G(H(raw))，共 8 视图。H 采用既有 residual/post_sync/noEQ 的 low_suburban、mid_urban、low_urban 参数；信道 seed20261016421，硬件 seed20261016423，固定 namespace `PAMF/restored_source/V/fixed`。视图按原物理记录配对，不视为新增独立样本。
- 预算：648000 次 F packet forward、108000 次 G packet forward，另列 L smoke。无 optimizer、训练、另训头、参数或 normalization 状态更新；不会用源 V 梯度更新状态。

## 输出与解释

1. raw 与每个视图的四格计数：均正确、纠正、破坏、均错误，逐 TX/RX/day 和 raw 共识子集列出。
2. 原默认头真类 margin、同物理特征 cosine 距离、按类和 RX 的特征中心几何、输入 IQ 功率。只用不拟合参数的分析，不建立新的诊断分类头。
3. 固定 .95 阈值下三个 raw 教师的精度、覆盖率及一致集合。源可行性要求共识精度≥99.5%、相对单模型平均接受集合保留率≥90%；不是目标性能保证。
4. 私有保存 logits/物理 ID/source V 标签以复核；公开仅汇总 CSV、分析和来源。执行前后精确比较冻结 F/G 的参数与 buffers。

源 V 结果不能证明目标提升，也不能仅凭冻结 F 在恢复输入上变差就断言 G 抹掉 TX。梯度剂量、恢复后弱扰动以及新 F 训练仍属于后续步骤，本次不执行。

## 发布与验证

代码/配置身份为本次 Git commit，发布 runtime 记录实际 commit；不追加逐文件/数据哈希。模型矩阵唯一维护于 experiment.json。

命令：`PYTHON -u -m experiments.pamf_channel_pl.restored_source dispatch --config PROJECT/releases/20261025-diagnostic-PAMF-DURA-paired-source-release-r01/runtime.json`。

CWD：上述不可覆盖 release；run root：`PROJECT/runs/20261025-diagnostic-PAMF-DURA-paired-source-m3-r01`。N607 普通账户、GPU0 一条本 run worker、全卡 CUDA/待启动 worker 少于 2 且空闲显存≥10000 MiB 才准入，CPU 2 threads，不改变其他作者任务。精确环境和私有路径在 local_artifacts/pamf_ssl_research_20261025/restored_source/runtime.json。

4 条针对目标句柄拒绝、跨契约/祖先/实际 EMA 标志、配对计数及相关伪标签错误的测试 PASS；一次独立限定 P0/P1 审查 PASS。真实权重 smoke 为 launcher 首步，通过后直接完整 V，不设额外审批。一次 archive 传输 SHA 比较、远端编译及启动后 PID/CWD/argv/GPU/log-growth 独立核实。

停止规则：本 run 任何技术、数据权限、来源、非有限或持久状态变化立即失败并保留产物；不自动重试，不干预健康旧任务。预期产物 smoke.json、progress.json、private source_V_predictions.npz、complete.json、pipeline_state.json，以及同 run 公开分析与登记。
