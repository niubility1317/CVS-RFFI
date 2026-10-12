# 独立 P0/P1 发布审查

结论：**PASS；已修复1项P1，剩余P0/P1为0。** 日期：2026-10-12；审查者：`/root/stage2_release_review`。

本次范围为 `20261012-phase1-finite-action-transfer-manysig-m3-r01` 的设计、实际读出与拟合、保存状态、源汇总及发布控制。审查者未参与实现，仅写本文件；未使用GPU、访问远端、启动正式诊断或读取真实目标产物。本结论确认执行正确性，不代表作用迁移有效或身份性能改善。

## 已关闭的问题

原 `run.py`、`preflight.py` 为复用 snapshot 导入 `cvs_source_identity_readout.run`，该运行入口立即导入未进入归档的 `cvs_source_conditional_geometry`，会使隔离发布的preflight报ModuleNotFoundError。实现者已改为本包 `state.snapshot`，run/preflight/checks不再依赖旧运行入口。定点复核：在新Python进程中拒绝导入归档外的experiments模块，run、preflight、checks、summarize均成功导入；原P1关闭。当前归档66个路径全部存在，直接依赖及发布入口正确。

## 设计与实际行为

- 三行登记config与 `design.config` 完全相等。参考身份网络固定为各seed合规noCFO纯CE final20；实际加载复用严格teacher验证，核对完整config、源数据契约、固定提交、seed、预算与空祖先。无目标排名或候选选择路径。身份网络eval/no_grad，三实际层的读出detached；前后核验全部参数及注册buffer，并拒绝梯度残留和训练模式。
- L6300可见TX，只用L拟合作用预测网络、标准化、ridge及输出PCA。13个split为PID80/20、6个TX、3个day、3个quality组。每个split的fit/eval物理ID不相交；质量分组使用全L观测，仅作为明确披露的源分层定义，预测器标准化仍只用fit。原身份编码器已见这些源TX/RX，报告没有称为未知身份或未知RX泛化。
- U56700仅做同包已知操作响应评估，拒绝TX真值字段，不进入任何预测器拟合；只用PID80/20对应的已拟合模型验证U。V从旧packed源缓存反序列化后即丢弃，不前向、不拟合。没有target读取、自动身份训练或自动目标测试入口。
- 频谱FIR、分数时延和四次私有随机噪声均按已知参数作用于真实256点IQ；前两者保持包RMS，噪声保留其增加的随机能量。三作用×三层×13split使用独立网络。参数线性、强双线性ridge、fit delta输出PCA rank8再ridge与MLP共用合法fit/eval；低秩基线没有被表述为最优RRR。
- noise目标是四次实现的经验均值和unbiased=False样本内方差。训练高斯期望NLL含均值误差、重复方差及log方差；物理尺度评估含明确variance floor。该流程没有把具体噪声实现作为可恢复状态。RX覆盖子空间只从fit预测有限差分求得，独立eval L条件均值用于覆盖评估；报告明确未配对分布几何，不作因果恢复解释。前导任务保留N/A，lag16仅质量代理。
- 独立复算预算：每行117个网络、2340个aux epoch、117180次aux更新、94500 L与396900 U模型样本读出、1929前向；三行351个网络、351540次aux更新、1474200模型样本读出。身份和头更新均0。共有1890条评估记录、135条覆盖记录；完整JSONL/CSV及闭合stdout进入自动汇总。
- 新run/release/output独占，唯一owner；共享容量统计已有CUDA所有者和启动前预约，每卡最多3个总进程并保留12GB空闲。失败保留产物、不自动重启、不干预健康任务；三行完成后自动调用source-only finish，更新报告、原登记及events。

## 验证证据

已读回登记 `evidence/finite_action_cuda.json` 的PASS：原生合成L240/U112实际读出、13split、零操作/RMS/私有RNG、fit-only统计、强双线性人工基线、U标签与覆盖PID泄漏拒绝，以及参考状态不变。未重复运行GPU检查。

独立本地CPU合成验证：

1. 三类作用各20个实际小样本辅助更新；按 `run.py` 的实际保存格式序列化并重建Action网络，五种方法的均值/方差推断均逐位相等。fit输入均值与fit-only手算一致；无身份更新或真实数据读取。
2. 实际执行 `summarize.analyze` 和 `finish`，完整解析三行共7020条JSONL及对应CSV，351个fit组、351540更新、1890条评估、135条覆盖全部闭合。报告/analysis/experiment状态/events正常写出；闭合stdout之后的错误哨兵未读取；identity_updates=1负例被拒绝。此测试没有mock源分析函数，但所有输入都是临时合成记录，不代表正式辅助拟合已完成。
3. 登记exact-config、独立预算复算、归档存在性及修复后的隔离导入检查通过。新增CSV日志字段与实际预测器epoch记录逐项对应。

剩余P0/P1为0；无额外审批或性能门槛。
