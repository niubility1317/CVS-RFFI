# 80% 全标签训练／20% 验证：E30 监督参考

用户于2026-10-12明确指定80/20划分、训练集全部有标签及E30。当前CVS residual_fusion身份网络保留内部门控、参考响应和cosine分类头；纯CE、label_smoothing=0，关闭U、EMA、伪标签、星地训练增强及辅助损失。从零训练，不加载旧checkpoint。

ManySig全部6TX；源RX索引1/3/4/6/8、day1/2/3。每TX/RX/day/eq按sig_i排序，前800训练、末200验证，72000/0/18000。每轮batch128完整打乱遍历，最后64条，563步；E30共16890步。AdamW，LR2e-4至1e-6，weight_decay1e-4；FP32。源V只观察健康，固定final30。

单seed2026092701。这是全标签监督参考，不是伪标签算法的数学上限；与默认7/63/30同时存在训练池、验证池和曝光成本差异，不能只归因标签质量。当前测试基准已暴露，不声称新独立盲测。测试成绩不回流训练、选模、排序或选择性重跑。

训练自动接固定模型、clean及六种practical预测，全部7x168000条预测完成后独立truth-last评分。复用既有VALIDATED_ONCE观测，改变源角色后核对source/target物理ID不交。矩阵、命令、环境、路径、owner及预期产物见experiment.json。commit由release_commit.txt和发布证据记录。

技术停止规则：数据角色/物理ID/预算错误、query越界、非有限训练、启动失败、输出碰撞或无法闭合合法prediction时保留产物并停止本run后续阶段。低精度不触发停止；不自动重跑，不干预健康旧任务。每GPU最多3个总进程、至少12000MB空闲。

当前状态：本地实现，未发布。一次真实自有scratch checkpoint无query smoke由launcher训练首步执行，PASS即继续。

本地聚焦验证PASS：80/0/20物理角色、负测、完整72000样本/563批、实际网络CE梯度及无辅助训练。唯一独立P0/P1审查通过，未发现阻断缺陷。

## 实际发布状态

2026-10-12T20:09:10.177076+08:00：VERIFIED／RUNNING。实际GPU0，从零E30/16890步，L/U/V=72000/0/18000。release archive校验、远端compile、实际source-only数据预检和自有scratch checkpoint smoke均通过。PID/CWD/argv/GPU及日志增长证据保存在本地evidence/launch_readback.json。执行提交`6d701688f40d4cda0862430cdf0b5281a5b14114`。首轮进度：1。

训练、七视图预测和独立truth-last评分仍在执行；尚无最终测试结果，不标为完成。远端run root与精确命令见experiment.json，自动训练→固定final30→clean与六practical预测→独立评分→scores/analysis.md，无需再次授权测试。

## 完成与测试结果

2026-10-12T20:49:47.281695+08:00：VERIFIED／ANALYZED。E30训练、固定final30、全部7×168000条预测和独立truth-last评分均完成。未发现技术失败。

|测试视图|样本数|准确率 (%)|Macro-F1 (%)|
|---|---:|---:|---:|
|clean|168000|82.3649|82.0748|
|practical_high|168000|63.4161|63.5374|
|practical_mid|168000|60.1137|60.0211|
|practical_low_suburban|168000|57.3262|57.0658|
|practical_high_urban|168000|60.4946|60.4333|
|practical_mid_urban|168000|47.3702|46.7421|
|practical_low_urban|168000|38.1524|37.1103|

六种practical场景准确率算术均值：54.4789%。这是单seed结果，不能替代多seed统计。
实际训练时间：29.25分钟；16890次更新。源验证集最终准确率：98.6444%。验证指标与目标测试分开报告。
该结果提供本数据划分、网络和E30预算下的全标签参考；clean82.36%不能被解释为伪标签算法的理论上限。与默认弱标注实验的训练/验证划分及曝光预算同时不同。
完整每场景/RX/day/TX指标保存在远端scores/metrics.json，本地只读回收副本位于local_artifacts/cvs_oracle80_e30_20261012_r01/results/；原始预测和权重保持远端。
