# 纯 F E20：训练流重建与新信道源拟合诊断

完成状态：VERIFIED。四个固定终轮模型共5761个源前向包（含1包真实checkpoint smoke），0优化器/EMA/分类头更新，V全部复用已有logits，目标IQ/truth和正式评分均未访问。运行 `20261019-diagnostic-F-observedChannel-e20-s2026092701-m4-r01`，代码/config `6864ce18aae696f551feadf953efeaa901e97a5c`，实际wall14.302秒。总体研发目标尚未完成。

关键发现：low urban的识别困难已经出现在训练期消费流的重建观测中；本轮风险权重没有修复它。以下是同一seed2026092701，不能解释为跨seed稳定结论。

## low urban 的默认头识别

“旧流”指FINAL20对训练期固定命名流的确定性重建，不存在原As IQ存档供独立字节核验。所选180个L ID每个TX×RX×day cell2个，其中low旧流69个；新draw严格匹配这69个ID和场景。V为1800个不同物理ID×2draw。

|模型|旧流正确数/69；准确率|同L新draw正确数/138；准确率|V新draw正确数/3600；准确率|
|---|---:|---:|---:|
|B0|31；44.928%|65；47.101%|1539；42.750%|
|B1|45；65.217%|89；64.493%|2237；62.139%|
|BALANCED|44；63.768%|90；65.217%|2223；61.750%|
|RISK|44；63.768%|91；65.942%|2226；61.833%|

B0没有接受As训练，不把这些重建观测称为B0见过的As。四模型在所选180个raw L上均180/180；这不是完整L6300拟合率，也不是ordinary增强在线训练准确率。

|模型|旧流真标签CE|同L新draw真标签CE|V新draw真标签CE|
|---|---:|---:|---:|
|B1|0.829691|0.862350|0.936680|
|BALANCED|0.837099|0.862484|0.938138|
|RISK|0.825490|0.858356|0.932725|

CE为默认logits的标准真标签交叉熵，不加温、不平滑；训练smoothing0.01的CE另保存在分析JSON中。RISK相对B1：旧流修正2/新增3，净少1个正确；同L新draw修正5/新增3，净多2个；V修正59/新增70，净少11个。轻微CE下降没有形成V准确率增量。

## 三场景与观测难度

|模型|low suburban旧流→新draw|mid urban旧流→新draw|low urban旧流→新draw|
|---|---:|---:|---:|
|B1|86.000% → 93.000%|85.246% → 71.311%|65.217% → 64.493%|
|BALANCED|88.000% → 94.000%|85.246% → 70.492%|63.768% → 65.217%|
|RISK|88.000% → 94.000%|85.246% → 72.951%|63.768% → 65.942%|

三个场景旧流n分别50/61/69，新draw n分别100/122/138，不能用不同观测难度的准确率差直接证明过拟合。mid urban新draw更难：BLOCKED由34.43%升至44.26%，unlocked由13.11%升至25.41%，signal_to_added_noise_db均值由18.649降至15.786 dB。该量是新增噪声比，不是总SNR。low suburban新draw反而更容易，BLOCKED由12%降至4%。low urban组成较接近：BLOCKED53.62%→56.52%，unlocked37.68%→39.13%，新增噪声比均值10.693→10.869 dB。

这支持低urban严重受损观测仍未被充分识别；不能据此证明物理上限、结构上限或所有优化路线无效。两draw和RX/day切片不充当模型seed。

![源拟合诊断](E:/type10-7/exports/cvs_pureF_sceneRisk_20261019/source_fit.png)

## 执行与保存

L6300/U56700/V27000、源RX1/3/4/6/8、day1/2/3、split392005保持。先验证L物理ID/索引再读取IQ；未遍历U。回放896包完整batch与180包独立单包逐元素一致，BALANCED/RISK保存的scene槽和RX/day/eq session对齐。两个新namespace固定，不改Config、receiver或正式信道。四模型均严格加载自身scratch FINAL20，所有权重和buffer前后完全相等。

一次P0/P1审查、本地窄负测、单归档传输比较和远端编译均PASS；首步真实源checkpoint smoke PASS后立即读出。代码与远端Git分支独立比对一致。任务在首个postlaunch快照前已结束，所以活动PID/CWD/argv回读为UNKNOWN；完整模型读出、冻结状态和完成产物为VERIFIED，没有重新运行来补快照。GPU5及普通账号约束在实际代码中检查。

完整metadata、预测、channel记录与JSON留本地 `local_artifacts/F_observedChannel_E20_20261019` 和不可覆盖远端运行根；不将IQ、权重、逐包数据和原日志提交Git。此诊断不改sceneRisk预登记规则，不救援失败路线。
