# 跨 RX 脆弱方向：源原型结果

**真实源原型执行完成，固定信号可用性条件 4/4 通过；泛化训练收益尚未验证。** Fixed BASE 首 seed 2026092701/final50，仅诊断；执行 commit `1bf44076deae732f4805199f564cf34548a4b952`。固定 CENTER_25K + SIG_TIME 的17个部署特征参数、原默认 6×160 读出、ρ=.05／系数=.1，无新增模型／头拟合／实际训练／U/V/target访问。

| current cell ordinal | CROSS 最大 otherRX CE gap | CROSS 辅助梯度范数 | CROSS-GLOBAL 梯度相对差 | 辅助实际增量改变 Sinc 核数／24 |
|---:|---:|---:|---:|---:|
| 0 | 0.000632533 | 0.00161637 | 3.53167 | 2 |
| 17 | 0.0040048 | 0.00355873 | 0.692271 | 2 |
| 35 | 0.00273487 | 0.00719309 | 0.441342 | 3 |
| 52 | 0.0025773 | 0.00482241 | 0.901887 | 2 |

4 个批次均满足预登记：存在正 gap>1e−4、auxgrad>1e−8、与GLOBAL梯度相对差>=.05，并且actualAdamW辅助增量（totalvsbase）改变至少1个Sinc核。浅中层 time_fuse/t1/t2 与实际SIG_TIME梯度均非零；Sinc坐标梯度亦非零。默认W/gain/其他非selected参数辅助梯度None，原mainW梯度非零。该证据表明部署路径存在不同于共同扰动方向的可用信号，不证明它是接收机专属噪声、具有独创性或能提升目标准确率。

ZERO／CLEAN_CE／GLOBAL_GAP／CROSS_GAP 的完整4×4结果与80个RX项已保留；sourcebank/current图均不改读出头，clean和perturbed两项保留特征梯度，δ停止梯度，始终固定5分母。4个不同currentL90均排除原L180bank的完整physicalIDs/base_index；它仍是已训练过的源L，不是未见RX或未训练holdout。CLEAN_CE weightedloss固定约.000552457，CROSS weightedloss约2.43e−5至1.56e−4；loss数值不能替代训练效能比较。

实际成本：540 unique源L IQ／6300metadata、14762 packet networkforwards、8temporarysteps／12kernel syntheses，0realoptimizer／EMA／head-fit更新。真实checkpoint no-querysmoke与唯一P0/P1 review PASS；archiveSHA／remotecompile VERIFIED；独立PID/CWD/argv/loggrowth/GPU检查确认CPU-only，summary与自然退出核实。参数／buffer／optimizer／gradient／RNG均保持。legacy counterfactual summary中的U_increment字段在此表示本辅助项增量，不代表访问U真值。

当前新主路线是跨RX feature loss gap：RX内监督梯度造方向，RX外且排除当前物理样本的L学习脆弱性差值。已开启E20匹配训练的代码准备；计划保留同预算纯CE、相同SSL、普通额外sourceCE、GLOBAL_GAP与CROSS_GAP控制，先首seed，再按固定源规则扩种子。当前尚未发布或完成该训练矩阵，阶段日程和预算在实际release前固定；不能把source原型或历史E50当新E20效能证据。84%默认头clean三种子目标仍active且未达成。

以下原预登记保留，阈值、矩阵和半径未追溯更改。

---

# 跨 RX 脆弱方向：源原型预登记

运行 `20261015-phase1-SIGTIME-crossRX-feature-sharpness-sourceL-manysig-s2026092701-r01`。固定匹配 BASE 首 seed 2026092701/final50 为诊断父状态，加载前检查原 source 角色／来源／固定选模，不据目标分数挑权重。CENTER_25K 和 SIG_TIME 各自正路线保留，不相加历史增益。原默认 6×160 分类头保持。这里只做 source-only counterfactual，0 实际训练／头拟合／EMA，不访问 U/V/目标 IQ 或 truth；后续实际训练遵循当前 E20 预算重新预登记，不默认沿用 E50。

固定 ZERO／CLEAN_CE／GLOBAL_GAP／CROSS_GAP 四控制 × currentL90 ordinal0/17/35/52。原 L180 bank ordinal68/69，与当前完整物理 ID/base_index 分离；曾用于 checkpoint 的源 L，不称未训练 holdout、未见 RX 或 V。540uniqueIQ／6300元信息。选择17个真实部署参数张量：primarySinc、time_fuse、t1、t2、SIG_TIME sequence_pool（除gain）。原构造origin／scale25000保持，不扫描单位／rho／权重／温度／margin。

ρ=.05、coefficient=.1、normalizeeps1e−12。CROSS 的每RX δ=.05*g_r.detach()/max(||g_r||,eps)，inner当前源RX CE，outer在otherRX分离源L计算relu(meanpertCE-meancleanCE)，两项都对selectedθ保留graph。所有其余params（含默认W/gain）只detach作固定读出，δ不接二阶；主原默认头联合训练方式不变。GLOBAL用完整currentL一条δ，其余masks相同；CLEAN仅同masks cleanCE；ZERO0。各mode均clean一次＋5pertotherRXforward，每mode900packets/cohort，共14762packetforwards（含2零IQ realCPsmoke）。对CROSS各cohort用saved onegroupAdamW核对实际辅助increment，共8temporarysteps/12kernel syntheses，0realupdate。

源可用性条件一次固定：至少2/4cohorts CROSS存在otherRX mean gap>1e−4且auxgrad>1e−8；至少2/4cohorts ||CROSSgrad−GLOBALgrad||/max(||GLOBALgrad||,1e−12)>=.05；至少2/4cohorts的actualAdamW辅助increment（totalvsbase）改变至少一个Sinc kernel。只检验非零且有区别的部署信号，不宣称效能或创新；失败不扫阈值、不按target重排。完整mainloss/U/EMA的动态作用仍需实际E20训练检验。

所有functionalstate克隆buffer，临时eval/保留RNG/逐次清空并恢复Sinc与归一化Wcache，结束检查liveparams/buffers/gradient/moment未变；辅助W/gain梯度None、其他非selectedNone，mainW梯度非零。当前prototype current cleanx直接生成logits，与metadata同view。未来Lhook默认labeled_inputs可能原clean而主logits来自augview；inner应明确使用原主logits，不误称输入完全同view。不安装覆盖额外functionalforward的永久capturehook。

代码/config唯一身份release_commit.txt。普通N607项目Python，CPU2threads/GPU0，CWD `releases/cvs_SIGTIME_crossRX_feature_sharpness_sourceL_20261015_r01`，唯一immutableoutput `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-crossRX-feature-sharpness-sourceL-manysig-s2026092701-r01/source_diagnostic`。发布 `python -X utf8 E:/type10-7/.codex_tmp/native_polyphase_runner.py experiments.cvs_crossrx_sharpness50.publish`；runner命令见experiment.json。一次N607preflight、一个archiveSHA比较、remotecompile；首步严格realCPno-querysmoke后直接执行；独立PID/CWD/argv/GPU/loggrowth。唯一P0/P1审查只覆盖下一次真实源原型。仅预定确切runowned系统技术错误停止，保留产物，不自动重启或动其他健康进程。预期runtime/4cohort全部控制/summary及same-row分析登记Git。

方法区别与原文依据见[机制说明](../../../analysis/phase1_crossrx_feature_sharpness_20261015.md)：SAM／SAGM／DISAM／DGSAM／2026DualAdaptiveSAM已有相关sharpness思想。本候选区别是feature-only、固定读出、RX内造方向／RX外且current-packet-disjoint学习gap；这是待证假设，尚无原创性或泛化收益结论。84%默认头clean三种子目标仍未完成。

唯一独立 P0/P1 审查（2026-10-15）PASS：detached delta、17 个特征梯度边界、跨 RX／跨包分离、缓存恢复、固定 BASE 来源和发布依赖均未发现下一次源原型的直接阻断错误，14,762 IQ／8 临时 AdamW／12 kernel 核算一致。聚焦 CPU 合成检查 VERIFIED：clean 两项梯度保留，W/gain/其他参数 auxNone，ZERO exact0，4模式相同前向预算，状态/RNG恢复及非法 role／物理 ID／scope 负例通过；尚不代表真实源有效或 DG 收益。
