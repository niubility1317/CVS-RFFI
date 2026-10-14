# 跨 RX 脆弱方向：源原型预登记

运行 `20261015-phase1-SIGTIME-crossRX-feature-sharpness-sourceL-manysig-s2026092701-r01`。固定匹配 BASE 首 seed 2026092701/final50 为诊断父状态，加载前检查原 source 角色／来源／固定选模，不据目标分数挑权重。CENTER_25K 和 SIG_TIME 各自正路线保留，不相加历史增益。原默认 6×160 分类头保持。这里只做 source-only counterfactual，0 实际训练／头拟合／EMA，不访问 U/V/目标 IQ 或 truth；后续实际训练遵循当前 E20 预算重新预登记，不默认沿用 E50。

固定 ZERO／CLEAN_CE／GLOBAL_GAP／CROSS_GAP 四控制 × currentL90 ordinal0/17/35/52。原 L180 bank ordinal68/69，与当前完整物理 ID/base_index 分离；曾用于 checkpoint 的源 L，不称未训练 holdout、未见 RX 或 V。540uniqueIQ／6300元信息。选择17个真实部署参数张量：primarySinc、time_fuse、t1、t2、SIG_TIME sequence_pool（除gain）。原构造origin／scale25000保持，不扫描单位／rho／权重／温度／margin。

ρ=.05、coefficient=.1、normalizeeps1e−12。CROSS 的每RX δ=.05*g_r.detach()/max(||g_r||,eps)，inner当前源RX CE，outer在otherRX分离源L计算relu(meanpertCE-meancleanCE)，两项都对selectedθ保留graph。所有其余params（含默认W/gain）只detach作固定读出，δ不接二阶；主原默认头联合训练方式不变。GLOBAL用完整currentL一条δ，其余masks相同；CLEAN仅同masks cleanCE；ZERO0。各mode均clean一次＋5pertotherRXforward，每mode900packets/cohort，共14762packetforwards（含2零IQ realCPsmoke）。对CROSS各cohort用saved onegroupAdamW核对实际辅助increment，共8temporarysteps/12kernel syntheses，0realupdate。

源可用性条件一次固定：至少2/4cohorts CROSS存在otherRX mean gap>1e−4且auxgrad>1e−8；至少2/4cohorts ||CROSSgrad−GLOBALgrad||/max(||GLOBALgrad||,1e−12)>=.05；至少2/4cohorts的actualAdamW辅助increment（totalvsbase）改变至少一个Sinc kernel。只检验非零且有区别的部署信号，不宣称效能或创新；失败不扫阈值、不按target重排。完整mainloss/U/EMA的动态作用仍需实际E20训练检验。

所有functionalstate克隆buffer，临时eval/保留RNG/逐次清空并恢复Sinc与归一化Wcache，结束检查liveparams/buffers/gradient/moment未变；辅助W/gain梯度None、其他非selectedNone，mainW梯度非零。当前prototype current cleanx直接生成logits，与metadata同view。未来Lhook默认labeled_inputs可能原clean而主logits来自augview；inner应明确使用原主logits，不误称输入完全同view。不安装覆盖额外functionalforward的永久capturehook。

代码/config唯一身份release_commit.txt。普通N607项目Python，CPU2threads/GPU0，CWD `releases/cvs_SIGTIME_crossRX_feature_sharpness_sourceL_20261015_r01`，唯一immutableoutput `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-crossRX-feature-sharpness-sourceL-manysig-s2026092701-r01/source_diagnostic`。发布 `python -X utf8 E:/type10-7/.codex_tmp/native_polyphase_runner.py experiments.cvs_crossrx_sharpness50.publish`；runner命令见experiment.json。一次N607preflight、一个archiveSHA比较、remotecompile；首步严格realCPno-querysmoke后直接执行；独立PID/CWD/argv/GPU/loggrowth。唯一P0/P1审查只覆盖下一次真实源原型。仅预定确切runowned系统技术错误停止，保留产物，不自动重启或动其他健康进程。预期runtime/4cohort全部控制/summary及same-row分析登记Git。

方法区别与原文依据见[机制说明](../../../analysis/phase1_crossrx_feature_sharpness_20261015.md)：SAM／SAGM／DISAM／DGSAM／2026DualAdaptiveSAM已有相关sharpness思想。本候选区别是feature-only、固定读出、RX内造方向／RX外且current-packet-disjoint学习gap；这是待证假设，尚无原创性或泛化收益结论。84%默认头clean三种子目标仍未完成。

唯一独立 P0/P1 审查（2026-10-15）PASS：detached delta、17 个特征梯度边界、跨 RX／跨包分离、缓存恢复、固定 BASE 来源和发布依赖均未发现下一次源原型的直接阻断错误，14,762 IQ／8 临时 AdamW／12 kernel 核算一致。聚焦 CPU 合成检查 VERIFIED：clean 两项梯度保留，W/gain/其他参数 auxNone，ZERO exact0，4模式相同前向预算，状态/RNG恢复及非法 role／物理 ID／scope 负例通过；尚不代表真实源有效或 DG 收益。
