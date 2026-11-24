# WiSig 平滑前导 DURA Main split 三种子正式测试

[预登记](experiment.json) · [运行事件](events.jsonl)

固定源训练 `preamble` 臂的第 40,000 步和源 U 门限，正式读出为 `preamble_gated`。同一权重、门限和规则覆盖两条路线、两个划分、三个冻结 Tri-SIG-TIME 种子。每条路线保留 `cross_rx` 的 126,000 个物理包和 `cross_both` 的 42,000 个物理包，两者互斥，合并口径为 168,000。沿用原六种 practical、clean、原直通以及在相同视图上重跑的原 DURA R69。 [原配对基线与监督](../../../docs/analysis/OBSERVABLE_SATELLITE_RECOVERY_20261123.md) · [源端训练](../20261124-phase1-wisig-preamble-dura-source-m4-r01/report.md)

末步冻结后，先做不读取目标的真实检查，记录 RTX 3090 单包恢复时延。同一包恢复一次后，三套冻结识别器共享该份 IQ。分四个路线与划分任务盲预测。两种原读出的真实无查询组件检查已用合法源 IQ 完成，在三个种子及 batch 128、48、16 下，恢复 IQ、分数和 576 次决策与原联合图实现逐值一致。候选的权重级检查仍在末步冻结后执行。 [正式盲预测](../../../experiments/pamf_unrolled_restore/wisig_preamble_dura_target.py)

独立 CPU 在四片预测固定后先验证物理 ID 和重跑的直通与原 R69 决策，再打开真值。逐种子、场景和接收机报告正确数、纠错、改错及同包不确定性。各路线按三种子合并口径判定相对原 R69 的六景净增和 clean 正确数，同时列出两个划分各自的结果。当前已预登记，正式任务等待末步权重及门限。 [独立评分与验收](../../../experiments/pamf_unrolled_restore/wisig_preamble_dura_score.py)
