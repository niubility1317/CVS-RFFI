# ADVB02 NTRS-V4反事实风险优化矩阵发布报告

## 发布结论

`phase1_advb02_ntrs_v4_counterfactual_matrix_20260821_r1`已于2026-08-21发布到N607并进入`RUNNING`。本轮固定seed=`392034`、`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，训练和最终测试均只使用`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，不使用历史`mixed_orbit`。

实际运行代码提交为`9634f72381be46e4f330c04367f6a5303dc4f6f7`，远端分支OID已与本地HEAD独立核对一致。首个release归档本地/远端SHA256均为`38186078fc0b27e1ab3873278296c28eff476eb774fa537e382a623db0c08b90`。归档落地后仅针对N607 NumPy2.2.5/PyTorch2.1.0 ABI不兼容同步了提交`9634f723`中的2个B0运行文件，没有重新打包或增加成员hash。

## 实现与验证

- 保留成熟D1 raw骨干与共享CosFace头，训练时raw参数冻结。
- IQ上下文使用精确Phase1`V_cal`一次性拟合的signed-log median/IQR。
- metadata只作为训练教师，独立测试强制IQ-only。
- 条件算子为`U[a(q)⊙(V^Tz)]`，锚点detach且修正上限为2%。
- 目标包括paired shift、paired cosine、clean-tail、raw-correct harm保护与clean-correct/satellite-wrong rescue。
- B2-A使用B0固定PCA公共shift，不是可训练additive残差。
- 本地Python编译和78项聚焦测试全部通过。
- 独立审查无P0；B2-A定义、B1-N统计和B0配对3项P1修复后定点复审均为`RESOLVED`。
- 真实D1无query smoke使用source训练日、接收机0–6的实际IQ及`leo_clear_weak`配对前向，`raw_trainable_parameters=0`、`unexpected_keys=0`，输出有限。

## B0诊断

B0只读取精确`V_cal`，共12,600个物理样本、37,800个clean/三场景配对；逐行核对`tx/rx/day/eq/sig`、checkpoint和manifest。

|rank|解释方差|oracle准确率|rescued|harmed|净救回|
|---:|---:|---:|---:|---:|---:|
|4|97.495%|98.101%|5,061|86|4,975|
|8|98.600%|98.204%|5,116|102|5,014|
|16|99.236%|98.185%|5,111|104|5,007|
|32|99.625%|98.201%|5,112|99|5,013|
|full shift|100%|98.214%|5,116|98|5,018|

raw准确率为84.939%。方差分解中场景主效应为0.105%，TX主效应为71.186%，TX×场景交互为0.208%。这说明source配对shift高度低秩，但身份条件差异显著，支持继续验证锚点条件算子；oracle不等同于可部署性能。

## 运行矩阵

|GPU|profile|候选|launcher PID|启动状态|
|---:|---|---|---:|---|
|0|`b0_constant`|`ADVB02_NTRS_B0_CONSTANT_r1_E200`|4146381|RUNNING|
|1|`b0_shuffled`|`ADVB02_NTRS_B0_SHUFFLED_r1_E200`|4146382|RUNNING|
|2|`b0_random_feature`|`ADVB02_NTRS_B0_RANDOM_FEATURE_r1_E200`|4146383|RUNNING|
|3|`b1_metadata`|`ADVB02_NTRS_B1_METADATA_r1_E200`|4146384|RUNNING|
|4|`b1_normalized`|`ADVB02_NTRS_B1_NORMALIZED_r1_E200`|4146385|RUNNING|
|5|`b2_additive`|`ADVB02_NTRS_B2_ADDITIVE_r1_E200`|4146386|RUNNING|
|6|`b2_operator`|`ADVB02_NTRS_B2_OPERATOR_r1_E200`|4146387|RUNNING|
|7|`b3_risk`|`ADVB02_NTRS_B3_RISK_r1_E200`|4146388|RUNNING|

8行均记录`active_before=1`，启动后每张GPU新增1个训练进程，未超过每GPU2个训练任务上限。PID、CWD、cmdline、run root、GPU映射和日志增长均已核对；E2时未发现Traceback、RuntimeError、ValueError、AssertionError或OOM。E2单epoch为27.9–30.1秒，训练主体ETA约1.6–1.8小时，最终clean与三种LEO_WEAK独立测试另计。

训练完成不等于实验完成。只有E200最终checkpoint完成clean和三个LEO_WEAK逐场景全量独立测试，并保存raw→fused、Strict UDU和rescue/harm证据后，run才可进入`ARTIFACTS_COMPLETE`或`ANALYZED`。
