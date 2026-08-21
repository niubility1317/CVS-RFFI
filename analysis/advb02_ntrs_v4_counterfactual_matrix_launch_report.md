# ADVB02 NTRS-V4反事实风险优化矩阵发布报告

## 发布结论

`phase1_advb02_ntrs_v4_counterfactual_matrix_20260821_r1`已于2026-08-21在N607完成并进入`ANALYZED`。本轮固定seed=`392034`、`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，训练和最终测试均只使用`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，不使用历史`mixed_orbit`。

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

## 最终完成状态

8行已全部完成E200与clean＋三种LEO_WEAK独立测试，状态从`RUNNING`更新为`ANALYZED`。8行均为`train_exit=0`、`eval_exit=0`，最终checkpoint加载均为`missing_keys=0`、`unexpected_keys=0`。每行CSV/JSONL均包含连续E1–E200共200条记录；8份完整stdout均为9,019行，未发现Traceback、RuntimeError、ValueError、AssertionError、OOM或Killed。

科学判定为：`ANALYZED_WEAK_POSITIVE_B3_RISK_B4_GATE_ONLY`。

## 最终矩阵

所有Δ均为同一独立最终评测中的`fused-raw`。冻结raw路径为clean88.2343%、LEO均值71.7681%、Strict UDU均值64.3794%。

|行|Clean fused/Δ(pp)|Clear fused/Δ(pp)|Low-elev fused/Δ(pp)|Rain fused/Δ(pp)|LEO均值/Δ(pp)|Strict均值/Δ(pp)|净救回|
|---|---:|---:|---:|---:|---:|---:|---:|
|B0-C|88.2363/+0.0020|73.9098/−0.0010|70.8039/+0.0000|70.5873/−0.0025|71.7670/−0.0011|64.3744/−0.0050|−7|
|B0-S|88.2338/−0.0005|73.9025/−0.0083|70.8132/+0.0093|70.5863/−0.0034|71.7673/−0.0008|64.3839/+0.0044|−5|
|B0-R|88.2373/+0.0029|73.8961/−0.0147|70.7926/−0.0113|70.5765/−0.0132|71.7551/−0.0131|64.3644/−0.0150|−80|
|B1-M|88.2363/+0.0020|73.9034/−0.0074|70.8059/+0.0020|70.5784/−0.0113|71.7626/−0.0056|64.3678/−0.0117|−34|
|B1-N|88.2338/−0.0005|73.9044/−0.0064|70.8157/+0.0118|70.5892/−0.0005|71.7698/+0.0016|64.3850/+0.0056|+10|
|B2-A|88.2245/−0.0098|73.8534/−0.0574|70.7574/−0.0466|70.5382/−0.0515|71.7163/−0.0518|64.3339/−0.0456|−317|
|B2-O|88.2368/+0.0025|73.9015/−0.0093|70.8059/+0.0020|70.5814/−0.0083|71.7629/−0.0052|64.3644/−0.0150|−32|
|B3-R|**88.2377/+0.0034**|**73.9201/+0.0093**|**70.8691/+0.0652**|**70.5956/+0.0059**|**71.7949/+0.0268**|**64.4378/+0.0583**|**+164**|

## 关键解释

B3-R是唯一同时保持clean不退化、三个LEO场景总体均为正增益的候选。它在612,000个LEO样本上救回2,531个、伤害2,367个，净救回164个，预测分歧率1.2982%。这证明harm/rescue反事实风险目标把B2-O的LEO均值Δ从−0.0052pp拉到+0.0268pp、Strict Δ从−0.0150pp拉到+0.0583pp；但救回在correct/wrong转换中只占51.67%，仍不够安全。

B3-R的主要收益来自low-elev和rain Strict UDU。rain的seen-day/unseen-rx切片仍从66.9933%降至66.8933%，回退0.1000pp、净伤害60个样本。因此B3-R只能判为机制弱阳性，允许进入B4条件门控，不能直接作为稳定最终方法，也不能进入Phase2 B5-RX。

B0-PCA诊断使用12,600个V_cal source样本、37,800个三场景配对。rank-8解释98.5997%shift方差，oracle准确率98.2037%，但TX主效应占71.1864%，场景公共主效应仅0.1051%。这解释了公共PCA平移B2-A为何三个场景全部下降：shift低秩不代表shift对所有TX相同。B0 runner未提供`learned_correction`，最终artifact没有`continuous_gate_oracle`字段，本报告不作该项声明。

训练期间raw参数可训练数与漂移均为0。B3-R最后20轮平均alpha为0.00421，adapter梯度范数峰值4.105；独立LEO评测平均alpha为0.01623、平均旋转0.881°。风险目标确实增强了条件修正，不是单纯评测噪声。训练终端heldout与后续独立最终评测的最大重复运行差异为0.0064pp；主结论采用同次评测的raw→fused转移，不使用跨次绝对值作因果归因。

训练、数据划分和训练期LEO增强seed为392034；独立测试统一使用固定`sat_seed base=2027`。本轮只支持单seed Phase1模拟信道筛选结论，不支持跨seed稳定性、真实在轨、Phase2、unknown或新类注册声明。

完整逐行机器可读数据见：

- `analysis/advb02_ntrs_v4_counterfactual_matrix_results.csv`
- `analysis/advb02_ntrs_v4_counterfactual_matrix_training_summary.csv`

下一步只建议B4条件门控：冻结B3-R算子，以source-only V_select选择IQ-only门控，重点消除rain seen-day/unseen-rx的有害修正，并继续使用clean＋`leo_clear_weak`＋`leo_low_elev_weak`＋`leo_rain_weak`完整测试；不使用`mixed_orbit`。
