# ADVB02 NTRS-V2恢复矩阵完整实验报告

## 结论

截至2026-08-20 17:02 CST，本轮预登记恢复矩阵已经全部完成训练和独立测试。D0、M1-DIAG、D1、D2、D3和V2-1共6行均有E200结果；D0、D1、D2、D3和V2-1完成clean及`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`三场景全量测试，M1-DIAG对历史E200 checkpoint完成同样的只读重评。最终状态为：

```text
ANALYZED_NEGATIVE_NO_GO_V2_1
```

数据支持三个结论。

1. 旧NTRS-V1的主要回退来自core低学习率。严格旁路D1的LEO均值为71.768%，保留旧低学习率的D2仅为52.825%，同行下降18.943pp；完整V1恢复公平core LR后的D3达到70.312%，比历史M1提高18.693pp。
2. 严格恒等V2外壳没有通过预登记的单seed实验等价门。D1相对D0的clean和LEO均值分别提高0.698pp和1.311pp，方向为正，但都超过`abs(delta)<=0.5pp`的等价带。代码级旁路仍满足logit与embedding最大绝对误差小于`1e-6`。因此，D0与D1之间约1pp量级的差异不能解释为NTRS方法收益。
3. 最小共享头残差V2-1没有晋级。其LEO均值为69.424%，相对D0下降1.033pp，相对D1下降2.344pp；clean下降1.481pp；三个LEO场景均下降超过0.5pp；累计rescued为2596、harmed为3169，净救回为-573。V2-1的四项闭集门槛全部失败。

V2-2至V2-6和多seed实验没有启动。最新指导把这些模块明确设为V2-1通过后才可进入的后续阶段；在V2-1失败后继续堆叠teacher basis、slow support、物理IQ校正或安全损失会偏离预登记路线。

## 实验协议

|项目|冻结值|
|---|---|
|阶段|Phase1 source-only weak-label/semi-supervised DG|
|数据角色|`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`|
|训练seed|`392034`|
|训练轮数|200，最终checkpoint固定E200|
|训练信道|`LEO_WEAK`三场景按Core90日程轮换|
|测试信道|clean对照＋三种`LEO_WEAK`逐场景全量测试|
|卫星测试seed|`2027`|
|数据集|ManySig Phase1 source数据|
|禁止路径|未使用`mixed_orbit`，未使用target receiver、target support/query/truth|
|测试规模|每个LEO场景204000个样本，其中strict UDU为60000个样本|

本实验中的LEO信道是物理启发的代理压力测试，不是真实在轨数据或真实卫星链路验证。

## 矩阵定义与完成状态

|行|方法差异|训练|clean＋三LEO测试|状态|
|---|---|---:|---:|---|
|D0|Core90同协议控制组|历史E200复用|完成|`ARTIFACTS_COMPLETE`|
|M1-DIAG|历史NTRS-V1，只读raw/robust/fused诊断|历史E200复用|完成|`ARTIFACTS_COMPLETE`|
|D1|V2严格identity bypass＋公平core LR|200/200|完成|`ARTIFACTS_COMPLETE`|
|D2|D1外壳＋旧V1低core LR|200/200|完成|`ARTIFACTS_COMPLETE`|
|D3|完整V1结构＋公平core LR|200/200|完成|`ARTIFACTS_COMPLETE`|
|V2-1|单身份前向＋共享CosFace头＋无LayerNorm＋最小有界残差|200/200|完成|`ARTIFACTS_COMPLETE_NO_GO`|

所有最终checkpoint均为E200，独立评测均为`missing_keys=0`、`unexpected_keys=0`。

## 最终准确率

单位均为百分比；`LEO均值`是三场景overall的等权平均，`strict均值`是三场景未见日＋未见接收机准确率的等权平均。

|行|clean|clear|low-elev|rain|LEO均值|strict clear|strict low|strict rain|strict均值|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|D0|87.536|72.490|69.442|69.439|70.457|65.795|63.032|62.945|63.924|
|M1|84.313|52.865|50.796|51.194|51.618|45.497|44.197|44.515|44.736|
|D1|88.234|73.911|70.804|70.590|**71.768**|66.332|63.580|63.227|**64.379**|
|D2|84.476|54.426|51.834|52.216|52.825|46.403|44.752|44.965|45.373|
|D3|86.632|72.485|69.267|69.183|70.312|64.838|61.757|61.520|62.705|
|V2-1|86.055|71.401|68.481|68.390|69.424|63.422|60.935|60.995|61.784|

D1取得本矩阵最高LEO均值，但它是严格旁路诊断，不是稳健层带来的方法增益。其作用是给公平学习率训练路径提供参照。

## 同行差分与因果诊断

|比较|clean差值|LEO均值差值|strict均值差值|解释|
|---|---:|---:|---:|---|
|M1−D0|-3.224|-18.839|-19.188|复现历史NTRS-V1大幅回退|
|D2−D1|-3.758|-18.943|-19.006|单独恢复旧低core LR即重现约19pp回退，学习率是主因|
|D3−M1|+2.320|+18.693|+17.969|公平core LR几乎完全挽回V1的LEO损失|
|D3−D1|-1.602|-1.456|-1.674|公平LR下，V1结构与损失仍有约1.5pp负贡献|
|V2-1−D1|-2.179|-2.344|-2.596|最小共享头残差训练损害raw身份路径，且融合没有救回|
|V2-1−D0|-1.481|-1.033|-2.140|V2-1未达到基线|

本次正交矩阵给出直接证据：同一严格旁路结构中，仅把core LR从baseline日程改为旧V1日程，就造成18.943pp下降。D3在保留完整V1结构与损失的条件下恢复公平LR后，LEO均值与D0只差0.145pp。core LR不公平是历史回退的决定性因素；LayerNorm、双头和附加损失仍有负作用，但不是18.8pp回退的主要来源。

## 收敛与断点

下表给出训练日志中的source-validation三场景LEO均值。全部数据来自200行`metrics_epoch.jsonl`，表中仅展示用于定位断点的关键epoch。

|行|E16|E40|E68|E90|E130|E200|
|---|---:|---:|---:|---:|---:|---:|
|D0|47.005|57.542|75.421|82.603|79.122|83.270|
|M1|44.402|55.085|60.479|61.616|65.209|70.045|
|D1|45.656|65.050|80.971|82.878|76.894|87.770|
|D2|44.085|51.770|57.918|63.437|68.653|72.323|
|D3|44.452|68.680|81.405|82.865|85.466|88.143|
|V2-1|48.484|59.632|69.198|82.228|83.519|87.989|

D1与D2在E16前使用相同core LR。E17后D2的core LR降到`4e-5`，D1维持`2e-4`；到E40，两者已相差13.280pp，到E68扩大为23.053pp。E69后D2进一步降到`2e-5`，E90仍落后19.442pp。该断点与学习率日程严格对齐。

V2-1在E1–90保持残差为0，E91开始打开残差。其source-validation LEO均值从E90的82.228%升至E200的87.989%，但最终target-like全量LEO测试仅为69.424%。source-validation上升没有转化为未见日/未见接收机的稳健收益。

## raw、robust与fused诊断

以下均值按三个LEO场景等权计算。`safe gate活跃率`表示`safe_gate>0`，并不等于修正有效。

|行|raw均值|robust均值|fused均值|raw/robust分歧率|safe gate活跃率|rescued|harmed|净救回率|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|M1|51.618|46.982|51.618|33.743|2.202|50287|78661|-4.636%|
|D1|71.768|71.768|71.768|0.000|0.000|0|0|0.000%|
|D2|52.825|52.825|52.825|0.000|0.000|0|0|0.000%|
|D3|70.312|70.300|70.312|3.254|2.210|6558|6628|-0.011%|
|V2-1|69.518|69.425|69.424|1.822|82.141|2596|3169|-0.094%|

M1的robust准确率比raw低4.636pp，raw/robust分歧率达到33.743%。其safe gate均值接近数值零、p95为0，因此fused完全退回raw；2.202%活跃率来自极小正浮点数，不代表有效修正。历史V1的直接问题不是融合把raw结果进一步拉低，而是低core LR已经把raw骨干训练到51.618%的LEO均值。

D3把core LR恢复公平后，raw均值升至70.312%。robust与raw只差0.011pp，fused仍等于raw；V1稳健分支没有产生可用净救回。

V2-1不再使用独立头和LayerNorm，但raw均值只有69.518%，比D1低2.250pp；fused又比raw低0.094pp。损失耦合先降低了身份骨干质量，残差修正随后产生573个净伤害样本。`alpha`三场景均值约0.200，safe gate活跃率约82%，说明残差使用强度接近上限，但方向不具备正净收益。

## 预登记门槛判定

### D1严格旁路等价门

|门槛|要求|实测|结果|
|---|---:|---:|---|
|代码logit旁路|最大绝对误差`<1e-6`|聚焦测试通过|PASS|
|代码embedding旁路|最大绝对误差`<1e-6`|聚焦测试通过|PASS|
|clean实验等价|`abs(delta)<=0.5pp`|`+0.698pp`|FAIL|
|LEO均值实验等价|`abs(delta)<=0.5pp`|`+1.311pp`|FAIL|

代码恒等契约成立，但单seed端到端训练结果没有落入预登记等价带。D1的偏差方向为正，不能解释为稳健层收益；它更可能反映GPU训练非确定性或D0与D1训练路径中尚未控制的微小差异。若需精确量化1pp以内效应，必须先做D0/D1重复性研究。

### V2-1闭集晋级门

|门槛|要求|实测|结果|
|---|---:|---:|---|
|LEO均值增益|相对D0`>=+1.0pp`|`-1.033pp`|FAIL|
|clean保持|相对D0`>=-0.5pp`|`-1.481pp`|FAIL|
|三场景非退化|每场景不低于D0超过0.5pp|clear`-1.089`、low`-0.961`、rain`-1.050pp`|FAIL|
|净救回|`rescued>harmed`|2596<3169，净-573|FAIL|

V2-1四项门槛全部失败，不能进入V2-2至V2-6，也不能进入多seed正式重复。

## 训练和artifact完整性

|行|结构化轮数|缺失/重复轮|optimizer step执行均值|最差单轮执行率|完整日志行数|致命错误标记|
|---|---:|---:|---:|---:|---:|---:|
|D0|200|0/0|99.878%|95.556%|9019|0|
|M1|200|0/0|99.911%|93.333%|9019|0|
|D1|200|0/0|99.911%|97.778%|9019|0|
|D2|200|0/0|99.922%|95.556%|9019|0|
|D3|200|0/0|99.800%|95.556%|9019|0|
|V2-1|200|0/0|99.867%|97.778%|9019|0|

致命错误扫描覆盖完整stdout中的Traceback、RuntimeError、CUDA OOM、Killed、KeyError和AssertionError。六行均为0。少量梯度step跳过已记录在结构化日志中，均值不超过0.20%，未造成训练中断或epoch缺失。

## 独立测试故障与修复

D1、D2和V2-1首次自动测试在训练结束后失败。checkpoint中的`ntrs_context.fast_encoder`宽度为32，而评测器错误地按V1默认结构重建为宽度64，触发`state_dict`尺寸不匹配。根因是`build_model_from_checkpoint_args`恢复了NTRS维度，却遗漏`ntrs_variant`与`ntrs_identity_bypass`。

修复提交为`b27e3b164c62cec231e4725283cd3393a8c0f3ef`。新增回归测试先在旧代码上观察到`v1 != v2_min`的预期失败，再补充两个checkpoint结构参数；19项聚焦测试和Python编译通过。新评测release归档SHA256为`927e09dab1ca56c9c78ac406f840a4960b43811267ce7e7b00ae726dd16238ae`，本地与N607一致。三组补测均使用原E200 checkpoint，不重训、不覆盖原失败日志，结果保存在`independent_final_eval_fix_b27e3b16`。

## 数据验证与限制

关键数值由`final_eval.json`中的分子/分母和`metrics_epoch.jsonl`的全部200行独立复算。D1 clear的150778/204000复算为73.9108%，与评测日志一致；六行三场景分母均为204000，strict UDU分母均为60000。报告状态为`Ready to share with caveats`，限制如下。

- 只有一个训练seed。18–19pp的学习率差异远大于D0/D1的约1pp旁路波动，主因判断可信；1–2pp的结构差异仍需重复实验才能估计方差。
- D1没有通过预登记端到端等价带，因此不能把D1−D0的正差解释为方法收益。
- 评测JSON保存了gate均值、p95和`safe_gate>0`活跃率，但没有序列化预期的`P(g_safe>0.01/0.05/0.10)`三个字段。M1/D3的gate均值接近0且p95为0，足以判定实际回退raw；三个精确阈值比例仍不可恢复。
- 最终评测中的receiver floor字段为NaN；overall和strict UDU基于明确分子/分母，未受影响，但本报告不声明receiver-floor结果。
- V2-1没有启用开放集安全损失，因此unknown FAR、OSCR、AUROC和真实unknown转移均不适用。
- WiSig＋LEO_WEAK仅支持代理场景结论，不支持真实在轨性能声明。

## 最终决策与下一步

本轮拒绝V2-1，保留D1作为公平学习率、严格旁路的诊断参照，不把D1包装成新方法。V1与V2都不应继续通过调`alpha_max`、rank或附加小损失来修补。

下一轮应执行指导报告中的路线A：从成熟D0/D1 checkpoint出发，冻结身份骨干训练共享头残差adapter，再以极小core LR联合微调。新候选应先证明冻结骨干时raw准确率位级保持，并在三个LEO_WEAK场景上均满足`rescued>harmed`。只有相对同行冻结基线达到LEO均值`+1pp`且clean下降不超过0.5pp，才重新讨论确定性teacher tangent basis。

## 证据位置

- 完整结构化复算：`E:/type10-7/automation_reports/CV-SincNet/phase1_advb02_ntrs_v2_recovery_matrix_20260820/matrix_analysis.json`
- 最终矩阵：`E:/type10-7/automation_reports/CV-SincNet/phase1_advb02_ntrs_v2_recovery_matrix_20260820/matrix_final_metrics.csv`
- raw/robust/fused及转移：`E:/type10-7/automation_reports/CV-SincNet/phase1_advb02_ntrs_v2_recovery_matrix_20260820/matrix_ntrs_telemetry.csv`
- 关键epoch曲线：`E:/type10-7/automation_reports/CV-SincNet/phase1_advb02_ntrs_v2_recovery_matrix_20260820/matrix_training_curve_points.csv`
- 完整日志完整性：`E:/type10-7/automation_reports/CV-SincNet/phase1_advb02_ntrs_v2_recovery_matrix_20260820/matrix_log_integrity.csv`
- 本地原始小型证据集：`E:/type10-7/automation_reports/CV-SincNet/phase1_advb02_ntrs_v2_recovery_matrix_20260820/retrieved_b27e3b16`
- N607原始训练输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_v2_recovery_20260820_*`

设计追踪最终计数：`verified=10`、`deferred=1`、`rejected=1`、`blocked=0`。最高风险剩余项是D0/D1单seed实验等价性未通过；当前实现属于严格设计落地，不是对V2-2至V2-6的近似实现。
