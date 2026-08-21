# ERBT-IDR M2.5 G0锚定交叉拟合残差完整125实验报告

日期：2026-08-21

run ID：`erbt_idr_m25_g0_anchored_residual_full125_20260821_v2`

当前状态：`ANALYZED`

实现分支：`work/m24-safe-residual`

## 一、问题与方法裁决

既有`erbt_idr_m24_invariance_break_full125_20260820_v1`已证明，G1–G4虽然打破了R2的代数等价，但整体替换D92 E0稳健中心、共享尺度、full/block融合和全类LDA校准后，H从0.537558降至0.297636/0.297678/0.285538/0.278228，四个候选在125/125个同row身份上均低于G0。本轮不再构造替代分类头，而以去RF32的D92 E0/R1作为不可破坏主分数：

\[
s_c(q)=s_c^{G0}(q)+\lambda g(q)\bar r_c(q).
\]

其中，\(\bar r_c\)由合法support构造的局部证据跨类中心化并归一化至\([-1,1]\)；\(g(q)\)只在G0 top-2 margin不超过0.10时开启；\(\lambda\in\{0,0.02,0.04,0.08\}\)只由support内部留一局部证据选择。任一query只读取自身G0分数和冻结局部状态，不更新模型、阈值、强度、原型或其他query状态。

本实现没有取得D92内部每个fold的完整held-support logits。强度选择使用固定全support G0锚点分数与留一重建的局部证据，因此属于`G0_FIXED_ANCHOR_PLUS_LOCAL_JACKKNIFE`，不是对指导中“复用D92 held logits”的严格同构实现。该差异写入证据边界，不影响support-only和query只读合法性。

## 二、完整矩阵

|arm|定义|
|---|---|
|B0|`M24-D1-COMPILE-PARITY`，去RF32的D92 E0/R1等价基线|
|B1|`M25-B1-G0-BOUNDED-LOCAL-RESIDUAL`，G0＋有界单原型局部残差|
|B2|`M25-B2-G0-SHRINKAGE-RADIUS-RESIDUAL`，B1＋query依赖的收缩类半径|
|B3|`M25-B3-G0-STABLE-DUAL-PROTOTYPE-RESIDUAL`，B2＋稳定性门控、按簇大小加权双原型|

每个arm运行5个receiver×5个method seed×5组K/new条件，共125个输入身份。总矩阵为500个方法行、1500个LEO弱场景单元。K/new条件固定为K1/new20、K2/new20、K5/new20、K10/new20和K10/new5。K1/K2的B1–B3强度固定为0，必须与B0逐query一致。

## 三、协议与本地状态

- `protocol_schema=p2_min_v1`；
- 复用`phase2_data_status=VALIDATED_ONCE`且匹配`capsule_id/split_id`的既有固定received IQ，不因方法变化重验；
- predictor只能读取冻结Phase1 bundle、当前row合法support和与数据无关的冻结配置；
- query truth、old/new真实角色、query batch统计、类别配额和跨query重排均不可达；
- prediction全部闭合后，独立scorer才连接truth；
- 本地聚焦与相邻回归56项通过，五个生产脚本/模块编译通过，`git diff --check`通过；
- 独立P0/P1审查初审发现3项P1，均已定点修复；定点复审结论为`NO_P0_P1`；
- v1在352/500个partial receipt后因局部原型MAC统计引用未定义`IF_DIM`而退出，未生成matrix index，truth未打开且无性能结果；v1全部证据原位保留；
- 新增非零残差MAC分支回归测试，定点修复改为使用当前row的真实`feature_dim`；57项聚焦与相邻回归通过；
- 实验代码固定提交：`e847e0d41883d39c70f9633292e6f87aabcb7349`；远端`origin/work/m24-safe-residual`已独立回读为同一OID。

## 四、N607路径与资源

|字段|路径|
|---|---|
|既有feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`|
|补充feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features`|
|既有scoring root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars`|
|补充scoring root|`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3`|
|release root|`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|资源|CPU，`--max-workers 2`；不占用GPU，不干预既有训练|

v2单一release归档本地与N607的SHA-256均为`c523d45633392ce2fdb8a71f775265d0e169b67cd33dff76a4b93d361db59583`，远端编译通过。真实cache无truth smoke先确认B0–B3在K5均可闭合；随后使用v1已知会进入非零分支的`rx20-1/m7282101/K10/new20/B1`定点验证，三场景强度为`[0.04,0,0]`，局部原型MAC为6656，状态为`PREDICTIONS_COMPLETE_TRUTH_UNOPENED`。两次smoke均位于独立目录，不进入正式矩阵。

正式v2 prediction于2026-08-21 05:16（Asia/Hong_Kong）只启动一次，父PID为`42383`。首次健康检查确认CWD、cmdline、run root绑定正确，2个worker属于该父进程，日志持续增长，8个正式receipt已闭合，异常指纹为0。

## 五、冻结命令

prediction：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/code
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/run_m25_anchored_residual_full125.py --run-id erbt_idr_m25_g0_anchored_residual_full125_20260821_v2 --feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features --supplemental-feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/predictions --device cpu --max-workers 2
```

truth-last scorer：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/score_m25_anchored_residual_full125.py --matrix-index /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/predictions/matrix_index.json --scoring-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars --supplemental-scoring-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3 --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/scores --bootstrap-repeats 2000
```

summary：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/summarize_m25_anchored_residual_full125.py --prediction-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/predictions --score-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/scores --output /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/results_summary.json
```

## 六、闭合、停止与分析要求

prediction闭合必须同时满足：

- `matrix_index.status=PREDICTIONS_COMPLETE_TRUTH_UNOPENED`；
- `row_count=500`；
- `paired_input_identity_count=125`；
- B0、B1、B2、B3各125行；
- 500个prediction和row receipt均存在；
- K1/K2的B1–B3相对B0 disagreement均为0。

仅在协议/query泄漏、错误矩阵、输出碰撞、错误checkout、无法启动、prediction不闭合、scorer连接错误、进程归属不清或至少两行相同确定性prediction前异常时停止并保留证据。低性能不得停止。

评分后生成总体、K/new、receiver、seed、scene、四状态、old/new、class、margin、中心角距、help/harm、`F_within/F_std`、残差强度分布、门控比例、双原型接受率和资源分析，并与去RF32 D92 E0、M2.4 R2及G1–G4完整125结果同表比较。

## 七、prediction与truth-last闭合

正式prediction只启动一次并完成闭合：

- `matrix_index.status=PREDICTIONS_COMPLETE_TRUTH_UNOPENED`；
- `row_count=500`、`paired_input_identity_count=125`、`scenario_unit_count=1500`；
- B0、B1、B2、B3各125行，500/500个row receipt和prediction完整；
- 125个输入身份均恰好覆盖4个arm；
- B0注册前／后parity disagreement均为0；
- K1/K2的B1–B3共150行相对B0逐query完全一致；
- prediction阶段`query_truth_opened=false`，确定性异常指纹为0。

只有完成上述核对后，预登记scorer才连接既有与补充scoring root。完整score root包含500行same-row评分、四状态评分和候选对B0的配对结果，scorer返回`status=PASS`；随后汇总器返回`status=ANALYZED`。未使用局部矩阵，也未重跑prediction。

## 八、总体结果与D92 E0比较

下表为172500条注册后query的加权均值；括号内为125行分布的总体标准差。B0与去RF32 D92 E0/R1完整125结果逐项一致。

|arm|`A_o_pre`|`A_o_post`|`A_n`|H|F|`min-old`|`min-new`|
|---|---:|---:|---:|---:|---:|---:|---:|
|B0：D92 E0去RF32|0.740929|0.573497|0.515814|0.537558（0.183905）|0.167432|0.254104|0.173606|
|B1：有界局部残差|0.741822|0.574859|0.515952|0.538319（0.184798）|0.166963|0.257061|0.173496|
|B2：收缩半径残差|0.741608|0.575253|0.516488|0.538796（0.185404）|0.166355|0.257832|0.174116|
|B3：稳定双原型残差|0.741608|0.575735|0.516887|0.539228（0.185870）|0.165872|0.259142|0.175061|

B3相对B0的query加权差为：`A_o_pre=+0.000679`、`A_o_post=+0.002239`、`A_n=+0.001073`、`H=+0.001669`、`F=-0.001559`、`min-old=+0.005038`、`min-new=+0.001455`。按125行等权差，B3的H为`+0.001969`，其中45行提高、4行降低、76行持平；F为`-0.001867`，34行降低、6行升高、85行持平。

这不是R2那种注册后完全等价：B3在172500条注册后query中纠正352条、破坏98条，净增254条正确预测，整体accuracy从0.528638升至0.530110。B1/B2的help/harm分别为186/92和285/113，效果随机制增强呈单调改善。另一方面，B3的H绝对增益只有0.001669，即0.167个百分点，不能写成大幅性能突破。

## 九、K/new、receiver、seed与scene

### 9.1K/new条件

|条件|B0 H|B3 H|H差|B0 F|B3 F|F差|
|---|---:|---:|---:|---:|---:|---:|
|K1/new20|0.332070|0.332070|0|0.211667|0.211667|0|
|K2/new20|0.418885|0.418885|0|0.193000|0.193000|0|
|K5/new20|0.607455|0.608943|+0.001489|0.151111|0.149111|−0.002000|
|K10/new20|0.691927|0.696017|+0.004090|0.148778|0.145667|−0.003111|
|K10/new5|0.773678|0.777943|+0.004265|0.085111|0.080889|−0.004222|

K1/K2严格回退B0，既验证了安全边界，也说明当前机制无法改善最困难的低K条件。增益随support增加而扩大，主要来自K10；这与稳定双原型和support留一强度选择需要足够样本的设计一致。

### 9.2receiver与seed

|receiver|B0 H|B3 H|H差|F差|
|---|---:|---:|---:|---:|
|20-1|0.538210|0.538788|+0.000577|−0.000106|
|3-19|0.386967|0.387083|+0.000117|−0.000072|
|7-14|0.587397|0.590020|+0.002623|−0.003092|
|7-7|0.598225|0.599763|+0.001539|−0.001686|
|8-8|0.576993|0.580484|+0.003491|−0.002841|

五个receiver的H均未退化，但最困难的`3-19`只有+0.000117，改进集中于`7-14`和`8-8`。五个method seed的H差分别为+0.001589、+0.001353、+0.001862、+0.001853和+0.001690，方向全部一致；对应F差也全部为负。

### 9.3LEO场景

|scene|B0 H|B3 H|H差|B0 F|B3 F|F差|
|---|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|0.591143|0.592797|+0.001655|0.155081|0.153835|−0.001246|
|`leo_low_elev_weak`|0.513026|0.515103|+0.002077|0.174157|0.170988|−0.003168|
|`leo_rain_weak`|0.508507|0.509783|+0.001276|0.173058|0.172794|−0.000264|

三个scene均为正向H差，且低仰角场景的遗忘改善最大；雨衰场景的F改善很小。该结果只适用于当前固定LEO弱信道模拟观测，不构成真实在轨性能证据。

## 十、四状态、遗忘口径与因果解释

|arm/state|old accuracy|new accuracy|H|
|---|---:|---:|---:|
|B0/B3 `DA0_REG0`|0.706206|N/A|N/A|
|B0/B3 `DA0_REG1`|0.538031|0.433253|0.475373|
|B0 `DA1_REG0`|0.740929|N/A|N/A|
|B3 `DA1_REG0`|0.741608|N/A|N/A|
|B0 `DA1_REG1`|0.573497|0.515814|0.537558|
|B3 `DA1_REG1`|0.575735|0.516887|0.539228|

B3保留了`DA0`状态，只在G0/D92 E0主决策链上叠加support-only残差，因此主收益出现在`DA1_REG0`和`DA1_REG1`，没有复现G1–G4接管整个分类头后使DA效应翻转为负的问题。

B3的`F_within=0.165872`，以B0注册前准确率标准化后的`F_std=0.165193`；两者都低于B0的0.167432。与R2不同，这不是通过降低自身注册前基线获得的表面改善：B3注册前准确率反而提高0.000679，注册后旧类提高0.002239。

## 十一、类别、margin、中心几何与残差行为

按26个真实类别汇总，B3相对B0有25类准确率提高、0类降低、1类持平。最大提升出现在`14-7`（+0.005333）、`14-10`（+0.004533）、`14-11`和`20-19`（各+0.003333）；该类别级汇总说明整体增益没有由少数类别的巨大收益抵消多数退化，但每类增幅仍小。

B0/B1/B2/B3的top-2 margin均值分别为0.469750/0.469851/0.469925/0.469949，中位数均为0.246166。B3只改变低margin决策，没有压塌全局margin。101625个类别中心对中，B0中心角距均值/中位数为28.409889°/26.996561°，B1–B3均为34.136466°/33.233715°。中心几何变化来自局部support表示，但最终分类仍由G0主分数锚定，因此没有重演G1–G4“角距增加但分类崩塌”的现象。

375个scene级fit中，B1/B2/B3选择非零强度的数量分别为70/82/93；B3强度分布为0:282、0.02:12、0.04:11、0.08:70。B3在8625个scene×class状态中接受745个双原型，接受率8.64%。B3相对B0有785/172500条预测变化，翻转率0.455%，涉及51/125行；这说明机制确实打破等价，但扰动仍被限制在少量边界query上。

## 十二、资源分析

|arm|state bytes|注册时间|batch query head延迟/row|query head MAC|
|---|---:|---:|---:|---:|
|B0|15688.70B|13.41ms|3.43ms|6288.70|
|B1|39394.87B|834.27ms|10.45ms|8009.02|
|B2|39394.87B|932.52ms|10.55ms|8260.56|
|B3|46519.36B|14180.74ms|32.50ms|8966.50|

B3相对B0约为2.97倍state bytes、1057倍注册时间、9.49倍当前batch head延迟和1.43倍query head MAC。B3注册时间长尾明显，125行中位数约8.05s、p95约53.69s、最大约58.02s。当前实现的双原型jackknife稳定性检查是主要资源代价，因此本轮只能证明科学机制的小幅收益，不能声明部署效率或实时性。

## 十三、与既有M2.4结果的统一裁决

|方法|H|H相对D92 E0|注册后help/harm|裁决|
|---|---:|---:|---:|---|
|D92 E0去RF32/B0|0.537558|0|—|当前主基线|
|M2.4 R1 compile parity|0.537558|0|0/0|无损编译路径|
|M2.4 R2 support refit|0.537558|0|0/0|`DO_NOT_PROMOTE`|
|M2.4 G1|0.297636|−0.239923|8746/50641|`DO_NOT_PROMOTE`|
|M2.4 G2|0.297678|−0.239880|8745/50632|`DO_NOT_PROMOTE`|
|M2.4 G3|0.285538|−0.252020|8606/52267|`DO_NOT_PROMOTE`|
|M2.4 G4|0.278228|−0.259331|8486/53093|`DO_NOT_PROMOTE`|
|M2.5 B1|0.538319|+0.000761|186/92|机制有效、非最佳|
|M2.5 B2|0.538796|+0.001238|285/113|机制有效、非最佳|
|M2.5 B3|0.539228|+0.001669|352/98|性能候选晋级，部署不晋级|

M2.5首次同时满足“非等价”和“完整125下优于去RF32 D92 E0”。核心原因不是重新拟合一个替代头，而是保留G0稳健中心、共享尺度和全类校准，只在低margin query上加入support-only有界证据。B3的五receiver、五seed和三scene方向一致，足以晋级为下一轮效率化和独立确认候选；但增益只有0.167个百分点H且资源显著恶化，不能替换D92 E0作为当前部署默认。

最终裁决：`PROMOTE_FOR_SCIENTIFIC_REFINEMENT / DO_NOT_PROMOTE_AS_DEPLOYMENT_DEFAULT`。下一步只应优化B3的稳定性筛选计算和状态布局，并保留同一G0锚定与低margin有界修改原则；不得再次整体替换分类头。

## 十四、证据路径与边界

|artifact|Git路径|
|---|---|
|正式报告|`automation_reports/CV-SincNet/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/report.md`|
|机器可读完整汇总|`automation_reports/CV-SincNet/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/results_summary.json`|
|prediction index|`automation_reports/CV-SincNet/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/evidence/matrix_index.json`|
|scored matrix index|`automation_reports/CV-SincNet/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/evidence/scored_matrix_index.json`|
|prediction日志|`automation_reports/CV-SincNet/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/evidence/prediction.log`|
|残差行为汇总|`automation_reports/CV-SincNet/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/evidence/residual_behavior_summary.json`|

本结果是`p2_min_v1`下同row完整125的Stage2-C证据，不是Phase3、unknown拒识、真实卫星数据或真实在轨部署证据。B3没有严格复用D92内部held-support logits；强度选择仍是固定G0锚点＋局部jackknife近似。资源数字来自当前N607 CPU实现，不等价于目标星载硬件WCET、能耗或端到端实时性。
