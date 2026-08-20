# ERBT-IDR M2.4非等价机制完整125实验报告

日期：2026-08-20

run ID：`erbt_idr_m24_invariance_break_full125_20260820_v1`

当前状态：`ANALYZED / DO_NOT_PROMOTE_G1_G4`

## 一、目标与机制

本实验针对提交`8d712e7cb35e4c908f9975357132c10656f26a76`复盘发现的代数等价问题：旧`M24-D1-REFIT`把IF256补零后重新送入P2-A1拟合器，实际仍使用同一特征、中心、共享协方差和LDA目标，注册后172500个query相对去RF32的D92 E0得到`N_help=0`、`N_harm=0`。

本轮实现提交为`703b7d07a2ec77e40f4f9e29e4b534af98c5dc34`。新路线不调用P2-A1协方差/LDA拟合器：G1冻结50%identity/50%FFT的平衡IF256余弦原型头；G2增加support-only、类中心张成空间正交的rank-1硬投影；G3增加类别对称的不确定性惩罚；G4在K≥5时增加确定性双原型和按类归一化log-mean-exp。K1直接使用自己的冻结原型头，K2使用投影单原型，不再强制退回历史F1。

## 二、完整矩阵

|arm|身份|
|---|---|
|G0|`M24-D0-HISTORICAL-F1`，当前D92 E0去RF32主基线|
|G1|`M24-G1-FROZEN-BALANCED-PROTOTYPE`|
|G2|`M24-G2-ORTHOGONAL-NUISANCE`|
|G3|`M24-G3-CLASS-UNCERTAINTY`|
|G4|`M24-G4-LOCAL-DUAL-PROTOTYPE`|

- receiver：`20-1`、`3-19`、`7-14`、`7-7`、`8-8`；
- method seed：`7282101`至`7282105`；
- 条件：`K1/new20`、`K2/new20`、`K5/new20`、`K10/new20`、`K10/new5`；
- 每个arm完整125组，共625个方法行、1875个场景单元；
- 所有arm复用相同`capsule_id`、`split_id`、support/query物理身份和固定received IQ。

## 三、协议与本地验证

- `protocol_schema=p2_min_v1`，`phase2_data_status=VALIDATED_ONCE`；
- 数据身份未改变，不因方法和头状态变化重验数据；
- 拟合API不接收query或truth，query逐样本对所有注册类独立决策；
- prediction完整后才允许独立scorer连接truth；
- 48项聚焦回归、Python编译、矩阵/scorer静态闭合和`git diff --check`通过；
- 一次独立P0/P1审查结论为`NO_P0_P1`；
- `REJECTED_EXTRA_GATE`：非等价检查仅作为科学诊断和停止规则，不增加发布审核门。

## 四、N607输入、输出与资源

|字段|路径|
|---|---|
|既有feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`|
|补充feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features`|
|既有scoring root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars`|
|补充scoring root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_packages`|
|release root|`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m24_invariance_break_full125_20260820_v1`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_invariance_break_full125_20260820_v1`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m24_invariance_break_full125_20260820_v1`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|资源|CPU，`--max-workers 2`；不占用GPU，不干预既有Phase1训练|

N607只读预检已确认项目根、两组feature root和两组scoring root可见，新run/log根不存在。

单一release归档本地与N607的SHA-256均为`ef29aad0e47a7c635b050d7be6efad66fa74804d6ba65cc2924da7ea9cff53fd`，远端编译通过。真实cache无query smoke使用`rx3-19/m7282101/K1/new20`，G0–G4共5行全部生成`PREDICTIONS_COMPLETE_TRUTH_UNOPENED` prediction并返回`PASS`；smoke位于本run的独立`smoke`子目录，不进入625行正式矩阵。

## 五、冻结命令

prediction命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m24_invariance_break_full125_20260820_v1/code
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/run_m24_invariance_breaking_full125.py --run-id erbt_idr_m24_invariance_break_full125_20260820_v1 --feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features --supplemental-feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_invariance_break_full125_20260820_v1/predictions --device cpu --max-workers 2
```

prediction闭合后运行truth-last scorer：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/score_m24_invariance_breaking_full125.py --matrix-index /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_invariance_break_full125_20260820_v1/predictions/matrix_index.json --scoring-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars --supplemental-scoring-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_packages --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_invariance_break_full125_20260820_v1/scores --bootstrap-repeats 2000
```

## 六、停止规则与预期artifact

仅在协议/query泄漏、错误矩阵身份、输出碰撞、错误checkout、无法启动、prediction不闭合、scorer连接错误或至少两行相同确定性prediction前异常时停止并保留证据。低性能不得停止。

prediction闭合要求：`matrix_index.status=PREDICTIONS_COMPLETE_TRUTH_UNOPENED`、`row_count=625`、`paired_input_identity_count=125`且G0–G4各125行。闭合后生成625个same-row score、625个four-state score、500个paired-vs-G0结果、500个`F_within/F_std`结果，并汇总总体、K/new、receiver、seed、scene、old/new、class、margin、中心角距、help/harm、状态差异和资源指标。

## 七、实验闭合与评分修复

正式prediction只启动一次，父PID为`4020112`，未重跑。2026-08-20 22:56完成625/625个正式receipt；`matrix_index.status=PREDICTIONS_COMPLETE_TRUTH_UNOPENED`、`row_count=625`、`paired_input_identity_count=125`，G0–G4各125行。125个同输入身份均恰好覆盖5个arm，625个prediction和receipt文件全部存在，完整`prediction.log`无确定性异常指纹。独立smoke的5行未进入正式矩阵。

truth-last评分发生两次可复现的连接失败，局部结果均保留且未进入统计：

1.预登记的补充root少写`artifacts/packages`，在75个方法行后因manifest未命中退出；
2.纠正路径后，补充truth sidecar仍标记为v2，在75行后被正式v3 loader拒绝为`truth sidecar schema drift`。

最终复用M2.4 D1 v3已验证的`scoring_root_repaired_v3`。该root只把20个补充truth sidecar的schema由v2发布为v3，22200条truth row内容不变。预检确认全部50个唯一scoring身份均恰好命中一个manifest。最终`scores_complete`完成625个same-row score、625个four-state score、500个paired-vs-G0结果和500个`F_within/F_std`结果，`scored_matrix_index.status=PASS`。汇总器另修复了G0–G4矩阵误读D1专属`d1_historical_parity`字段的问题；修复提交为`d5b004a396b4ac306601129645072a9e6e317718`，19项聚焦测试和N607远端编译通过，最终汇总状态为`ANALYZED`。

## 八、总体性能

下表为三个LEO弱扰动场景按query数加权的均值；括号内为125行等权H均值和行间总体标准差。

|arm|`A_o_pre`|`A_o_post`|`A_n`|H|F|`min-old`|`min-new`|H相对G0|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|G0：D92 E0去RF32|0.740929|0.573497|0.515814|0.537558（0.564803±0.183905）|0.167432|0.254104|0.173606|0|
|G1：冻结平衡原型|0.624155|0.419335|0.240555|0.297636（0.318275±0.109963）|0.204819|0.104493|0.017397|−0.239923|
|G2：正交nuisance|0.624352|0.419521|0.240530|0.297678（0.318383±0.110196）|0.204831|0.104429|0.017333|−0.239880|
|G3：类别不确定性|0.619536|0.417281|0.227317|0.285538（0.306286±0.106692）|0.202255|0.091867|0.013270|−0.252020|
|G4：局部双原型|0.620356|0.410793|0.222049|0.278228（0.298423±0.099551）|0.209562|0.077693|0.012336|−0.259331|

G1–G4均显著低于G0。表现最好的候选是G2，但其H仍下降23.988个百分点，相对下降44.62%；G4下降25.933个百分点，相对下降48.24%。全部四个候选在125/125个同row身份上的H都低于G0，没有单行持平或提升。因此本轮不是“平均无提升”，而是跨完整矩阵的一致性退化。

## 九、与D92 E0及既有M2.4 D1/R2对比

本轮G0与`erbt_idr_m24_d1_refit_full125_20260820_v3`中的R0在125个身份和七项指标上逐row完全一致，最大绝对差为0，证明本轮D92 E0去RF32基线接线正确。

|方法|`A_o_pre`|`A_o_post`|`A_n`|H|F|注册后help/harm|裁决|
|---|---:|---:|---:|---:|---:|---:|---|
|D92 E0去RF32/R0|0.740929|0.573497|0.515814|0.537558|0.167432|基线|保留主基线|
|M2.4 R1 compile parity|0.740929|0.573497|0.515814|0.537558|0.167432|0/0|通过，决策等价|
|M2.4 R2 support refit|0.740821|0.573497|0.515814|0.537558|0.167325|0/0|不晋级，无注册后收益|
|M2.4 G1|0.624155|0.419335|0.240555|0.297636|0.204819|8746/50641|不晋级|
|M2.4 G2|0.624352|0.419521|0.240530|0.297678|0.204831|8745/50632|不晋级|
|M2.4 G3|0.619536|0.417281|0.227317|0.285538|0.202255|8606/52267|不晋级|
|M2.4 G4|0.620356|0.410793|0.222049|0.278228|0.209562|8486/53093|不晋级|

R2的问题是代数等价、没有改变注册后决策；G1–G4确实打破了等价，但改变方向错误。由此可见，“非等价”只是必要条件，不能替代对G0稳健中心、共享尺度和全局类间校准的保护。

## 十、K/new分层

|条件|G0 H|G1 H|G2 H|G3 H|G4 H|最佳候选相对G0|
|---|---:|---:|---:|---:|---:|---:|
|K1/new20|0.332070|0.227895|0.227895|0.227895|0.227895|−0.104175|
|K2/new20|0.418885|0.252259|0.252093|0.233005|0.233005|−0.166626|
|K5/new20|0.607455|0.299791|0.300005|0.284757|0.274159|−0.307450|
|K10/new20|0.691927|0.334919|0.334805|0.320420|0.303804|−0.357008|
|K10/new5|0.773678|0.476514|0.477116|0.465355|0.453252|−0.296562|

退化幅度随K增大而扩大：K1仅下降10.42个百分点，K10/new20下降35.70个百分点。候选头没有把更多support转化为更好的类间决策，反而在G0最能利用support的区域损失最大。这排除了“只是低K估计不稳”的解释。

## 十一、receiver、seed与scene

### 11.1Receiver H

|receiver|G0|G1|G2|G3|G4|
|---|---:|---:|---:|---:|---:|
|20-1|0.538210|0.277104|0.277131|0.265700|0.266051|
|3-19|0.386967|0.192868|0.192839|0.189139|0.195331|
|7-14|0.587397|0.360448|0.360783|0.342143|0.319484|
|7-7|0.598225|0.338244|0.337999|0.324382|0.314775|
|8-8|0.576993|0.319515|0.319641|0.306327|0.295498|

`3-19`仍是绝对性能最差receiver，但所有receiver均大幅退化；G1相对G0的H下降范围为19.41–26.11个百分点，问题不是单一receiver异常。

### 11.2Seed H

|seed|G0|G1|G2|G3|G4|
|---|---:|---:|---:|---:|---:|
|7282101|0.536860|0.298874|0.298935|0.285911|0.283015|
|7282102|0.533063|0.295502|0.295668|0.282783|0.272867|
|7282103|0.543628|0.305471|0.305586|0.293805|0.283557|
|7282104|0.531410|0.296901|0.296932|0.285705|0.278844|
|7282105|0.542831|0.291430|0.291270|0.279488|0.272856|

五个seed方向完全一致，排除单seed偶然性。

### 11.3Scene H

|scene|G0|G1|G2|G3|G4|
|---|---:|---:|---:|---:|---:|
|`leo_clear_weak`|0.591143|0.319703|0.319934|0.303852|0.299672|
|`leo_low_elev_weak`|0.513026|0.281535|0.281449|0.271268|0.259744|
|`leo_rain_weak`|0.508507|0.291669|0.291652|0.281495|0.275268|

G1在clear场景下降27.14个百分点，反而比low-elev和rain的下降更大；因此不能把失败归因于恶劣LEO场景本身。

## 十二、四状态因果分析与遗忘

|arm/state|`DA0_REG0`旧类|`DA1_REG0`旧类|`DA0_REG1`旧类/新类/H|`DA1_REG1`旧类/新类/H|
|---|---:|---:|---:|---:|
|G0|0.706206|0.740929|0.538031/0.433253/0.475373|0.573497/0.515814/0.537558|
|G1|0.694416|0.624155|0.516738/0.401068/0.446253|0.419335/0.240555/0.297636|
|G2|0.694416|0.624352|0.516738/0.401068/0.446253|0.419521/0.240530/0.297678|
|G3|0.694416|0.619536|0.516738/0.401068/0.446253|0.417281/0.227317/0.285538|
|G4|0.694416|0.620356|0.516738/0.401068/0.446253|0.410793/0.222049/0.278228|

G0的DA在注册前使旧类提高3.47个百分点，在注册后使旧类提高3.55个百分点；G1–G4的DA却在注册前降低7.01–7.49个百分点，在注册后降低9.72–10.59个百分点。也就是说，失败主要发生在候选DA头接管`DA1`状态后，而不是共同的`DA0`初始化。

|候选|`F_within`|`F_std`|解释|
|---|---:|---:|---|
|G1|0.204819|0.321593|自身注册前已低于G0，标准化遗忘更差|
|G2|0.204831|0.321408|与G1几乎相同|
|G3|0.202255|0.323647|较小`F_within`来自较低起点，不是保护|
|G4|0.209562|0.330135|标准化遗忘最差|

跨方法必须使用`F_std`。若只看G3的`F_within`较小，会错误地把注册前性能损失解释成遗忘改善。

## 十三、机制增量、help/harm与类别

G2相对G1的125行等权H均值只增加0.000107，34行提升、42行下降、49行不变，说明rank-1正交nuisance投影基本没有形成稳定作用。G3相对G2的H均值下降0.012096，仅1行提升、99行下降、25行不变；类别不确定性惩罚稳定有害。G4相对G3再下降0.007863，28行提升、47行下降、50行不变；局部双原型未弥补尺度和全局校准问题。

|候选|query数|`N_help`|`N_harm`|总体accuracy delta|125行McNemar p<0.05|
|---|---:|---:|---:|---:|---:|
|G1|172500|8746|50641|−0.242870|125|
|G2|172500|8745|50632|−0.242823|125|
|G3|172500|8606|52267|−0.253107|125|
|G4|172500|8486|53093|−0.258591|125|

按角色看，G1对127500个新类query的accuracy下降27.21个百分点，对45000个旧类query下降16.02个百分点；G4分别下降29.03和16.86个百分点。损害同时覆盖旧类和新类，不是单纯的旧新权衡。

631个class切片中，G1/G2各有595个负差、32个正差、4个不变；G3为594/33/4；G4为596/33/2。最差单class下降达到0.883333。少量局部class提升不能抵消大范围退化，机器汇总保留全部class明细。

## 十四、margin与中心角距

|arm|top-2 margin均值|中位数|margin≤0.001|margin≤0.01|margin≤0.05|中心角距均值|中心角距中位数|
|---|---:|---:|---:|---:|---:|---:|---:|
|G0|0.469750|0.246166|0.3391%|3.0904%|14.2713%|28.4099°|26.9966°|
|G1|0.043680|0.021913|3.8812%|30.4701%|71.9901%|47.1916°|49.4619°|
|G2|0.043941|0.022056|3.8620%|30.3710%|71.8046%|47.2596°|49.5463°|
|G3|0.046823|0.022818|3.6614%|29.4719%|70.3699%|47.2596°|49.5463°|
|G4|0.040824|0.021440|3.4186%|29.2557%|74.2754%|47.2596°|49.5463°|

候选的top-2 margin中位数约为G0的9%，超过70%的query落在0.05以内，决策显著变得拥挤。中心角距增加约18.8°说明support中心几何发生了实质变化，但结合margin和准确率可知这种变化是失配位移，不是有效分离。

## 十五、资源分析

|arm|state bytes|相对G0|registration time|query head MAC|候选head批量延迟/row|
|---|---:|---:|---:|---:|---:|
|G0|14493.48|基线|0|7074.78|未同口径测量|
|G1|26473.57|+82.65%|16.03ms|6288.70|21.75ms|
|G2|26473.57|+82.65%|157.26ms|6288.70|18.63ms|
|G3|26473.57|+82.65%|156.80ms|6288.70|17.03ms|
|G4|39589.67|+173.15%|178.78ms|6288.70|24.70ms|

G1–G4的query head MAC相对G0下降11.11%，但state bytes明显增加且性能大幅下降。G0缺少同口径head延迟，`deployment_state_bytes`也未验证，因此不能声称候选推理加速或部署收益。

## 十六、最终裁决与后续方向

1.实验完整跑完，625行、1875场景单元和500组配对比较均形成合法同row证据。
2.G1–G4全部`DO_NOT_PROMOTE`。G2是数值上最好的候选，但相对G0仍退化23.99个百分点H，不具备继续扩展价值。
3.D92 E0去RF32继续作为主基线。既有R1保留为等价编译路径；R2保留为“注册后无变化”的否定证据。
4.根因是候选用低margin的平衡余弦原型头整体替换了G0的稳健中心、共享尺度和全局类间校准；正交投影、类别惩罚和局部双原型都没有恢复这一校准。
5.下一候选应保留G0主决策，只允许support-only、可界定幅度的残差修改，并以`DA1_REG1`同row收益为唯一晋级依据；优先在K5/K10验证能否真正利用额外support，再扩展完整125。不得把几何发生变化本身视为成功。

## 十七、证据路径

|artifact|路径|
|---|---|
|正式报告|`automation_reports/CV-SincNet/erbt_idr_m24_invariance_break_full125_20260820_v1/report.md`|
|机器可读完整汇总|`automation_reports/CV-SincNet/erbt_idr_m24_invariance_break_full125_20260820_v1/results_summary.json`|
|prediction index|`automation_reports/CV-SincNet/erbt_idr_m24_invariance_break_full125_20260820_v1/evidence/matrix_index.json`|
|scored matrix index|`automation_reports/CV-SincNet/erbt_idr_m24_invariance_break_full125_20260820_v1/evidence/scored_matrix_index.json`|
|prediction完整日志|`automation_reports/CV-SincNet/erbt_idr_m24_invariance_break_full125_20260820_v1/evidence/prediction.log`|
|评分与汇总日志|同目录`scoring*.log`和`summarize*.log`|

本报告支持的是`p2_min_v1`下完整125、同row、truth-last的Stage2-C结论，不支持Phase3、真实在轨或部署性能声明。
