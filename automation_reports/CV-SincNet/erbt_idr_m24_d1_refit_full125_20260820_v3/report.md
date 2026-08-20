# ERBT-IDR M2.4 D1-REFIT完整125修复重跑报告

日期：2026-08-20

run ID：`erbt_idr_m24_d1_refit_full125_20260820_v3`

当前状态：`ANALYZED / COMPLETE / D1_COMPILE_PARITY_PASS / R2_NO_POST_REG_GAIN`

## 一、目标与修复身份

本run是v2启动路径技术失败后的不可覆盖继任run。算法与代码不变，继续验证D1源精度保持修复：

- FP32注册前源头使用F0 IF256主状态；
- 已量化F3注册后源头直接裁剪前256维code/scale，不重新量化；
- 注册前／后R1相对R0必须逐query零差异；
- D92 E0主基线固定为去RF32的`P2-A1_NO_RF32`。

实现提交：`1ca297dc1d5c44f6ec993abc58c8c1dc4208e89b`。

冻结release：`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m24_d1_refit_full125_20260820_v2`，对应archive SHA-256：`c5974d71fff4c04a3e2fed81c9a73ffddc29152f038e7f6673b84847731480a1`。该release已完成本地／远端SHA一致性和远端编译验证，不因仅修正输入根路径而重新发布。

## 二、完整矩阵

- receiver：`20-1`、`3-19`、`7-14`、`7-7`、`8-8`；
- method seed：`7282101`至`7282105`；
- 条件：`K1/new20`、`K2/new20`、`K5/new20`、`K10/new20`、`K10/new5`；
- R0/R1/R2各125行，总计375行、1125个场景单元；
- 禁止跨run复用旧prediction，完整重算375行。

## 三、协议与验证

- `protocol_schema=p2_min_v1`；
- `phase2_data_status=VALIDATED_ONCE`；
- 数据身份未改变，不重验received IQ或split；
- prediction不读取truth，scorer仅在375行闭合后连接truth；
- 本地57项相关回归、Python编译和`git diff --check`通过；
- 独立P0/P1审查及定点复审PASS；
- v2真实失败行smoke已达到before/after 0/1560差异；
- v3启动前重新执行同一smoke并使用独立输出根。

## 四、路径与命令

|字段|路径|
|---|---|
|既有feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`|
|补充feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features`|
|既有scoring root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars`|
|补充scoring root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_packages`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m24_d1_refit_full125_20260820_v3`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|设备|`cpu`，最多2个worker|

完整prediction命令使用`run_m24_d1_refit_matrix.py --run-id erbt_idr_m24_d1_refit_full125_20260820_v3`及上述两个feature root，输出到v3的`predictions`目录。

## 五、停止规则与预期artifact

仅在协议/query泄漏、错误矩阵身份、输出碰撞、错误checkout、无法启动、无prediction闭合或至少两行相同确定性prediction前异常时停止；低性能不得停止。

prediction闭合标准：

- `matrix_index.status=PREDICTIONS_COMPLETE_TRUTH_UNOPENED`；
- `row_count=375`；
- `paired_input_identity_count=125`；
- R0/R1/R2各125；
- R1 before/after disagreement均为0。

闭合后运行truth-last scorer，并生成总体、K/new、receiver、seed、scene、old/new、class、margin、中心角距、help/harm、`F_within`和`F_std`完整分析。

## 六、结果

### 6.1总体裁决

完整125输入身份、375方法行和1125场景单元已全部完成prediction与truth-last评分，375/375行均为`PASS`。R1在注册前与注册后相对R0均为0个prediction disagreement，修复目标成立。

R2没有产生任何注册后收益：相对R0，172500个注册后query中`N_help=0`、`N_harm=0`，`A_o_post`、`A_n`、H、`min-old`和`min-new`逐row全部相同。R2仅在4/125行降低了注册前旧类准确率，因此其`F_within`看似减少0.000107并不代表保护增强；统一R0注册前基线计算的`F_std`与R0完全相同。当前证据不支持晋级R2。

R1的科学角色是“去RF32后保持D92 E0决策等价”的实现验证，不是精度增益模块。后续D92 E0继续以`P2-A1_NO_RF32`为主基线。

### 6.2闭合与评分边界

- prediction index：`PREDICTIONS_COMPLETE_TRUTH_UNOPENED`；
- prediction行：375；输入身份：125；R0/R1/R2各125；
- R1注册前disagreement：0；R1注册后disagreement：0；
- scorer：375个same-row score、375个four-state score、250个paired-vs-R0结果、250个`F_within/F_std`结果；
- truth仅在完整prediction index闭合后打开，最终scorer状态为`PASS`。

run根目录下共有376个`row_execution_receipt.json`，其中375个属于正式矩阵，另1个属于预登记smoke；正式矩阵计数严格为375，不把smoke混入性能分析。

## 七、总体指标与D92 E0去RF32基线对比

下表为三场景query加权总体均值；括号内为125行总体标准差。

|arm|`A_o_pre`|`A_o_post`|`A_n`|H|F|`min-old`|`min-new`|
|---|---:|---:|---:|---:|---:|---:|---:|
|R0：D92 E0去RF32|0.740929（0.126972）|0.573497（0.160753）|0.515814（0.203110）|0.537558（0.183905）|0.167432（0.053283）|0.254104（0.176422）|0.173606（0.201896）|
|R1：compile parity|0.740929（0.126972）|0.573497（0.160753）|0.515814（0.203110）|0.537558（0.183905）|0.167432（0.053283）|0.254104（0.176422）|0.173606（0.201896）|
|R2：support refit|0.740821（0.126961）|0.573497（0.160753）|0.515814（0.203110）|0.537558（0.183905）|0.167325（0.053454）|0.254104（0.176422）|0.173606（0.201896）|

严格同row、125行等权的R2−R0差值为：`A_o_pre=-0.000133`、`A_o_post=0`、`A_n=0`、`H=0`、`F=-0.000133`、两个floor均为0。R1−R0的七项差值全部为0。

R2的4个非零注册前行集中于K10：

- `rx3-19/m7282104/K10/new20`与`K10/new5`各下降0.002778；
- `rx8-8/m7282105/K10/new20`与`K10/new5`各下降0.005556；
- K1、K2、K5全部零差异；注册后125/125行全部零差异。

## 八、K/new、receiver、seed与scene分层

### 8.1R2绝对结果

|条件|`A_o_pre`|`A_o_post`|`A_n`|H|F|
|---|---:|---:|---:|---:|---:|
|K1/new20|0.603889|0.392222|0.298633|0.332070|0.211667|
|K2/new20|0.674556|0.481556|0.377967|0.418885|0.193000|
|K5/new20|0.784222|0.633111|0.589667|0.607455|0.151111|
|K10/new20|0.853111|0.704667|0.683300|0.691927|0.148444|
|K10/new5|0.853111|0.768333|0.784533|0.773678|0.084778|

K增加带来稳定改善；K10下把新类数从20降到5，H从0.691927升至0.773678。R2相对R0的注册后指标在每个条件内都为0差值，K10的`F_within`微降只来自注册前损失。

|receiver|`A_o_pre`|`A_o_post`|`A_n`|H|F|
|---|---:|---:|---:|---:|---:|
|20-1|0.748903|0.562575|0.520542|0.538210|0.186329|
|3-19|0.610580|0.449464|0.352490|0.386967|0.161116|
|7-14|0.768275|0.615947|0.572128|0.587397|0.152329|
|7-7|0.815275|0.640841|0.567307|0.598225|0.174435|
|8-8|0.761072|0.598657|0.566603|0.576993|0.162415|

`3-19`仍是最困难receiver。R2注册前的4个负差异行只涉及`3-19`和`8-8`；其他receiver完全等价。

|scene|`A_o_pre`|`A_o_post`|`A_n`|H|F|
|---|---:|---:|---:|---:|---:|
|`leo_clear_weak`|0.780557|0.625690|0.570031|0.591143|0.154867|
|`leo_low_elev_weak`|0.724061|0.549904|0.489649|0.513026|0.174157|
|`leo_rain_weak`|0.717846|0.544896|0.487762|0.508507|0.172951|

低仰角与雨衰仍显著弱于clear。R2相对R0的注册后差值在三个scene均为0；注册前负差异只出现在clear和rain。

五个method seed的R2 H依次为0.536860、0.533063、0.543628、0.531410和0.542831。注册前负差异只出现在seed7282104和7282105，前三个seed全等价。

## 九、四状态与遗忘口径

|arm/state|old accuracy|new accuracy|H|
|---|---:|---:|---:|
|R0/R1 `DA0_REG0`|0.706206|N/A|N/A|
|R2 `DA0_REG0`|0.694416|N/A|N/A|
|R0/R1 `DA1_REG0`|0.740929|N/A|N/A|
|R2 `DA1_REG0`|0.740821|N/A|N/A|
|R0/R1 `DA0_REG1`|0.538031|0.433253|0.475373|
|R2 `DA0_REG1`|0.516738|0.401068|0.446253|
|R0/R1 `DA1_REG1`|0.573497|0.515814|0.537558|
|R2 `DA1_REG1`|0.573497|0.515814|0.537558|

R1在四个状态均与R0完全一致。R2在`DA0_REG0`和`DA0_REG1`明显更差；在主评估状态`DA1_REG1`完全回到R0，没有净收益。

`F_within`方面，R1为0.167432，R2为0.167325；`F_std`方面，两者均为0.167432。R2的`F_within`降低0.000107来自自身注册前基线下降，不应解释为注册保护改善。跨方法主比较应使用`F_std`。

## 十、help/harm、类别、margin与中心角距

R1和R2相对R0的注册后paired结果完全一致：每个候选均覆盖172500个query，`N_help=0`、`N_harm=0`、McNemar p值均为1。按old/new role、三个scene、五个receiver、五个seed、五个K/new条件和true class分层后仍全部为0差异。

因此，注册后的逐类别准确率与混淆方向也完全不变。R2 pooled类别准确率最低的五个TX为`4-10`（0.359333）、`14-11`（0.370933）、`10-10`（0.378333）、`20-12`（0.386333）和`18-10`（0.390167）；最高的五个为`19-6`（0.619833）、`1-16`（0.665200）、`11-19`（0.674667）、`20-15`（0.738667）和`8-20`（0.914800）。这些是共同基线的类别难度，不是R2造成的变化。

|诊断|R0|R1|R2|
|---|---:|---:|---:|
|top-2 margin均值|0.469750|0.469750|0.469750|
|top-2 margin中位数|0.246166|0.246166|0.246167|
|margin≤0.001比例|0.3391%|0.3391%|0.3380%|
|中心角距均值|28.409889°|28.409889°|28.409889°|
|中心角距中位数|26.996561°|26.996561°|26.996561°|

R1的极小浮点margin变化没有造成预测翻转；R2的margin分布与R0近似重合。101625个中心对的角距在三臂完全一致，说明本轮差异不来自support中心几何变化。

## 十一、资源与实现含义

|arm|state bytes|registration time|head latency|query head MAC|
|---|---:|---:|---:|---:|
|R0|14493.48B|0|未单独计时|7074.78|
|R1|15688.70B|13.12ms|3.071ms|6288.70|
|R2|15680.48B|57992.74ms|2.905ms|6288.70|

R1/R2相对R0的query head MAC下降11.11%，但当前序列化状态字节反而增加约8.2%；R0没有同口径head latency，因此不能从该表声称推理加速。R2每行平均注册时间约58.0秒，远高于R1，且没有注册后性能收益。

## 十二、评分数据修复与异常保留

prediction本身没有重跑。首次scorer启动因`PYTHONPATH`缺失在0行退出；第二次使用了补充package根而非末级`artifacts/packages`，在45行退出；纠正路径后发现补充truth sidecar仍为v2，而正式scorer要求v3。

修复没有改动原始sidecar、prediction或query truth rows。我们在run内新建`scoring_root_repaired_v3`，对20个补充scoring manifest做只读schema-only发布：22200条truth row除schema从v2改为v3外保持不变，重新计算对应truth SHA并通过正式loader验证。随后以全新不可覆盖输出根`scores_complete`重新评分完整375行。两个45行partial score根和全部失败日志原地保留，未进入结果汇总。

## 十三、最终裁决

1.R1修复通过完整125验证，可作为D92 E0去RF32实现的决策等价编译路径。
2.R2不晋级。它没有任何注册后help或精度收益，却在4行损害注册前旧类，并引入约58秒/行的refit成本。
3.后续方法比较继续以R0=`P2-A1_NO_RF32`为D92 E0主基线，并保持每个方法完整125行。
4.若继续探索refit，必须改变能够影响`DA1_REG1`的机制，而不是只改变最终被适应路径抵消的注册前状态。

## 十四、正式artifact

|artifact|路径|
|---|---|
|根目录正式报告|`E:/type10-7/automation_reports/CV-SincNet/erbt_idr_m24_d1_refit_full125_20260820_v3/report.md`|
|Git镜像正式报告|`automation_reports/CV-SincNet/erbt_idr_m24_d1_refit_full125_20260820_v3/report.md`|
|机器可读完整汇总|`automation_reports/CV-SincNet/erbt_idr_m24_d1_refit_full125_20260820_v3/results_summary.json`|
|prediction index|`evidence/matrix_index.json`|
|scored matrix index|`evidence/scored_matrix_index.json`|
|scoring修复摘要|`evidence/scoring_repair_summary.json`|
|完整分析脚本|`code/scripts/summarize_m24_d1_refit_full125.py`|
