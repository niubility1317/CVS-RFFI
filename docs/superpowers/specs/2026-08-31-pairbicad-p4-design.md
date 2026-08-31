# ADV3B02-PairBiCAD-P4设计冻结

## 目标

在现有ADV3B02-BiCAD-XR双骨干、`z_id/z_dom`、RCN和DANN训练入口上落地报告定义的第一阶段核心方法`ADV3B02-PairBiCAD-P4`，并发布可归因的`P0→P4`source-only完整矩阵。

## 协议边界

- 仅Phase1 source数据；禁止访问Phase2、target receiver、support、query或truth。
- 数据为`Dataset_WigSig/ManySig.pkl`，训练day1、day2、day3。
- source角色固定`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，物理样本ID两两不交。
- 星地增强保持`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`。
- 旧D0–F3继续保持`concat_sat_ce_only/E80/lambda_sat_cls=0.68/lambda_sat_cons=0`，不得静默改变。
- 新P0–P4显式预登记`satellite_supervision_mode=ce_only_plus_pair_selfsup`：卫星视图的TX标签监督仍只允许CE，但从首个update生成同物理样本pair；P3/P4允许不读取U_s标签的pair一致性、VICReg和delta信道自监督。该模式不产生硬伪标签TX CE、不更新TX原型，也不得声称与旧`concat_sat_ce_only`训练语义完全相同。
- 每个物理batch为48条，clean+LEO拼接后一次网络前向为96条；不得执行第二次骨干前向。

## 候选递进

|候选|唯一主要增量|必须可观测的运行证据|
|---|---|---|
|P0|ADV3B02双骨干+每update严格clean/LEO成对拼接；有标签双视图TX CE|physical/network batch计数、单前向计数、LEO场景|
|P1|P0+receiver/day/channel因素化域表示+`z_int`+shared-stem gradient firewall|各因素域CE、firewall应用数、`z_int`不进入TX头|
|P2|P1+有标签class-conditional DANN+`z_dom`主因素TX adversary|`CAdv-r/d/c`、TXAdv调用/有效样本、GRL剂量|
|P3|P2+L/U clean-LEO pair hinge+U预测JS+projector VICReg|pair有效数、hinge/JS/VICReg三项有限值、无U标签访问|
|P4|P3+pair-delta identity channel adversary+domain channel predictor+delta-norm hinge|identity/domain delta信道损失、delta范数、信道标签来源|

P4使用现有动态梯度比控制器记录`rho_adv`，目标区间为0.15–0.25；`z_dom` TX adversary目标区间为0.05–0.10。无法在单步安全取得精确比率时，控制器只缩放对抗分量并记录实际比率，不允许改变TX标签或数据角色。

## 兼容性裁决

报告要求与旧BiCAD默认存在三处候选级差异：P0从首个update启用LEO pair并采用0.5→1.0的卫星TX CE权重；P3加入pair/VICReg；P4加入delta对抗和channel等变。当前任务对报告的实现授权允许这些差异，但只对新P0–P4生效。三种`LEO_WEAK`、source-only和L/U信息权限不变；旧候选、旧checkpoint和旧runtime协议保持严格兼容。

## 因素化表示

`z_dom`经独立投影得到`z_r/z_d/z_c/z_int`。receiver/day/channel监督分别作用于前三者；TX adversary只作用于`[z_r,z_d,z_c]`；`z_int`维度固定为24，只作低维交互容器，不进入公共TX分类器，也不进入TX adversary。

## Pair目标

- projector维度为128。
- identity hinge容忍`epsilon_p=0.05`。
- VICReg方差目标`gamma=1.0`，在clean/satellite projector输出上计算。
- P3总pair权重起点：identity hinge 0.08、U预测JS 0.05、VICReg组0.03。
- P4起点：identity delta adversary 0.08、domain delta prediction 0.15、delta-norm hinge 0.05；delta半径0.25。
- L样本可用于TX CE和全部pair目标；U样本只用于元数据域监督和无标签pair目标。

## 正式矩阵

- Run ID：`phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r1`。
- 候选：P0、P1、P2、P3、P4。
- folds：1、8，对应现有两个source-LORO划分。
- seeds：392001、392002、392003。
- 共30行，每行4000 optimizer updates。
- GPU0–7，每GPU最多2个本run训练进程；初始最多16行并发，余下行由同一dispatcher续发。
- 每行必须保存final checkpoint、严格重建结果、clean和三种`LEO_WEAK`source V_select指标、逐类下界、训练资源与PairBiCAD调用审计。
- 只能按source V_select比较和冻结；低性能是科学结果，不是技术失败。

## 明确延期

`P5`无标签soft conditional DANN、`P6`XDC、`P7`margin-tail、`P8`hard-LEO mining、`P9`SWAD、FastTrust、Fishr、MixUp、MixStyle、HCF transport和CSD均不进入本轮代码路径或正式矩阵。

## 验收

1. P0–P4候选只产生递进差异且旧D0–F3行为保持兼容。
2. 聚焦测试先失败后通过；真实checkpoint no-query smoke通过。
3. 独立P0/P1审查不发现会直接导致错误数据、错误候选、输出覆盖、无法启动或无artifact闭合的问题。
4. Git代码/config提交已push且远端OID等于本地HEAD。
5. N607 release归档仅做一次本地/远端SHA比较并远端编译。
6. 启动后精确核对dispatcher、worker、CWD、cmdline、GPU与日志增长。
