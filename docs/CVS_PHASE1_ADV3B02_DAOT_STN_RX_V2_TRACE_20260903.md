# ADV3B02-DAOT-STN-RX-V2报告实现追踪

## 1.范围

- 方法ID：`ADV3B02-DAOT-STN-RX-V2`。
- 当前阶段：详细计划已冻结，代码尚未实施，实验尚未启动。
- 科学边界：Phase1 source-only弱标签/半监督域泛化；target/query只用于最终只读测试。
- 默认教师：两个新鲜视图加Temporal Orbit Memory；三新鲜教师只属于A2/A3上界实验及其明确消融。
- 用户排除：不使用上一轮A4/A7等checkpoint执行任何非LEO_WEAK场景测试。

## 2.数据与评测协议

- 数据集：`Dataset_WigSig/ManySig.pkl`，equalized=`true`。
- split：`tx_rx_day_1_7_2`，seed=`392005`。
- source receiver：`[1,3,4,6,8]`；source day：`[1,2,3]`。
- source pool：90000；`L=6300`、`U=56700`、单一只读`V=27000`。
- target receiver：`[0,2,5,7,9,10,11]`；target day：`[0,1,2,3]`。
- 最终场景白名单：`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- 非LEO_WEAK、cross-family、真实轨道外推：`REJECTED_BY_USER_SCOPE`。

## 3.追踪状态

|ID|机制|计划代码入口|状态|边界|
|---|---|---|---|---|
|RXV2-01|锚定非对称轨道教师|`code/cvsrffi/orbit_teacher.py`、`daot_training.py`|pending|只用source/固定received IQ|
|RXV2-02|coverage-aware可恢复度权重|`code/cvsrffi/deployment_orbit.py`|pending|缺失元数据显式mask|
|RXV2-03|方向注册表、逐方向delta与budget|`deployment_orbit.py`、`selective_tangent.py`、`tangent_calibration.py`|pending|TX方向不做nulling|
|RXV2-04|随机单方向TX干预与Jacobian路由|`selective_tangent.py`、`daot_training.py`|pending|禁止固定组合shortcut|
|RXV2-05|Source-only Receiver Style Bank|`receiver_style_bank.py`|pending|拒绝target/test角色|
|RXV2-06|TX条件RX原型与CVaR|`receiver_conditioned_alignment.py`|pending|禁止跨TX全局对齐|
|RXV2-07|连续U可信度与三态选择|`daot_unlabeled_trust.py`|pending|函数不接收真实label|
|RXV2-08|分支专属不变性预算|`branch_invariance.py`|pending|不对所有分支统一null|
|RXV2-09|S0～S6独立调度与冲突保护|`orbit_teacher.py`、`daot_gradient_control.py`|pending|仅持续冲突时投影identity backbone|
|RXV2-10|向量化Temporal Memory与两fresh默认|`orbit_teacher.py`、`identity_only_forward.py`|pending|三fresh非默认|
|RXV2-11|选择性nuisance子空间|`selective_nuisance_subspace.py`|pending/default-off|`lambda_subspace=0`|
|RXV2-12|结构化batch与source-only选择|`balanced_tx_rx_sampler.py`、`daot_source_selection.py`|pending|单V只读，不拆V_cal/V_select|
|RXV2-XF|旧checkpoint非LEO_WEAK测试|无|rejected|用户明确不需要|

## 4.实施依据

- 详细任务、测试顺序、参数范围、后续最小实验矩阵和验收阈值见`docs/superpowers/plans/2026-09-03-adv3b02-daot-stn-rx-v2.md`。
- 实施完成后在本文件逐项记录实际文件、测试、状态、偏差及理由。
- 在用户明确授权前，状态不得从“本地实现完成”推进到N607实验发布。

