# ADV3B02-PairBiCAD-P4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现报告冻结的PairBiCAD P0–P4并发布30行source-only正式矩阵。

**Architecture:** 复用现有`phase1_bicad_xr`包和SSDG独立入口，新增向后兼容的P0–P4候选、因素投影与pair目标；训练入口以48条物理样本生成clean/LEO并一次拼接前向，P3/P4再接入U_s同物理样本pair。矩阵launcher复用现有严格四场景闭合逻辑。

**Tech Stack:** Python、PyTorch、pytest、现有SSDG/ManySig/LEO_WEAK训练与N607 launcher。

**Spec:** `docs/superpowers/specs/2026-08-31-pairbicad-p4-design.md`

## Global Constraints

- Phase1 source-only；禁止Phase2/target/support/query/truth。
- day1/2/3；`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- P0–P4使用`ce_only_plus_pair_selfsup`候选级扩展；卫星TX标签监督只走CE，pair目标不得读取U_s TX真值；旧D0–F3语义不得改变。
- 物理batch48、网络batch96、每步一次双骨干前向。
- P5–P9全部延期。
- 正式矩阵30行、U4000、每GPU最多2行。

---

### Task 1: P0–P4候选与冻结配置

**Files:**
- Modify: `code/cvsrffi/phase1_bicad_xr/config.py`
- Test: `code/tests/phase1_bicad_xr/test_config.py`

**Interfaces:**
- Produces: `candidate_config("P0"..."P4") -> BiCADXRConfig`
- Produces: 配置字段`strict_pair_concat/pair_vicreg/pair_delta/factor_interaction_dim/pair_projector_dim`

- [ ] 写测试断言P0–P4逐级只增加规范中的主要开关，batch size为48、updates为4000、start epoch为1，旧D0–F3不变。
- [ ] 运行配置测试并确认因未知P候选或缺失字段失败。
- [ ] 最小实现P0–P4 registry和条件化配置校验。
- [ ] 重跑配置测试并确认通过。

### Task 2: 因素投影与PairBiCAD纯损失

**Files:**
- Modify: `code/cvsrffi/phase1_bicad_xr/heads.py`
- Create: `code/cvsrffi/phase1_bicad_xr/pair.py`
- Test: `code/tests/phase1_bicad_xr/test_pair.py`
- Test: `code/tests/phase1_bicad_xr/test_heads.py`

**Interfaces:**
- Produces: `FactorizedDomainProjector.forward(z_dom) -> DomainFactors`
- Produces: `pair_identity_hinge(clean, satellite, epsilon)`
- Produces: `vicreg_pair_loss(clean, satellite, gamma)`
- Produces: `pair_delta_objectives(clean_id, sat_id, clean_c, sat_c, channel, ...)`

- [ ] 写真实Tensor测试覆盖维度、有限值、梯度方向、`z_int`隔离、batch<2的VICReg安全退化。
- [ ] 运行新测试并确认因API不存在失败。
- [ ] 实现最小投影与损失，禁止内部第二次模型前向或TX伪标签。
- [ ] 重跑新测试和既有heads/losses测试。

### Task 3: Trainer与SSDG混合L/U单前向接入

**Files:**
- Modify: `code/cvsrffi/phase1_bicad_xr/trainer.py`
- Modify: `code/SSDG/train_ssdg.py`
- Modify: `code/tests/phase1_bicad_xr/test_trainer.py`
- Modify: `code/tests/phase1_bicad_xr/test_ssdg_entry.py`

**Interfaces:**
- Consumes: Task1配置、Task2因素投影和pair目标。
- Produces: P0–P4可训练路径及`pairbicad_runtime/components/effective_counts/skip_reasons`。

- [ ] 写失败测试：P0每步pair单前向；P3使用L/U pair且U的TX为None；P4记录delta目标；P0–P2不调用后续机制。
- [ ] 写失败入口测试：labeled/unlabeled loader按16L+32U组成48物理样本，拼接后96，U标签不可达。
- [ ] 运行定点测试确认预期失败。
- [ ] 实现trainer组件路由、梯度剂量记录及SSDG L/U循环，不改变旧候选入口。
- [ ] 重跑trainer/entry/model/protocol聚焦测试。

### Task 4: 30行矩阵launcher与artifact闭合

**Files:**
- Modify: `code/scripts/launch_phase1_bicad_xr_matrix_20260830.py`
- Create: `code/scripts/launch_phase1_pairbicad_p0p4_n607_20260831.sh`
- Modify: `code/tests/phase1_bicad_xr/test_launcher.py`
- Modify: `code/tests/phase1_bicad_xr/test_n607_launch_script.py`

**Interfaces:**
- Produces: P0–P4×fold1/8×seed392001/2/3的30行计划。
- Produces: 每GPU最多2行的不可覆盖N607 dispatcher命令。

- [ ] 写失败测试断言30行、唯一row ID、U4000、day1/2/3、两slot/GPU及四场景闭合。
- [ ] 运行launcher测试确认失败。
- [ ] 实现矩阵stage和N607脚本，复用严格checkpoint重建和四场景评估。
- [ ] 重跑launcher、dry-run和JSON计划测试。

### Task 5: 聚焦验证、审查、Git与N607发布

**Files:**
- Update: `automation_reports/CV-SincNet/phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r1/report.md`

**Interfaces:**
- Consumes: Task1–4全部实现。
- Produces: Git commit、release、RUNNING状态和首轮绑定证据。

- [ ] 激活`ssr-gpu`并运行全部`code/tests/phase1_bicad_xr`聚焦测试、py_compile、launcher dry-run。
- [ ] 运行一次真实checkpoint no-query smoke。
- [ ] 执行一次独立P0/P1审查；只修复直接导致跑错、越权、覆盖、不能启动或不能闭合的问题。
- [ ] 精确stage实现、测试、spec、plan和报告，commit、push并核对远端OID。
- [ ] 运行N607 preflight、release归档单次SHA对比、远端编译并启动正式矩阵。
- [ ] 回读dispatcher/worker/CWD/cmdline/GPU/log增长，报告最高交付状态。
