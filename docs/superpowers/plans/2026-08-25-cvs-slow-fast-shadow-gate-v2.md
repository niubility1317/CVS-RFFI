# CVS Slow-Fast Shadow Gate V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不重训Phase1.5的前提下输出非零适配影子状态，建立分层cross-fit连续风险门控，并用同rowtruth-last实验区分gate误拒绝与Adapter无上界。

**Architecture:** 统一余弦logit和Adapter强度语义；selection模块只消费support并返回正式gate状态、固定候选状态与完整审计；runner只提取一次query特征并批量输出预注册影子状态；独立scorer在prediction完整后连接truth。P1／P2由本轮评分结果触发，不在P0实现中混入。

**Tech Stack:** Python、PyTorch、NumPy、pytest、N607、`p2_min_v1`。

**Spec:** `docs/CVS_SLOW_FAST_SHADOW_GATE_V2_DESIGN_20260825.md`

## Global Constraints

- Phase2只允许target support更新和选择；query只读且逐样本，不得访问source、query truth／role或跨query配额。
- 复用既有`VALIDATED_ONCE` capsule／split，不因方法变化重验数据。
- K1固定DA0；K10 cross-fit每fold逐类平衡；所有输出不可覆盖。
- 先本地`ssr-gpu`验证和Git提交，再发布N607；只运行单seed同row诊断。

---

### Task 1: 统一logit与强度语义

**Files:**
- Modify: `code/cvsrffi/slow_fast_objectives.py`
- Modify: `code/cvsrffi/slow_fast_adapter.py`
- Modify: `code/cvsrffi/slow_fast_bundle.py`
- Test: `tests/test_slow_fast_adapter.py`
- Test: `tests/test_slow_fast_phase15.py`

**Interfaces:**
- Produces: `prototype_logits(features,prototypes,logit_scale)`；零中心`tanh`方向门控；COMMON_SHIFT由`rho`单独控制强度。

- [ ] 写失败测试：非8.0scale改变logit；LOWRANK gate零值关闭、正负值反向；COMMON_SHIFT rho=0／0.5／1满足手算。
- [ ] 串行运行聚焦测试并确认因旧实现失败。
- [ ] 最小修改统一目标函数、forward和bundle版本兼容。
- [ ] 运行聚焦测试和现有bundle负测至通过。

### Task 2: 分层cross-fit连续风险门控

**Files:**
- Modify: `code/cvsrffi/slow_fast_selection.py`
- Test: `tests/test_slow_fast_selection.py`

**Interfaces:**
- Produces: `select_support_only_state(...,logit_scale,trust_radius,crossfit_seed,repeats)`；返回正式状态、每lambda状态和完整审计。

- [ ] 写失败测试：K10每fold5／5且逐类平衡；K5双折平衡；K1回退。
- [ ] 写失败测试：最小风险而非最大lambda；风险改善不足、accuracy容差或trust越界时回退。
- [ ] 写失败测试：回退时仍记录fold数、尝试更新次数和每lambda完整轨迹。
- [ ] 运行测试确认正确RED。
- [ ] 实现fold生成、MacroCE、class CVaR、move风险、容差约束和argmin选择。
- [ ] 运行selection全部测试至GREEN。

### Task 3: 影子状态runner和scorer

**Files:**
- Modify: `code/cvsrffi/stage2_slow_fast_runner.py`
- Modify: `code/cvsrffi/stage2_slow_fast_matrix.py`
- Modify: `code/cvsrffi/slow_fast_scorer.py`
- Modify: `code/scripts/run_stage2_slow_fast_matrix.py`
- Modify: `code/scripts/score_stage2_slow_fast_matrix.py`
- Test: `tests/test_stage2_slow_fast_runner.py`
- Test: `tests/test_stage2_slow_fast_matrix.py`
- Test: `tests/test_slow_fast_scorer.py`

**Interfaces:**
- Produces: 固定lambda／J／步长倍率状态prediction、`DA1_GATE_LEGACY_REG0`、`DA1_GATE_CF_REG0`、扩展receipt和多状态truth-last summary。

- [ ] 写失败测试：query只提取一次且所有影子状态在truth未知时输出。
- [ ] 写失败测试：receipt含score type、scale、trust、参数／特征移动和计算量字段。
- [ ] 写失败测试：scorer对所有状态同一opaque-ID join并报告per-state／per-class delta和翻转。
- [ ] 运行测试确认RED。
- [ ] 实现批量影子状态预测和多状态评分，保持旧pair scorer兼容。
- [ ] 运行runner／matrix／scorer测试至GREEN。

### Task 4: 配置、smoke、追踪和本地闭合

**Files:**
- Create: `configs/stage2_slow_fast_shadow_diag_s392002_20260825.json`
- Create: `configs/stage2_slow_fast_shadow_smoke_s392002_20260825.json`
- Modify: `code/cvsrffi/slow_fast_no_query_smoke.py`
- Modify: `analysis/cached_slow_fast_shadow_gate_v2_traceability_20260825.md`
- Test: `tests/test_slow_fast_no_query_smoke.py`

**Interfaces:**
- Produces: 同一receiver20-1／K10-new10／三scene的不可覆盖诊断配置和真实checkpoint无query核验。

- [ ] 写失败测试：smoke核对160维、class mapping、原型判决一致性且无query能力。
- [ ] 运行测试确认RED后实现最小核验。
- [ ] 运行全部相关测试、compileall和config dry-run。
- [ ] 更新追踪状态，完成一次独立P0/P1审查；仅修直接运行问题并最多定点复审一次。
- [ ] 精确stage、commit、push并回读远端OID。

### Task 5: N607发布、评分和条件决策

**Files:**
- Create: `E:/type10-7/automation_reports/CV-SincNet/<run-id>/report.md`
- Create mirror: `docs/experiments/<run-id>_report.md`

**Interfaces:**
- Consumes: Task 4固定提交和配置。
- Produces: Phase2影子prediction、truth-last score、gate诊断树和P1／P2决策。

- [ ] 写最小预登记报告，记录run、commit、命令、CWD、路径、GPU、停止规则和artifact。
- [ ] 运行N607只读preflight，制作单一release归档并完成一次SHA对比和一次远端编译。
- [ ] launcher先执行真实checkpoint无query smoke，PASS后立即启动prediction并核对PID／CWD／cmdline／GPU／日志增长。
- [ ] prediction完整后由独立scorer连接truth；不得依据评分重跑或改变状态。
- [ ] 若固定非零状态存在query收益，停在gate／优化器结论；若全部无上界，才建立P1轻型Phase1.5后续实现批次。
- [ ] 完成中文详细报告、追踪反审计、精确提交推送和远端OID回读。
