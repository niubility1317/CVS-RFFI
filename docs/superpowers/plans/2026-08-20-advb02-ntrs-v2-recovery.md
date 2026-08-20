# ADVB02 NTRS-V2 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复NTRS-v1的主干低学习率和非恒等公共路径，发布D1、D2、D3、V2-1完整单seed矩阵，并补做M1只读三头诊断。

**Architecture:** 保留v1历史路径，新增正交配置`ntrs_variant`、`ntrs_core_lr_mode`和`ntrs_identity_bypass`。V2最小路径只做一次身份骨干前向，使用确定性fast context、共享原CosFace头和零初始化有界嵌入残差；评估器扩展gate阈值比例，launcher冻结四个profile。

**Tech Stack:** Python、PyTorch、pytest、Bash、Git、N607 CUDA训练。

**Spec:** `docs/superpowers/specs/2026-08-20-advb02-ntrs-v2-recovery-design.md`

## Global Constraints

- seed固定为`392034`。
- Phase1角色固定为`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- 训练与最终测试只允许`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- `mixed_orbit`、target receiver、target support/query/truth均不可进入训练、选模或测试。
- 所有run训练200轮，最终checkpoint必须完成clean和三种LEO_WEAK逐场景测试。
- v1历史profile和checkpoint加载行为保持不变。
- 本地测试串行使用`conda run -n ssr-gpu`，不得使用PowerShell或pwsh。

---

### Task 1: LR模式与V2阶段控制

**Files:**
- Modify: `code/cvsrffi/ntrs_training.py`
- Modify: `code/SSDG/train_ssdg.py`
- Test: `code/tests/test_ntrs_training.py`

**Interfaces:**
- Produces: `ntrs_training_stage(epoch, variant="v1") -> NTRSTrainingStage`
- Produces: `set_ntrs_optimizer_learning_rates(..., core_lr_mode="v1", variant="v1") -> dict[str, float]`
- Consumes: optimizer参数组中的`group_name=core|ntrs`

- [ ] **Step 1: 写入失败测试**

```python
def test_ntrs_baseline_core_lr_never_decays():
    optimizer = _optimizer_with_named_groups()
    for epoch in (1, 16, 17, 40, 41, 68, 69, 90, 91, 130, 200):
        rates = set_ntrs_optimizer_learning_rates(
            optimizer, epoch=epoch, base_lr=2e-4,
            core_lr_mode="baseline", variant="v2_min",
        )
        assert rates["core"] == 2e-4

def test_v2_stage_is_identity_through_epoch_90_then_ramps():
    assert ntrs_training_stage(90, variant="v2_min").geometry_scale == 0.0
    assert 0.0 < ntrs_training_stage(91, variant="v2_min").geometry_scale < 1.0
    assert ntrs_training_stage(130, variant="v2_min").geometry_scale == 1.0
    assert ntrs_training_stage(200, variant="v2_min").geometry_scale == 1.0
```

- [ ] **Step 2: 运行RED**

Run: `conda run -n ssr-gpu python -m pytest code/tests/test_ntrs_training.py -q`

Expected: 因函数尚不接受`variant/core_lr_mode`而失败。

- [ ] **Step 3: 实现最小阶段控制**

保留v1阶段返回值；新增V2阶段E1–90全0、E91–130线性ramp、E131–200全1。baseline模式只覆盖core组为`base_lr`，NTRS组仍读取variant阶段比例。

- [ ] **Step 4: 运行GREEN**

Run: `conda run -n ssr-gpu python -m pytest code/tests/test_ntrs_training.py -q`

Expected: 全部通过。

### Task 2: 严格旁路与共享头V2模型

**Files:**
- Modify: `code/ntrs.py`
- Modify: `code/model_dual_cvsincnet.py`
- Modify: `code/model.py`
- Test: `code/tests/test_ntrs_core.py`
- Test: `code/tests/test_advb02_ntrs_model.py`

**Interfaces:**
- Produces: `NTRSMinimalResidual.forward(z_anchor, q_fast, stage_scale) -> NTRSOutput`
- Produces: model参数`ntrs_variant`和`ntrs_identity_bypass`
- Preserves: v1的`NTRSRobustifier`、独立head及checkpoint键

- [ ] **Step 1: 写入严格恒等失败测试**

```python
def test_v2_identity_bypass_matches_raw_path_exactly():
    base, bypass = _paired_models_with_shared_raw_state()
    base.eval(); bypass.eval()
    with torch.no_grad():
        raw = base(FIXED_X, return_aux=True)
        out = bypass(FIXED_X, return_aux=True)
    assert (raw["tx_logits"] - out["tx_logits"]).abs().max() < 1e-6
    assert (raw["z_id"] - out["z_id"]).abs().max() < 1e-6
```

- [ ] **Step 2: 写入共享头和单前向失败测试**

```python
def test_v2_uses_shared_cosface_and_one_identity_forward():
    model = _tiny_model(use_ntrs=True, ntrs_variant="v2_min")
    assert model.ntrs_robust_head is None
    calls = _count_identity_backbone_calls(model)
    model.eval()(FIXED_X, return_aux=True, ntrs_epoch=130)
    assert calls.value == 1
```

- [ ] **Step 3: 运行RED**

Run: `conda run -n ssr-gpu python -m pytest code/tests/test_ntrs_core.py code/tests/test_advb02_ntrs_model.py -q`

Expected: v1仍执行LayerNorm、独立head和第二次身份前向，新增测试失败。

- [ ] **Step 4: 实现V2最小路径**

新增零初始化`NTRSMinimalResidual`。其`z_rob=z_anchor-stage_scale*delta_z`，不调用LayerNorm或额外normalize。V2模型复用raw CosFace头计算robust logits；identity bypass直接返回raw输出和零遥测；V2不实例化物理corrector、slow support、独立robust head及v1 factor/safety heads。

- [ ] **Step 5: 运行GREEN**

Run: `conda run -n ssr-gpu python -m pytest code/tests/test_ntrs_core.py code/tests/test_advb02_ntrs_model.py -q`

Expected: 全部通过，v1原测试仍通过。

### Task 3: V2最小loss和评估遥测

**Files:**
- Modify: `code/cvsrffi/ntrs_training.py`
- Modify: `code/cvsrffi/ntrs_evaluation.py`
- Modify: `code/cvsrffi/eval.py`
- Modify: `code/SSDG/train_ssdg.py`
- Test: `code/tests/test_ntrs_training.py`
- Test: `code/tests/test_ntrs_evaluation.py`

**Interfaces:**
- Produces: `ntrs_minimum_correction_loss(delta_z, anchor_z) -> Tensor`
- Extends telemetry: `safe_gate_gt_001/005/010_rate`
- Preserves: raw/robust/fused accuracy和transition字段

- [ ] **Step 1: 写入loss与遥测失败测试**

```python
def test_v2_minimum_correction_is_zero_for_zero_delta():
    anchor = torch.randn(4, 16)
    zero = torch.zeros_like(anchor)
    assert ntrs_minimum_correction_loss(zero, anchor).item() == 0.0

def test_ntrs_telemetry_reports_gate_threshold_rates():
    summary = _accumulate_safe_gates([0.0, 0.02, 0.06, 0.20])
    assert summary["known"]["safe_gate_gt_001_rate"] == 0.75
    assert summary["known"]["safe_gate_gt_005_rate"] == 0.50
    assert summary["known"]["safe_gate_gt_010_rate"] == 0.25
```

- [ ] **Step 2: 运行RED**

Run: `conda run -n ssr-gpu python -m pytest code/tests/test_ntrs_training.py code/tests/test_ntrs_evaluation.py -q`

Expected: 新函数和字段缺失而失败。

- [ ] **Step 3: 接通V2 loss**

V2只接通共享头robust CE、直接minimum correction、sat-KL和共享prototype margin；v1继续使用原loss bundle。训练遥测必须记录raw和加权值。

- [ ] **Step 4: 扩展只读评估**

在现有accumulator中增加三个safe-gate阈值计数，保持eval/no-grad且不更新模型状态。

- [ ] **Step 5: 运行GREEN**

Run: `conda run -n ssr-gpu python -m pytest code/tests/test_ntrs_training.py code/tests/test_ntrs_evaluation.py -q`

Expected: 全部通过。

### Task 4: 完整矩阵launcher与协议负测

**Files:**
- Modify: `code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh`
- Modify: `code/tests/test_phase1_advb02_ntrs_leo_weak_launcher.py`
- Modify: `code/tests/test_ntrs_protocol_negatives.py`
- Create: `automation_reports/CV-SincNet/phase1_advb02_ntrs_v2_recovery_matrix_20260820/report.md`

**Interfaces:**
- Consumes: `NTRS_PROFILE=v2_identity_bypass|v2_identity_bypass_v1_lr|v1_fair_core_lr|v2_min_shared_head`
- Produces: 四个不可覆盖candidate output和对应final evaluation

- [ ] **Step 1: 写入四profile失败测试**

每个profile运行launcher dry-run，逐项断言candidate、variant、core LR模式、identity bypass、seed、四角色、三LEO场景和final eval命令；断言`mixed_orbit`不存在。

- [ ] **Step 2: 运行RED**

Run: `conda run -n ssr-gpu python -m pytest code/tests/test_phase1_advb02_ntrs_leo_weak_launcher.py code/tests/test_ntrs_protocol_negatives.py -q`

Expected: 新profile尚不存在而失败。

- [ ] **Step 3: 实现profile映射和parser参数**

候选分别命名为`ADVB02_NTRS_V2_D1_BYPASS_E200`、`ADVB02_NTRS_V2_D2_BYPASS_V1LR_E200`、`ADVB02_NTRS_V1_D3_FAIRLR_E200`、`ADVB02_NTRS_V2_MIN_SHARED_E200`。

- [ ] **Step 4: 运行GREEN和语法检查**

Run: `conda run -n ssr-gpu python -m pytest code/tests/test_phase1_advb02_ntrs_leo_weak_launcher.py code/tests/test_ntrs_protocol_negatives.py -q`

Run: `bash -n code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh`

Expected: 全部通过。

### Task 5: 集成验证、审查、提交和N607发布

**Files:**
- Update: `docs/superpowers/specs/2026-08-20-advb02-ntrs-v2-recovery-design.md`
- Update: `automation_reports/CV-SincNet/phase1_advb02_ntrs_v2_recovery_matrix_20260820/report.md`

- [ ] **Step 1: 运行聚焦回归**

Run: `conda run -n ssr-gpu python -m pytest code/tests/test_ntrs_core.py code/tests/test_advb02_ntrs_model.py code/tests/test_ntrs_training.py code/tests/test_ntrs_evaluation.py code/tests/test_ntrs_protocol_negatives.py code/tests/test_phase1_advb02_ntrs_leo_weak_launcher.py -q`

Run: `conda run -n ssr-gpu python -m py_compile code/ntrs.py code/model_dual_cvsincnet.py code/cvsrffi/ntrs_training.py code/cvsrffi/ntrs_evaluation.py code/SSDG/train_ssdg.py`

- [ ] **Step 2: 执行一次独立P0/P1审查**

审查范围只包括会让下一次真实实验跑错、越权、覆盖输出、不能启动或不能生成合法最终测试的问题。发现P0/P1后只做一次定点修复和定点复审。

- [ ] **Step 3: 提交并推送**

只stage本次代码、测试、launcher、spec追踪和矩阵报告；提交后核对远端分支OID等于本地HEAD。

- [ ] **Step 4: 本地打包并发布release**

创建一个release归档，做一次本地/远端SHA256比较；远端解包后运行Python编译和launcher dry-run。

- [ ] **Step 5: 真实checkpoint无query smoke**

使用现有ADV3B02 checkpoint和source样本，验证`query_samples=0`、真实模型前向、D1恒等及V2输出有限。

- [ ] **Step 6: 启动完整矩阵**

在资源preflight后为M1-DIAG、D1、D2、D3、V2-1分配唯一run ID和GPU；立即核对PID、CWD、cmdline、GPU和日志增长。训练结束后launcher自动执行clean和三种LEO_WEAK最终测试。
