# ADVB02 NTRS LEO弱信道增强实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在Core90和当前Phase1协议上实现独立的`ADVB02_NTRS_LEO_WEAK_E200`，发布到N607并闭合clean与三种LEO弱场景测试。

**Architecture:** 新增独立`ntrs.py`模块，先从raw IQ构造快慢上下文和近恒等复数粗校正，再在160维`z_id`端以rank-8干扰切空间限制有界修正。双头可校正性与安全门生成最终logits，训练器接通结构保持、条件去相关和安全损失，现有CRRA代码保持不变。

**Tech Stack:** Python 3、PyTorch、pytest、CV-SincNet/SSDG、Git Bash、`ssr-gpu`、N607。

**Spec:** `docs/superpowers/specs/2026-08-20-advb02-ntrs-leo-weak-design.md`

## Global Constraints

- 训练与最终测试仅使用`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。
- Phase1角色固定为`0.07/0.63/0.15/0.15`，seed为`392034`，不得访问target/query。
- `use_ntrs`与`use_crra`互斥；旧checkpoint在两者关闭时保持兼容。
- 每个生产行为先写失败测试并确认因缺少该行为而失败。
- 现有CRRA run和所有历史输出只读。

### Task 1: NTRS核心物理与上下文模块

**Files:**
- Create: `code/ntrs.py`
- Create: `code/tests/test_ntrs_core.py`

**Interfaces:**
- Produces: `compute_grouped_physical_descriptors(x)->Tensor[B,D]`
- Produces: `FastSlowContext.forward(x, metadata, domains, update_slow)->NTRSContext`
- Produces: `BoundedWidelyLinearCorrector.forward(x,q)->PhysicalCorrection`
- Produces: `NuisanceTangentBasis.update(clean_z,sat_z)`和`project(coefficients)`
- Produces: `NTRSRobustifier.forward(z_anchor,z_phys,q,...)->NTRSOutput`

- [ ] 写失败测试，覆盖40维有限描述符、source-only慢EMA、评估只读、L=3近恒等校正、PA/raw输入不变、rank-8基正交、修正严格落在切空间、alpha上界和S1零gate。
- [ ] 运行`PYTHONPATH=code python -m pytest -q code/tests/test_ntrs_core.py`，确认每项因NTRS接口不存在而失败。
- [ ] 实现最小核心模块并再次运行同一测试至通过。
- [ ] 增加数值边界测试：零信号、常数信号、NaN/Inf输入、单样本和缺失metadata均输出有限值。

### Task 2: 双骨干路径与安全双头

**Files:**
- Modify: `code/model.py`
- Modify: `code/model_dual_cvsincnet.py`
- Create: `code/tests/test_advb02_ntrs_model.py`

**Interfaces:**
- `model.py`新增可选`original_iq`和`frequency_dual_mix`，使时域读校正IQ、频域读双视图、PA读原始IQ。
- `DualCVSincNetDisentangle`新增`use_ntrs`及NTRS配置，输出`ntrs_z_anchor/ntrs_z_rob/ntrs_raw_logits/ntrs_robust_logits`和安全遥测。

- [ ] 写失败测试，证明NTRS只存在于identity模型、domain模型读raw IQ、PA路径读raw IQ、raw/robust头独立、分歧时fused logits回退raw、未知救回默认关闭。
- [ ] 运行新模型测试确认失败。
- [ ] 接入NTRS上下文、校正前向、第二身份视图、z_id切空间修正和双头安全融合。
- [ ] 运行新模型测试及现有`test_advb02_crra_model.py`，证明CRRA行为未回归。

### Task 3: NTRS损失和分阶段优化

**Files:**
- Create: `code/cvsrffi/ntrs_training.py`
- Modify: `code/cvsrffi/losses.py`
- Modify: `code/SSDG/train_ssdg.py`
- Create: `code/tests/test_ntrs_training.py`
- Create: `code/tests/test_ntrs_protocol_negatives.py`

**Interfaces:**
- Produces: `ntrs_stage(epoch)`、`build_optimizer_with_ntrs_groups(...)`、`set_ntrs_optimizer_learning_rates(...)`
- Produces: `margin_preservation_loss`、`relation_distillation_loss`、`conditional_decorrelation_loss`、`ntrs_correctability_loss`和`ntrs_safety_loss`

- [ ] 写失败测试，覆盖S1/S2-a/S2-b/S3门控和`1:5`学习率比例、KL=`0.01`、margin=`0.03`、relation=`0.02`、correctability标签、结构损失、source-only状态更新及NTRS/CRRA互斥。
- [ ] 运行测试确认失败原因来自缺少NTRS实现。
- [ ] 实现训练配置、损失、优化器分组和主训练循环接线。
- [ ] 运行NTRS训练测试、协议负测和现有CRRA训练回归测试。

### Task 4: 独立评估、checkpoint恢复与launcher

**Files:**
- Create: `code/cvsrffi/ntrs_evaluation.py`
- Modify: `tools/eval_cvs_checkpoint_sat_channel.py`
- Create: `code/tests/test_ntrs_evaluation.py`
- Create: `code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh`
- Create: `code/tests/test_phase1_advb02_ntrs_leo_weak_launcher.py`

**Interfaces:**
- 评估JSON新增`ntrs_telemetry`，保持现有accuracy字段兼容。
- launcher产生唯一run/output/log根，训练成功后无条件运行clean和三种LEO独立测试。

- [ ] 写失败测试，覆盖checkpoint恢复epoch/basis/slow/support buffer、评估状态不变、逐场景遥测、固定Core90参数和不可覆盖输出。
- [ ] 实现评估归纳、checkpoint恢复和launcher。
- [ ] 运行评估/launcher测试、`bash -n`、完整命令解析和`py_compile`。

### Task 5: 追踪、审查、版本和N607发布

**Files:**
- Modify: `analysis/advb02_ntrs_traceability.md`
- Create: `automation_reports/CV-SincNet/phase1_advb02_ntrs_leo_weak_20260820/report.md`

- [ ] 将所有`pending`条目更新为`verified/deferred/rejected/blocked`之一，反向核对不存在遗漏。
- [ ] 运行聚焦协议负测、真实Core90 checkpoint无query smoke及一次独立P0/P1审查；若有直接P0/P1，只修复并定点复审一次。
- [ ] 只stage本候选文件，提交并自动push，独立核对远端分支OID等于本地HEAD。
- [ ] 生成一个release归档，完成一次本地/远端SHA比较和一次远端编译。
- [ ] 在N607选择不超过每GPU两个训练任务的GPU，启动唯一run ID并检查PID/CWD/cmdline/GPU/log增长。
- [ ] 训练完成后由独立评估器保存clean和三种LEO逐场景结果，再按同row证据更新报告和Git。

