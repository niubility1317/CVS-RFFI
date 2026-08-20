# ADVB02 NTRS Adapter-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在成熟ADVB02 checkpoint上实现冻结raw身份空间、训练q和低秩类共享残差的NTRS adapter-only路线，并发布LEO_WEAK实验矩阵。

**Architecture:** 新增`v3_adapter`模型variant，训练时主输出保持raw，仅显式adapter损失消费robust输出；评测时always-on robust作为主输出。训练器以专用冻结策略保证raw骨干/head不进入优化器，launcher冻结A0到A4的配置和顺序门槛。

**Tech Stack:** Python、PyTorch、pytest、Bash、N607 CUDA训练。

**Spec:** `docs/superpowers/specs/2026-08-20-advb02-ntrs-adapter-only-design.md`

## Global Constraints

- seed固定`392034`。
- `L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- 训练和最终测试只用`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`，禁止`mixed_orbit`。
- 首轮A1/A2冻结raw backbone/head，A3/A4不得绕过前序门槛。
- 本地`ssr-gpu`验证后才同步N607；正式结果必须包含E200 checkpoint、clean和三LEO全量测试。

---

### Task 1: q-only低秩adapter核心

**Files:**
- Modify: `code/ntrs.py`
- Test: `code/tests/test_ntrs_core.py`

**Interfaces:**
- Produces: `NTRSAdapterOnlyResidual(embedding_dim, q_dim, rank, alpha_max, support_domains, support_tau)`。
- Produces: `forward(z_anchor, q, epoch, update_source_support, source_domains, source_support_mask)`返回`NTRSOutput`。

- [ ] 写失败测试：残差只接受q，不接受z_anchor拼接；对q反向产生梯度；raw anchor无梯度。
- [ ] 运行聚焦测试，确认旧实现因缺少类和q梯度失败。
- [ ] 实现rank-8共享basis和q系数头、相对范数边界、always-on gate及可选source support。
- [ ] 验证alpha不超过0.02/0.05、无LayerNorm、无独立分类头。

### Task 2: 模型接线和raw主路径隔离

**Files:**
- Modify: `code/model_dual_cvsincnet.py`
- Modify: `code/model.py`
- Modify: `code/post_stage_common.py`
- Test: `code/tests/test_advb02_ntrs_model.py`

**Interfaces:**
- Consumes: `NTRSAdapterOnlyResidual`。
- Produces: `ntrs_variant=v3_adapter`和`ntrs_q_trainable`checkpoint字段。

- [ ] 写失败测试：train模式`tx_logits/z_id`严格等于raw，robust输出可反向到q/adapter；eval模式主输出等于always-on robust。
- [ ] 写失败测试：残差不依赖z_anchor梯度，共享CosFace head没有重复prototype参数。
- [ ] 实现构造、forward和checkpoint参数传播。
- [ ] 运行模型测试确认train/raw与eval/robust语义。

### Task 3: Adapter-only冻结和损失分离

**Files:**
- Modify: `code/cvsrffi/ntrs_training.py`
- Modify: `code/SSDG/train_ssdg.py`
- Test: `code/tests/test_ntrs_training.py`

**Interfaces:**
- Produces: `configure_ntrs_trainable_parameters(model,args)`。
- Produces: loss keys`clean_zero`和`satellite_relative`。

- [ ] 写失败测试：A1只让q/adapter可训练，A1-R只让adapter可训练，raw backbone/head参数为0个optimizer成员。
- [ ] 写失败测试：adapter variant的robust CE只消费satellite，clean-zero只消费clean correction。
- [ ] 写失败测试：A1 q梯度非零、A1-R q参数不更新、raw参数最大漂移为0。
- [ ] 实现专用冻结策略、loss分离和raw伪标签路径。
- [ ] 增加q梯度、adapter梯度和raw漂移训练遥测。
- [ ] 运行训练聚焦测试。

### Task 4: 评测遥测与checkpoint重建

**Files:**
- Modify: `code/cvsrffi/ntrs_evaluation.py`
- Modify: `code/evaluation/collaborative_inference_eval.py`
- Test: `code/tests/test_ntrs_evaluation.py`
- Test: `code/tests/test_collaborative_inference_eval.py`

**Interfaces:**
- Produces: relative correction和rotation angle的p50/p95，clean/satellite分开汇总。

- [ ] 写失败测试：遥测包含p50、p95和rotation angle；checkpoint重建恢复v3结构、rank、alpha、q模式。
- [ ] 实现分布统计和结构参数恢复。
- [ ] 运行评测测试和Python编译。

### Task 5: 矩阵launcher和协议负测

**Files:**
- Create: `code/scripts/launch_phase1_advb02_ntrs_adapter_only_20260820.sh`
- Create: `code/tests/test_phase1_advb02_ntrs_adapter_only_launcher.py`
- Modify: `analysis/advb02_ntrs_adapter_only_traceability.md`

**Interfaces:**
- Produces profiles:`a0_control,a0_bypass,a1_random_q,a1_trainable_q,a2_teacher_margin,a3_support_gate,a4_joint_core`。

- [ ] 写失败测试：七个profile、checkpoint初始化、冻结策略、alpha、loss、顺序门、E200独立测试和不可覆盖输出。
- [ ] 实现launcher；A0/A0-B从头，A1-R/A1/A2从成熟D1 checkpoint，A3/A4要求前序晋级标志。
- [ ] 验证seed、角色比例、三LEO训练测试和mixed_orbit拒绝。
- [ ] 运行dry-run并更新追踪状态。

### Task 6: 发布前验证和N607发布

**Files:**
- Create: `automation_reports/CV-SincNet/phase1_advb02_ntrs_adapter_only_matrix_20260820/report.md`
- Modify: `analysis/advb02_ntrs_adapter_only_traceability.md`

**Interfaces:**
- Produces: Git固定commit、release归档和首轮矩阵run ID。

- [ ] 在`ssr-gpu`运行全部NTRS聚焦测试、`py_compile`、`bash -n`和七profile dry-run。
- [ ] 使用真实D1 checkpoint做无query smoke，核对raw零漂移和q梯度。
- [ ] 完成一次独立P0/P1审查，只处理会导致真实实验跑错的问题。
- [ ] 写最小预登记报告并提交、自动push、核对远端OID。
- [ ] 生成单一release归档，完成一次本地/远端SHA比对和远端编译。
- [ ] N607资源预检后启动A0/A0-B/A1-R/A1/A2首轮；A3/A4保留顺序门。
- [ ] 启动后核对PID、CWD、cmdline、GPU和日志增长。

