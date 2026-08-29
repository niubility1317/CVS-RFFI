# BiNOVA-D92 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现support-only的NOVA-DA和NOVA-REG、精确D92桥、四状态预测、最小阶段门槛及truth-last实验闭环。

**Architecture:** 冻结ADV3B02负责一次特征提取，阶段A和阶段B分别持久化独立低秩残差状态。训练使用可微D92代理，最终预测复用现有`D92-E0-NORF32`精确拟合；任何选择在query打开前由support cross-fit冻结。

**Tech Stack:** Python、PyTorch、NumPy、pytest、现有CVS-RFFI Phase2 runner与D92 E0实现。

**Spec:** `docs/superpowers/specs/2026-08-29-binova-d92-design.md`

## Global Constraints

- `protocol_schema=p2_min_v1`且`phase2_data_status=VALIDATED_ONCE`。
- 每个物理样本只有一份固定LEO weak received IQ；support/query物理ID不相交。
- query没有标签、角色、配额、scorer或状态更新入口。
- 阶段A只使用六个旧类support；阶段B完全冻结`phi_D`。
- 最终D92固定`identity160+FFT96`、RF32关闭。
- floor和所有损失对类别标签置换保持同一公式。

---

### Task 1: 冻结特征与协议对象

**Files:**
- Create: `code/cvsrffi/stage2_binova_features.py`
- Test: `tests/test_stage2_binova_features.py`

**Interfaces:**
- Produces: `BiNOVAFeatures`、`BiNOVASupport`、`extract_binova_features(model, received_iq)`、`class_balanced_domain_context(...)`。

- [ ] 写失败测试：零/非有限IQ、重复物理ID、非法句柄、query字段进入support对象均被拒绝；类内重复行不改变各类等权上下文。
- [ ] 运行聚焦测试并确认因模块不存在而失败。
- [ ] 实现冻结特征抽取、6维物理统计、support/query分离对象和类均衡几何中位数。
- [ ] 运行聚焦测试并确认通过。

### Task 2: 可微D92训练代理

**Files:**
- Create: `code/cvsrffi/stage2_binova_d92.py`
- Test: `tests/test_stage2_binova_d92.py`

**Interfaces:**
- Consumes: `[N,256]`support特征、标签、old class mask。
- Produces: `DifferentiableD92State`、`fit_differentiable_d92(...)`、`d92_geometry_features(...)`、`exact_d92_fit(...)`。

- [ ] 写失败测试：task-balanced协方差手算一致、特征梯度非零、Cholesky正定、held行删除会改变其cross-fit几何。
- [ ] 确认RED。
- [ ] 实现unit拼接、OAS式收缩、old/new各0.5共享协方差、Cholesky判别行和六维几何条件。
- [ ] 通过聚焦测试。

### Task 3: NOVA-DA阶段A

**Files:**
- Create: `code/cvsrffi/stage2_binova_da.py`
- Test: `tests/test_stage2_binova_da.py`

**Interfaces:**
- Consumes: `BiNOVASupport`。
- Produces: `NOVA_DA_Config`、`NOVA_DA_State`、`fit_nova_da(...)`、`apply_nova_da(...)`、`evaluate_nova_da_crossfit(...)`。

- [ ] 写失败测试：零初始化逐行identity；4+2轮换均衡；五fold每类8/2互斥；affine-leak对纯仿射残差高于非仿射残差；实际新类标签被拒绝。
- [ ] 确认RED。
- [ ] 实现late-time rank16、identity rank32、D-A/B/C损失和支持内cross-fit。
- [ ] 通过聚焦测试并运行梯度有限值回归。

### Task 4: NOVA-REG阶段B

**Files:**
- Create: `code/cvsrffi/stage2_binova_reg.py`
- Test: `tests/test_stage2_binova_reg.py`

**Interfaces:**
- Consumes: 冻结`NOVA_DA_State`及old/new support。
- Produces: `NOVA_REG_Config`、`NOVA_REG_State`、`fit_nova_reg(...)`、`apply_nova_reg(...)`。

- [ ] 写失败测试：`phi_R`零初始化、`phi_D`无梯度/无变化、old/new双向margin手算、冲突梯度投影后与旧类梯度内积非负。
- [ ] 确认RED。
- [ ] 实现rank16条件残差、D92条件、注册损失、拓扑与梯度投影。
- [ ] 通过聚焦测试。

### Task 5: 生命周期、四状态和回退

**Files:**
- Create: `code/cvsrffi/stage2_binova_lifecycle.py`
- Test: `tests/test_stage2_binova_lifecycle.py`

**Interfaces:**
- Produces: `freeze_binova_support_states(...)`、`predict_binova_query_read_only(...)`和S0/S1/S2选择审计。

- [ ] 写失败测试：支持状态冻结前query不可打开；S2失败回退S1/S0；四状态键完整；REG0新类指标为N/A；query调用前后state逐字节不变。
- [ ] 确认RED。
- [ ] 实现精确D92重拟合桥、状态选择和只读query。
- [ ] 通过聚焦测试。

### Task 6: CLI、计划和truth-last评分

**Files:**
- Create: `code/scripts/run_stage2_binova_d92.py`
- Create: `tests/test_run_stage2_binova_d92.py`
- Create: `configs/stage2_binova_d92_minimal_20260829.json`

**Interfaces:**
- Produces: `inspect-plan`、`adapt-a`、`adapt-b`、`predict`、`score`。

- [ ] 写失败测试：输出根已存在、truth传入predict、门槛未通过时adapt-b、错误capsule/split均失败。
- [ ] 确认RED。
- [ ] 实现不可覆盖artifact、A/B自动门槛和独立score入口。
- [ ] 通过CLI聚焦测试。

### Task 7: 本地闭环、审查和Git发布

**Files:**
- Update: `analysis/binova_d92_traceability_20260829.md`
- Create: `automation_reports/CV-SincNet/binova_d92_stagea_minimal_20260829/report.md`

- [ ] 在`ssr-gpu`运行全部BiNOVA聚焦测试和相关D92/四状态回归。
- [ ] 用真实checkpoint执行一次无query smoke。
- [ ] 完成一次独立P0/P1审查；若有直接问题，修复后只做一次定点复审。
- [ ] 更新追踪表状态和验证证据，精确stage、commit、push并核对远端OID。

### Task 8: N607阶段A及条件阶段B

**Files:**
- Update: `automation_reports/CV-SincNet/binova_d92_stagea_minimal_20260829/report.md`
- Conditional create: `automation_reports/CV-SincNet/binova_d92_stageb_minimal_20260829/report.md`

- [ ] 预登记A0–A4、命令、GPU、输入输出、停止规则和artifact。
- [ ] N607 preflight、单release SHA、远端编译和不可覆盖启动。
- [ ] 核对PID/CWD/cmdline/GPU/log增长，等待prediction闭合。
- [ ] 独立scorer连接truth并写同row四状态结果。
- [ ] 读取预先冻结的support门槛；通过则自动发布B0–B3，否则记录`NOT_RUN_GATE_NOT_MET`。
- [ ] 阶段B如运行，重复prediction闭合和独立评分，不因低性能停止。
- [ ] 镜像最终报告、commit、push并核对远端OID。
