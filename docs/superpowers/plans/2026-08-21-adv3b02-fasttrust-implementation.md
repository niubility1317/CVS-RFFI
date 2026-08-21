# ADV3B02 FastTrust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完整实现FastTrust伪标签、U侧ADV3B02 CORE90星地身份监督、训练提速、source validation修复、双空间审计和16条双进程GPU实验发布链。

**Architecture:** 纯路由和配额逻辑放入`cvsrffi.muse_ssdg`，数据角色与feature contract校验放在训练/导出边界，`train_ssdg.py`只负责组织真实前向、损失和遥测。新launcher消费机器可读矩阵，为每张GPU顺序拉起两个独立候选并在训练完成后执行一次联合评测，再拆分为clean和三种LEO弱信道artifact。

**Tech Stack:** Python 3、PyTorch、pytest、Git Bash、现有SSDG训练器、现有LEO联合评测器。

**Spec:** `docs/superpowers/specs/2026-08-21-adv3b02-fasttrust-pseudo-design.md`

## Global Constraints

- source角色固定为`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，物理样本互斥，target不参与训练、校准或选模。
- 所有候选使用seed392002、200epoch和不可覆盖输出目录；训练完成后测试`final_ssdg.pth`。
- 星地增强固定为ADV3B02 CORE90同款拼接CE：`lambda_sat_cls=0.68`、`lambda_sat_cons=0`和三段LEO弱信道日程。
- U侧仅`U_H=high reliability∩temporal stable∩three-head agreement∩class-balanced cap`使用hard TX CE与satellite hard TX CE；`U_M`仅soft/candidate，`U_L`无唯一身份梯度。
- E1–E16不得建立U identity融合/loss/prototype更新图；遥测必须报告零选择、零loss和零identity梯度。
- U loader每epoch完整覆盖，主配置U batch256；关闭或权重为0的分支不产生对应view/前向。
- 所有测试使用`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`；每个生产改动必须先有按预期失败的测试。
- 完成全部本地验证和一次独立P0/P1审查后再创建项目提交、自动push、同步N607和启动实验。

---

### Task 1: Source validation与feature contract

**Files:**
- Modify: `code/SSDG/train_ssdg.py`
- Modify: `code/cvsrffi/phase2_prototypes.py`
- Create: `code/tests/test_phase1_fasttrust_protocol.py`
- Modify: `code/tests/test_phase2_prototypes.py`

**Interfaces:**
- Produces: `_partition_source_validation_roles(dataset, validation, cal_fraction, min_class_samples)`，按TX、receiver/day和稳定物理键分层产生互斥`V_cal/V_select`。
- Produces: `Phase1CalibrationError(code, details)`；错误码为`MISSING_CLASS_IN_V_CAL`、`INSUFFICIENT_CLASS_SAMPLES`、`ZERO_DIRECTION_FEATURE`、`NONFINITE_FEATURE`、`CLASS_ORDER_MISMATCH`。
- Produces: `audit_identity_feature_contract(z_id, feat_joint, labels, domains, logits)`，输出class coverage、finite/nonzero、类间几何和两空间分类一致性审计。

- [x] **Step 1:** 写失败测试，证明按全局TX排序切半会缺类，而新接口必须让每个足量类别在`V_cal/V_select`各至少4条且物理索引互斥。
- [x] **Step 2:** 运行聚焦测试，确认失败原因是接口缺失或现有全局切片缺类。
- [x] **Step 3:** 实现分层划分并在`split_tx_rx_day_1_7_2_roles`中替换全局切片；不足8条的类别以`INSUFFICIENT_CLASS_SAMPLES`失败关闭。
- [x] **Step 4:** 写校准错误分类失败测试，分别构造缺类、样本不足、零向量、非有限值和logit类别顺序不匹配。
- [x] **Step 5:** 实现结构化错误与双空间审计，并在endpoint calibration前调用；审计结果写入导出metadata。
- [x] **Step 6:** 运行两组测试并确认GREEN。

### Task 2: FastTrust严格路由

**Files:**
- Modify: `code/cvsrffi/muse_ssdg.py`
- Modify: `code/tests/test_muse_ssdg_routing.py`

**Interfaces:**
- Produces: `FastTrustRoute(hard, soft, candidate, no_identity, agreement, class_cap)`。
- Produces: `route_fasttrust(reliability, stable, evidence_probabilities, high_threshold, low_threshold, hard_max_fraction=0.25, identity_max_fraction=0.50)`。

- [x] **Step 1:** 写失败测试，证明高可靠但不稳定、三头不同意或超过逐类/全批cap的样本不能进入hard。
- [x] **Step 2:** 运行测试并确认现有`route_muse_reliability`不能满足严格交集。
- [x] **Step 3:** 实现确定性排序：先按reliability降序，再按样本索引稳定打破平局；hard总量不超过`floor(B×0.25)`，hard+soft/candidate不超过`floor(B×0.50)`，逐预测类配额采用同一公式。
- [x] **Step 4:** 加入空批、单类、并列可靠度和非有限输入测试并确认GREEN。

### Task 3: S1零身份图与U侧星地身份监督

**Files:**
- Modify: `code/SSDG/train_ssdg.py`
- Modify: `code/tests/test_muse_ssdg_train_integration.py`
- Modify: `code/tests/test_muse_ssdg_satellite.py`

**Interfaces:**
- `_compute_muse_unlabeled_losses`消费`FastTrustRoute`并返回`u_identity_selected_count`、`u_identity_loss`、`u_satellite_identity_selected_count`、`three_head_agreement_rate`和`prototype_update_count`。
- satellite loss仅为`F.cross_entropy(satellite_logits[hard_mask], pseudo[hard_mask])×lambda_u×0.68`，伪标签detach；mid/low不得进入。

- [x] **Step 1:** 写E1/E16/E17失败测试，断言E1–E16不调用local/prototype融合或temporal observe，identity/satellite/prototype更新均为0；E17才开放。
- [x] **Step 2:** 写U satellite失败测试，构造high但不稳定、mid和严格hard三类，仅严格hard产生卫星CE和梯度。
- [x] **Step 3:** 运行测试确认RED。
- [x] **Step 4:** 最小改造loss主链：S1在base domain/self/nuisance后提前返回零身份结果；S2+使用FastTrust；prototype与cross-RX只消费严格hard stable。
- [x] **Step 5:** 将每epoch首批identity gradient norm接入训练遥测，非首批记录`nan`以避免重复反向；边界epoch强制运行断言。
- [x] **Step 6:** 运行集成与卫星测试确认GREEN。

### Task 4: 速度与稳定性

**Files:**
- Modify: `code/SSDG/train_ssdg.py`
- Create: `code/tests/test_phase1_fasttrust_speed.py`

**Interfaces:**
- Parser增加`--muse_unlabeled_batch_size`、`--muse_fused_student_forward`、`--muse_lr_schedule fasttrust`。
- Produces: `_fasttrust_lr_scale(epoch)`，边界为E1线性warmup、E5=1、E6–160 cosine、E161–180 backbone×0.2、E181–200 backbone×0.05。
- Produces: `_forward_muse_student_views(model, strong_x, nuisance_x, ...)`，拼接一次前向并按batch边界拆分结构化输出。

- [x] **Step 1:** 写失败测试，断言U loader使用独立batch256并完整覆盖；batch128/384只改变step数，不丢样本。
- [x] **Step 2:** 写拼接前向等价测试，断言一次模型调用的拆分输出与两次确定性前向一致；分支关闭时不执行nuisance前向。
- [x] **Step 3:** 写LR边界和`max_grad_norm=5`launcher参数失败测试。
- [x] **Step 4:** 运行测试确认RED后实现独立U batch、拼接前向和逐epochLR更新；保留现有AMP与梯度裁剪路径。
- [x] **Step 5:** 接入`epoch_time_s`、U samples/s、forward样本数和CUDA峰值显存遥测并确认GREEN。

### Task 5: 16条矩阵launcher与评测闭环

**Files:**
- Create: `code/scripts/launch_phase1_adv3b02_fasttrust16_20260821.sh`
- Create: `code/tests/test_phase1_fasttrust_launcher.py`
- Modify: `configs/phase1_adv3b02_fasttrust16_s392002_20260821.json`

**Interfaces:**
- launcher读取矩阵并支持`--dry-run`、`--only=<candidate>`和`--gpu=<0..7>`。
- 每张GPU的A/B候选由同一GPU worker并发启动，单卡最多2个训练进程；不同GPU worker可并发。
- 每个候选只调用一次联合评测器，拆分生成clean和三种LEO的JSON/log；只有四场景完整才写`ARTIFACTS_COMPLETE`。

- [x] **Step 1:** 写launcher失败测试，执行dry-run并断言16条、每GPU2条、统一seed/E200、独立U batch、ADV3B02星地参数、不可覆盖目录和四场景输出。
- [x] **Step 2:** 写fake trainer/evaluator行为测试，覆盖训练失败、bundle失败不阻断评测、任一评测缺失失败、完整四场景成功和同卡最多2进程。
- [x] **Step 3:** 运行测试确认RED后实现launcher与矩阵参数映射。
- [x] **Step 4:** 运行`bash -n`、dry-run和行为测试确认GREEN。

### Task 6: 完成验证、审查与发布

**Files:**
- Modify: `automation_reports/CV-SincNet/phase1_adv3b02_fasttrust16_s392002_20260821/report.md`

**Interfaces:**
- 追踪表每项最终为`verified`或明确的非实施状态；发布只接受全部必要项`verified`。

- [x] **Step 1:** 运行FastTrust聚焦测试、既有MUSE联合回归、`py_compile`、`bash -n`和16条dry-run。
- [x] **Step 2:** 使用真实ADV3B02 checkpoint执行单batch、无target query的本地或N607 smoke，验证strict恢复和输出路径。
- [x] **Step 3:** 执行一次独立P0/P1正确性审查；只修复会导致真实实验跑错、越权、覆盖、误杀、不能启动或不能形成合法prediction的问题。
- [ ] **Step 4:** 更新追踪表、实际命令、验证结果、风险和release映射；显式stage本轮文件，提交、自动push并核对远端OID。
- [ ] **Step 5:** 创建单个release归档并做一次本地/远端SHA比较，远端编译与dry-run通过后启动8个GPU worker。
- [ ] **Step 6:** 启动后核验PID、CWD、run-root、GPU映射和日志增长；状态进入`RUNNING`。
