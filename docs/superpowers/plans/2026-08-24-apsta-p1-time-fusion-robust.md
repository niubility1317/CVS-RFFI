# APSTA-P1 Time Fusion Robust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建并验证只用合法target support、真实更新ADV3B02时间/融合非分类层、以LOO最差类风险安全选择checkpoint的Stage2-B方法。

**Architecture:** 新建独立APSTA runtime，保留现有CAPTA-P0和TIME_FUSION_V1路径不变；runner复用既有严格checkpoint、support/query和row-binding加载器；matrix launcher执行15个同rowprediction后调用既有truth-last paired scorer，并用独立aggregator生成诊断。

**Tech Stack:** Python、PyTorch、NumPy、pytest、ADV3B02 checkpoint、现有Stage2-B loaders/scorer。

**Spec:** `docs/superpowers/specs/2026-08-24-apsta-p1-time-fusion-robust-design.md`

## Global Constraints

- Phase2只读`p2_min_v1/VALIDATED_ONCE`固定LEO IQ、合法support标签、冻结原型/映射、checkpoint和配置。
- 不得读取source/clean/cache或query truth/role/quota，不得新增可训练/持久分类头。
- Query打开前完成训练与checkpoint选择，query只读且逐样本面对全部冻结注册类。
- 只更新`t3+t_proj+fuse`，其余参数、buffer、CosFace头和ground prototypes不变。
- checkpoint固定`0/10/30/100/300`，选择只使用support robust LOO证据。

---

### Task 1: Robust support objectives

**Files:**
- Create: `code/cvsrffi/stage2_apsta_time_robust.py`
- Test: `tests/test_stage2_apsta_time_robust.py`

**Interfaces:**
- Produces: `anchored_loo_objective(features, targets, frozen_prototypes, scale, config)`。

- [ ] 写失败测试：手算两类LOO严格排除自身，弱类损失升高时tail风险升高，类别置换保持结果。
- [ ] 运行聚焦测试确认因接口缺失而RED。
- [ ] 实现NumPy2/Torch2.1安全输入、锚定LOO、per-class CE、tail和topology。
- [ ] 运行聚焦测试确认GREEN。

### Task 2: Multi-step partial-backbone adaptation

**Files:**
- Modify: `code/cvsrffi/stage2_apsta_time_robust.py`
- Test: `tests/test_stage2_apsta_time_robust.py`

**Interfaces:**
- Produces: `ApstaConfig`、`ApstaAudit`、`adapt_on_target_support`、`predict_query_read_only`。

- [ ] 写失败测试：只有`t3+t_proj+fuse`真实改变，head/prototypes/nonselected/buffer不变，训练trace到300步检查点。
- [ ] 写失败测试：安全选择会选support robust改善状态，有害状态回退step0，源码/API无source/query/可训练头表面。
- [ ] 实现AdamW训练、LOO/tail/topology/L2-SP、checkpoint内存快照与Pareto选择。
- [ ] 实现冻结teacher/student只读query分数和query状态不变审计。
- [ ] 运行聚焦及原late-block邻近回归。

### Task 3: Protocol-bound runner

**Files:**
- Create: `code/scripts/run_stage2_apsta_p1.py`
- Test: `tests/test_stage2_apsta_row_binding.py`

**Interfaces:**
- Consumes: 既有late-block `_read_context/_read_row_binding/_load_support_only/_load_query_received_iq/_load_prototypes/_exact_adv3b02`。
- Produces: `smoke`和`run-row`CLI。

- [ ] 写失败测试：smoke无query参数；support/context与query row四句柄错配时query IQ打开前失败。
- [ ] 写失败测试：prediction同时保存selected/student/teacher scores和完整审计，但无truth/role。
- [ ] 实现CLI并运行RED→GREEN。

### Task 4: Target5 launcher and diagnostic aggregation

**Files:**
- Create: `code/scripts/run_stage2_apsta_target5_matrix.py`
- Create: `code/scripts/summarize_stage2_apsta_target5.py`
- Create: `configs/stage2b_apsta_p1_target5_s713101_20260824.json`
- Test: `tests/test_run_stage2_apsta_target5_matrix.py`
- Test: `tests/test_summarize_stage2_apsta_target5.py`

**Interfaces:**
- Produces: 15份DA1 prediction、15份paired score、matrix summary和truth-last diagnostic summary。

- [ ] 写失败测试：配置穷尽式schema、15个唯一Target5 row、不可覆盖输出根。
- [ ] 写失败测试：aggregator给出row/class/scenario指标、selected step、disagreement和全矩阵mean/floor。
- [ ] 实现launcher和aggregator并运行GREEN。

### Task 5: Local closure and release

**Files:**
- Update: `analysis/apsta_p1_traceability.md`
- Create: `automation_reports/CV-SincNet/adv3b02_stage2b_apsta_p1_t5_s713101_20260824_v1/report.md`

- [ ] 运行全部新增测试与late-block/CAPTA邻近回归、`py_compile`和`git diff --check`。
- [ ] 运行真实checkpoint无query smoke，回读训练参数/改变状态/selected checkpoint/query=0。
- [ ] 完成唯一一次独立P0/P1审查；直接P0/P1修复后最多一次定点复审。
- [ ] 精确stage、commit、自动push并独立核对远端OID。

### Task 6: N607 Target5 and publication

**Files:**
- Update: Git镜像及根目录同名`report.md`

- [ ] N607只读preflight，创建不可覆盖run根，单release归档SHA对比，远端编译和无query smoke。
- [ ] GPU0启动15-row矩阵，核对PID/CWD/cmdline/GPU/log增长，短连接监控到artifact闭合。
- [ ] 独立scorer连接truth，运行diagnostic aggregator并判断`+1.0pp/+0.5pp`门槛。
- [ ] 更新报告与追溯表，提交、push并回读GitHub远端OID。

## Plan self-review

- 覆盖复盘的P1立即启动、真实梯度、LOO、tail、topology、L2-SP、多步选择、teacher保留和诊断要求。
- 可训练目标头、多域原型包明确rejected；物理一致性、class/sample gate、频率第二候选和meta-training明确deferred。
- 未包含`TODO/TBD`占位；接口、文件、状态和验证路径一致。
