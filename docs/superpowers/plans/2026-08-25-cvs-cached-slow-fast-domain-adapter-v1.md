# CVS Cached Slow-Fast Domain Adapter V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结ADV3B02和冻结原型上实现source-only Phase1.5慢参数学习、Phase2 4～24维快参数适配和9-row真实诊断。

**Architecture:** 新模块在正式`z_id`之后工作，不修改ADV3B02主干。地面缓存训练与Phase2部署bundle严格分离；runner复用现有validated support、prediction封存和truth-last评分合同。

**Tech Stack:** Python 3、PyTorch、NumPy、pytest、现有ADV3B02和Stage2矩阵工具。

**Spec:** `docs/CVS_CACHED_SLOW_FAST_DOMAIN_ADAPTER_V1_DESIGN_20260825.md`

## Global Constraints

- `项目.md`和`p2_min_v1`数据/query边界优先。
- Phase2 bundle不得包含source样本、source特征缓存或样本级派生状态。
- 实际特征维度由冻结原型宽度决定；当前真实宽度为160。
- K=1固定回退`DA0_REG0`；不得由同一物理样本数学view增加K。
- 首次实验固定receiver`20-1`、seed`392002`、`K10/new10`和三种LEO weak场景。
- 所有生产代码先有一个因缺失行为而失败的测试，再写最小实现。

---

### Task 1: Slow-Fast核心和地面bundle

**Files:**
- Create: `code/cvsrffi/slow_fast_adapter.py`
- Create: `code/cvsrffi/slow_fast_bundle.py`
- Test: `tests/test_slow_fast_adapter.py`
- Test: `tests/test_slow_fast_bundle.py`

**Interfaces:**
- `SlowFastCandidate`枚举三个候选。
- `SlowFastAdapterState`保存慢参数和快参数，不保存source行。
- `apply_slow_fast(features, state) -> Tensor`。
- `save_slow_fast_bundle/load_slow_fast_bundle_strict`。

- [x] 写160维shape、手算COMMON_SHIFT、16/24快参数、错误宽度和bundle禁止source字段的失败测试。
- [x] 在`ssr-gpu`运行测试，确认因模块缺失RED。
- [x] 实现最小核心与严格bundle schema。
- [x] 运行聚焦测试并确认GREEN。

### Task 2: Phase1.5缓存和慢参数训练

**Files:**
- Create: `code/cvsrffi/slow_fast_cache.py`
- Create: `code/cvsrffi/slow_fast_phase15.py`
- Create: `code/cvsrffi/slow_fast_objectives.py`
- Test: `tests/test_slow_fast_phase15.py`

**Interfaces:**
- `GroundFeatureCache`严格保存`z_id/label/receiver/day/scene/physical_sample_id/view`。
- `fit_common_shift_basis`和`train_slow_fast_basis`只消费source cache与冻结原型。
- 输出bundle只含聚合慢参数和配置。

- [x] 写角色拒绝、物理ID隔离、类中心化SVD、pair、floor和区间trust失败测试。
- [x] 确认RED后实现缓存验证和训练循环。
- [x] 用合成域偏移验证COMMON闭式基以及两类FAST候选降低冻结原型CE且不改变原型。

### Task 3: Phase2 support-only选择和query隔离

**Files:**
- Create: `code/cvsrffi/slow_fast_selection.py`
- Create: `code/cvsrffi/stage2_slow_fast_runner.py`
- Test: `tests/test_slow_fast_selection.py`
- Test: `tests/test_stage2_slow_fast_runner.py`

**Interfaces:**
- `select_support_only_state`执行K≥2 LOO/lambda选择；K=1返回DA0。
- runner严格按base/bundle→support→冻结DA1→query顺序，输出两状态prediction。

- [x] 写K1零更新、LOO回退、非法source/query字段、query顺序不变和状态只读失败测试。
- [x] 确认RED后实现选择器和runner。
- [x] 复用现有scorer格式运行同rowfixture GREEN。

### Task 4: CLI、矩阵和真实smoke

**Files:**
- Create: `code/scripts/train_slow_fast_phase15.py`
- Create: `code/scripts/run_stage2_slow_fast.py`
- Create: `code/scripts/run_stage2_slow_fast_matrix.py`
- Create: `code/scripts/smoke_slow_fast_no_query.py`
- Create: `configs/stage2_slow_fast_diag9_s392002_20260825.json`
- Test: `tests/test_stage2_slow_fast_matrix.py`

- [x] 写9-row笛卡尔积、不可覆盖output和固定K/scene/seed测试并确认RED。
- [x] 实现薄CLI和矩阵调度，运行GREEN及邻近Meta-Adapter回归。
- [ ] 使用真实ADV3B02和冻结原型完成一次无query smoke，核对160维与4/16/24快参数。

### Task 5: 审查、发布和评分

**Files:**
- Update: `analysis/cached_slow_fast_domain_adapter_traceability_20260825.md`
- Create: `automation_reports/CV-SincNet/cvs_cached_slow_fast_diag9_s392002_20260825_r1/report.md`
- Mirror: `docs/experiments/cvs_cached_slow_fast_diag9_s392002_20260825_r1_report.md`

- [ ] 完成一次仅限直接P0/P1的一次独立审查；若有问题只做一次定点复审。
- [ ] 精确stage、commit、push并回读remote OID。
- [ ] 执行N607 preflight、单release归档SHA核对和远端编译。
- [ ] 启动9-row prediction，核验PID/CWD/cmdline/GPU/log增长。
- [ ] prediction闭合后独立scorer连接truth，按预注册门槛给出晋级或科学失败结论。
