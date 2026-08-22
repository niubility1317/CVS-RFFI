# SAT-Anchor-SSL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前Phase1协议下实现SAT-Anchor-SSL，取消固定50%伪身份补齐，用全部`U_s`的clean-satellite配对关系和少量严格可信星地身份监督提升LEO鲁棒性，同时保持clean性能并降低训练开销。

**Architecture:** 在`MUSE/FastTrust`现有训练入口新增`SAT_ANCHOR`模式。冻结Core90教师和EMA教师只读clean `U_s`，由`V_cal`冻结类别条件confidence/margin阈值；学生对clean和必要的satellite行做一次拼接前向，训练期SimSiam投影头与clean KL锚定不进入部署推理。A5通过模型内零初始化低秩identity residual adapter和logit correction限制U侧梯度范围。

**Tech Stack:** Python、PyTorch、pytest、Bash、JSON。

**Spec:** `E:/codex/home/attachments/58a9e689-7563-4993-909a-9513e0d87975/pasted-text.txt`

## Global Constraints

- 严格保持`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`、source-only训练、`R_s∩R_t=∅`和`U_s`TX真值隐藏。
- 保持`LEO_WEAK`场景、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`及E1–40/E41–90/E91–200日程。
- 取消通用U clean hard/soft/candidate CE、U prototype更新、prototype/local/temporal证据、cross-RX attraction和nuisance regression。
- trusted U satellite CE以完整`B_U`为分母；没有可信样本时U身份梯度严格为0。
- U batch固定256且每epoch完整覆盖；训练加速不能靠丢弃U样本。
- 每个候选使用不可覆盖输出；训练完成后自动闭合clean和三种LEO弱信道评测。

---

### Task 1: 可信选择与V_cal校准

**Files:**
- Modify: `code/cvsrffi/muse_ssdg.py`
- Modify: `code/SSDG/train_ssdg.py`
- Test: `code/tests/test_sat_anchor_ssl_routing.py`

**Interfaces:**
- Produces: `calibrate_sat_anchor_thresholds(...) -> SATAnchorThresholds`
- Produces: `route_sat_anchor_trusted(...) -> SATAnchorRoute`
- Consumes: 冻结教师与EMA教师的class probability、source receiver/domain metadata。

- [ ] **Step 1: Write the failing tests**

测试必须证明：anchor/EMA不一致、confidence不足或margin不足均不会进入trusted；无样本通过时不补齐；class cap和class×receiver cap均确定性生效；`fill_to_fraction=0.5`只用于命名对照。

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest -q code/tests/test_sat_anchor_ssl_routing.py`
Expected: FAIL，因为SAT-Anchor路由与校准接口尚不存在。

- [ ] **Step 3: Implement minimal routing and calibration**

实现类别条件confidence/margin阈值、anchor/EMA几何均值融合、无填充trusted mask和可选class×receiver cap；所有排序只读取预测、score和source receiver/domain，不读取`U_s`TX。

- [ ] **Step 4: Run tests to verify GREEN**

Run: `pytest -q code/tests/test_sat_anchor_ssl_routing.py`
Expected: PASS。

### Task 2: 配对损失、clean锚定与加速前向

**Files:**
- Modify: `code/cvsrffi/muse_ssdg.py`
- Modify: `code/SSDG/train_ssdg.py`
- Test: `code/tests/test_sat_anchor_ssl_losses.py`
- Test: `code/tests/test_phase1_sat_anchor_speed.py`

**Interfaces:**
- Produces: `MUSETrainingHeads.sat_anchor_pair_loss(clean_z, sat_z)`。
- Produces: `_forward_sat_anchor_student_views(...)`，一次模型调用返回clean和选定satellite输出。

- [ ] **Step 1: Write failing behavior tests**

测试必须证明：pair loss使用真实clean/satellite张量且stop-gradient对称；anchor KL不产生teacher梯度；trusted satellite CE按完整`B_U`归一化；pair interval关闭时不生成全U satellite前向；拼接前向与分离前向等价。

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest -q code/tests/test_sat_anchor_ssl_losses.py code/tests/test_phase1_sat_anchor_speed.py`
Expected: FAIL，因为新损失和前向接口尚不存在。

- [ ] **Step 3: Implement minimal losses and one-call student forward**

学生clean与satellite行合并前向；pair step覆盖全部U satellite，非pair step只前向trusted satellite行。冻结教师与EMA教师均在AMP inference/no-grad下运行；关闭pair、anchor或trusted CE时对应张量和前向不创建。

- [ ] **Step 4: Run tests to verify GREEN**

Run: `pytest -q code/tests/test_sat_anchor_ssl_losses.py code/tests/test_phase1_sat_anchor_speed.py`
Expected: PASS。

### Task 3: A5零初始化identity adapter与U梯度作用域

**Files:**
- Modify: `code/model_dual_cvsincnet.py`
- Modify: `code/post_stage_common.py`
- Modify: `code/SSDG/train_ssdg.py`
- Test: `code/tests/test_sat_anchor_identity_adapter.py`

**Interfaces:**
- Produces: 可选`SatAnchorIdentityAdapter`，checkpoint state属于模型本体。
- Produces: U侧梯度掩码，只允许adapter与identity tail参数接收pair/trusted-satellite梯度。

- [ ] **Step 1: Write failing adapter tests**

测试必须证明：adapter零初始化时输出与旧模型一致；U目标对early backbone梯度为0而adapter/tail非0；L_s监督仍能更新完整模型；关闭adapter不改变旧checkpoint加载结构。

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest -q code/tests/test_sat_anchor_identity_adapter.py`
Expected: FAIL，因为adapter接口尚不存在。

- [ ] **Step 3: Implement minimal adapter and gradient scope**

在identity特征与分类logit之间增加低秩零初始化residual/correction；A5对U目标使用detach后的backbone feature进入adapter，其他候选保持完整U梯度或按矩阵关闭。

- [ ] **Step 4: Run tests to verify GREEN**

Run: `pytest -q code/tests/test_sat_anchor_identity_adapter.py`
Expected: PASS。

### Task 4: 8条最小矩阵、launcher与闭合评测

**Files:**
- Create: `configs/phase1_adv3b02_sat_anchor_ssl8_s392002_20260822.json`
- Create: `code/scripts/launch_phase1_adv3b02_sat_anchor_ssl8_20260822.sh`
- Test: `code/tests/test_phase1_sat_anchor_launcher.py`
- Modify: `automation_reports/CV-SincNet/phase1_adv3b02_sat_anchor_ssl8_s392002_20260822/report.md`

**Interfaces:**
- Produces: A0、A1、A2、A3_ADAPTIVE_NO_FILL、A3_FIXED_50_FILL、A3_PAIR_INTERVAL2、A4_CLASS_RX_CAP、A5_ADAPTER_TAIL。
- Produces: 每GPU一条、同seed392002、E200、U256、final-only、四场景自动评测。

- [ ] **Step 1: Write launcher tests first**

测试解析dry-run并断言8条唯一候选、GPU0–7各一条、固定协议/seed/U256、A0–A5单因素差异、四场景输出和不可覆盖路径。

- [ ] **Step 2: Verify RED, implement config/launcher, verify GREEN**

Run: `pytest -q code/tests/test_phase1_sat_anchor_launcher.py`
Expected: RED后GREEN。

### Task 5: 聚焦验证、发布与N607启动

**Files:**
- Modify: 本计划列出的全部实现、测试、配置、launcher和report。

**Interfaces:**
- Produces: Git提交、远端OID读回、唯一release归档、N607 run/log root、PID/GPU/log健康证据。

- [ ] **Step 1: Run focused local verification under `ssr-gpu`**

Run: SAT-Anchor新增测试、现有FastTrust协议/速度/launcher/训练集成测试、`py_compile`、`bash -n`、8条dry-run和`git diff --check`。

- [ ] **Step 2: Run one real-checkpoint no-query smoke and one P0/P1 review**

Smoke必须严格恢复Core90、query迭代=0、target truth读取=0，并验证至少一个optimizer step、有限梯度与前向计数。

- [ ] **Step 3: Commit, push and verify remote OID**

只stage本计划明确文件；不得stage既有未跟踪产物。

- [ ] **Step 4: N607 preflight, sync, dry-run and launch**

执行普通账户直连预检、资源/CWD/路径核对、唯一归档本地—远端SHA、远端编译和8条dry-run；通过后启动并立即核对PID/CWD/cmdline/GPU/log增长。
