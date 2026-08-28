# SF-TAPFT-PACE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在D0 Compact安全底座上实现受保护的Norm容量升级、support-only cross-fitted 6参数零和偏置校准、按需升级状态机和OARC资源回执，并完成E0–E3最小实验。

**Architecture:** 保留既有SF-TAPFT部署runner和delta-only加载路径。适配阶段先形成D0教师状态，再仅对预登记Norm和head执行120步保护训练；最终embedding只计算一次，4-fold仅重拟合head并产生OOF logits，随后拟合零和类别偏置。广义prefix cache按最早可训练Norm选择边界，query路径只消费最终delta，不新增推理分支。

**Tech Stack:** Python、PyTorch、pytest、JSON配置、N607 CUDA运行环境。

**Spec:** `E:\codex\home\attachments\371afff5-5f16-4be4-81a3-e14fef0b10ca\pasted-text.txt`

## Global Constraints

- Phase2固定为`p2_min_v1/VALIDATED_ONCE`，support和query物理ID不交，query只测试。
- 旧6类K=10，共60条support；本轮不注册新类，只报告`DA0_REG0`和`DA1_REG0`。
- E0–E3固定`lambda_tail=0.03`、`lambda_preserve=0.10`、阶段B 120步、bias-only校准40步。
- 晋级要求BA、floor不低于E0，任一类别下降不超过5pp，NLL不高于E0+0.02，warm-resident中位数不超过60秒，delta不超过16KB。
- 禁止HardPair、Adapter、完整t3、frequency/domain更新、EMA和query驱动选择。

---

### Task 1: 配置、损失与阶段状态

**Files:**
- Modify: `code/cvsrffi/target_only_progressive_adapt.py`
- Test: `tests/test_sf_tapft_pace.py`

**Interfaces:**
- Produces: `stable_support_weights(...)`、`stable_preservation_kl(...)`、PACE配置字段和support-only风险摘要。

- [ ] **Step 1: 写失败测试**：覆盖稳定权重、Top2类别尾部、KL只作用于稳定样本、非法步数/权重拒绝。
- [ ] **Step 2: 运行定点测试并确认因缺少PACE接口失败。**
- [ ] **Step 3: 实现最小损失和配置校验**：稳定度为D0正确类概率乘归一化margin，保持损失使用D0教师概率，教师张量全部detach。
- [ ] **Step 4: 运行定点测试并确认通过。**

### Task 2: 广义time-Norm suffix cache

**Files:**
- Modify: `code/cvsrffi/target_only_progressive_adapt.py`
- Test: `tests/test_sf_tapft_pace.py`

**Interfaces:**
- Produces: `encode_trainable_suffix_prefix(model, support, earliest_trainable_node)`和`CompactTimeNormSuffix`。

- [ ] **Step 1: 写失败测试**：分别验证`t2.norm`和`time_fuse.1`边界的logit、梯度与完整路径一致，并断言suffix不持有完整模型引用。
- [ ] **Step 2: 运行测试并确认接口缺失失败。**
- [ ] **Step 3: 实现边界捕获和suffix重放**：缓存最早Norm输入、冻结frequency/domain/fusion尾部和identity head辅助量；只复制所需time suffix。
- [ ] **Step 4: 运行测试并确认等价性通过。**

### Task 3: PACE阶段B与bias-only cross-fit

**Files:**
- Modify: `code/cvsrffi/target_only_progressive_adapt.py`
- Modify: `code/cvsrffi/target_only_progressive_runner.py`
- Test: `tests/test_sf_tapft_pace.py`
- Test: `tests/test_target_only_progressive_runner.py`

**Interfaces:**
- Produces: D0教师快照、120步受保护扩展、`fit_support_oof_head_bias(...)`、带`head_bias`的delta v3。

- [ ] **Step 1: 写失败测试**：阶段B前早层Norm不变、阶段B后只许可Norm变化；OOF fold组不交；bias严格零和；delta加载前后logit一致。
- [ ] **Step 2: 运行测试并确认预期失败。**
- [ ] **Step 3: 实现阶段切换和保护损失**：阶段B总损失为CE+0.5 LOO-proto+1e-4 L2-SP+0.03 tail+0.10 preserve。
- [ ] **Step 4: 实现head-only OOF和40步零和偏置**：每fold只训练head，最终bias减去均值后写入head；不重跑backbone。
- [ ] **Step 5: 扩展delta schema和严格加载器**：保存`head_bias`，旧v1/v2继续只读兼容。
- [ ] **Step 6: 运行定点及回归测试并确认通过。**

### Task 4: 状态机、OARC与E0–E3矩阵

**Files:**
- Modify: `code/cvsrffi/target_only_progressive_runner.py`
- Modify: `code/scripts/run_sf_tapft_slim_matrix_row.py`
- Create: `configs/stage2_sf_tapft_pace_e0_e3_s392002_20260828.json`
- Test: `tests/test_sf_tapft_pace.py`
- Test: `tests/test_sf_tapft_p1_compact_matrix.py`

**Interfaces:**
- Produces: `BASE_MODEL→D0_COMPACT_ADAPT→SUPPORT_RISK_CHECK→SAFE_EXPAND→HEAD_BIAS_CALIBRATION→COMMIT/ROLLBACK`receipt和OARC字段。

- [ ] **Step 1: 写失败测试**：验证E0–E3冻结矩阵、禁止项、风险触发、失败回滚、元素/cache/delta/forward/head-only计数。
- [ ] **Step 2: 运行测试并确认预期失败。**
- [ ] **Step 3: 实现状态机和最小OARC**：记录wall-clock、RSS、CUDA allocated/reserved、cache、optimizer状态、完整forward、suffix forward/backward、head-only steps、NaN/Inf和非许可变化；能量/温度没有传感器时写`NOT_CAPTURED`。
- [ ] **Step 4: 实现E0–E3配置并运行全部聚焦测试。**

### Task 5: N607最小发布、评分与报告

**Files:**
- Create: `automation_reports/CV-SincNet/<run-id>/report.md`并镜像到`docs/experiments/`
- Update: `docs/experiments/stage2_sf_tapft_pace_20260828_traceability.md`

**Interfaces:**
- Consumes: 已提交代码/config、合法未暴露capsule或明确工程回放边界。
- Produces: E0–E3 support artifact、prediction、truth-last同row评分、资源证据和最终Git发布。

- [ ] **Step 1: 本地真实checkpoint无query smoke并完成一次P0/P1审查。**
- [ ] **Step 2: 创建最小预登记报告，提交并推送代码/config。**
- [ ] **Step 3: N607 preflight、单归档SHA核对、远端编译、按GPU容量启动。**
- [ ] **Step 4: 核对PID/CWD/cmdline/GPU/log增长并等待prediction闭合。**
- [ ] **Step 5: 独立scorer连接truth，生成逐类准确率、NLL、ECE、floor、配对变化和OARC结果。**
- [ ] **Step 6: 更新报告和追踪表，只stage本轮文件，提交、自动push并核对远端OID。**
