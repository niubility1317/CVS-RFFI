# Gaskin Tweak Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**实现Gaskin等人2023年Tweak的论文忠实、可测试PyTorch复现，并提供明确标记的WiSig替代数据配置。

**Architecture:**`model.py`只负责1D metric encoder；`triplet.py`负责batch-hard triplet选择与损失；`calibration.py`持有无梯度的centroid/radius并执行论文三种决策；`metrics.py`从预先形成的决策分数计算指标。训练/数据适配不混入该核心实现。

**Tech Stack:**Python、PyTorch、pytest、scikit-learn（可选AUROC）。

**Spec:**`docs/superpowers/specs/2026-08-27-two-paper-rffi-reproduction-design.md`

## Global Constraints

- 仅使用`[B,2,128]`IQ输入、12维embedding、triplet margin=0.1和无权重校准。
- 纸面未提供的学习率通过配置记录为implementation choice。
- WiSig运行必须标记`METHOD_REPRODUCTION_ON_SURROGATE_DATA`。
- PDF只写入用户本地目录，不纳入Git。

---

### Task 1:建立包与论文来源说明

**Files:**
- Create:`paper_reproduction/gaskin_tweak_2023/__init__.py`
- Create:`paper_reproduction/gaskin_tweak_2023/PAPER_SOURCE.md`
- Create:`paper_reproduction/gaskin_tweak_2023/README.md`
- Create:`paper_reproduction/gaskin_tweak_2023/configs/wisig_surrogate.json`

- [ ] 写入目录和数据边界测试。
- [ ] 运行测试确认其先失败。
- [ ] 创建最小目录元数据与配置。
- [ ] 重跑测试并提交本任务。

### Task 2:实现metric encoder与triplet学习

**Files:**
- Create:`paper_reproduction/gaskin_tweak_2023/model.py`
- Create:`paper_reproduction/gaskin_tweak_2023/triplet.py`
- Create:`tests/test_gaskin_tweak_2023.py`

- [ ] 写入embedding形状、输入拒绝、hardest positive/negative和margin loss的失败测试。
- [ ] 运行测试确认失败原因是缺少实现。
- [ ] 以最小PyTorch实现使测试通过。
- [ ] 运行全部Tweak测试。

### Task 3:实现校准、决策与指标

**Files:**
- Create:`paper_reproduction/gaskin_tweak_2023/calibration.py`
- Create:`paper_reproduction/gaskin_tweak_2023/metrics.py`
- Modify:`tests/test_gaskin_tweak_2023.py`

- [ ] 写入centroid/radius、closed-set两分支、open-set、参数无更新和5-trial均值的失败测试。
- [ ] 确认测试失败。
- [ ] 实现无梯度校准和指标函数。
- [ ] 运行Tweak测试、编译检查和配置解析检查。
