# Hu Feature Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**实现Hu等人2024年基于特征分离的少样本跨接收机RFFI，并把论文未指定参数与实现选择明确隔离。

**Architecture:**`representation.py`形成I/Q+Welch PSD；`model.py`实现图6的共享注意力ResNet18和TX/RX分支；`losses.py`实现论文总损失；`finetune.py`仅使用TX标签。训练入口保持为后续工作，不嵌入CVS扩展层。

**Tech Stack:**Python、PyTorch、pytest。

**Spec:**`docs/superpowers/specs/2026-08-27-two-paper-rffi-reproduction-design.md`

## Global Constraints

- 输入固定`[B,2,256]`或融合后`[B,3,256]`。
- 论文明确值：30 samples/TX、Adam lr=0.005、batch size=256、25 samples/TX fine-tuning。
- λ、epoch、seed、增强概率和冻结策略一律记录为implementation choice。
- PDF只写入用户本地目录，不纳入Git。

---

### Task 1:建立包与输入表示

**Files:**
- Create:`paper_reproduction/hu_feature_separation_2024/__init__.py`
- Create:`paper_reproduction/hu_feature_separation_2024/PAPER_SOURCE.md`
- Create:`paper_reproduction/hu_feature_separation_2024/README.md`
- Create:`paper_reproduction/hu_feature_separation_2024/configs/manysig_paper_choices.json`
- Create:`paper_reproduction/hu_feature_separation_2024/representation.py`
- Create:`tests/test_hu_feature_separation_2024.py`

- [ ] 写入融合形状、Welch PSD确定性和无效输入失败测试。
- [ ] 运行测试确认失败。
- [ ] 实现无状态融合表示与目录配置。
- [ ] 重跑测试。

### Task 2:实现特征分离网络与损失

**Files:**
- Create:`paper_reproduction/hu_feature_separation_2024/model.py`
- Create:`paper_reproduction/hu_feature_separation_2024/losses.py`
- Modify:`tests/test_hu_feature_separation_2024.py`

- [ ] 写入三路输入、TX/RX输出形状、相关性损失、熵项和总损失梯度的失败测试。
- [ ] 确认失败。
- [ ] 最小实现注意力ResNet18、双分支与损失。
- [ ] 运行完整Hu模块测试。

### Task 3:实现只用TX标签的微调步骤与本地镜像

**Files:**
- Create:`paper_reproduction/hu_feature_separation_2024/finetune.py`
- Modify:`tests/test_hu_feature_separation_2024.py`
- Create:`E:\type10-7\paper_reproduction\gaskin_tweak_2023\...`
- Create:`E:\type10-7\paper_reproduction\hu_feature_separation_2024\...`

- [ ] 写入不要求RX标签且只更新指定参数的失败测试。
- [ ] 确认失败。
- [ ] 实现微调步骤并通过测试。
- [ ] 在本地用户目录复制两个PDF和已验证代码，确认文件可读且与Git代码一致。
