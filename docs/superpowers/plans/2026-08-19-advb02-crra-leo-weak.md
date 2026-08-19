# ADVB02 CRRA-S LEO弱信道增强实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现`ADVB02_CRRA_S_LEO_WEAK_E200`，在固定Phase1角色协议下以三种`leo_weak`场景训练与独立逐场景测试，修复训练完成后历史非技术终态阻断测试的问题。

**Architecture:** CRRA-S只位于身份路径共享Sinc/IQ后的时间特征。它由每对I/Q的收缩白化、FiLM条件低秩时域残差、源域多中心支持门和q条件的时间/频率/PA可靠度融合组成。域支路和PA原始特征保持不受CRRA重构影响。训练器提供唯一一致性KL权重、E1–16/E17–46/E47+日程、CRRA专用学习率组及不阻断合法后测的终态语义。

**Tech Stack:** Python 3、PyTorch、pytest、现有CV-SincNet/SSDG、Git Bash、`ssr-gpu`、N607。

**Spec:** `docs/superpowers/specs/2026-08-19-advb02-crra-leo-weak-design.md`。

## 全局约束

- 训练和最终测试只允许`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，不得回退到`mixed_orbit`。
- 角色固定为`0.07/0.63/0.15/0.15`，seed固定为`392034`；无target访问或CRRA-C。
- 实现必须测试先行：先增加失败测试、确认失败，再修改生产代码。
- 最终launcher自动执行clean和三个LEO逐场景独立测试，并保存对应遥测。

### Task 1: 规格追溯和基线

**Files:**
- Create: `docs/superpowers/specs/2026-08-19-advb02-crra-leo-weak-design.md`
- Create: `docs/superpowers/plans/2026-08-19-advb02-crra-leo-weak.md`
- Modify: `analysis/advb02_crra_traceability.md`

- [ ] 更新追溯表，将新LEO协议、CRRA-S完整结构、唯一KL和后测终态逐条映射到文件与测试。
- [ ] 在`ssr-gpu`中运行现有CRRA聚焦测试作为基线，并记录任何既有失败。

### Task 2: 每对I/Q的CRRA-S核心

**Files:**
- Modify: `code/crra.py`
- Modify: `code/tests/test_crra_adapter.py`

- [ ] 先写失败测试：逐对alpha、FiLM零初始化近似恒等、多中心`exp(-d²/tau)`支持门、q停止梯度。
- [ ] 实现逐对alpha白化和FiLM低秩残差；保留输入均值和数值稳定性。
- [ ] 实现源域多中心对角Mahalanobis统计，仅允许source mask更新。
- [ ] 运行核心与现有CRRA测试。

### Task 3: 身份路径融合与分支隔离

**Files:**
- Modify: `code/model.py`
- Modify: `code/model_dual_cvsincnet.py`
- Modify: `code/tests/test_advb02_crra_model.py`

- [ ] 先写失败测试：身份路径启用、域路径原始、PA旁路、时间/频率/PA可靠度为有效凸权重。
- [ ] 移除新路径中的均匀频率CRRA层，接入q条件残差可靠度融合。
- [ ] 从双模型向身份adapter传递域中心数量和新遥测，保持旧checkpoint关闭CRRA时兼容。
- [ ] 运行模型集成和相邻稳定性测试。

### Task 4: LEO协议、训练损失和终态

**Files:**
- Modify: `code/cvsrffi/crra_training.py`
- Modify: `code/SSDG/train_ssdg.py`
- Modify: `code/tests/test_crra_protocol_negatives.py`
- Modify: `code/tests/test_crra_training_plumbing.py`
- Modify: `code/tests/test_crra_mixed_orbit_metadata.py`

- [ ] 先写失败测试：只接受三种LEO弱场景、拒绝target adapter、唯一KL、CRRA学习率组和非正式合法后测终态。
- [ ] 实现`leo_weak`族验证、唯一KL权重、E47后的0.25 CRRA学习率组和诊断式P0/P1终态。
- [ ] 验证LEO信道元数据与同一次视图绑定，覆盖三种场景。
- [ ] 运行聚焦协议、训练和元数据测试。

### Task 5: 独立评估遥测

**Files:**
- Create: `code/cvsrffi/crra_evaluation.py`
- Modify: `code/tools/eval_cvs_checkpoint_sat_channel.py`
- Create or Modify: `code/tests/test_crra_evaluation.py`

- [ ] 先写失败测试：评估不更新模型状态，输出每种LEO场景的修正、支持、泄漏和可靠度统计。
- [ ] 实现无梯度遥测归纳以及clean-satellite距离、同类跨域半径。
- [ ] 扩展独立评估器的JSON，保持现有clean/各场景准确率字段兼容。
- [ ] 运行评估器单元测试及checkpoint无query smoke。

### Task 6: 训练与测试发布器

**Files:**
- Create: `code/scripts/launch_phase1_advb02_crra_leo_weak_20260819.sh`
- Create: `automation_reports/CV-SincNet/phase1_advb02_crra_leo_weak_20260819/report.md`
- Modify: `analysis/advb02_crra_traceability.md`
- Test: `code/tests/test_phase1_advb02_crra_leo_weak_launcher.py`

- [ ] 先写失败测试：冻结seed、四角色、Core90超参数、LEO日程、CRRA权重、三个独立测试命令和不可覆盖run root。
- [ ] 实现单候选launcher和最小预登记报告；训练成功且checkpoint存在时必跑测试，即使历史P0/P1仅诊断失败。
- [ ] 本地进行launcher语法/参数审查和真实checkpoint无query smoke。

### Task 7: 本地验证、版本化与N607发布

- [ ] 运行全部聚焦测试和必要的相邻回归；记录结果。
- [ ] 只stage本计划列出的正式文件，提交并推送；独立核对远端分支OID与本地HEAD一致。
- [ ] 对N607进行直接只读preflight；打包单一release归档并进行一次本地/远端SHA核对，远端编译后启动唯一run ID。
- [ ] 启动后核查PID/CWD/cmdline/GPU/log增长一次；训练完成后保留逐场景评估、同row结果和下一步判断。
