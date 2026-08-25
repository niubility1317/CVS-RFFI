# PHASE1 JMRS01 Mechanism Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development when policy permits; otherwise use superpowers:executing-plans and execute each task inline.

**Goal:** 在冻结Core90且完全source-only的条件下，分别检验接收机校正、无符号多尺度谱商和相位创新是否降低receiver/channel变化，同时保留TX身份间隔，并仅在单机制通过预注册门槛后进入联合阶段。

**Architecture:** Core90原始IQ路径保持不变。R1在冻结`z_id`上做有界低秩校正；R2仅向自身分支提供低秩平滑幅相校正视图；D1从未知符号IQ提取带分母屏蔽与裁剪的多尺度双向谱商；P1/P2提取幅度门控的一阶/二阶相位创新；S1提供同容量sham。每个机制独立输出32维表示、分类logits和可观测性，不做早期拼接，不训练联合gate。

**Tech Stack:** Python、PyTorch、NumPy、scikit-learn、pytest；现有CV-SincNet数据与Core90加载接口；N607单归档发布。

---

## Task 1：固化可追踪设计和实验边界

**Files:**
- Create: `docs/CVS_PHASE1_JMRS01_TRACE_20260826.md`
- Create: `docs/experiments/PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A_REPORT.md`

1. 把指导中的每一项映射为接受、修正、延期或否定。
2. 明确删除D2：数据无已知符号与同符号跨时刻配对，配置层必须拒绝。
3. 冻结S0矩阵、source-only嵌套LORO、统一预算、输出artifact和停止条件。
4. 记录当前分支、基线测试和不可覆盖run ID。

## Task 2：TDD实现机制提取器和统一契约

**Files:**
- Create: `code/tests/test_jmrs01.py`
- Create: `code/cvsrffi/jmrs01.py`

1. 先写失败测试：合法行集合、D2拒绝、参数预算、输出shape/有限值、RC边界、DSQ零陷稳定、PI幅度覆盖、sham可复现、损失有限。
2. 运行聚焦测试并确认因模块缺失而失败。
3. 最小实现R1/R2/D1/P1/P2/S1及统一`MechanismOutput`。
4. 运行测试至通过，再做小步重构。

## Task 3：TDD实现协议、runner和独立scorer

**Files:**
- Create: `code/tests/test_phase1_jmrs01_runner.py`
- Create: `code/tests/test_phase1_jmrs01_scorer.py`
- Create: `code/audit_phase1_jmrs01.py`
- Create: `code/score_phase1_jmrs01.py`

1. 先写失败测试：source角色白名单、target/query拒绝、外层held receiver隔离、prediction先闭合后truth评分、不可覆盖输出、七个S0行、四场景逐项输出。
2. 实现冻结Core90特征缓存、7折LORO训练、统一指标和prediction-only产物。
3. 实现独立scorer，输出稳定性、receiver probe、LORO、clean-sat一致性、互补性、可观测性、成本和决策JSON。
4. 运行聚焦测试至通过。

## Task 4：TDD实现launcher与最小真实smoke

**Files:**
- Create: `code/tests/test_phase1_jmrs01_launcher.py`
- Create: `code/scripts/launch_phase1_jmrs01_20260826.sh`

1. 先写失败测试：run目录/log不可覆盖、固定CWD、真实checkpoint smoke先于正式运行、D2无法从参数进入。
2. 实现单run owner launcher和系统技术失败状态。
3. 在本地完成语法、聚焦测试和历史回归。

## Task 5：发布与N607实验

1. 进行一次仅针对会跑错、越权、覆盖、误杀、无法启动或无法产出prediction的P0/P1审查。
2. 提交并自动push，核对远端分支OID等于本地HEAD。
3. 生成一个release归档，仅比较一次本地/远端SHA；远端编译一次。
4. 运行一次真实checkpoint无query smoke；PASS后立即启动唯一新run。
5. 检查一次PID/CWD/cmdline/GPU/log增长，不干预无关进程。

## Task 6：评分、结论和GitHub完整报告

1. 等待prediction闭合；独立scorer连接truth。
2. 检查日志、NaN/OOM/Traceback、每行/每receiver/每场景数据完整性。
3. 按预注册阈值决定哪些机制入池；未通过则不启动S1/S2。
4. 补全方法、落地文件、实验数据、问题、否定项和下一步。
5. 精确stage报告与必要代码，提交push并核对远端OID。
