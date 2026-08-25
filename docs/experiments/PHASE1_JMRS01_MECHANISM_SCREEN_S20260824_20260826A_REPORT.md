# PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A实验报告

## 当前状态

`LOCAL_VERIFIED`

这是JMRS01首个source-only单机制筛选实验。旧PA-M2.1实验及其产物保持只读，不重复、不覆盖。

## 候选与矩阵

- Run ID：`PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A`
- S0行：`M0,R1,R2,D1,P1,P2,S1`
- 已删除：`D2`线性/对数差分谱比值，原因是数据无已知符号与合法同符号跨时刻配对。
- 延期：`D3,I1,S1联合、S2三机制、P4残差gate`
- Core90：冻结，原始IQ路径不变。
- 协议：7个source receiver嵌套LORO；其余receiver内使用既有`L_s/V_select/V_cal`，held receiver只作audit。
- 场景：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`逐项保存。
- 统一预算：32维embedding、可训练参数不超过5万、同epoch/batch/optimizer。

## 预注册停止与晋级规则

- 只允许因query/target泄漏、错误角色/split/receiver/scene、输出碰撞、错误checkout、同一确定性系统异常或prediction无法闭合而技术停止。
- 不允许因中途性能低停止。
- 单机制须同时满足追踪表中的8项阈值才入池。
- 少于2个机制入池时，不启动两两联合；少于2个pairwise synergy的CI下界大于0时，不启动三机制联合。

## 版本与路径

- 分支：`codex/phase1-jmrs01-20260826`
- 设计起点：`2d833538d077e8bebdf8adaa70887732c7985565`
- 实验代码commit：`d17cf6fb8128b47f505fbd80e2fabfb7c8421284`
- 本地工作树：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\ccoi-pa-v1`
- 本地正式报告：`E:\type10-7\automation_reports\CV-SincNet\PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A\report.md`
- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- 远端输出根：`runs/phase1_jmrs01_20260826/PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A`
- 远端日志：`logs/phase1_jmrs01_20260826/PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A.out`
- release目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A_d17cf6fb`
- release归档：本地`E:\type10-7\releases\PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A_d17cf6fb.tar.gz`映射到远端项目`releases/`；仅做一次本地/远端SHA-256比较。
- GPU：0。2026-08-26预检时8张RTX3090均为0%利用率、显存1MiB；不干预无关进程。

## 精确发布命令

远端CWD固定为上述release目录，使用原项目的只读checkpoint/WiSig路径和全新run/log根：

```bash
env ROOT=<release目录> CHECKPOINT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs01_20260826 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_jmrs01_20260826 GPU=0 RUN_ID=PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A bash code/scripts/launch_phase1_jmrs01_20260826.sh
```

launcher第一步是真实checkpoint无query smoke，PASS后立即继续正式S0；之后独立scorer连接truth。预期owner、smoke、正式和score日志均位于JMRS01专属日志根。

## 输入与预期artifact

- WiSig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- Core90 checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 预期输出：`predictions.jsonl`、`run_manifest.json`、`training_history.json`、`mechanism_identity_stability.json`、`mechanism_receiver_probe.json`、`mechanism_loro_metrics.json`、`mechanism_clean_sat_consistency.json`、`mechanism_complementarity.json`、`mechanism_observability.json`、`mechanism_cost.json`、`mechanism_decision.json`。

## 当前验证

- 修改前相关历史测试：通过。
- TDD RED：机制模块、runner与scorer、launcher依次因实现缺失而失败，证明测试能够拦截缺失行为。
- TDD GREEN：JMRS01聚焦测试23项通过；覆盖D2拒绝、RC边界、DSQ零陷、PI低幅度、LORO隔离、prediction/truth闭合、fold-local probe/geometry、四场景和launcher调用顺序。
- 既有CCOI/PA回归：93项通过。
- Python语法：`jmrs01.py`、runner、scorer通过。
- dry-run：矩阵严格为`M0,R1,R2,D1,P1,P2,S1`，四场景完整，`target_or_query_access=false`。
- D2负测：CLI在创建输出前以“需要known transmitted symbols”拒绝。
- 一次P0/P1审查发现并修复：不同LORO fold的embedding坐标不能混合训练receiver probe；身份几何也必须在fold内计算后汇总。定点测试通过。
- 本地Git Bash路由：`FAILED`。请求的Git Bash被替换为`/bin/bash`和`/mnt/e/...`，按Windows执行规则停止；不把WSL结果当作Git Bash证据。release落到N607后执行一次远端shell编译。
- `ruff`：环境未安装，记为`NONBLOCKING`；未新增依赖。
- 真实checkpoint无query smoke：待执行。
- 正式实验：未启动。

## 已落地实现

- `code/cvsrffi/jmrs01.py`：统一机制契约；RC-Feature32、RC-Smooth16、MS-DSQ、MS-PI-1、MS-PI-12和Sham32；每分支32维、≤5万参数。
- R1：低秩有界校正、Core90高置信正确样本KL身份保持、可学习可靠性。
- R2：16项DCT低秩幅相曲线、幅度与相位状态联合估计、二阶平滑惩罚、分支专用校正视图。
- D1：shift=1/2/4/8双向商；原始幅度、log幅度和相位六类通道；分母floor、双端mask、幅值clip和有效频点coverage。
- P1/P2：加权二次趋势去除；一阶/二阶创新；幅度mask；4/8/16/32尺度的方差、MAD、自相关、三段PSD、差分能量、突变率、类Allan方差、三分位数、超额峰度和coverage。
- 统一损失：clean/satellite CE、精确clean-sat表示一致性、类条件receiver均值对齐、TX margin、R1身份保持、可靠性目标和R2平滑正则。
- `code/audit_phase1_jmrs01.py`：冻结Core90、四固定视图缓存、7折source receiver LORO、内层train/select/cal、held audit、不可覆盖输出、safe residual只读诊断和prediction先闭合。
- `code/score_phase1_jmrs01.py`：truth后连接；逐receiver/逐场景LORO、fold-local三类receiver probe、fold-local身份几何、互补性、coverage曲线、成本和8项入池决策。
- `code/scripts/launch_phase1_jmrs01_20260826.sh`：不可覆盖smoke/正式run/log，smoke通过后正式运行，再由独立scorer评分。

## 最终结果

待prediction闭合并由独立scorer连接truth后补充。`RUNNING`和日志增长不视为性能结果。
