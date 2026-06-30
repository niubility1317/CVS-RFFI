# Stage2 OPGAC V4控制面更新报告

## 基本信息

| 字段 | 内容 |
|---|---|
| 时间 | 2026-07-01 01:13 Asia/Hong_Kong |
| 操作 | 本地V4自动化控制面优化 |
| 目标 | 将阶段二优先路线调整为OPGAC-first，并固定阶段二模型基底为`JREF_C9_MULTICOMP_M2_E220` |
| 远端动作 | 未访问N607，未SCP，未启动实验 |

## 已读取控制文件

| 文件 | 用途 |
|---|---|
| `AGENTS.md` | 安全、环境、Git、N607和报告规则 |
| `项目.md` | CVS场景、Stage2-A/B/C、OLD80_FIRST和指标声明边界 |
| `tools/optimizer_control_manifest.md` | V4控制面读取顺序 |
| `automation_reports/CV-SincNet/automation_prompt_backups/20260615_001820_stage2_closed_loop_v4/stage2_prompt.md` | V4版本化prompt |
| `tools/optimizer_workflow_contract.md` | runner/validator/矩阵合同 |
| `automation_reports/CV-SincNet/stage2_optimizer_state.json` | 当前状态和机器可读策略 |

## 变更摘要

| 文件 | 变更 |
|---|---|
| `stage2_prompt.md` | Phase2路线改为OPGAC-first；OA-MSE降为对照/诊断；写入`JREF_C9_MULTICOMP_M2_E220`、`OLD80_FIRST`和OPGAC指标bundle要求 |
| `optimizer_workflow_contract.md` | 新增OPGAC矩阵字段、Stage2-A/B/C映射、基底边界和OPGAC指标定义 |
| `stage2_optimizer_state.json` | 新增`stage2_opgac_priority_policy`，保持Phase1 FLOORREPAIR完成审计前不进入Phase2启动的边界 |
| `optimizer_validate_matrix.py` | 新增OPGAC行检测和字段校验，包括基底、support-only记忆、query禁用、同row排名和指标bundle |
| `test_optimizer_workflow_tools.py` | 新增OPGAC缺字段失败和JREF_C9+OLD80_FIRST关键字段通过检查 |
| `test_monitor_optimizer_closed_loop_prompt.py` | 新增OPGAC/JREF_C9/OLD80 token检查，并允许自动化当前`PAUSED`状态 |
| `E:\codex\home\automations\cv-sincnet-n607-monitor-optimizer-v4-2\automation.toml` | 通过Codex自动化工具追加OPGAC-first和指标deficit vector执行说明，状态保持`PAUSED` |

额外状态对齐：`idle_lane_execution_policy.status`从旧的dual-idle审计状态对齐为`MONITOR_ONLY_PHASE1_ACTIVE_PHASE2_IDLE`，并把该策略内`required_next_action`对齐到当前顶层`MONITOR_PHASE1_FLOORREPAIR_TO_COMPLETION_AND_ANALYZE_FULL_TRAINING_LOGS`。这不授权Phase2启动；它只防止状态机把当前Phase1 FLOORREPAIR活跃边界误读成可重启的dual-idle。

## 指标优化口径

阶段二OPGAC当前先服务`OLD80_FIRST`。主门槛是`old80_gap=max(0,0.80-old_acc)`归零；在此之前，低FAR、高AUROC或高coverage不能抵消`old_acc<0.80`。

OPGAC行必须按同一candidate row输出并排名：`old_acc`、`old80_gap`、`unknown_FAR`、`old_unknown_hmean`、`coverage`、`old_FRR`、AUROC、FPR95、rollback/defer、confusion counts和score-table diagnostics。Stage2-C合法时再加入`seen_new_acc`、`H_old_new`和unknown-to-seen-new confusion。

## 版本与快照

根目录`E:\type10-7`不是Git仓库。已创建本地快照：

`E:\type10-7\code\snapshots\20260701_011304_stage2_opgac_v4_control_update`

可Git追踪的工具/测试文件已镜像到：

`E:\type10-7\github_publish\CVS-RFFI-repo`

## 本地验证

| 命令 | 结果 |
|---|---|
| `conda run -n ssr-gpu python -c "import json; json.load(open(...)); print('json ok')"` | PASS，`stage2_optimizer_state.json`可解析 |
| `conda run -n ssr-gpu python -m py_compile tools\optimizer_validate_matrix.py tools\optimizer_state_current_view.py` | PASS |
| `conda run -n ssr-gpu python -m pytest -p no:cacheprovider -q code\tests\test_opgac_net.py code\tests\test_optimizer_workflow_tools.py tests\test_monitor_optimizer_closed_loop_prompt.py` | PASS，73 passed |
| `conda run -n ssr-gpu python tools\optimizer_state_current_view.py` | PASS，输出`stage2_optimizer_current_state_view_v1` |

## 后续边界

当前状态仍要求先完成`PHASE1_FLOORREPAIR`完成审计和完整训练日志/loss分析。只有该边界清除后，Phase2 optimizer才应按本次策略生成OPGAC-first候选矩阵。OLD80达成只是中间门槛，不是Stage2-C成功、部署成功或论文主结论。
