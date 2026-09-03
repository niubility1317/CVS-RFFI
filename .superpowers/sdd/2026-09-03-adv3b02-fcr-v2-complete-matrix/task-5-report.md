# Task 5报告

日期：2026-09-03

状态：DONE_WITH_CONCERNS

## 改动文件

- `code/cvsrffi/phase1_fcr_v2_diagnostics.py`
- `code/train.py`
- `code/tests/test_phase1_fcr_v2_diagnostics.py`

## 实现内容

- 新增`phase1_fcr_v2_diagnostics.py`，在现有V1 detached probe基础上补齐V2诊断聚合与JSON写出。
- V2输出包含：matched pair计数/覆盖率、same-TX cross-domain pair覆盖、`eta_valid_coverage`、`eta_component_error`、decoder zero-nuisance敏感度、swap输出差异、`z_tx_state`线性probe、梯度范数比例、per-TX source metrics、epoch时间、激活lambda与capability原因。
- 对拿不到的数值项统一输出`N/A`并附带`*_reason`，不伪造0值。
- 扩展`collect_fcr_diagnostic_artifacts()`，补采`z_tx_state`、`eta_target/eta_pred`、decoder full/zero-nuisance/swap输出，以及fingerprint response quality。
- 新增`finalize_fcr_v2_diagnostics_before_return()`，在`defer_target_evaluation`返回前先写source-only diagnostics；非deferred路径复用同一helper，避免双口径。
- helper优先加载`best_save_path` checkpoint后再做diagnostics，因此保持原有“基于best checkpoint写diagnostics”的语义，而不是退化成final-epoch权重。

## Root Cause

- 当前HEAD中`defer_target_evaluation`在`code/train.py:5233-5239`直接`return`。
- FCR predictions/diagnostics写出逻辑位于其后的`try`块内，因此deferred路径永远不会落`fcr_diagnostics.json`。
- 2026-09-03新增回归测试先稳定复现了这两个缺口：V2模块缺失、deferred前diagnostics helper缺失。

## TDD记录

### Red

先新增`code/tests/test_phase1_fcr_v2_diagnostics.py`，覆盖两类行为：

- V2 schema与关键字段可写出；
- deferred target evaluation返回前，diagnostics已经落盘。

失败命令：

```text
conda run -n ssr-gpu python -m pytest E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\adv3b02-fcr-20260901\code\tests\test_phase1_fcr_v2_diagnostics.py -q
```

失败结果：

```text
AssertionError: Task 5 V2 diagnostics module is missing
AssertionError: deferred finalization helper is missing
```

### Green

通过命令1：

```text
conda run -n ssr-gpu python -m pytest E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\adv3b02-fcr-20260901\code\tests\test_phase1_fcr_v2_diagnostics.py -q
```

结果：`2 passed`

通过命令2：

```text
conda run -n ssr-gpu python -m pytest E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\adv3b02-fcr-20260901\code\tests\test_phase1_fcr_v2_diagnostics.py E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\adv3b02-fcr-20260901\code\tests\test_phase1_fcr_diagnostics.py E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\adv3b02-fcr-20260901\code\tests\test_phase1_fcr_review_fix.py -q
```

结果：`12 passed`

## 关注点

- 当前V2里`grad_clean_leo_cosine`仍默认`N/A`，因为本任务只允许改diagnostic collection/finalization顺序相关区段，没有把训练中逐batch梯度向量本身持久化到finalization资源里；测试已要求该字段显式给理由，而不是伪造数值。
- `zs_content_probe`仍沿用V1行为，若上游没有合法`content_labels`则继续输出`N/A`。本任务没有扩展source loader契约，也没有引入任何target数据或truth sidecar访问。
- `train.py`工作树还存在他人未提交的无关未跟踪文件（例如`code/scripts/launch_phase1_adv3b02_fcr_v2_complete_s392005_20260903.sh`和`code/tests/test_phase1_fcr_v2_complete_launcher.py`）；本任务未触碰，也不会stage它们。
