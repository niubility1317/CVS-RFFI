# RIEI/DRIFT Stage2-B SourceLogitBias-CDA修复实验

## 基本信息

- 实验ID：`riei_drift_current_sat_supervised_r010_stage2b_targetold_source_logit_bias_20260707_225826`
- 时间：2026-07-07 22:58 CST
- 操作者：Codex
- 目标：修复RIEI/DRIFT在Stage2-B目标旧类域适应中使用域对齐后性能低于预期的问题，验证“源分类器logit仍比embedding prototype/head更保留旧类决策面”这一假设。
- 协议：CVS Stage2-B target-old few-shot domain adaptation；`R_s=day0/day1,rx0-rx6`源域训练，目标域为剩余receiver；support/query均来自目标域旧类`Y_old`；启用LEO星地信道视图；不做新类学习，不做未知类拒识主线。
- 声明边界：本实验只报告target-old域适应；unknown/open-set字段如存在只作Phase3备用诊断，不参与阈值拟合、排序或成功声明。

## 假设与对比目标

前序诊断说明：

| 路线 | 最好结果 | 解释 |
|---|---:|---|
| raw Euclidean ProtoNet-CDA | RIEI K5约0.5111 | 未归一化距离把embedding尺度差异误当类别差异，是明确实现问题 |
| normalized Euclidean ProtoNet-CDA | RIEI K5 0.5422 | 修复尺度问题后小幅改善，但仍远低于OLD80 |
| normalized prototype-initialized support head | DRIFT K10 0.5650，RIEI K5 0.5556 | support-only线性头明显优于prototype距离，尤其DRIFT改善约20pp |
| support head K20/K50 | DRIFT K50 0.5878，RIEI K50 0.5756 | 增加support样本后饱和在0.58-0.59附近，说明瓶颈不只是K-shot样本量 |

本次路线：`source_logit_bias_calibration`冻结source classifier，只用目标域support学习每个旧类的bias，不更新backbone、不用query标签、不启用新类或未知类拒识。若结果显著高于support head，说明源分类器logit保留的决策面比当前`z_id`距离空间更可靠；若不升反降，则说明源分类器本身已与目标LEO视图错位，后续应转向representation/adapter级修复。

## 本地变更

Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`

提交：

- `ba99a5c Add source logit bias calibration for RIEI DRIFT`

变更文件：

- `paper_reproduction/cvs_aligned/evaluate.py`：新增`_source_logits`和`_source_logit_bias_predict`，为RIEI/DRIFT抽取source classifier logits，并用support-only bias校准进行预测。
- `tests/test_paper_reproduction_cvs_aligned.py`：新增source-logit bias单测。
- `paper_reproduction/configs/*source_logit_bias_n607.json`：新增RIEI/DRIFT K5/K10四个Stage2-B配置。
- `run_riei_drift_current_sat_supervised_r010_cvs_stage2b_leo_source_logit_bias_n607.sh`：新增N607启动脚本。

本地快照：

- `E:\type10-7\code\snapshots\riei_drift_current_sat_supervised_r010_stage2b_targetold_source_logit_bias_20260707_225826\`

本地哈希：

| 文件 | SHA256 |
|---|---|
| `evaluate.py` | `8FCA9CFAE69B91A3D408CEA5B25CA0D349FBB271900436FD9B9FFE2825192D85` |
| `test_paper_reproduction_cvs_aligned.py` | `7501EB7F7E467632E93136A7BF798A9B4C9A4B985FEFDBFD0DA096301CC86830` |
| `run_riei_drift_current_sat_supervised_r010_cvs_stage2b_leo_source_logit_bias_n607.sh` | `95E7C63A386F8567ED17E5B3B86227EE872C8666AAA644DA0D4A37C4A98E3EA1` |
| `drift_current_sat_supervised_r010_cvs_stage2b_k5_leo_source_logit_bias_n607.json` | `4F8CECC4B4A520C1549B59444669C58BB19D051D42896A5291F8B7100C6043E9` |
| `drift_current_sat_supervised_r010_cvs_stage2b_k10_leo_source_logit_bias_n607.json` | `80E7D89F769D34B79F10CF2F9EA57882F8B5F419CD9FC7CFCDDD8B127BF9C808` |
| `riei_fd_current_sat_supervised_r010_cvs_stage2b_k5_leo_source_logit_bias_n607.json` | `9AB909DB7DEE6A6562CC8BBF68182860F9C96FFF643CFB267B5ADE921CAC2416` |
| `riei_fd_current_sat_supervised_r010_cvs_stage2b_k10_leo_source_logit_bias_n607.json` | `B66991843AC35F3E0242451D66FA86BEE6AC01D17C61C8DC7C9E4A2CCE23886F` |

## 本地验证

| 命令 | 结果 |
|---|---|
| `conda run -n ssr-gpu python -m py_compile paper_reproduction\cvs_aligned\evaluate.py` | PASS |
| `conda run -n ssr-gpu python -m pytest tests\test_paper_reproduction_cvs_aligned.py -k "source_logit or support_head or prototype_predict" -q` | 初次遇到conda GBK输出编码错误；设置`PYTHONIOENCODING=utf-8`和`PYTHONUTF8=1`后PASS，6 passed、12 deselected，只有pytest cache权限warning |
| `bash -n run_riei_drift_current_sat_supervised_r010_cvs_stage2b_leo_source_logit_bias_n607.sh` | PASS |
| 四个`*source_logit_bias_n607.json`配置dry-run | PASS，先前并发命令出现过conda临时锁，串行重跑通过 |

## N607计划

- 远端根目录：`/home/szu2070436088/2510044040/CV-SincNet`
- 同步目标：
  - `paper_reproduction/cvs_aligned/evaluate.py`
  - `tests/test_paper_reproduction_cvs_aligned.py`
  - `paper_reproduction/configs/*source_logit_bias_n607.json`
  - `run_riei_drift_current_sat_supervised_r010_cvs_stage2b_leo_source_logit_bias_n607.sh`
  - 本报告和`code/SYNC_MANIFEST.txt`
- 远端验证：
  - `sha256sum`匹配本地哈希
  - `bash -n` launcher
  - `DRY_RUN=1` launcher
- 启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
RUN_ID=riei_drift_current_sat_supervised_r010_stage2b_targetold_source_logit_bias_20260707_225826 bash run_riei_drift_current_sat_supervised_r010_cvs_stage2b_leo_source_logit_bias_n607.sh
```

## 结果

完成状态：4个候选均已完成，GPU回到空闲；本地artifact已拉回：

- `E:\type10-7\automation_reports\CV-SincNet\riei_drift_current_sat_supervised_r010_stage2b_targetold_source_logit_bias_20260707_225826\artifacts\`

主结果：

| candidate | baseline | K | old_acc | clear | low | rain | coverage | lat_ms | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `riei_fd_current_sat_k5_leo_source_logit_bias_seed1337` | RIEI | 5 | 0.5422 | 0.5550 | 0.5533 | 0.5183 | 1.0000 | 0.676 | below best support-head |
| `riei_fd_current_sat_k10_leo_source_logit_bias_seed1337` | RIEI | 10 | 0.5361 | 0.5317 | 0.5450 | 0.5317 | 1.0000 | 0.568 | below best support-head |
| `drift_current_sat_k5_leo_source_logit_bias_seed1337` | DRIFT | 5 | 0.5333 | 0.5350 | 0.5450 | 0.5200 | 1.0000 | 0.583 | below best support-head |
| `drift_current_sat_k10_leo_source_logit_bias_seed1337` | DRIFT | 10 | 0.5183 | 0.5217 | 0.5150 | 0.5183 | 1.0000 | 0.594 | below best support-head |

逐类和逐receiver诊断：

| candidate | weak TX | strong TX | weak receiver | strong receiver | top confusion |
|---|---|---|---|---|---|
| DRIFT K5 | `6-15` 0.247、`14-7` 0.290、`14-10` 0.410 | `8-20` 0.917、`20-15` 0.890 | `20-1` 0.408、`3-19` 0.403 | `7-14` 0.731、`8-8` 0.647 | `14-7->3`、`20-19->1`、`6-15->1` |
| DRIFT K10 | `14-7` 0.277、`6-15` 0.283、`14-10` 0.380 | `8-20` 0.907、`20-15` 0.820 | `3-19` 0.350、`20-1` 0.392 | `7-14` 0.722、`8-8` 0.625 | `14-7->3`、`6-15->1`、`14-10->4` |
| RIEI K5 | `6-15` 0.283、`14-7` 0.320、`14-10` 0.350 | `8-20` 0.937、`20-15` 0.863 | `20-1` 0.358、`3-19` 0.419 | `7-14` 0.778、`8-8` 0.614 | `14-7->3`、`14-10->1`、`6-15->1` |
| RIEI K10 | `6-15` 0.313、`14-7` 0.317、`14-10` 0.360 | `8-20` 0.917、`20-15` 0.840 | `20-1` 0.367、`3-19` 0.414 | `7-14` 0.794、`8-8` 0.603 | `14-7->3`、`6-15->1`、`14-10->4` |

日志扫描：

- 4个`.out`均完整解析；无`Traceback`、`RuntimeError`、CUDA OOM、`Killed`或参数错误。
- 日志中出现`NaN`只来自`AUROC/FPR95/unknown_FAR`在unknown disabled场景下的JSON字段，不是训练或推理数值异常。

## 解释

SourceLogitBias-CDA没有修复性能差问题：最好的RIEI K5仅0.5422，与normalized Euclidean基本相同，低于support-head的0.5556；DRIFT从support-head K10的0.5650退回到0.5183。结论是源分类器logit并没有比目标support上重新拟合的head保留更好的LEO目标域决策面。

更关键的诊断信号是误差同时按TX和receiver集中：`20-15`/`8-20`稳定高，而`14-7`/`6-15`/`14-10`稳定低；`7-14`明显好于`20-1`/`3-19`。这说明当前全局prototype/head把5个目标receiver的support合并后，可能混掉receiver-specific响应。下一步已转向`ReceiverConditionedSupportHead-CDA`，即每个目标receiver只用本receiver的support训练轻量head并预测本receiver query。

## 风险与观察点

- 若source-logit路线明显低于support head，说明源分类器logit也被目标LEO视图破坏，后续需要representation或小adapter修复，而不是继续调prototype距离。
- 若source-logit路线高于support head，下一步可做bias+temperature、小型affine classifier或BN affine only校准，但仍需保持support-only、query不可见和无unknown阈值拟合。
- 本实验不改变`项目.md`协议，不涉及Stage2-C seen-new，也不声明部署成功。
