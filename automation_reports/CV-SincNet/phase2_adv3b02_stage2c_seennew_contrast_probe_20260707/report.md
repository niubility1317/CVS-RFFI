# qKNNV42 Stage2-C seen-new contrast probe

## 基本信息

- experiment_id:phase2_adv3b02_stage2c_seennew_contrast_probe_20260707
- timestamp:2026-07-07
- operator:Codex
- status:local_verified,pending_n607_sync_launch
- diagnostic_only:true
- deployment_success_claim:false
- stage2_success_claim:false

## 目标和假设

目标是在qKNNV42现有Stage2-C性能基础上，优先缓解seen-new增多后的识别坍塌和最低类性能过低，同时观察old类目标域适应是否受损。协议仍为K=5/K=10目标域old+seen-new少量support用于域适应和新类注册，target unknown只用于eval-only，目标接收样本均为LEO星地信道视图。

假设:过去rescue/veto和support_center仍把真实seen-new大量判成unknown，说明问题主要在新类注册打分层。新增`seen_new_old_contrast`只对非old标签加分，且仅当该seen-new原型相似度超过所有old原型包络时生效，用于强化“新类原型确实离开old包络”的证据，而不是无条件提升新类标签。

## 本地变更

| file | purpose |
|---|---|
| `code/scripts/phase2_collaborative_open_set_qknn_eval.py` | 增加默认关闭的`seen_new_old_contrast_weight/margin`，接入qKNN打分、label score matrix、阈值校准、class verifier、事件证据和metadata |
| `code/scripts/phase2_frozen_manytx_unknown_diagnostic.py` | 暴露并透传contrast扫参参数 |
| `code/tests/test_phase2_collaborative_open_set_qknn_eval.py` | 验证contrast只提升old包络外的非old标签，old分数保持不变 |
| `code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py` | 验证薄包装器CLI能解析contrast参数 |
| `code/scripts/launch_phase2_adv3b02_stage2c_seennew_contrast_probe_20260707.sh` | 新增N607只读冻结特征诊断启动器，扫K=5/K=10、NORM/HEAD和三组contrast profile |

## 本地验证

| command | result |
|---|---|
| `python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\scripts\phase2_frozen_manytx_unknown_diagnostic.py` | PASS |
| `python -m pytest code\tests\test_phase2_collaborative_open_set_qknn_eval.py code\tests\test_phase2_frozen_manytx_unknown_diagnostic.py` | PASS,61 passed |
| `python code\scripts\phase2_frozen_manytx_unknown_diagnostic.py --help \| Select-String -Pattern 'seen_new_old_contrast'` | PASS,CLI参数可见 |
| `bash -n code/scripts/launch_phase2_adv3b02_stage2c_seennew_contrast_probe_20260707.sh` | PASS |
| `ROOT=/tmp/cvs-rffi-dryrun PYTHON=python bash code/scripts/launch_phase2_adv3b02_stage2c_seennew_contrast_probe_20260707.sh --dry-run` | PASS,12个诊断组合展开 |

## N607计划

| item | value |
|---|---|
| remote_root | `/home/szu2070436088/2510044040/CV-SincNet` |
| source_run_id | `phase2_adv3b02_stage2c_normsep_protocol_20260707` |
| case_id | `PHASE2_STAGE2C_RX7_14` |
| run_id | `phase2_adv3b02_stage2c_seennew_contrast_probe_20260707` |
| variants | `STAGE2C_NORM_SEP`,`STAGE2C_HEAD_SEP` |
| K-shot | `5`,`10` |
| profiles | `CONTRAST_W025_M00`,`CONTRAST_W050_M02`,`CENTER_CONTRAST_W025_M00` |
| query_per_class | `70` |
| qknn_k | `8` |
| outputs | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_seennew_contrast_probe_20260707` |
| logs | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_seennew_contrast_probe_20260707` |

Planned remote command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
bash code/scripts/launch_phase2_adv3b02_stage2c_seennew_contrast_probe_20260707.sh
```

## 成功/止损判据

| criterion | interpretation |
|---|---|
| `seen_new_acc`和`min_seen_new_class_acc`较center-veto/rescue-veto结果提升 | 说明注册打分层缓解坍塌 |
| `old_acc`和`min_old_class_acc`不明显低于已有最好诊断行 | 说明未牺牲old类目标域适应 |
| `unknown_FAR`保持可解释，尤其是否存在`<=0.05`可行行 | 决定是否进入低FAR约束扫参 |
| 每行metrics必须同candidate/run上下文一起解释 | 禁止用孤立max/min宣称成功 |

## 当前风险

- contrast可能提高seen-new分数但同步提高unknown误收，需要后续低FAR门控复查。
- `CENTER_CONTRAST_W025_M00`可能重复support_center对seen-new的伤害，因此单独标注为组合诊断，不作为默认路线。
- 本次为冻结特征诊断，不是部署成功证据。
