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

## N607完成状态

- launch_pid:4118154
- remote_status:completed
- remote_command:`cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash code/scripts/launch_phase2_adv3b02_stage2c_seennew_contrast_probe_20260707.sh > logs/phase2_adv3b02_stage2c_seennew_contrast_probe_20260707/launch_background.out 2>&1 &`
- remote_json_count:12/12
- summary_json:`E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_seennew_contrast_probe_20260707\stage2c_seennew_contrast_probe_summary.json`
- summary_csv:`E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_seennew_contrast_probe_20260707\stage2c_seennew_contrast_probe_summary.csv`
- ssh_cleanup:checked,no local`ssh.exe`,no ESTABLISHED TCP22 after preflight,sync,launch,monitor,pull

## 结果表

以下每行均为同一candidate/run上下文内的联合指标。`unknown_FAR<=0.05`没有可行行。

| variant | profile | K | old_acc | min_old | seen_new | min_seen_new | unknown_FAR | known_coverage | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HEAD | CENTER_CONTRAST_W025_M00 | 5 | 0.8143 | 0.6571 | 0.3107 | 0.1429 | 0.8196 | 0.9459 | 注册恢复但FAR过高 |
| HEAD | CENTER_CONTRAST_W025_M00 | 10 | 0.7952 | 0.5714 | 0.3518 | 0.2286 | 0.9446 | 0.9969 | min_seen最高但FAR过高 |
| HEAD | CONTRAST_W050_M02 | 10 | 0.7929 | 0.6000 | 0.3554 | 0.2143 | 0.9554 | 0.9959 | seen-new恢复但FAR过高 |
| HEAD | CONTRAST_W025_M00 | 10 | 0.7929 | 0.6000 | 0.3536 | 0.2143 | 0.9554 | 0.9959 | seen-new恢复但FAR过高 |
| NORM | CONTRAST_W025_M00 | 10 | 0.8024 | 0.5143 | 0.3875 | 0.1857 | 0.9661 | 0.9918 | seen_new最高但FAR过高 |
| NORM | CONTRAST_W050_M02 | 10 | 0.8024 | 0.5143 | 0.3875 | 0.1857 | 0.9661 | 0.9918 | seen_new最高但FAR过高 |
| NORM | CENTER_CONTRAST_W025_M00 | 10 | 0.8143 | 0.5286 | 0.3768 | 0.1571 | 0.9607 | 0.9929 | old高但FAR过高 |
| NORM | CENTER_CONTRAST_W025_M00 | 5 | 0.8071 | 0.6571 | 0.2714 | 0.0714 | 0.8411 | 0.9367 | K5较稳但seen_new不足 |
| HEAD | CONTRAST_W025_M00 | 5 | 0.8071 | 0.6286 | 0.2893 | 0.0714 | 0.8696 | 0.9398 | K5注册不足,FAR过高 |
| HEAD | CONTRAST_W050_M02 | 5 | 0.8071 | 0.6286 | 0.2893 | 0.0714 | 0.8696 | 0.9398 | K5注册不足,FAR过高 |
| NORM | CONTRAST_W025_M00 | 5 | 0.8071 | 0.6429 | 0.2768 | 0.0571 | 0.8696 | 0.9469 | K5注册不足,FAR过高 |
| NORM | CONTRAST_W050_M02 | 5 | 0.8048 | 0.6286 | 0.2750 | 0.0571 | 0.8696 | 0.9459 | K5注册不足,FAR过高 |

## 解释

- 与此前rescue-veto/center-veto几乎全拒绝seen-new不同，本次contrast打分层能把seen_new_acc恢复到0.27-0.39区间，说明“新类原型超过old包络”是有效注册证据。
- 最强seen_new行是NORM K10的`CONTRAST_W025_M00/W050_M02`:seen_new=0.3875,min_seen_new=0.1857,old_acc=0.8024,min_old=0.5143，但unknown_FAR=0.9661，不能作为开放集部署路线。
- 最强min_seen_new行是HEAD K10的`CENTER_CONTRAST_W025_M00`:seen_new=0.3518,min_seen_new=0.2286,old_acc=0.7952,min_old=0.5714，但unknown_FAR=0.9446，仍不可部署。
- 最低FAR行是HEAD K5的`CENTER_CONTRAST_W025_M00`:unknown_FAR=0.8196，仍远高于`<=0.05`门槛。
- 结论:contrast是qKNNV42下一轮的注册打分组件候选，但必须配套“contrast-aware unknown gate”，不能沿用旧veto把seen-new打回unknown，也不能无门控直接接受高FAR结果。

## 下一步

1. 固定HEAD/NORM K10的contrast打分作为候选证据，新增low-FAR门控:对unknown使用old-envelope和seen-new-contrast margin的双阈值，而不是只看全局unknown risk。
2. 对比`seen_new_old_contrast_delta`、`old_reference_score`、`seen_new_centroid_score`的事件级分布，找出unknown误收是否来自unknown靠近seen-new原型还是门控阈值过松。
3. 只在低FAR门控出现`unknown_FAR<=0.05`且seen_new/min_seen不坍塌后，才进入更大矩阵或部署候选判定。
