# RIEI/DRIFT ProtoNet-CDA归一化欧氏修复实验报告

## 基本信息

- 实验ID：`riei_drift_current_sat_supervised_r010_stage2b_targetold_protonet_cda_normeuclid_20260707_222833`
- 时间：2026-07-07 22:28:33 Asia/Hong_Kong
- 操作者：Codex
- 目标：修复RIEI/DRIFT使用ProtoNet-CDA域适应后性能反而下降的问题，并解释下降原因。
- 阶段：Stage2-B target-old few-shot domain adaptation。
- 对照：上一轮未归一化欧氏ProtoNet-CDA，run `riei_drift_current_sat_supervised_r010_stage2b_targetold_protonet_cda_20260707_212633`。

## 协议边界

| 字段 | 设置 |
|---|---|
| source receivers | `1-1,1-19,14-7,18-2,19-2,2-1,2-19` |
| target receivers | `20-1,3-19,7-14,7-7,8-8` |
| source days | `0,1` |
| target day | `0` |
| source train ratio | `0.1` |
| source checkpoint | `riei_drift_no_unlabeled_r010_sat_20260707_191029` |
| target support | K-shot old TX only |
| target query | held-out old TX only |
| K grid | `5,10` |
| target-new | disabled |
| unknown rejection | disabled |
| train/support/query satellite augmentation | enabled |
| target channel scenarios | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` |

该实验不声明新类识别，不使用unknown query调阈值，也不把clean view作为部署成功证据。

## 失败原因诊断

上一轮域适应变差不是运行错误，而是度量几何错误。RIEI的`z_e`和DRIFT的`z_tx`来自分类/域对齐训练，不是episodic ProtoNet度量学习空间。直接使用未归一化欧氏距离时，距离会同时受到feature norm、接收机响应和星地信道幅度扰动影响。结果是target support prototype被少数大范数方向吸附，query按幅度近邻而不是身份方向近邻分类。

上一轮同协议结果：

| candidate | K | prototype_metric | old_acc_mean | clear | low | rain | 主要现象 |
|---|---:|---|---:|---:|---:|---:|---|
| `drift_current_sat_k5` | 5 | `euclidean` | 0.2911 | 0.2983 | 0.2633 | 0.3117 | 预测集中到少数类，class4严重丢失 |
| `drift_current_sat_k10` | 10 | `euclidean` | 0.3344 | 0.3567 | 0.3283 | 0.3183 | class5吸附明显，rx `3-19`最低 |
| `riei_fd_current_sat_k5` | 5 | `euclidean` | 0.5111 | 0.5283 | 0.5100 | 0.4950 | class2/class5保留较好，class1/3/4弱 |
| `riei_fd_current_sat_k10` | 10 | `euclidean` | 0.4861 | 0.4217 | 0.5133 | 0.5233 | K增加未稳定提升，说明prototype估计受norm扰动 |

修复策略：保留原`euclidean`用于历史复现，新加`normalized_euclidean`。support/query先L2归一化，按归一化support求prototype，再归一化prototype，最后用欧氏距离分类。该修复等价于在单位球面上做prototype距离，保留方向性身份信息，降低feature norm和receiver/channel幅度对分类的支配。

## 本地改动

| 文件 | 目的 |
|---|---|
| `paper_reproduction/cvs_aligned/evaluate.py` | 新增`normalized_euclidean/norm_euclidean/l2_euclidean` ProtoNet-CDA metric |
| `tests/test_paper_reproduction_cvs_aligned.py` | 新增feature norm偏置单测 |
| `paper_reproduction/configs/*normeuclid_n607.json` | 新增RIEI/DRIFT K5/K10修复版配置 |
| `run_riei_drift_current_sat_supervised_r010_cvs_stage2b_leo_protonet_cda_normeuclid_n607.sh` | 新增N607 launcher |

Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`  
代码提交：`09f5335 Add normalized ProtoNet CDA for RIEI DRIFT`  
报告已镜像并提交到Git承载面。  
根目录：`E:\type10-7`不是Git仓库，改动已镜像到Git承载面并提交。

## 本地验证

| 命令 | 结果 |
|---|---|
| `python -m py_compile paper_reproduction/cvs_aligned/evaluate.py` | PASS |
| `python -m pytest tests/test_paper_reproduction_cvs_aligned.py -k prototype_predict -q` | PASS，3 passed，13 deselected；仅`.pytest_cache`写权限警告 |
| `bash -n run_riei_drift_current_sat_supervised_r010_cvs_stage2b_leo_protonet_cda_normeuclid_n607.sh` | PASS |
| 4个`*normeuclid_n607.json` dry-run | PASS，`stage2_protocol_valid=true`，`rs_rt_disjoint=true`，`unknown_query_used_for_threshold=false` |

## N607计划

- 远端工程根：`/home/szu2070436088/2510044040/CV-SincNet`
- 远端运行目录：`runs/riei_drift_current_sat_supervised_r010_stage2b_targetold_protonet_cda_normeuclid_20260707_222833`
- 远端日志目录：`logs/riei_drift_current_sat_supervised_r010_stage2b_targetold_protonet_cda_normeuclid_20260707_222833`
- 启动命令：

```bash
RUN_ID=riei_drift_current_sat_supervised_r010_stage2b_targetold_protonet_cda_normeuclid_20260707_222833 bash run_riei_drift_current_sat_supervised_r010_cvs_stage2b_leo_protonet_cda_normeuclid_n607.sh
```

## 完成状态

远端任务已完成。launcher返回`RUN_ROOT`和`LOG_ROOT`，4个候选均生成`metrics.json`、`score_table.csv`、`resolved_config.json`和`split_manifest.json`。日志扫描未发现`Traceback`、`RuntimeError`、`ValueError`、`CUDA out of memory`或`Killed`。`NaN`仅出现在禁用unknown/open-set后的备用指标字段中，不是运行异常。

远端结果：

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/riei_drift_current_sat_supervised_r010_stage2b_targetold_protonet_cda_normeuclid_20260707_222833`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/riei_drift_current_sat_supervised_r010_stage2b_targetold_protonet_cda_normeuclid_20260707_222833`
- 本地artifact：`E:\type10-7\automation_reports\CV-SincNet\riei_drift_current_sat_supervised_r010_stage2b_targetold_protonet_cda_normeuclid_20260707_222833\artifacts`

## 结果对照表

| candidate | K | metric | old_acc_mean | clear | low | rain | old_acc_delta_vs_euclidean | verdict |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `riei_fd_current_sat_k5_leo_protonet_cda_normeuclid` | 5 | `normalized_euclidean` | 0.5422 | 0.5483 | 0.5400 | 0.5383 | +3.11 pp | best in this batch，仍未达OLD80 |
| `riei_fd_current_sat_k10_leo_protonet_cda_normeuclid` | 10 | `normalized_euclidean` | 0.5189 | 0.5217 | 0.5050 | 0.5300 | +3.28 pp | improves，但低于K5 |
| `drift_current_sat_k10_leo_protonet_cda_normeuclid` | 10 | `normalized_euclidean` | 0.3600 | 0.3867 | 0.3517 | 0.3417 | +2.56 pp | improves，但整体弱 |
| `drift_current_sat_k5_leo_protonet_cda_normeuclid` | 5 | `normalized_euclidean` | 0.2961 | 0.3017 | 0.2650 | 0.3217 | +0.50 pp | minimal rescue |

同row对照：

| candidate | old euclidean mean | normalized mean | delta pp | clear delta pp | low delta pp | rain delta pp |
|---|---:|---:|---:|---:|---:|---:|
| DRIFT K5 | 0.2911 | 0.2961 | +0.50 | +0.33 | +0.17 | +1.00 |
| DRIFT K10 | 0.3344 | 0.3600 | +2.56 | +3.00 | +2.33 | +2.33 |
| RIEI K5 | 0.5111 | 0.5422 | +3.11 | +2.00 | +3.00 | +4.33 |
| RIEI K10 | 0.4861 | 0.5189 | +3.28 | +10.00 | -0.83 | +0.67 |

## 细分诊断

| candidate | receiver acc `20-1/3-19/7-14/7-7/8-8` | class acc `0/1/2/3/4/5` |
|---|---|---|
| DRIFT K5 | 0.3167 / 0.1889 / 0.2861 / 0.3639 / 0.3250 | 0.2300 / 0.2000 / 0.4433 / 0.2733 / 0.0433 / 0.5867 |
| DRIFT K10 | 0.3583 / 0.2167 / 0.4194 / 0.4083 / 0.3972 | 0.2033 / 0.4267 / 0.4567 / 0.1533 / 0.1733 / 0.7467 |
| RIEI K5 | 0.4806 / 0.4056 / 0.6389 / 0.5556 / 0.6306 | 0.5500 / 0.2467 / 0.9233 / 0.3467 / 0.3367 / 0.8500 |
| RIEI K10 | 0.4722 / 0.3972 / 0.6306 / 0.5222 / 0.5722 | 0.4600 / 0.2967 / 0.8167 / 0.4067 / 0.2667 / 0.8667 |

预测分布说明：

- DRIFT修复后不再比上一轮更差，但仍明显偏向class2/class5，class4和class3仍弱；K10相对K5提高，说明更多target support能缓解但不能解决DRIFT表征的类别塌缩。
- RIEI修复后最强，尤其class2/class5稳定，receiver `7-14/8-8`较高；弱点仍在class1/class3/class4，receiver `3-19`为主要floor。
- K10没有超过RIEI K5，说明问题不是单纯support数量不足，而是support prototype与query类内结构仍不匹配。

## 结论

“使用域对齐反而性能更差”的直接原因是：上一版把RIEI/DRIFT的分类训练embedding当成ProtoNet度量空间，直接使用未归一化欧氏距离。RIEI/DRIFT的`z_e/z_tx`携带feature norm和接收机/信道幅度信息，欧氏prototype会被norm和target receiver偏移牵引，导致少数类别吸附和K增加不稳定。这个问题不是N607运行失败，也不是unknown拒识阈值错误，因为本轮Stage2-B没有启用unknown拒识。

修复后，4个候选全部相对未归一化欧氏提升，证明metric geometry是主故障点。当前最佳为RIEI K5 normalized Euclidean，old_acc_mean=0.5422，比上一轮RIEI K5高+3.11 pp，比上一轮RIEI K10高+5.61 pp。

但这还不是完整解决：RIEI/DRIFT仍远低于OLD80，说明仅靠support prototype重标定不足以完成目标接收机旧类适应。下一步应优先做两类实验：一是target-old only小adapter/BN affine或support-head微调上限诊断，二是class-balanced/shrinkage prototype，避免class1/3/4继续被class2/5吸附。
