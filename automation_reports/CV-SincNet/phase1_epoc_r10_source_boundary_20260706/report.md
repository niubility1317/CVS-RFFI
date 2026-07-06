# phase1_epoc_r10_source_boundary_20260706

## Objective

设计并启动R10 source-only特征边界修复实验。R10继续以`ADV3B02_CORE90_SOFT_E200`为teacher/底座，目标是在不接触真实未知类、不使用目标接收机样本的前提下，增强`z_id`源域紧致性、LEO视图稳定性和后续qknn8开集边界。最终仍必须在Stage2-C中使用qknn8、`M=1..target receiver count`协同推理、真实unknown eval-only进行验证。

## Rationale

R7 base qknn8协同推理已经证明单纯多接收机融合不足。R8强proxy/riskgate诊断能提高unknown拒识，但通过误拒旧类和新类换取，旧类和seen-new严重不足。R9改为source-anchor路线，但仍需等待prototype。R10在R9基础上进一步收紧source-only类内半径和导出prototype半径，同时把真实unknown相关路径保持为eval-only。

可用文献依据：

|方法|可迁移点|合规性|
|---|---|---|
|OpenMax, Bendale and Boult, 2016, https://arxiv.org/abs/1511.06233|用特征尾部分布估计unknown概率，启发后端EVT/半径门控。|可用于后端阈值，不需要真实未知训练。|
|Mahalanobis OOD, Lee et al., 2018, https://arxiv.org/abs/1807.03888|用类条件Gaussian/Mahalanobis距离做异常检测，适合prototype/qknn8旁路打分。|只用已知类训练特征即可，符合source-only边界。|
|Energy-based OOD, Liu et al., 2020, https://arxiv.org/abs/2010.03759|相比softmax置信度，energy更适合作为OOD分数和训练约束。|可作为已知类能量面塑形和后端评分，unknown query仍eval-only。|
|VOS, Du et al., 2022, https://arxiv.org/abs/2202.01197|在特征空间合成虚拟outlier，学习“未知边界”而不接触真实未知类。|只能使用虚拟outlier；不得把ManyTx真实unknown当训练负样本。|

## Candidate Design

|candidate|GPU|seed|机制|预期|
|---|---:|---:|---|---|
|`EPOC_R10_BOUNDARY_NOPROXY`|4|7061001|无proxy主线；强teacher clean/sat/zid蒸馏；更强`zid_compact`和`source_episode_three_sigma`；更小Phase2 prototype半径。|优先保护旧类/seen-new特征几何，降低未知误吸附风险。|
|`EPOC_R10_GENTLE_VOS_LATE`|5|7061011|同上，但E120后加入极弱虚拟outlier/VOS压力。|观察轻量虚拟边界能否改善unknown，同时避免R8的强proxy数值风险。|

## Protocol Boundaries

|item|value|
|---|---|
|ground-training dataset|`ManySig.pkl` only|
|teacher/base|`ADV3B02_CORE90_SOFT_E200`|
|target receiver samples in training|0|
|real unknown classes in training|0|
|ManyTx in training|0|
|virtual unknown|allowed, synthetic/feature-space only|
|true unknown query|Stage2-C eval-only|
|claim boundary|no Stage2-C or deployment success until qknn8 same-row eval completes|
|downstream evaluator|qknn8, `M=1..all target receivers`, LEO views|

## Local Files

|file|purpose|
|---|---|
|`code/scripts/launch_phase1_epoc_r10_source_boundary_20260706.sh`|R10 launcher|
|`code/tests/test_phase1_epoc_r10_source_boundary_launcher.py`|launcher protocol tests|
|`code/snapshots/phase1_epoc_r10_source_boundary_20260706/launch_phase1_epoc_r10_source_boundary_20260706.sh`|local snapshot before N607 sync|
|`automation_reports/CV-SincNet/phase1_epoc_r10_source_boundary_20260706/report.md`|experiment report|

## Verification Plan

|check|expected|
|---|---|
|`bash -n`|launcher syntax PASS|
|dry-run|prints source-only, no real unknown, no target receiver, no deployment claim|
|forbidden `WISIG_PKL=/tmp/ManyTx.pkl`|fail-closed|
|pytest|R10 launcher tests pass under`ssr-gpu`|
|py_compile|`train_ssdg.py` and test compile|
|remote verification|hash match, remote `bash -n`, dry-run, fail-closed guard, py_compile|

## Local Verification

|check|result|
|---|---|
|`bash -n code/scripts/launch_phase1_epoc_r10_source_boundary_20260706.sh`|PASS|
|dry-run `--only=EPOC_R10_BOUNDARY_NOPROXY`|PASS；打印`ManySig_only`、`real_unknown_classes_in_training=0`、`target_receiver_samples_in_training=0`、`stage2_success_claim=0`、`deployment_success_claim=0`。|
|`WISIG_PKL=/tmp/ManyTx.pkl` guard|PASS；launcher非零退出并拒绝非source Phase1输入。|
|pytest|PASS；`conda run -n ssr-gpu python -m pytest code\tests\test_phase1_epoc_r10_source_boundary_launcher.py -q` => 3 passed，仅有`.pytest_cache`权限警告。|
|py_compile|PASS；`conda run -n ssr-gpu python -m py_compile code\tests\test_phase1_epoc_r10_source_boundary_launcher.py code\SSDG\train_ssdg.py`。|
|local hashes before sync|launcher/snapshot`0EF3CA60876262955932DD69EEC2CE40786E2A1E505F7E2A982D0D12981B96C2`；test`3108E62C569AC3FF219F36E99DE63D77B99A27F805E74140B63816489A9C8EB9`；report`32ADD84CFA94BD27E07E9516B7E55ABBF3D61B81013C0B546822163D2155B509`。|

## Remote Sync and Verification

|item|result|
|---|---|
|N607 preflight|PASS at 2026-07-06 09:23 CST；project root visible；GPU4/GPU5 idle at约10MiB。|
|remote sync|launcher、test、snapshot、report和`code/SYNC_MANIFEST.txt`已同步到`/home/szu2070436088/2510044040/CV-SincNet`。|
|remote hash verify|PASS；launcher/snapshot`0ef3ca60876262955932dd69eec2ce40786e2a1e505f7e2a982d0d12981b96c2`；test`3108e62c569ac3ff219f36e99de63d77b99a27f805e74140b63816489a9c8eb9`；report`5337152e2d510c11fb4c2574beb0c4b91d3556cfa259fa3b56aad841ca60fd3b`。|
|remote syntax/dry-run|PASS；远端`bash -n`通过；dry-run打印`ManySig_only`、`real_unknown_classes_in_training=0`、`proxy_unknown_real_tx_calibration=0`、`deployment_success_claim=0`。|
|remote fail-closed guard|PASS；`WISIG_PKL=/tmp/ManyTx.pkl`返回码4并拒绝非source Phase1输入。|
|remote py_compile|PASS；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/tests/test_phase1_epoc_r10_source_boundary_launcher.py code/SSDG/train_ssdg.py`。|

## Launch Plan

If verification passes and N607 GPU4/GPU5 remain low-memory, launch:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
nohup bash code/scripts/launch_phase1_epoc_r10_source_boundary_20260706.sh > logs/phase1_epoc_r10_source_boundary_20260706/driver.out 2>&1 &
```

Expected logs:

```text
logs/phase1_epoc_r10_source_boundary_20260706/EPOC_R10_BOUNDARY_NOPROXY.out
logs/phase1_epoc_r10_source_boundary_20260706/EPOC_R10_GENTLE_VOS_LATE.out
```

## Remote Launch

|item|result|
|---|---|
|launch time|2026-07-06 09:24 CST|
|launch command|`cd /home/szu2070436088/2510044040/CV-SincNet; nohup bash code/scripts/launch_phase1_epoc_r10_source_boundary_20260706.sh > logs/phase1_epoc_r10_source_boundary_20260706/driver.out 2>&1 &`|
|driver PID|`3349101`，driver submitted both candidates and exited after submit.|
|candidate PIDs|`EPOC_R10_BOUNDARY_NOPROXY` main PID`3349113` on GPU4；`EPOC_R10_GENTLE_VOS_LATE` main PID`3349554` on GPU5。|
|logs|`logs/phase1_epoc_r10_source_boundary_20260706/EPOC_R10_BOUNDARY_NOPROXY.out`；`logs/phase1_epoc_r10_source_boundary_20260706/EPOC_R10_GENTLE_VOS_LATE.out`。|
|GPU after startup|09:28 CST GPU4约2033MiB，GPU5约2039MiB；GPU6/7仍约10MiB。|

## Startup Health

|timestamp|candidate|epoch|files|status|
|---|---|---:|---|---|
|2026-07-06 09:28 CST|`EPOC_R10_BOUNDARY_NOPROXY`|9/200|`latest_safe_ssdg.pth`、`latest_ssdg.pth`|running；无Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed；梯度total为有限值，`aux=nan`为既有辅助字段现象。|
|2026-07-06 09:28 CST|`EPOC_R10_GENTLE_VOS_LATE`|9/200|`latest_safe_ssdg.pth`、`latest_ssdg.pth`|running；无Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed；早期出现`total=inf`梯度行但后续仍继续到E009，列为数值稳定性风险监控。|

当前边界：R10只有启动健康和训练中证据；尚未导出prototype，不能声明Stage2-C成功、部署成功或未知类拒识改善。后续必须在`phase2_zid_prototypes.pt/json`导出后执行qknn8协同推理`M=1..all target receivers`，真实unknown仍仅作eval-only。
