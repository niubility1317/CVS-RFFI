# Phase1 soft unknown mixup实现报告

## 基本信息

|字段|内容|
|---|---|
|run_id|`phase1_soft_unknown_mixup_impl_20260701_171500`|
|时间|2026-07-01|
|操作者|Codex|
|目标|按方案A实现3个不同TX样本的软标签mixup未知类代理,并把同一批mixup样本接入`proxy_unknown`扩展项和`source_episode_three_sigma_loss`|
|N607状态|未启动、未同步、未中断任何远程任务|

## 假设与机制

本次修改把旧的`proxy_unknown_energy_loss`保持为leave-one-TX-out与virtual outlier基线,新增`soft_unknown_mixup_loss`作为显式数据增强未知代理。每个合成样本从3个不同TX类各采一个样本,按mixup权重生成归一化特征,并构造软标签。训练目标包含软标签CE、相对旧类原型的能量边界、以及可选vacuum隔离项,使混合样本更难被旧类接受。

`source_episode_three_sigma_loss`新增`mixup_features`入口。它复用source leave-domain得到的旧类3σ半径,对mixup样本施加反向约束:如果mixup样本落入任一旧类episode半径内,则产生隔离惩罚。训练循环中同一批`soft_unknown_mixup_batch`同时供`soft_unknown_mixup_loss`和`source_episode_three_sigma_loss`使用。

## 文件变更

|文件|用途|
|---|---|
|`code/cvsrffi/losses.py`|新增`SoftUnknownMixupBatch`、`make_soft_unknown_mixup`、`soft_unknown_mixup_loss`;扩展`source_episode_three_sigma_loss`支持mixup隔离|
|`code/SSDG/train_ssdg.py`|新增CLI参数、权重表、telemetry、训练循环接线、日志摘要|
|`tests/test_soft_unknown_mixup_losses.py`|新增3-TX软未知mixup与source episode组合单测|

## 本地验证

|命令|结果|
|---|---|
|`conda run --no-capture-output -n ssr-gpu python -m pytest tests\test_soft_unknown_mixup_losses.py -q`|通过,2 passed|
|`conda run --no-capture-output -n ssr-gpu python -m py_compile code\cvsrffi\losses.py code\SSDG\train_ssdg.py`|通过|
|`conda run --no-capture-output -n ssr-gpu python code\SSDG\train_ssdg.py --help`|通过,新增参数可见|
|`conda run --no-capture-output -n ssr-gpu python code\SSDG\train_ssdg.py --output_dir runs\dry_soft_unknown_mixup --dry_run --lambda_open_world_feat 0.0018 --lambda_zid_compact 0.024 --lambda_proxy_unknown 0.002 --lambda_soft_unknown_mixup 0.003 --soft_unknown_mixup_count 16 --soft_unknown_mixup_order 3 --soft_unknown_mixup_alpha 0.5 --soft_unknown_mixup_vacuum_weight 0.35 --lambda_source_episode 0.002 --source_episode_mixup_weight 0.75`|通过,参数解析成功且未构造数据/模型|

一次并行`conda run`触发Windows临时文件锁,随后按项目规则串行重跑`--help`通过。该现象不是代码错误。

## 推荐下一轮实验参数

|参数|建议值|目的|
|---|---:|---|
|`--lambda_soft_unknown_mixup`|`0.0025`到`0.004`|开启方案A未知代理|
|`--soft_unknown_mixup_order`|`3`|每个合成样本固定来自3个不同TX|
|`--soft_unknown_mixup_count`|`16`或`24`|保持批内未知代理密度|
|`--soft_unknown_mixup_ce_weight`|`0.5`到`1.0`|软标签拉到旧类尾部/低密度区域|
|`--soft_unknown_mixup_energy_weight`|`1.0`|训练混合样本能量边界|
|`--soft_unknown_mixup_vacuum_weight`|`0.25`到`0.5`|对旧类半径附近加隔离带|
|`--source_episode_mixup_weight`|`0.5`到`1.0`|把source episode3σ与已有mixup结合|
|`--lambda_zid_compact`|较当前候选提高约25%到50%|增强旧类紧致性|
|`--lambda_open_world_feat`|较当前候选提高约25%到50%|增强特征空间开放世界边界|

## 风险与观察点

主要风险是拒识增强后旧类准确率下降,尤其是高强度vacuum或source episode mixup权重过大时。下一轮应重点观察`train/soft_unknown_mixup_virtual_accept_rate`、`train/source_episode_mixup_overflow_rate`、旧类准确率、unknown AUROC/FAR、以及joint-safe指标是否同向改善。

## 版本与同步

本地快照保存于`code/snapshots/phase1_soft_unknown_mixup_impl_20260701_171500/`。同名代码与测试已镜像到Git发布仓库`github_publish/CVS-RFFI-repo`等待提交。本次未执行`scp`,未改变N607服务器文件。
