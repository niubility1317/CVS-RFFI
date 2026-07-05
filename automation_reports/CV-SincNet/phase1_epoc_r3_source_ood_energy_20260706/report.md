# Phase1 EPOC-R3 Source OOD Energy Repair Report

## 基本信息

|项目|值|
|---|---|
|run_id|`phase1_epoc_r3_source_ood_energy_20260706`|
|时间|2026-07-06|
|operator|Codex|
|目标|在`ADV3B02_CORE90_SOFT_E200`基础上，针对R2暴露的虚拟未知分离弱、旧类保持不足和LEO strict floor不足，启动下一轮source-only底座修复|
|定位|Phase1/source-only地面训练修复；不是Stage2-C成功，不是部署成功，不是真实未知类训练|
|比较目标|`phase1_epoc_r2_leo_unknown_separation_20260705`的`EPOC_R2_OLD_FLOOR`与`EPOC_R2_BALANCED_SEP`|

## 协议边界

|边界|执行方式|
|---|---|
|训练数据|只加载`ManySig.pkl`；不加载`ManyTx.pkl`；不使用`--new_wisig_pkl`|
|真实未知类|`Y_unknown`和`target_unknown`不进入训练、阈值、prototype、adapter、profile、receiver权重或early stopping|
|目标接收机|地面训练不接触`R_t`样本、统计、BN、阈值、prototype、adapter或验证结果|
|底座模型|`ADV3B02_CORE90_SOFT_E200`作为`baseline_ckpt`和`teacher_ckpt`|
|LEO视图|使用源域派生的`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|后续成功判定|必须重新进入真实Stage2-C qknn8/OSPR-CI或等价评估，同一行满足`old_acc>=0.99`、`min_old>=0.95`、`seen_new_acc>=0.97`、`min_seen>=0.93`、`unknown_reject>=0.99`、`unknown_FAR<=0.01`才可升级声明|

## R2触发证据

|问题|R2证据|R3修复目标|
|---|---|---|
|虚拟未知分离近随机|`proxy_auc≈0.47-0.48`|强化source-only energy、低密度、半径比和virtual outlier边界|
|虚拟未知误接受过高|`virtual_accept≈0.82`，`soft_virtual_accept≈0.996-0.999`|压低open-space误接受，不用真实未知类调参|
|旧类保持不足|best `test_tx≈89%-90%`|增强ADV3B02教师保真、旧类floor和类中心紧致|
|LEO strict floor弱|`sat_strict_floor≈69%-71%`|加强源域LEO强视图一致性和source episode外推|

## 候选矩阵

|候选|GPU|定位|关键机制|
|---|---:|---|---|
|`EPOC_R3_ENERGY_VOS_GUARD`|2|energy/VOS优先|更强virtual outlier、energy in/out、proxy energy quantile、low-density accept、radius-inter ratio约束|
|`EPOC_R3_TIGHT_CORE_MARGIN`|3|旧类核心紧致优先|更强teacher clean/zid保持、更小source/core半径、更高类间margin和local component prototype导出|

## 本地文件变更

|文件|用途|
|---|---|
|`code/scripts/launch_phase1_epoc_r3_source_ood_energy_20260706.sh`|新增R3 source-only energy/VOS/geometric repair启动器|
|`code/tests/test_phase1_epoc_r3_source_ood_energy_launcher.py`|新增启动器协议与dry-run测试|
|`automation_reports/CV-SincNet/phase1_epoc_r3_source_ood_energy_20260706/report.md`|本报告|
|快照|`E:\type10-7\code\snapshots\phase1_epoc_r3_source_ood_energy_20260706`|

## 本地验证

|命令|结果|
|---|---|
|`bash -n code/scripts/launch_phase1_epoc_r3_source_ood_energy_20260706.sh`|PASS|
|`conda run -n ssr-gpu pytest code\tests\test_phase1_epoc_r3_source_ood_energy_launcher.py -q`|PASS，2 passed；仅`.pytest_cache`权限warning|
|`bash code/scripts/launch_phase1_epoc_r3_source_ood_energy_20260706.sh --dry-run --only=EPOC_R3_ENERGY_VOS_GUARD`|PASS；显示`ManySig.pkl`、`CUDA_VISIBLE_DEVICES=2`、energy/proxy/soft/source_episode参数；未出现`ManyTx.pkl`、`--new_wisig_pkl`或`target_unknown`|

## 本地hash

|文件|SHA256|
|---|---|
|`code/scripts/launch_phase1_epoc_r3_source_ood_energy_20260706.sh`|`D59A1FA43B7075867605819CEEEFDEE5280595A1DF3037A8CD082F80B3155365`|
|`code/tests/test_phase1_epoc_r3_source_ood_energy_launcher.py`|`12705A317B6524FC7242C7E44995FBBE03E667A61D381D679D8A7173EE197E09`|

## N607计划

|项目|计划|
|---|---|
|远端根目录|`/home/szu2070436088/2510044040/CV-SincNet`|
|远端环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|同步文件|R3启动器、R3测试、本报告、`code/SYNC_MANIFEST.txt`|
|远端验证|hash一致性、`bash -n`、dry-run协议字段、run/log路径碰撞检查|
|启动计划|若GPU2/3仍低占用且run/log路径不存在，启动两个R3候选；R2继续monitor-only，不中断|
|日志路径|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_epoc_r3_source_ood_energy_20260706/<candidate>.out`|
|输出路径|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_epoc_r3_source_ood_energy_20260706/<candidate>`|
|启动健康|检查`[CONFIG-LOSS]`、`[CONFIG-TEACHER]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`、Traceback/OOM/NaN/unrecognized/stale log|

## 资源约束查漏

|项目|结论|
|---|---|
|原文文件|未在当前工作区找到`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`|
|可用替代口径|沿用OSPR-CI与`collaborative_open_set_qknn_eval.py`中的代理字段：`bytes_per_event`、`latency_ms`、`max_event_bytes`、`max_event_latency_ms`、`qknn8_support_int8_bytes`、`total_fp16_state_bytes`|
|默认预算|OSPR-CI CLI默认`max_event_bytes=1152.0`、`max_event_latency_ms=20.0`；底层融合函数默认`0.0`表示不启用该上限|
|声明边界|这些字段是本地融合/证据包/状态大小代理估算，不是真实星间链路、端侧batch=1 profiler或网络排队p95实测；不得写`resource_real_pass=true`|

## 声明边界

R3若启动健康，只能证明下一轮source-only底座修复开始运行。即使Phase1指标改善，也必须通过后续真实Stage2-C qknn8/OSPR-CI同row评估后，才能讨论旧类、新类和未知拒识目标。

## N607同步与远端验证

|项目|结果|
|---|---|
|N607 preflight|2026-07-06 00:14 CST与00:18 CST direct preflight PASS；项目根目录可见，GPU可见|
|远端占用|启动前GPU0/GPU1承载R2，GPU2-GPU7低显存占用；按用户允许的低显存占用GPU启动|
|同步目标|`/home/szu2070436088/2510044040/CV-SincNet`|
|启动器hash|远端`d59a1fa43b7075867605819ceeefdee5280595a1df3037a8cd082f80b3155365`，与本地一致|
|测试hash|远端`12705a317b6524fc7242c7e44995fbbE03e667a61d381d679d8a7173ee197e09`，与本地一致；大小写差异只来自显示|
|报告hash|初次同步远端`68da9a77ed2d998c0ffeaae63eb0cd5a031f906d2ac3506469077a28fc642cd3`，与本地一致|
|manifest hash|初次同步远端`a819f9374ba87ccdbb5e7a2149de84b5c2c8ab2ced2d1bcafa9f0893b7078813`，与本地一致|
|远端语法检查|`bash -n code/scripts/launch_phase1_epoc_r3_source_ood_energy_20260706.sh` PASS|
|远端dry-run|`EPOC_R3_TIGHT_CORE_MARGIN` dry-run PASS；显示`CUDA_VISIBLE_DEVICES=3`、`ManySig.pkl`、energy/proxy参数；未出现`ManyTx.pkl`或`target_unknown`|

## N607启动记录

|项目|值|
|---|---|
|启动命令|`cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/launch_phase1_epoc_r3_source_ood_energy_20260706.sh --only=EPOC_R3_ENERGY_VOS_GUARD,EPOC_R3_TIGHT_CORE_MARGIN`|
|远端环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|`EPOC_R3_ENERGY_VOS_GUARD`|PID`3064573`，GPU2，log`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_epoc_r3_source_ood_energy_20260706/EPOC_R3_ENERGY_VOS_GUARD.out`，output`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_epoc_r3_source_ood_energy_20260706/EPOC_R3_ENERGY_VOS_GUARD`|
|`EPOC_R3_TIGHT_CORE_MARGIN`|PID`3064995`，GPU3，log`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_epoc_r3_source_ood_energy_20260706/EPOC_R3_TIGHT_CORE_MARGIN.out`，output`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_epoc_r3_source_ood_energy_20260706/EPOC_R3_TIGHT_CORE_MARGIN`|
|协议打印|启动器输出确认`source_only_phase1_ground_training_repair`、`real_unknown_classes_in_training=0`、`manytx_in_training=0`|

## 启动健康与监控

|时间|候选|状态|关键证据|
|---|---|---|---|
|2026-07-06 00:16 CST|`EPOC_R3_ENERGY_VOS_GUARD`|健康启动|主PID`3064573`存活；log/outdir存在；`[CONFIG-LOSS]`、`[CONFIG-TEACHER]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`存在；hard error计数0；已到E004/180；`metrics_epoch.jsonl`与`latest_safe_ssdg.pth`存在|
|2026-07-06 00:16 CST|`EPOC_R3_TIGHT_CORE_MARGIN`|健康启动|主PID`3064995`存活；log/outdir存在；同上配置标记存在；hard error计数0；已到E002/180；`metrics_epoch.jsonl`与`latest_safe_ssdg.pth`存在|
|2026-07-06 00:20 CST|`EPOC_R3_ENERGY_VOS_GUARD`|运行中|E013/180，best E010，`best_score=85.4310`，`val_tx_acc=98.5119`；开集loss尚未进入启动epoch，因此proxy unknown字段仍为null|
|2026-07-06 00:20 CST|`EPOC_R3_TIGHT_CORE_MARGIN`|运行中|E011/180，best E010，`best_score=85.8973`，`val_tx_acc=98.6548`；开集loss尚未进入启动epoch，因此proxy unknown字段仍为null|

GPU状态采样：2026-07-06 00:20 CST，GPU0`2487/24576MiB`，GPU1`2411/24576MiB`，GPU2`2305/24576MiB`，GPU3`2317/24576MiB`，GPU4-GPU7均约`10/24576MiB`。

## R2并行监控结论

|候选|时间|epoch|best|当前未知代理|判读|
|---|---:|---:|---|---|---|
|`EPOC_R2_BALANCED_SEP`|2026-07-06 00:20 CST|E109/190|best E100，`best_score=85.5742`|`proxy_auc=0.4694`，`proxy_accept=0.0324`，`virtual_accept=0.8293`，`soft_virtual_accept=0.9971`|仍未解决虚拟未知误接受，继续运行但不作为成功证据|
|`EPOC_R2_OLD_FLOOR`|2026-07-06 00:20 CST|E111/180|best E080，`best_score=85.1636`|`proxy_auc=0.4801`，`proxy_accept=0.0060`，`virtual_accept=0.8281`，`soft_virtual_accept=0.9966`|proxy AUC仍近随机，继续运行但不作为成功证据|

R2未被中断。R3是基于R2负面趋势追加的source-only底层修复路线。

## 00:24 CST监控更新

N607 direct preflight于2026-07-06 00:24 CST通过。GPU采样：GPU0`2487/24576MiB`，GPU1`2411/24576MiB`，GPU2`2315/24576MiB`，GPU3`2321/24576MiB`，GPU4-GPU7均约`10/24576MiB`。R2和R3均保持运行，未发现Traceback、RuntimeError、CUDA OOM或unrecognized arguments。日志尾部出现`nan`字样主要来自结构化指标/非有限计数字段，需继续关注`nonfinite_*_metric_count`，但当前没有训练崩溃证据。

|候选|epoch|best|旧类保持|开集代理状态|判读|
|---|---:|---|---|---|---|
|`EPOC_R3_ENERGY_VOS_GUARD`|E022/180|best E020，`best_score=85.5712`|`val_tx_acc=98.5357`|`proxy_unknown`尚未启动，`train_loss_proxy_unknown=0.0`，`proxy_auc=null`|仍在开集loss正式启动前；旧类保持未塌陷，需等E030后再判断未知分离|
|`EPOC_R3_TIGHT_CORE_MARGIN`|E021/180|best E010，`best_score=85.8973`|`val_tx_acc=98.6071`|`proxy_unknown`尚未启动，`train_loss_proxy_unknown=0.0`，`proxy_auc=null`|仍在开集loss正式启动前；当前best略高，但未知拒识尚无证据|
|`EPOC_R2_BALANCED_SEP`|E116/190|best E100，`best_score=85.5742`|`val_tx_acc=98.6429`|`proxy_auc=0.4772`，`virtual_accept=0.8291`，`soft_virtual_accept=0.9976`，`radius_to_inter_ratio=0.9504`|R2仍未突破未知代理分离，继续作为负面参照|
|`EPOC_R2_OLD_FLOOR`|E118/180|best E080，`best_score=85.1636`|`val_tx_acc=98.5476`|`proxy_auc=0.4785`，`virtual_accept=0.8212`，`soft_virtual_accept=0.9976`，`radius_to_inter_ratio=0.9753`|R2仍未突破未知代理分离，继续作为负面参照|

当前判断：R3运行健康但尚未到关键观测点。下一次有效判断点应在`EPOC_R3_TIGHT_CORE_MARGIN`达到E028-E030、`EPOC_R3_ENERGY_VOS_GUARD`达到E030之后，重点比较R3是否相对R2显著提升`proxy_auc`并压低`virtual_accept`/`soft_virtual_accept`，同时旧类`val_tx_acc`不能明显下降。

## 00:30-00:31 CST首轮proxy观测

N607 direct preflight于2026-07-06 00:27 CST通过。GPU采样显示R3仍在低显存运行：GPU2约`2459/24576MiB`，GPU3约`2465/24576MiB`。两条R3进程均存活，日志尾部未发现Traceback、RuntimeError、CUDA OOM或unrecognized arguments。

|候选|时间|epoch|旧类保持|首轮proxy unknown证据|初步判读|
|---|---:|---:|---|---|---|
|`EPOC_R3_ENERGY_VOS_GUARD`|2026-07-06 00:31 CST|E030/180|`val_tx_acc=98.5655`，`test_tx_acc=89.8348`|`proxy_auc=0.4362`，`proxy_accept=0.0068`，`virtual_accept=0.8183`，`soft_virtual_accept=0.9990`，`radius_to_inter_ratio=0.9778`|首轮proxy启动后没有优于R2，AUC低于R2约0.47-0.48；旧类未塌陷但未知分离仍弱|
|`EPOC_R3_TIGHT_CORE_MARGIN`|2026-07-06 00:30 CST|E029/180|`val_tx_acc=98.5655`|`proxy_auc=0.4070`，`proxy_accept=0.0166`，`virtual_accept=0.8272`，`soft_virtual_accept=0.9997`，`radius_to_inter_ratio=0.9681`|首轮proxy比R2更差；紧致核心没有立刻把虚拟未知推出已知邻域|

当前结论：R3首轮proxy证据是负面的早期信号，不能升级为有效路线。仍应继续运行到warmup后半段，因为`proxy_unknown_warmup_epochs=30`，但下一步若E040-E050仍维持`proxy_auc<0.55`且`virtual_accept>0.5`，应停止把R3作为主路线推进，转向更底层的两类方案：第一，重新设计source-only负样本构造与能量边界，避免当前virtual outlier仍落在已知类邻域；第二，准备Stage2只读诊断，用已冻结R3/R2/ADV3B02特征包验证真实`ManyTx`未知是否与source proxy指标一致，但阈值仍不得使用`Y_unknown`调参。

## 00:35-00:41 CST warmup后段监控

N607 direct preflight于2026-07-06 00:35 CST通过。GPU2/GPU3显存约`2459/2465MiB`，两条R3进程存活；R2仍在GPU0/GPU1运行。所有采样日志尾部均未发现Traceback、RuntimeError、CUDA OOM或unrecognized arguments。

|候选|时间|epoch|旧类保持|proxy unknown证据|判读|
|---|---:|---:|---|---|---|
|`EPOC_R3_ENERGY_VOS_GUARD`|2026-07-06 00:41 CST|E041/180|`val_tx_acc=98.5655`|`proxy_auc=0.4330`，`proxy_accept=0.0296`，`virtual_accept=0.8123`，`soft_virtual_accept=0.9997`，`radius_to_inter_ratio=1.0045`|warmup后段仍低于R2参照；旧类保持可以，但未知代理分离失败|
|`EPOC_R3_TIGHT_CORE_MARGIN`|2026-07-06 00:41 CST|E041/180|`val_tx_acc=98.5655`|`proxy_auc=0.4044`，`proxy_accept=0.0075`，`virtual_accept=0.8297`，`soft_virtual_accept=0.9994`，`radius_to_inter_ratio=0.9399`|比R2更差，紧致核心策略没有产生开放空间分离|
|`EPOC_R2_BALANCED_SEP`|2026-07-06 00:39 CST|E132/190|`val_tx_acc=98.6250`|`proxy_auc=0.4811`，`virtual_accept=0.8305`，`soft_virtual_accept=0.9971`|R2自身仍弱，但R3未超过此负面参照|
|`EPOC_R2_OLD_FLOOR`|2026-07-06 00:39 CST|E133/180|`val_tx_acc=98.5476`|`proxy_auc=0.4730`，`virtual_accept=0.8233`，`soft_virtual_accept=0.9966`|R2自身仍弱，但R3未超过此负面参照|

路线判断：按预设判断条件，R3已满足“`proxy_auc<0.55`且`virtual_accept>0.5`”的降级触发。R3应继续跑完以保留完整负面证据和后续checkpoint，但不应继续作为主路线追加同类超参。下一步主线应转为更底层的算法修复：构造不贴近旧类流形的source-only负样本、显式约束unknown energy高于known且低密度区域不可接受，并新增冻结特征包Stage2只读诊断来验证source proxy失败是否对应真实ManyTx未知失败；该诊断只读评估，不使用`Y_unknown`调阈值。

## SSH清理状态

|检查点|结果|
|---|---|
|首次启动健康检查后|本地无残留`ssh.exe`；无到`172.31.111.215:22`或`172.31.105.18:22`的ESTABLISHED连接|
|两次失败监控命令后|均无残留`ssh.exe`；无到N607或桥接机的ESTABLISHED连接|
|00:20 CST紧凑监控后|本地无残留`ssh.exe`；无到N607或桥接机的ESTABLISHED连接|
|00:24 CST紧凑监控后|本地无残留`ssh.exe`；无到N607或桥接机的ESTABLISHED连接|
