# Phase1早期拒识课程14候选实验报告

## 基本信息

|字段|内容|
|---|---|
|run_id|`phase1_early_reject_curriculum_gpu1_7_20260701_1643`|
|时间|2026-07-01|
|操作者|Codex|
|目标|在GPU1-7每卡落地2个候选,验证未知拒识约束从训练前期进入表示学习是否优于后期补丁|
|阶段|Phase1 source-only weak-label/semi-supervised DG|
|协议边界|使用source receivers地面训练数据;不使用target receiver样本、统计、阈值、prototype或early stopping信息|
|对照|GPU0当前`phase1_soft_unknown_mixup_gpu0_20260701_1605`与本矩阵内`E200_LATE_CONTROL`|

## 假设

历史配置主要在中后期才开启`open_world_feature_space_loss`、`zid_compactness_loss`、`proxy_unknown_energy_loss`和`source_episode_three_sigma_loss`。如果旧类原型和闭集决策区域已经形成,后期弱拒识损失更像边界修补,很难解决旧类尾部/低密度区域误接受。本轮把`zid/open_world/source_episode/soft_unknown_mixup/proxy_unknown`拆成多种启动时间与强度,用14候选验证早期拒识课程是否改变学习路线。

## 本地文件变更

|文件|用途|
|---|---|
|`code/SSDG/train_ssdg.py`|新增`--soft_unknown_mixup_start_epoch`和`--soft_unknown_mixup_warmup_epochs`,使soft unknown mixup可早于proxy unknown单独启动;默认`-1`保持回落到proxy配置兼容旧脚本|
|`code/scripts/launch_phase1_early_reject_curriculum_gpu1_7_20260701.sh`|新增GPU1-7的14候选launcher,支持`--dry-run`和`--only=`|
|`code/snapshots/phase1_early_reject_curriculum_gpu1_7_20260701_1643/`|因`code`子目录不是单独Git仓库,保存本地快照|

Git发布仓库`github_publish/CVS-RFFI-repo`已提交本次变更,提交信息为`Launch early rejection curriculum matrix`。

## 本地验证

|命令|结果|
|---|---|
|`conda run --no-capture-output -n ssr-gpu python -m py_compile code\SSDG\train_ssdg.py code\cvsrffi\losses.py`|通过|
|`conda run --no-capture-output -n ssr-gpu python code\SSDG\train_ssdg.py --help \| Select-String -Pattern 'soft_unknown_mixup_start_epoch\|soft_unknown_mixup_warmup_epochs'`|通过,新增参数可见|
|`bash -n code/scripts/launch_phase1_early_reject_curriculum_gpu1_7_20260701.sh`|通过|
|`bash code/scripts/launch_phase1_early_reject_curriculum_gpu1_7_20260701.sh --dry-run`|通过,展开14条候选命令|

## 候选矩阵

|GPU|候选ID|epoch配置|启动epoch:`zid/ow/source/soft/proxy`|核心变量|
|---:|---|---|---|---|
|1|`E120_EARLY_LITE`|120=80+40|5/10/15/20/40|轻量早期拒识,低风险保旧类|
|1|`E160_EARLY_MAIN`|160=100+60|8/12/20/25/50|中短训练主线|
|2|`E200_EARLY_MAIN`|200=130+70|8/12/20/25/60|200epoch早期课程主线|
|2|`E240_EARLY_MAIN`|240=150+90|10/15/25/30/70|更长巩固|
|3|`E300_EARLY_CONSOL`|300=180+120|10/15/25/30/75|长训上限|
|3|`E200_VERYEARLY_STRONG`|200=120+80|1/5/10/15/35|极早强拒识|
|4|`E200_MID_CURRIC`|200=130+70|20/30/45/60/80|中期课程|
|4|`E200_LATE_CONTROL`|200=150+50|80/100/120/140/150|后期补丁对照|
|5|`E200_CE_HEAVY_MIX`|200=130+70|8/12/20/20/65|mixup软标签主导|
|5|`E200_ENERGY_HEAVY`|200=130+70|8/12/20/25/35|energy/proxy主导|
|6|`E220_VACUUM_STRONG`|220=140+80|10/15/25/35/55|强vacuum隔离|
|6|`E220_3SIGMA_STRONG`|220=140+80|8/12/15/25/65|source episode三西格玛主导|
|7|`E200_MIXUP_DOMINANT`|200=130+70|8/12/20/20/80|soft unknown mixup主导|
|7|`E200_PROXY_DOMINANT`|200=130+70|8/12/20/25/30|proxy unknown主导|

## 同步计划

|本地文件|N607目标|
|---|---|
|`E:\type10-7\code\SSDG\train_ssdg.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/SSDG/train_ssdg.py`|
|`E:\type10-7\code\scripts\launch_phase1_early_reject_curriculum_gpu1_7_20260701.sh`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_early_reject_curriculum_gpu1_7_20260701.sh`|

## 远端命令

计划远端启动命令:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && RUN_ID=phase1_early_reject_curriculum_gpu1_7_20260701_1643 STAGE2_MAX_ACTIVE_PER_GPU=2 bash code/scripts/launch_phase1_early_reject_curriculum_gpu1_7_20260701.sh
```

预期输出:

|路径|内容|
|---|---|
|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/<candidate>/`|模型、metrics、prototype export|
|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/<candidate>.out`|训练日志|

## 观察指标

重点观察`train/soft_unknown_mixup_virtual_accept_rate`、`train/source_episode_mixup_overflow_rate`、`train/zid_compact_tail_cvar_deg`、`train/w_loss_soft_unknown_mixup`、`train/w_loss_source_episode`、strict UDU、receiver floor和`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`stress结果。若早期候选在epoch80前已明显降低mixup接受率且旧类指标未塌陷,支持“拒识约束必须早期参与表示学习”的假设。

## 远端状态

|阶段|状态|
|---|---|
|N607预检|通过; direct `N607`可达,项目根存在,GPU1-7空闲,GPU0已有2个训练进程|
|同步|完成;远端备份`code/SSDG/train_ssdg.py.pre_early_reject_20260701_1643`|
|远端语法/参数验证|通过;`py_compile`、help参数检查、`bash -n`、launcher `--dry-run`均通过|
|启动|完成;14个候选已提交到GPU1-7,每卡2个训练进程|
|startup health|通过;14个候选均有CUDA计算进程、`[EPOCH-BEGIN]`进展、配置/loss标记,未发现Traceback、RuntimeError、unrecognized arguments或OOM|

## 启动记录

远端启动命令已执行:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && RUN_ID=phase1_early_reject_curriculum_gpu1_7_20260701_1643 STAGE2_MAX_ACTIVE_PER_GPU=2 bash code/scripts/launch_phase1_early_reject_curriculum_gpu1_7_20260701.sh
```

同步后远端哈希:

|文件|SHA256|
|---|---|
|`code/SSDG/train_ssdg.py`|`5dc404f3deacfff8f4b9d306243980aeacdfd35e86cda7a7cba716cc70c09f2f`|
|`code/scripts/launch_phase1_early_reject_curriculum_gpu1_7_20260701.sh`|`5ae7ca006ddd5171e59210b324122c22b5a9c9de2be5abf6dbbe33c41c935f4e`|

|GPU|PID|候选ID|startup epoch|日志|
|---:|---:|---|---|---|
|1|2657202|`E120_EARLY_LITE`|E010/120|`logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/E120_EARLY_LITE.out`|
|1|2661388|`E160_EARLY_MAIN`|E007/160|`logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/E160_EARLY_MAIN.out`|
|2|2657609|`E200_EARLY_MAIN`|E010/200|`logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/E200_EARLY_MAIN.out`|
|2|2662004|`E240_EARLY_MAIN`|E006/240|`logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/E240_EARLY_MAIN.out`|
|3|2658032|`E300_EARLY_CONSOL`|E010/300|`logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/E300_EARLY_CONSOL.out`|
|3|2662452|`E200_VERYEARLY_STRONG`|E005/200|`logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/E200_VERYEARLY_STRONG.out`|
|4|2658866|`E200_MID_CURRIC`|E009/200|`logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/E200_MID_CURRIC.out`|
|4|2663018|`E200_LATE_CONTROL`|E005/200|`logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/E200_LATE_CONTROL.out`|
|5|2659289|`E200_CE_HEAVY_MIX`|E009/200|`logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/E200_CE_HEAVY_MIX.out`|
|5|2663771|`E200_ENERGY_HEAVY`|E005/200|`logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/E200_ENERGY_HEAVY.out`|
|6|2659714|`E220_VACUUM_STRONG`|E009/220|`logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/E220_VACUUM_STRONG.out`|
|6|2664311|`E220_3SIGMA_STRONG`|E004/220|`logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/E220_3SIGMA_STRONG.out`|
|7|2660137|`E200_MIXUP_DOMINANT`|E009/200|`logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/E200_MIXUP_DOMINANT.out`|
|7|2665004|`E200_PROXY_DOMINANT`|E003/200|`logs/phase1_early_reject_curriculum_gpu1_7_20260701_1643/E200_PROXY_DOMINANT.out`|

当前GPU状态:GPU1-7每卡2个本轮训练进程;GPU0保留既有`phase1_soft_unknown_mixup_gpu0_20260701_1605`两条训练,未新增GPU0任务。

## 2026-07-01 17:48 ETA监控

只读监控时间:`2026-07-01 17:48:42`。N607上本轮`phase1_early_reject_curriculum_gpu1_7_20260701_1643`仍有14个训练进程,日志扫描未发现Traceback、RuntimeError、unrecognized arguments或OOM。ETA按各候选最近epoch耗时与全局均值加权估算;最后20个epoch存在更密集评估,实际完成时间可能比表中晚约10-40分钟。

|候选ID|当前epoch|估算秒/epoch|剩余时间|预计完成|
|---|---:|---:|---:|---|
|`E120_EARLY_LITE`|E65/120|60.9s|0.95h|07-01 18:45|
|`E160_EARLY_MAIN`|E65/160|59.5s|1.59h|07-01 19:23|
|`E200_LATE_CONTROL`|E98/200|38.1s|1.09h|07-01 18:54|
|`E200_MID_CURRIC`|E75/200|58.1s|2.03h|07-01 19:50|
|`E200_CE_HEAVY_MIX`|E66/200|60.4s|2.26h|07-01 20:04|
|`E200_EARLY_MAIN`|E66/200|61.0s|2.29h|07-01 20:05|
|`E200_ENERGY_HEAVY`|E62/200|61.1s|2.36h|07-01 20:10|
|`E200_PROXY_DOMINANT`|E61/200|60.9s|2.37h|07-01 20:10|
|`E200_MIXUP_DOMINANT`|E63/200|62.2s|2.38h|07-01 20:11|
|`E200_VERYEARLY_STRONG`|E59/200|62.3s|2.46h|07-01 20:16|
|`E220_VACUUM_STRONG`|E68/220|60.3s|2.56h|07-01 20:22|
|`E220_3SIGMA_STRONG`|E61/220|60.9s|2.71h|07-01 20:31|
|`E240_EARLY_MAIN`|E65/240|60.6s|2.96h|07-01 20:46|
|`E300_EARLY_CONSOL`|E68/300|60.2s|3.90h|07-01 21:42|

整批完成受`E300_EARLY_CONSOL`限制,点估计约`2026-07-01 21:42`,保守窗口为`2026-07-01 22:00-22:30`。

## 风险

强vacuum和三西格玛隔离可能损害旧类准确率;长训300epoch占用时间更长;如果GPU1-7已有其他训练进程,launcher将按每卡最多2个训练进程等待而不是强行超额提交。
