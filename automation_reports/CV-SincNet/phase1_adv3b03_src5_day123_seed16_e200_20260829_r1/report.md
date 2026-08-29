# Phase1 ADV3B03 SRC5 DAY123十六种子实验预登记

## 实验身份与边界

- Run ID：`phase1_adv3b03_src5_day123_seed16_e200_20260829_r1`
- 当前状态：`RUNNING`
- 方法：仅`ADV3B03_MU10_ALPHA20_E200`
- 类型：纯Phase1接收机域泛化训练；不读取Phase2 capsule、support、query、truth、prototype或split，不做目标域适配和新类注册。
- 初始化：`from_scratch=true`，不复用历史ADV3B02 checkpoint。
- Git承载面：`codex/phase1-fasttrust-eff-src5-20260828`
- code/config commit：`0c1bb2344479eb3a12f1bfbf83f050aa1f5e7649`。
- Git发布：自动push为`VERIFIED`；独立远端回读`origin/codex/phase1-fasttrust-eff-src5-20260828=0c1bb2344479eb3a12f1bfbf83f050aa1f5e7649`，与本地HEAD一致。
- 启动所有者：当前主Agent；不得重复启动同一Run ID。

## 数据与矩阵

- 数据：`Dataset_WigSig/ManySig.pkl`，`wisig_equalized=1`。
- 源接收机：`1-19,18-2,19-2,2-19,3-19`，ManySig索引`1,3,4,6,8`。
- 目标接收机：`1-1,14-7,2-1,20-1,7-14,7-7,8-8`，ManySig索引`0,2,5,7,9,10,11`；本次十六行扫描不构建或访问其loader。
- 训练日期：严格为`1,2,3`；`day=0`排除。
- 源角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，物理样本两两不交；选种只读取`V_select`。
- 矩阵：种子`713101–713116`共16行；GPU0–7各绑定2行，映射为GPU`g`运行`713101+g`和`713109+g`。
- 每行：`epochs=200`、`label_epochs=130`、`pseudo_epochs=70`、`batch_size=128`、`eval_batch_size=512`。
- 并发：每GPU两个训练进程；正式启动前若任一GPU已有训练任务导致超过上限，则不启动并等待资源。

## 方法冻结

- ADV3B03差异参数：`lambda_proxy_unknown=0.0050`、`proxy_unknown_core_quantile=0.85`、`proxy_unknown_accept_quantile=0.80`、`proxy_unknown_core_accept_weight=0.35`、`proxy_unknown_vaccept_cvar_alpha=0.20`、`proxy_unknown_unknown_margin=0.10`。
- 拼接星地增强：`use_concat_sat_channel_aug=true`、`concat_sat_ce_only=true`、`concat_sat_ce_weight=1.0`、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`、`lambda_zid_channel_invariance=0`、`sat_cons_start_epoch=80`。
- 训练和评估场景：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- 星地课程：`1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- checkpoint选择：`final_only`；不使用目标接收机表现保存或重排模型。

## 环境、路径与启动命令

- 本地环境：`ssr-gpu`。
- 本地CWD：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\phase2-canonical-union-maxq`。
- N607普通账户CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b03_src5_day123_seed16_e200_20260829_r1_release2`。
- N607 Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 运行根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b03_src5_day123_seed16_e200_20260829_r1`。
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b03_src5_day123_seed16_e200_20260829_r1`。
- release归档本地路径：`E:\type10-7\release_archives\phase1_adv3b03_src5_day123_seed16_e200_20260829_r1_release2.zip`。
- release归档远端路径：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b03_src5_day123_seed16_e200_20260829_r1_release2.zip`。

正式命令：

```bash
nohup /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b03_src5_day123_seed16_e200_20260829_r1_release2/code/scripts/launch_phase1_adv3b03_src5_day123_seed16_e200_20260829.py --root /home/szu2070436088/2510044040/CV-SincNet --code-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b03_src5_day123_seed16_e200_20260829_r1_release2 --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --run-id phase1_adv3b03_src5_day123_seed16_e200_20260829_r1 --runs-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b03_src5_day123_seed16_e200_20260829_r1 --log-root /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b03_src5_day123_seed16_e200_20260829_r1 > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b03_src5_day123_seed16_e200_20260829_r1.dispatcher.log 2>&1 < /dev/null &
```

## 验证与停止规则

- 本地聚焦测试：当前`10 passed`，包含真实`train_ssdg`参数解析、严格四场景artifact拆分、不可覆盖和正式矩阵fail-closed检查；测试发现并修复`days_label`为列表时清洁场景去重键不可哈希的问题。独立审查P1指出正式入口可被参数缩减，现已定点修复为正式模式只接受固定Run ID、种子`713101–713116`和E200；显式`--smoke`只接受独立smoke Run ID、seed713101和E1。Python编译与`git diff --check`通过。
- 原审查者定点复审：`RESOLVED`；无剩余P0/P1。
- release1单归档SHA：本地与N607均为`e821a8a1d89e04db38864ee9ad3b1c103a513bc5c6bfbfeb9c9a001d43d3874c`，传输与远端编译`VERIFIED`。
- release2单归档SHA：本地与N607均为`2899ca25983de40183eac720973d30361ddade78ac971ed5ea447801f1d8cfe6`，传输与远端编译`VERIFIED`。
- 真实smoke r1：`phase1_adv3b03_src5_day123_smoke_e1_20260830_r1`在GPU0完成E1并写出`final_ssdg.pth`，随后训练器以`NON_PROMOTABLE_P0_DISABLED/exit=8`结束，启动器正确保存partial artifact并标记`TRAIN_FAILED`，外部四场景评估未运行，因此为`NO_PERFORMANCE_RESULT`。
- 根因：共享训练器只允许MUSE声明“外部最终评估”，纯ADV3B03虽按`final_only`写出checkpoint，仍被Phase1-V2机制门阻断。定点修复新增通用`phase1_external_final_eval`，仅委托最终严格评估并跳过与本方法无关的promotion/endpoint门，不改变训练数据、模型、损失或参数。
- 修复验证：新专用测试与MUSE兼容测试共`56 passed`；Phase1-V2终态测试`51 passed`，只有既有AMP弃用warning。smoke r1原路径完整保留；修复后使用全新release2和smoke r2，不覆盖、不续跑r1。
- 真实smoke r2：`phase1_adv3b03_src5_day123_smoke_e1_20260830_r2`在GPU0从头完成E1，`final_ssdg.pth`为`15025055`字节，训练终态`COMPLETE/exit=0`，启动器终态`ARTIFACTS_COMPLETE`。配置回读确认`train_days=[1,2,3]`、`train_receivers=[1,3,4,6,8]`、`target_access=false`、target day/receiver均为空，并启用冻结的ADV3B03参数与B02同款拼接星地信道增强。
- smoke r2严格全量`V_select`评估：每个场景`13,500`条；clean=`33.5704%`、`leo_clear_weak=26.0741%`、`leo_low_elev_weak=25.7333%`、`leo_rain_weak=23.6074%`。四个artifact均记录checkpoint epoch1、`strict_requested=true`、`checkpoint_load_strict=true`、`fallback_used=false`、missing/unexpected/shape mismatch均为0。E1数值只验证闭合，不用于性能判断或正式选种。
- 正式启动前只读回读：8张GPU均无compute process；正式run root和log root均不存在。
- 正式启动回执：N607普通账户于2026-08-30启动，dispatcher PID=`1764497`，release2与正式Run ID绑定正确。首次回读为16个直属主训练进程、16个候选目录、16份`config.json`、16份`status.txt=RUNNING`；GPU0–7均严格为2个训练进程。
- 启动日志增长：16份`train.log`由每份`6598`字节增长到`12129–12130`字节，总计`194077`字节；未匹配到`Traceback`、`CUDA out of memory`、`RuntimeError`、`ValueError`、`AssertionError`或`TRAIN_FAILED`。当前属于健康训练，不得因中间性能停止、重启或热补丁。
- 正式发布前仅执行：聚焦协议负测、一次真实checkpoint无目标域E1 smoke、一次独立P0/P1审查、N607资源/路径preflight、单release归档本地/远端SHA比较和一次远端编译。
- 只在错误stage/receiver/day/seed/场景、目标域或Phase2数据被访问、输出碰撞、错误checkout、命令不能启动、同一确定性异常至少两行复现、无最终checkpoint、严格重建失败或任一必要评估artifact缺失时停止该Run ID精确进程树并保留partial artifact。
- 低性能、收敛慢或中间指标差不得停止、重启或热补丁。

## 预期artifact与完成标准

每行必须具备：

- `config.json`
- `train.log`
- `final_ssdg.pth`且checkpoint epoch为200
- `eval_joint.log`、`metrics_joint.json`
- `eval_clean.log`、`metrics_clean.json`
- `eval_leo_clear_weak.log`、`metrics_leo_clear_weak.json`
- `eval_leo_low_elev_weak.log`、`metrics_leo_low_elev_weak.json`
- `eval_leo_rain_weak.log`、`metrics_leo_rain_weak.json`
- `status.txt=ARTIFACTS_COMPLETE`

16行全部达到`ARTIFACTS_COMPLETE`后才进入分析。按`H(clean,min(LEO三场景))`、LEO mean、LEO floor、clean、seed升序冻结最佳种子。冻结后只对该种子做一次纯Phase1零适应目标接收机确认，目标测试同样限定DAY123并逐接收机报告clean和三个LEO场景；目标结果不得反馈选种、调参、重训或选择性重跑。
