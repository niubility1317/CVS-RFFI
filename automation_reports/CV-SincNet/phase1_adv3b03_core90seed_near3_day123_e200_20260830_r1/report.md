# ADV3B03复用CORE90原seed及邻域3-seed正式实验

- Run ID：`phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1`
- 当前状态：`LOCAL_VERIFIED`
- 类型：纯Phase1接收机域泛化训练；与Phase2无关。
- 启动所有者：当前主Agent；同一Run ID只允许一次正式启动。
- 方法：仅`ADV3B03_MU10_ALPHA20_E200`，从头训练，不加载ADV3B02 checkpoint；只复用CORE90的随机种子中心。
- CORE90原训练seed证据：历史严格checkpoint配置`checkpoint_args.seed=392002`，候选为`ADV3B02_CORE90_SOFT_E200`。
- 正式seed：`392001`、`392002`、`392003`。
- Git分支：`codex/phase1-fasttrust-eff-src5-20260828`。
- code/config commit：提交后补记。

## 冻结配置

- 数据：`Dataset_WigSig/ManySig.pkl`。
- source receivers：`1,3,4,6,8`。
- 训练天：day1、day2、day3；day4不用于训练。
- source角色：`L_s=0.07`、`U_s=0.63`、`V_cal=0.15`、`V_select=0.15`。
- epoch：200；batch size：128；final checkpoint：`final_ssdg.pth`。
- 星地增强：与ADV3B02同款concat masked/CE-only拼接增强，`concat_sat_ce_weight=1.0`；课程为`1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- source选种：仅按同seed的`V_select` clean与`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`严格结果比较，目标接收机结果不得反馈选种、调参、重训或重跑。
- 训练完成要求：每行E200、final checkpoint、严格重建、source clean及3种LEO场景artifact完整。
- 后续测试：冻结source最佳seed后，测试全部7个目标接收机在day1/day2/day3/day4上的clean及3种LEO场景；零适配、无状态更新。

## 环境、路径与命令

- 本地环境：`ssr-gpu`。
- 本地CWD：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\phase2-canonical-union-maxq`。
- N607账户：普通账户`szu2070436088`，禁止使用管理员账户。
- N607项目根：`/home/szu2070436088/2510044040/CV-SincNet`。
- release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1_release1`。
- 运行根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1`。
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1`。
- GPU：预检后按资源状态使用GPU0、GPU1、GPU2各1个实验；若存在无关任务，只要每卡不超过2个训练实验且无显存冲突即可，不得影响无关进程。

正式命令：

```bash
nohup /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1_release1/code/scripts/launch_phase1_adv3b03_core90seed_near3_day123_e200_20260830.py --root /home/szu2070436088/2510044040/CV-SincNet --code-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1_release1 --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --run-id phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1 --runs-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1 --log-root /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1 > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1.dispatcher.log 2>&1 < /dev/null &
```

## 本地验证与停止规则

- 聚焦测试：`13 passed`。
- Python语法编译：通过。
- `git diff --check`：通过。
- 独立P0/P1审查：无P0/P1；审查者未修改文件。
- 预期artifact：3个候选各自的`config.json`、`train.log`、`final_ssdg.pth`、`metrics_joint.json`、4份场景指标与日志、`status.txt=ARTIFACTS_COMPLETE`；日志根含`plan.json`和`final_status.json`。
- 只在错误stage/receiver/day/seed/场景、目标域或Phase2被访问、输出碰撞、错误checkout、命令不能启动、同一确定性异常至少两行复现、无final checkpoint、严格重建失败或必要评估artifact缺失时，才绑定并停止该Run ID的精确进程树，同时保留partial artifact。
- 低性能、收敛慢或中间指标差不得停止、重启或热补丁；不得触碰任何无关进程。
