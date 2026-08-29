# ADV3B03 CORE90邻域新增5-seed满卡实验

- Run ID：`phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1`
- 当前状态：`LOCAL_VERIFIED`
- 目的：在不干扰已运行near3任务的前提下，使用当前空闲GPU3–7各新增1个邻近seed实验，使8张GPU各有1个正式训练实验。
- 类型：纯Phase1接收机域泛化训练；与Phase2无关。
- 启动所有者：当前主Agent；同一Run ID只允许一次正式启动。
- 方法：仅`ADV3B03_MU10_ALPHA20_E200`，从头训练。
- CORE90原训练seed：`392002`。
- 新增seed与GPU：`392000→GPU3`、`392004→GPU4`、`391999→GPU5`、`392005→GPU6`、`391998→GPU7`。
- 已运行但不属于本Run ID的seed：`392001/392002/392003`位于GPU0/1/2，禁止触碰。
- Git分支：`codex/phase1-fasttrust-eff-src5-20260828`。
- code/config commit：提交后补记。

## 冻结配置

- 数据：`Dataset_WigSig/ManySig.pkl`。
- source receivers：`1,3,4,6,8`。
- 训练天：day1、day2、day3；day4不用于训练。
- source角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- epoch：200；batch size：128；final checkpoint：`final_ssdg.pth`。
- 星地增强：与ADV3B02同款concat masked/CE-only拼接增强，`concat_sat_ce_weight=1.0`；课程为`1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- 训练完成要求：每行E200、final checkpoint、严格重建、source V_select clean及3种LEO场景artifact完整。
- 选种：near3与near5共8个seed全部完成后，仅按source V_select冻结最佳seed；目标接收机结果不得反馈选种、调参、重训或重跑。
- 后续测试：冻结最佳seed后测试全部7个目标接收机×day1/2/3/4的clean与3种LEO场景，零适配且无状态更新。

## 路径与正式命令

- 本地CWD：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\phase2-canonical-union-maxq`。
- N607账户：普通账户`szu2070436088`，禁止管理员账户。
- release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1_release1`。
- 运行根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1`。
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1`。

```bash
nohup /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1_release1/code/scripts/launch_phase1_adv3b03_core90seed_near5_day123_e200_20260830.py --root /home/szu2070436088/2510044040/CV-SincNet --code-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1_release1 --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --run-id phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1 --runs-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1 --log-root /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1 > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1.dispatcher.log 2>&1 < /dev/null &
```

## 验证、artifact与停止规则

- 聚焦测试：`15 passed`。
- Python语法编译与`git diff --check`：通过。
- 独立P0/P1审查：无P0/P1；固定seed与GPU3–7映射、独立Run ID、E200及源域四场景闭合路径均正确，审查者未修改文件。
- 预期artifact：5个候选各自的`config.json`、`train.log`、`final_ssdg.pth`、`metrics_joint.json`、4份场景指标与日志、`status.txt=ARTIFACTS_COMPLETE`；日志根含`plan.json`和`final_status.json`。
- 只在错误stage/receiver/day/seed/GPU/场景、目标域或Phase2被访问、输出碰撞、错误checkout、命令不能启动、同一确定性异常至少两行复现、无final checkpoint、严格重建失败或必要评估artifact缺失时，才绑定并停止该Run ID的精确进程树，同时保留partial artifact。
- 低性能、收敛慢或中间指标差不得停止、重启或热补丁；不得触碰near3或其他无关进程。
