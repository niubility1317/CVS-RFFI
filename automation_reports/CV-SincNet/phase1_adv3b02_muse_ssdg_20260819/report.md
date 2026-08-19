# ADV3B02-MUSE-SSDG Phase1最小预登记

## 候选矩阵

| 候选 | 固定基座 | 能力 | seed | epoch | source角色比例 | checkpoint选择 |
|---|---|---|---:|---:|---|---|
| M0 | `ADV3B02_CORE90_SOFT_E200` | 同协议ADV3B02控制；不进入MUSE能力路径 | 392002 | 200 | `0.07/0.63/0.15/0.15` | `final_only` |
| M1 | 同M0 | 基础domain/GRL/self/nuisance | 392002 | 200 | 同M0 | `final_only` |
| M2 | 同M0 | M1+fusion+H/M/L路由 | 392002 | 200 | 同M0 | `final_only` |
| M3 | 同M0 | M2+satellite student+cross-receiver+classification prototype | 392002 | 200 | 同M0 | `final_only` |

四个候选固定同一`tx_rx_day_1_7_2`数据split及`L_s/U_s/V_cal/V_select`角色定义。

## Commit

- Task 7基线提交：`198ba655a52f04bb63cbca4e92e6dedc936227af`。
- Task 7交付提交：本报告、launcher与测试所在的`feat: add MUSE Phase1 experiment launcher`提交；最终OID由Git提交与远端分支回读记录。

## 命令

```bash
bash code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh --only=M0,M1,M2,M3
```

本次Task 7仅执行本地`bash -n`、pytest与`--dry-run --only=M3`，不连接N607、不启动正式训练。

## 环境与CWD

- 计划环境：N607普通账户，`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 计划CWD：`/home/szu2070436088/2510044040/CV-SincNet`。
- 本地验证环境：`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`。
- 本地验证CWD：`E:/type10-7/github_publish/CVS-RFFI-repo/.worktrees/adv3b02-muse-ssdg`。

## 输入、输出与GPU

- 输入：`${ROOT}/Dataset_WigSig/ManySig.pkl`。
- 输出根：`${ROOT}/runs/phase1_adv3b02_muse_ssdg_20260819/{M0,M1,M2,M3}`；已存在的候选根禁止覆盖。
- GPU：默认`GPU=0`，所有子命令映射为`CUDA_VISIBLE_DEVICES=${GPU}`与进程内`cuda:0`。

## 停止规则

- 训练命令非零退出、`final_ssdg.pth`缺失或为空时停止当前候选并保留全部产物。
- 任一clean/LEO评测命令失败，或对应日志/metrics为空时停止，写`EVAL_FAILED_<SCENARIO>`并保留训练产物。
- 不因中间或最终性能高低停止。

## 预期artifact

每个候选根必须包含非空`train.log`、`config.json`、`final_ssdg.pth`、`eval_clean.log`、`eval_leo_clear_weak.log`、`eval_leo_low_elev_weak.log`、`eval_leo_rain_weak.log`、`metrics_clean.json`、`metrics_leo_clear_weak.json`、`metrics_leo_low_elev_weak.json`、`metrics_leo_rain_weak.json`。仅当四组评测日志与metrics均非空时，`status.txt`才写`ARTIFACTS_COMPLETE`。
