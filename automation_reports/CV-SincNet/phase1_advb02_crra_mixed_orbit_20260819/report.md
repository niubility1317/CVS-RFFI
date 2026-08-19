# ADVB02 CRRA`mixed_orbit`Phase1最小验证报告

## 预登记

- 实验ID：`phase1_advb02_crra_mixed_orbit_20260819`。
- 目标：在当前Phase1数据协议下，验证ADVB02 CRRA对历史`mixed_orbit`星地信道视图的训练闭环、稳定性遥测和同row性能方向。
- 数据比例：WiSig `0.07/0.63/0.30`，分别对应labeled/unlabeled/source-validation；不访问目标接收机、目标阈值、目标校准或query truth。
- 星地信道：仅使用历史`mixed_orbit`；本任务不使用`leo_*_weak`作为训练星地信道。
- 候选：`ADV3B02_MIXED_ORBIT_E200`控制行与`ADVB02_CRRA_MIXED_ORBIT_E200`改造行，共用seed=`392033`、split=`tx_rx_day_1_7_2`和同一run ID。
- CRRA配置：identity-only；rank=`8`；alpha上限=`0.25`；shrinkage=`0.10`；condition dim=`32`；nuisance dim=`9`；E1–16关闭、E17–46线性启用、E47+固定；PA旁路；domain分支不使用CRRA。
- 新损失起点：pair=`0.05`、satellite KL=`0.05`、correction energy=`0.001`、gate L1=`0.001`、nuisance Huber=`0.02`、condition TX adversary=`0.02`。
- 预期artifact：每个候选的最终checkpoint、训练日志、metrics CSV/JSONL、同row source-validation satellite指标和CRRA遥测（pair cosine、correction energy、gate、alpha、support distance、nuisance loss、condition TX accuracy）。

## 固定路径与命令

| 项目 | 值 |
|---|---|
| 本地Git承载 | `E:/type10-7/github_publish/CVS-RFFI-repo` |
| launcher | `code/scripts/launch_phase1_advb02_crra_mixed_orbit_20260819.sh` |
| 远端项目根 | `/home/szu2070436088/2510044040/CV-SincNet` |
| 远端run根 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_crra_mixed_orbit_20260819` |
| 远端log根 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_crra_mixed_orbit_20260819` |
| GPU | `0`，不超过该GPU默认两个训练进程 |
| 环境 | 远端既有`CVS-RFFI`Python环境；本地验证使用`ssr-gpu` |
| 技术停止 | 仅协议越权、错误checkout、输出覆盖、launcher/确定性启动故障、无prediction闭合；不因低性能停止 |

本地固定提交：`2d31cca4`（CRRA实现及本次Phase1启动交接）。

唯一启动入口：

```bash
bash code/scripts/launch_phase1_advb02_crra_mixed_orbit_20260819.sh --only=ADVB02_CRRA_MIXED_ORBIT_E200
```

先做dry-run：

```bash
bash code/scripts/launch_phase1_advb02_crra_mixed_orbit_20260819.sh --dry-run
```

## 本地验证

- `MSYSTEM=MINGW64`已核验；终端使用Git Bash，测试环境为`ssr-gpu`。
- CRRA聚焦与相邻测试：17个CRRA测试通过；协议负测、旧checkpoint loader和基础view回归通过。
- 真实checkpoint无query冒烟：本地`best_joint_safe_ssdg.pth`旧模型严格重建通过；CRRA关闭与CRRA结构补入后分别对clean和同一次生成的`mixed_orbit`视图前向，logits有限且shape一致。
- 预登记前应重新记录本报告对应Git提交、N607只读preflight、传输归档和远端compile结果；不增加额外seal、authority或逐文件hash门。

## 结果记录

待预测闭合后只记录同row结果：clean TX、`mixed_orbit`source-validation TX、receiver/day分层（若该row产生）、CRRA遥测和最终判定。不得拼接不同row的单指标最高值，也不得把本报告的单批冒烟当作性能证据。

## 当前状态

`LOCAL_VERIFIED`：代码、配置、launcher和报告已在本地Git工作区固定；远端旧历史任务保持monitor-only，尚未因本任务被停止、重启或修改。
