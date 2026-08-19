# ADVB02 CRRA`mixed_orbit`Phase1拼接监督实验报告

## 预登记

- 实验ID：`phase1_advb02_crra_mixed_orbit_20260819_r2`；前两次技术失败尝试分别保留为`phase1_advb02_crra_mixed_orbit_20260819`和`phase1_advb02_crra_mixed_orbit_20260819_r1`。
- 目标：在当前Phase1数据协议下，验证ADVB02 CRRA对历史`mixed_orbit`星地信道视图的拼接监督训练闭环、稳定性遥测和同row性能方向。
- 数据比例：WiSig `0.07/0.63/0.15/0.15`，分别对应`L_s/U_s/V_cal/V_select`；`V_select`只用于源侧选模，`V_cal`只用于校准/导出，不访问目标接收机、目标阈值或query truth。
- 星地信道：训练增强仅使用历史`mixed_orbit`（代码规范名）；本任务不使用`leo_*_weak`作为训练星地信道。
- 候选：`ADV3B02_MIXED_ORBIT_E200`控制行与`ADVB02_CRRA_MIXED_ORBIT_E200`改造行，共用seed=`392033`、split=`tx_rx_day_1_7_2`和同一run ID。
- 卫星训练模式：`concat_masked`。clean分支保留完整ADVB02主损失；satellite分支单独前向，只承载有监督TX CE、同视图nuisance Huber和有界类别壳层约束；每步为`B+B`两次前向，不把卫星样本复制进clean主分支。
- 首轮损失：`lambda_sat_cls=0.50`、`lambda_sat_cons=0`；CRRA点对点pair=`0`、satellite KL=`0`；correction energy=`0.001`、gate L1=`0.001`、nuisance Huber=`0.02`、condition TX adversary=`0.02`、satellite shell=`0.15`、shell width=`12°`。代码保留非零KL的独立可选路径，但本轮不启用。
- CRRA配置：identity-only；rank=`8`；alpha上限=`0.25`；shrinkage=`0.10`；condition dim=`32`；nuisance dim=`9`；E1–16关闭、E17–46线性启用、E47+固定；PA旁路；domain分支不使用CRRA。
- 预期artifact：每个候选的最终checkpoint、训练日志、metrics CSV/JSONL、V_select source-validation satellite指标、四角色split receipt和CRRA遥测（satellite CE、shell、nuisance、correction energy、gate、alpha、support distance、condition TX accuracy）。本轮未启用Phase2 prototype导出，因此不宣称已生成V_cal校准artifact；若后续启用导出，校准输入固定为V_cal。

## 固定路径与命令

| 项目 | 值 |
|---|---|
| 本地Git承载 | `E:/type10-7/github_publish/CVS-RFFI-repo` |
| launcher | `code/scripts/launch_phase1_advb02_crra_mixed_orbit_20260819.sh` |
| 远端项目根 | `/home/szu2070436088/2510044040/CV-SincNet` |
| 远端run根 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_crra_mixed_orbit_20260819_r2` |
| 远端log根 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_crra_mixed_orbit_20260819_r2` |
| GPU | `3`；最终发布前复核GPU3仅有一个既有轻量Stage2预测进程且利用率为0，因此再放置本run后不超过两个进程；不超过单GPU默认两个训练进程 |
| 环境 | 远端既有`CVS-RFFI`Python环境；本地验证使用`ssr-gpu` |
| 技术停止 | 仅协议越权、错误checkout、输出覆盖、launcher/确定性启动故障、无prediction闭合；不因低性能停止 |

前两次启动artifact均已保留，不覆盖。最终归档：本地`E:/type10-7/local_artifacts/phase1_advb02_crra_mixed_orbit_20260819/cvs-rffi-a5ca5594.tar.gz`→远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_crra_mixed_orbit_20260819_r1_a5ca5594.tar.gz`；单次归档SHA256=`fb09605f4ed92f9327d67428e8e39948c8563d714039235ba9ff0c36bf3faca6`。远端展开目录为`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_crra_mixed_orbit_20260819_r1_a5ca5594`，远端Python编译与launcher语法检查已通过；训练数据固定读取项目根的`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。

本地固定提交：实现`cd970f1f`（`exp: align ADVB02 CRRA concat supervision with Phase1 roles`），启动修复`21a23878`（`fix: remove stale ADVB02 launch argument`）。

唯一启动入口（最终重试run）：

```bash
ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_crra_mixed_orbit_20260819_r1_a5ca5594 WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl GPU=3 RUN_ID=phase1_advb02_crra_mixed_orbit_20260819_r2 bash code/scripts/launch_phase1_advb02_crra_mixed_orbit_20260819.sh
```

先做dry-run：

```bash
bash code/scripts/launch_phase1_advb02_crra_mixed_orbit_20260819.sh --dry-run
```

## 本地验证

- `MSYSTEM=MINGW64`已核验；终端使用Git Bash，测试环境为`ssr-gpu`。
- CRRA、Phase1四角色拆分、协议负测、旧checkpoint兼容性和launcher聚焦集：30 passed；无失败；两条dry-run命令均通过当前argparse参数解析。
- 真实checkpoint无query冒烟：本地`best_joint_safe_ssdg.pth`旧模型严格重建通过；CRRA关闭与CRRA结构补入后分别对clean和同一次生成的`mixed_orbit`视图前向，logits有限且shape一致。
- 拼接监督：`concat_masked`保持clean主损失完整，satellite只进入独立CE/nuisance/shell路径；首轮`lambda_crra_pair=0`、`lambda_crra_sat_kl=0`，不会引入clean-sat点对点/KL监督。
- CRRA卫星KL回归：代码验证`lambda_sat_cons=0`时仍可由独立`lambda_crra_sat_kl>0`在E17后物化；本轮显式置零，E1–16不提前计算。
- 元数据契约：`snr_db,cfo_hz,residual_cfo_hz,fD_hz,pl_db,K_db,theta_deg,h_km,state`始终固定9维；缺字段整行无效，不做左移拼接或截断回归。
- Phase1角色：训练日志打印`L_s/U_s/V_cal/V_select`；`V_select`进入选模，`V_cal`进入校准/导出，四个角色物理样本索引两两不交。
- 本地实现已固定为`cd970f1f`，启动修复为`21a23878`；首次失败是旧launcher传入不存在的`--dom_feature_key feat_imp`，第二次`_r1`失败是发布目录不含外部WiSig数据路径，二者均未进入有效训练且run/log完整保留。最终启动只显式绑定项目根只读数据文件并改用不可覆盖`_r2`run ID，不改变模型、数据协议或损失设定；不增加额外seal、authority或逐文件hash门。

## 结果记录

待预测闭合后只记录同row结果：clean TX、V_select上的`mixed_orbit`source-validation TX、receiver/day分层（若该row产生）、CRRA遥测和最终判定。不得拼接不同row的单指标最高值，也不得把本报告的单批冒烟当作性能证据。

## 当前状态

`LANDED`：前两次技术启动失败已被保留并隔离；最终`_r2`启动命令已绑定已验证的数据路径，代码归档已落地并完成远端编译，待启动唯一run owner。所有远端旧历史任务保持monitor-only。
