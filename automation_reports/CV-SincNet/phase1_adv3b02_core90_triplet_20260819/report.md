# ADV3B02 CORE90 三组E200重跑

## 运行前登记

- 目的：在同一ADV3B02 CORE90训练配置下，比较`seed=392034`的`0.07/0.63/0.30`与`0.10/0.70/0.20`两种数据比例，并复现历史`ADV3B02_CORE90_SOFT_E200`（`seed=392002`、`0.10/0.70/0.20`）。
- 三组实验均使用历史CORE90的弱LEO训练视图：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；使用`use_concat_sat_channel_aug=1`、`concat_sat_ce_only=1`、`lambda_sat_cls=0.68`、`sat_cons_start_epoch=80`。
- 三组均训练`200`轮（`label_epochs=130`、`pseudo_epochs=70`），`split_mode=tx_rx_day_1_7_2`，ManySig，模型`M/lite_d`，历史CORE90 loss、MixStyle、伪标签、开放世界和原型相关参数从远端历史checkpoint参数重建。
- 按用户最新要求，训练完成后三组统一使用各自的`final_ssdg.pth`进行独立测试和结果汇总，不使用`best_joint_safe_ssdg.pth`作为最终报告checkpoint。
- 远端代码：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_cipg_mixed_screen_20260819/code/SSDG/train_ssdg.py`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD：`/home/szu2070436088/2510044040/CV-SincNet`。
- 训练输出根（不可覆盖）：
  - 当前代码入口实际运行根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_core90_match_seed392034_073_20260819_retry2`
  - 历史入口实际运行根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_core90_match_seed392034_102_20260819_legacy`
  - 历史入口实际运行根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_core90_repro_seed392002_102_20260819_legacy`
- GPU安排：GPU1、GPU2、GPU3各运行一组；启动前直连N607预检显示8张3090均空闲。
- 预期产物：`final_ssdg.pth`、`metrics_epoch.csv`、`metrics_epoch.jsonl`、`phase2_zid_prototypes.pt`及训练日志；训练结束后另建独立evaluation-only输出目录并测试clean与三个弱LEO场景。
- 技术停止规则：仅在参数/入口错误、Traceback、CUDA OOM、输出覆盖、进程归属错误、无日志增长或prediction/test无法生成时停止；不因中间性能高低停止。

## 配置矩阵

| run_id | candidate_id | seed | labeled/unlabeled/source-val | GPU | final checkpoint |
|---|---|---:|---|---:|---|
| `phase1_adv3b02_core90_match_seed392034_073_20260819_retry2` | `ADV3B02_CORE90_MATCH073_E200_S392034_R2` | 392034 | 0.07/0.63/0.30 | 1 | `final_ssdg.pth` |
| `phase1_adv3b02_core90_match_seed392034_102_20260819_legacy` | `ADV3B02_CORE90_MATCH102_E200_S392034_LEGACY` | 392034 | 0.10/0.70/0.20 | 2 | `final_ssdg.pth` |
| `phase1_adv3b02_core90_repro_seed392002_102_20260819_legacy` | `ADV3B02_CORE90_REPRO_FINAL_E200_S392002_LEGACY` | 392002 | 0.10/0.70/0.20 | 3 | `final_ssdg.pth` |

## 启动命令说明

启动时以历史CORE90 checkpoint的CLI参数为基线，仅覆盖上表的`run_id`、`candidate_id`、`seed`、三项数据比例、`output_dir`、metrics路径、GPU和`--checkpoint_selection final_only`；历史配置中`use_concat_sat_channel_aug=1`、`concat_sat_ce_only=1`、弱LEO三场景训练/评估、`best_metric=joint_safe`、`test_eval_policy=interval_final`等均显式保留。

## 运行记录

## 首次启动技术失败与重启

2026-08-19首次启动的三组进程均在训练前被同一项现有启动校验拒绝：`checkpoint_selection=final_only`时不允许`best_metric=joint_safe`，日志为`Phase1 source-only checkpoint selection forbids test/receiver/satellite-test best metrics`。三组均未进入E001，GPU未产生训练占用；失败输出保留在原三个输出根中。

重启保持用户指定的`final_ssdg.pth`最终checkpoint不变，仅将训练过程中不参与最终checkpoint选择的`best_metric`改为合法的`source_val_sat_hmean`，并使用后缀`_retry1`的新输出根，避免覆盖失败尝试。

重启输出根：

- `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_core90_match_seed392034_073_20260819_retry1`
- `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_core90_match_seed392034_102_20260819_retry1`
- `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_core90_repro_seed392002_102_20260819_retry1`

`_retry1`三组同样在训练前被当前脚本拒绝，原因是历史`enable_joint_safe_guard=True`会启用held-out test joint guard，而Phase1 source-only入口要求该项关闭。三组仍未进入E001，GPU未产生训练占用；失败日志保留。

因此`_retry2`只关闭`enable_joint_safe_guard`，保留`best_metric=source_val_sat_hmean`和`checkpoint_selection=final_only`，其余CORE90参数、弱LEO增强、seed和数据比例不变。

`_retry2`输出根：

- `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_core90_match_seed392034_073_20260819_retry2`
- `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_core90_match_seed392034_102_20260819_retry2`
- `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_core90_repro_seed392002_102_20260819_retry2`

三组实际训练均已完成`E200/200`，训练进程已退出，GPU1/2/3释放；当前代码组已生成`final_ssdg.pth`。两条历史入口按其原始实现生成`latest_ssdg.pth`而不生成`final_ssdg.pth`，将只在确认E200后以不覆盖方式复制为最终测试文件；随后按最新`AGENTS.md`分别执行clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`测试。

## 训练完成与checkpoint

三组日志均包含`[EPOCH-END] E200/200`，训练进程已正常退出；没有Traceback、CUDA OOM、未识别参数或日志停滞。当前发布版组直接产生`final_ssdg.pth`；两个历史入口的`latest_ssdg.pth`在确认E200后复制为同一run目录下的`final_ssdg.pth`，原始`latest_ssdg.pth`和`latest_safe_ssdg.pth`均保留。最终测试没有使用`best_joint_safe_ssdg.pth`或`best_epoch`权重。

| run | E200 final checkpoint | checkpoint epoch | checkpoint加载 | 训练metrics行数 |
|---|---|---:|---|---:|
| `match073` | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_core90_match_seed392034_073_20260819_retry2/final_ssdg.pth` | 200 | missing=0，unexpected=0 | 200 |
| `match102` | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_core90_match_seed392034_102_20260819_legacy/final_ssdg.pth` | 200 | missing=0，unexpected=0 | 200 |
| `repro102` | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_core90_repro_seed392002_102_20260819_legacy/final_ssdg.pth` | 200 | missing=0，unexpected=0 | 200 |

## 训练后final checkpoint测试

按照最新`AGENTS.md`，三组均使用独立evaluation-only入口和同一测试配置：`tools/eval_cvs_checkpoint_sat_channel.py`、`device=cuda:4`、`eval_batch_size=256`、`eval_max_batches=0`、`sat_eval_max_batches=0`、`eval_sat_channel=true`、`eval_sat_on=main`、`eval_sat_scenarios=leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。每个LEO场景均为完整204000条测试样本；clean结果为同一final checkpoint上的完整held-out测试。评估输出根为：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_core90_triplet_20260819_eval_final/`。

其中`clean overall TX`是clean测试的三种receiver/day组合聚合TX准确率；每个LEO场景同时报告聚合TX准确率和`strict_udu`，不以场景均值替代逐场景结果。

| 候选 | seed | L/U/V | clean overall TX | `leo_clear_weak` TX / strict UDU | `leo_low_elev_weak` TX / strict UDU | `leo_rain_weak` TX / strict UDU | 三LEO均值 TX / strict UDU |
|---|---:|---|---:|---:|---:|---:|---:|
| `match073` | 392034 | 0.07/0.63/0.30 | 83.9559% | 70.4020% / 63.8233% | 67.6377% / 61.2250% | 67.5451% / 60.8167% | 68.5283% / 61.9550% |
| `match102` | 392034 | 0.10/0.70/0.20 | 85.6593% | 73.5907% / 64.3133% | 71.0711% / 62.3967% | 70.4127% / 61.7917% | 71.6915% / 62.8339% |
| `repro102` | 392002 | 0.10/0.70/0.20 | 89.3627% | 77.2549% / 71.2217% | 74.7569% / 68.9283% | 74.5206% / 68.5667% | 75.5108% / 69.5722% |

两组差值保持同一测试字段、同一final checkpoint口径：

| 对比 | clean TX差 | clear TX差 | low-elev TX差 | rain TX差 | 三LEO均值TX差 | 三LEO strict UDU均值差 |
|---|---:|---:|---:|---:|---:|---:|
| `match102 - match073`（同seed，L/U/V变化） | +1.7034pp | +3.1887pp | +3.4333pp | +2.8676pp | +3.1632pp | +0.8789pp |
| `repro102 - match102`（同L/U/V，历史seed392002对seed392034） | +3.7034pp | +3.6642pp | +3.6858pp | +4.1078pp | +3.8193pp | +6.7383pp |
| `repro102 - match073`（总跨度） | +5.4069pp | +6.8529pp | +7.1191pp | +6.9755pp | +6.9825pp | +7.6172pp |

## 解释与边界

1. `match102 - match073`是在同一seed=392034、同一CORE90损失/增强/训练视图下改变L/U/V比例的对照；当前数据中0.10/0.70/0.20高于0.07/0.63/0.30，三场景TX均提高，strict UDU均值提高0.8789个百分点。
2. `repro102 - match102`只保持L/U/V不变，seed从392034换为历史seed392002；clean和三个LEO场景均更高，说明seed差异在本三行中不可忽略，不能把这部分收益归因于数据比例。
3. `repro102 - match073`同时包含比例和seed变化，只作为总跨度，不作为单一因素因果结论。
4. 三组训练增强配置一致：均为三个`leo_*_weak`训练视图、`use_concat_sat_channel_aug=1`、`concat_sat_ce_only=1`、`mode=ce_only_aux`、`lambda_sat_cls=0.68`、`sat_cons_start_epoch=80`；`mixed_orbit`没有作为本次默认训练或测试场景。
5. `0.07/0.63/0.30`满足当前协议按`L/(L+U)`计算的0.10标注率；用户要求的历史`0.10/0.70/0.20`按同一计算为0.125，因此后两组使用历史CORE90入口以复现指定历史配置，并在本报告中标为历史复现/比较，不把它们写成当前`p2_min_v1`标注率门槛下的严格protocol-compliant新基线。
6. 训练完成才执行测试；三组最终状态为`ARTIFACTS_COMPLETE`，clean与三个LEO场景均有JSON/TXT证据。独立测试证据已归档在本报告目录的`artifacts/eval_final/{match073,match102,repro102}/`，训练CSV和日志在`artifacts/training_logs/`。

## 归档与版本状态

- 远端训练与测试产物保留在N607上述路径；未复制checkpoint本体到本地报告目录。
- 本地报告和小型证据文件已写入`E:\\type10-7\\automation_reports\\CV-SincNet\\phase1_adv3b02_core90_triplet_20260819`。
- `E:\\type10-7`根目录不是Git仓库；按最新`AGENTS.md`，完成报告后将只把本报告及必要小型证据说明镜像到`github_publish/CVS-RFFI-repo`，不stage checkpoint、dataset、runs或无关未跟踪文件。

## 运行状态结论

三组训练和三组final checkpoint测试均已完成，evaluation launcher已正常退出，GPU1/2/3/4均已释放。最终状态：`ANALYZED`。旧的首次启动失败和`_retry1`失败输出仅作为启动诊断保留，不计入三组实验结果。
