# WiSig 无门控恢复补齐 clean 的主对比

运行 `20261125-phase1-wisig-dura-ungated-clean-m2-r01`。沿用原 Main split 的三份冻结 Tri-SIG-TIME 识别器，以及 Estimated 和 Blind 各 40,000 步的原冻结权重。使用原半量更新。新运行只为 clean 实收 IQ 补算恢复，所有 clean 包都实际调用两份模型并将各自输出交给冻结识别器。直通单列，使用当前实收 IQ。

固定 cross_rx 126,000 包、cross_both 42,000 包，合并 168,000 个互斥物理包。每个划分逐批核对 Full 和 Residual 的物理顺序及 clean IQ，验证一致后共用一次 clean 推理。复用 `20261125-phase1-wisig-dura-force-main-m2-r01`（提交 `5b78d9524`）的常规六景强制恢复预测，逐包验证原 R69 和直通与已实跑的 `20261125-phase1-wisig-r69-fresh-main-m3-r01` 一致。原强制实验的 clean 直通产物保留。恢复与直通分别报告准确率、同包净纠错、各景各机结果及配对区间。两路均报告相对同包 R69 的六景和 clean 验收，以及每种子 clean 明细。加强扰动的强制结果仍随原运行报告。

N607 普通账号，进入归档强制恢复运行环境，使用固定新输出根 `runs/20261125-phase1-wisig-dura-ungated-clean-m2-r01`。发布入口 `python -m experiments.pamf_unrolled_restore.wisig_forced_clean publish --repo <checkout> --output <new-release-artifacts> --context <registered-WiSig-context> --ssh-config <direct-config>`。先对合法源 U 的 12 个 clean 包做实际无门控检查，之后按实际负载选择有显存余量的 GPU，同时最多两个划分 worker。待新 clean 和引用的强制矩阵均冻结且原强制运行完成独立评分，另起 CPU 进程验证覆盖与物理配对，再读取真值。来源、实收 IQ 一致性、覆盖、数值或闭合失败时保留产物并停止本运行下游。预期产物为 `smoke.json`、两份 clean 冻结回执及决策数组，以及独立 `scores.json`。 [原 R69 报告](WISIG_ESTIMATED_DURA_MAIN_20261125.md)

## 常规预测核验

两个 clean 分片及四个常规实收分片均已冻结。每条路线的每个候选在 clean 调用 168,000 次，六景调用 1,008,000 次。已逐项核对两划分的物理 ID、恢复调用计数、三个种子的直通和原 R69 决策。直通和原 R69 在各自配对行与实跑结果完全一致。核验读取真值为零。

| 模型 | Clean 改变类别 | Full 六景改变类别 | Residual 六景改变类别 |
| --- | ---: | ---: | ---: |
| Estimated | 4,372 | 1,089,569 | 1,089,493 |
| Blind | 4,619 | 1,082,651 | 1,082,542 |

各数为相对直通的类别变化，跨三个种子统计。Clean 的分母为 504,000 次判定，六景每路线分母为 3,024,000 次判定。

## 独立正式评分

**VERIFIED**。引用运行的八个分片及本轮两个 clean 分片全部冻结后，独立 CPU 完成评分。表中每个数是对应划分三个冻结识别器的六景准确率。

| 路线 | 划分 | 直通 | 原 R69 | Estimated | Blind |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | cross_rx | 71.1670% | 79.9027% | 57.2291% | 57.4884% |
| Full | cross_both | 68.8069% | 76.8267% | 54.3201% | 54.5759% |
| Full | 合并 | 70.5770% | 79.1337% | 56.5018% | 56.7603% |
| Residual | cross_rx | 71.1649% | 79.9047% | 57.2259% | 57.4865% |
| Residual | cross_both | 68.8083% | 76.8291% | 54.3212% | 54.5849% |
| Residual | 合并 | 70.5758% | 79.1358% | 56.4997% | 56.7611% |

两条路线的 clean 相同。直通正确数为 438,071，原 R69 为 438,286，Estimated 为 439,365，Blind 为 439,816，分母均为 504,000。两候选相对原 R69 分别净增 1,079 和 1,530，三个识别种子的 clean 正确数各自也达到原 R69。

| 路线 | 模型 | 六景改对／改错（相对原 R69） | 六景差值与配对 95% 区间 | 低强度城区差值 |
| --- | --- | ---: | ---: | ---: |
| Full | Estimated | 229,892／914,280 | −22.63 [−22.78，−22.48] 个百分点 | −26.12 个百分点 |
| Full | Blind | 229,162／905,734 | −22.37 [−22.52，−22.22] 个百分点 | −25.98 个百分点 |
| Residual | Estimated | 229,846／914,361 | −22.64 [−22.79，−22.48] 个百分点 | −26.13 个百分点 |
| Residual | Blind | 229,151／905,762 | −22.37 [−22.53，−22.22] 个百分点 | −25.99 个百分点 |

两候选达到了 clean 条件，两条路线的六景均显著低于同包原 R69，未通过主对比验收。原强制任务的加强扰动结果得到一致方向，原 R69 约为 74.92%，Estimated 和 Blind 约为 54.83% 和 55.07%，其 clean 在该任务口径中单列直通。本运行的 `scores.json` 保留常规两路线、划分、逐场景、逐接收机和逐种子的结果。

## 原 R69 的训练来源与推理输入

R69 推理使用接收窗口的复数 IQ，内部估计信道作用和噪声方差，再通过两阶段先验与共轭梯度求解恢复同一窗口。历史部署入口另有短训练字段重复差分门控。R69 本身从 R65 `prune_after10` 复制全部保留参数，使用配对源 clean IQ 的归一化均方误差、频谱损失和冻结 R65 的分阶段 IQ 蒸馏。R69 训练更新可用的输入为源域配对 IQ，TX 标签、识别器和目标数据均未参与这些更新。 [R69 结构与初始化](../../experiments/pamf_r65_four_route/stage_polyphase.py) [R69 训练](../../experiments/pamf_r65_four_route/training.py)

R65 从 M35 初始化并使用同一 M35 作为蒸馏教师，谱系继续追溯至 R07 `learned_operator`。R07 从模拟信道的实际作用构造白探针监督，并以仿真器计算的真实噪声方差监督方差估计器。R69 登记还披露其 R65 父版本由人工在已看过历史正式结果后指定。这两点决定 R69 保留为同包冻结对照。后续新候选用当前合法源 IQ 从头训练，沿用适用结构时也须重新学习信道估计器。 [R65 初始化](../../experiments/pamf_r46_delayed_prune/runtime.py) [R07 权重继承](../../experiments/pamf_uncertainty_restore/runtime.py) [R07 仿真监督构造](../../experiments/pamf_action_restore/data.py) [R07 实际损失](../../experiments/pamf_action_restore/runner.py) [R69 登记](../../automation_reports/CV-SincNet/20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69/experiment.json)
