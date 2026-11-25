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

各数为相对直通的类别变化，跨三个种子统计。Clean 的分母为 504,000 次判定，六景每路线分母为 3,024,000 次判定。独立正确率仍按原登记等待引用运行完整评分后计算。
