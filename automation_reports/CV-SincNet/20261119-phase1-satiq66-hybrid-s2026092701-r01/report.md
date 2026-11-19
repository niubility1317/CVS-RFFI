# SatIQ Oxford 66-class Single-Source Hybrid

- run_id：`20261119-phase1-satiq66-hybrid-s2026092701-r01`
- group_id：`phase1-satiq-oxford-cross-time-hybrid`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Matched supervised mechanisms followed by matched ordinary PL and Hybrid SSL

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## Source input and validation

Balanced physical IDs obtained from 10 original archives. Hidden-U public order is shuffled independently of classes. Source-only template median absolute correlations over 3,520 L messages were 0.694 (raw) and 0.851 (narrow). A freshly serialized 66-class model passed both real L-batch smokes locally, including five finite nonzero branch gradients and 132 valid anchors for each supervised mechanism. Independent P0/P1 review found a CPU fallback route, now closed by requiring CUDA. Actual extra noise SNR is logged.

## N607 launch

VERIFIED controller PID 1774733. Run path and CWD read back independently. Both code and source-public archive SHA256 values matched. The N607 raw and narrow real-checkpoint smokes each activated 132 same-packet, cross-day and prototype anchors and all five branch gradients. The launch controller owns GPUs 2 and 3 and advances input pilots to the supervised pair, then to the SSL pair after supervised artifact and source V checks.

Both 300-update input pilots completed and were near chance. V macro F1 was 0.4946% for 25 MS/s and 0.2097% for 0.8 MS/s, so the short raw window won the fixed source-only selection rule. S0 and S1 launched on GPU 2 and 3 (PIDs 1776524 and 1776525). In S1 E1, same-packet, cross-day and prototype terms each had 13,200 anchor exposures. The Q builder exported 293,371 late Oxford records and verified disjoint full-IQ hashes. Future inputs are stored separately from scorer truth.

The independent future readout process, PID 1779115, was verified in its release directory. It waits for all four fixed source endpoints, launches the four prediction jobs in pairs, then invokes the independent scorer. Candidate training commands received no Q or truth paths.

## Supervised endpoint and SSL launch

| Row | Updates | Source V accuracy | Source V macro F1 |
|---|---:|---:|---:|
| S0 | 5000 | 1.4899% | 1.4606% |
| S1 | 5000 | 1.5606% | 1.5270% |

S1 accumulated 646,800 same-record physical contrast exposures, 660,000 cross-day exposures, and 646,800 prototype exposures. The future test set has not been scored. End-of-training V accuracy remained around the 66-class uniform chance value of 1.5152%. The plan requires technical validation before SSL dispatch, with no performance stop rule. U0 and U1 were verified running at independently initialized warmup on GPU 2 and GPU 3, PID 1781675 and 1781676.

At U0/U1 epoch 14, each run evaluated 25,600 hidden-U exposures and accepted zero pseudo-labels at 0.95. U1 physical routing, auxiliary U losses and branch protection start at epoch 21. The source-only tests confirmed the high-confidence and empty-mask U routes with finite gradients. Continued runs use the fixed threshold and endpoint.

## Recorded technical failure

U1 failed at the first E21 update because N607 PyTorch 2.1 does not implement Tensor.any with tuple dimensions. The failing tensor was the per-record nonzero perturbation mask. Exact behavior reproduced on the N607 runtime using synthetic inputs. R02 replaces U1 only from scratch and keeps the three healthy fixed endpoints for common future scoring. No target data were inspected as part of the repair.
