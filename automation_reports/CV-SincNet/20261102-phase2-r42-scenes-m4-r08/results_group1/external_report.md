# 原生外部方法与冻结 R42 的 raw 主对比

只纳入九个既有原生 profile 和 MoPC-HR。ISSL 按用户停止指令排除，Tweak／CSIL 保留 N/A；没有把旧全局队列的 FAILED／BLOCKED 视为其他方法失败。

## 八组 MIX 主工作点 K5/N20

| 外部方法 | 原生阶段 | 原生主指标（%） | 四 seed 均值样本 SD（百分点） | R42 raw 配对差（百分点） | LocalRidge raw 配对差（百分点） | 外部旧／新标签 | R42 旧／新标签 |
|---|---|---:|---:|---:|---:|---:|---:|
| dadda_sda | B | 72.981 | 0.550 | 1.261 | -6.113 | 30/0 | 30/0 |
| feature_separation_hu2024 | B | 45.791 | 4.071 | 28.451 | 21.076 | 30/0 | 30/0 |
| lora_triplet_knn | C | 4.807 | 0.858 | 53.049 | 43.499 | 0/100 | 30/100 |
| mopc_hr | C | 1.643 | 0.387 | 56.212 | 46.663 | 0/100 | 30/100 |
| mrior_sda | B | 71.626 | 0.577 | 2.616 | -4.758 | 30/0 | 30/0 |
| orthogonal | C | 23.411 | 0.258 | 34.445 | 24.895 | 0/100 | 30/100 |
| poster_ft | B | 35.954 | 1.203 | 38.288 | 30.914 | 30/0 | 30/0 |
| protonet_cda | B | 72.823 | 1.080 | 1.419 | -5.956 | 30/0 | 30/0 |
| radionet_ada_knn | B | 65.255 | 1.034 | 8.987 | 1.612 | 30/0 | 30/0 |
| twostage | B | 66.376 | 0.515 | 7.866 | 0.491 | 30/0 | 30/0 |

B 阶段使用旧类宏准确率，C 阶段使用宏 H，并另列旧／新类宏准确率、宏 F1 和逐类下限。原生 CIL 使用 0 个旧类目标标签与 N×K 个新类标签；R42／LocalRidge 使用 6×K 个旧类标签与 N×K 个新类标签，不能称为标签等价。DA 的目标旧类预算相同，也不意味着源训练条件等价。

外部 P0 与 R42／LocalRidge 的 S0Q0 均为原始 received 输入。没有把 DURA-only、融合或恢复 support 路径计入外部同输入收益。原生 DA 的 C、原生 CIL 的 B 均为 N/A，没有填零。

主 MIX、仅 MIX01 的 K/N 曲线、LS／MU／LU 五组诊断分别汇总；每 pool、RX、episode 与逐单元配对差见 CSV。资源只引用既有真实 JSON 和实际优化步，缺失记录保留未知，不重测，不推测 MACs／FLOPs，共享 GPU 不称孤立测量。

汇总数据：[原生方法得分、预算与训练步](native_scores.csv)、[R42／LocalRidge 同 raw 输入配对差](native_paired.csv)、[源模型实测元数据](native_source_costs.csv)。包含全部 MIX 主点、MIX01 的 K/N 曲线及三种单场景。缺项保持空白。逐 pool／RX／episode 聚合与逐单元配对保留在本机私有分析目录。本聚合器只读取冻结评分及运行元数据，不读取 truth、预测数组或 query IQ。

Hu 结果保留原始 `feature_separation_hu2024` profile 身份，不将其改标为另行运行的 scaled 版本。ISSL 用户停止、Tweak／CSIL 缺合法绑定的状态保持不变。
