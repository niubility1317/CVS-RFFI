# 差发射机优化：从首次诊断到完整三种子结果

截至 2026-10-21，本页汇总本对话从首次差发射机风险诊断开始的 27 条实验、诊断与技术恢复记录。早期实现和报告已在本分支；本次补齐最新九行的完整三种子结果、终态登记和统一分析文件。**当前没有获得同时稳定改善两个弱发射机的方法，研究目标尚未达成。** 所有负结果、源域不晋级记录和技术失败均保留。

## 从哪里分析

- [逐种子目标结果 CSV](target_rows.csv)：44 条报告记录，含 clean 准确率、Macro-F1、六个 TX recall、最差 RX/day、同种子配对差与成本。MEM4 首种子在确认 run 中复用，因此共 43 份不同的预测产物，不能把这两条报告记录当成额外种子。
- [分组统计 CSV](group_summary.csv)：30 个 run×条件组，含实际种子数、均值、样本标准差和正差数。n=1 的标准差留空，不混充三种子确认。
- [完整分析 JSON](analysis_data.json)：逐 TX/RX/day、混淆矩阵、同种子基线、成本、27 条来源/配置索引和源日志汇总。
- [完整源训练曲线 CSV](source_epoch_curves.csv)：原 50 轮路线的 40 条报告行×50 轮，共 2000 条 epoch 记录；其中 MEM4 首种子的 50 条为复用，已标记。早期 E20 曲线不伪装为 E50，也未补造，详见该历史 run 的源训练报告。
- [101 次既有代码与报告提交](code_history.csv)：从首次源风险诊断提交 511b5c60b 开始，到本次同步前的 9aa2f2dd9；本次同步提交由 Git 历史继续记录。
- [完整三种子次级比较](secondary_complete_3seed.json)：在 STAT/NORM 全部完成后补齐预先声明的 CENTER 对照，不改模型、选模或预测。

所有百分数为准确率/recall/F1 的百分值；配对差和标准差单位为百分点。样本标准差使用 n−1。三种子共享同一组测试物理样本，描述的是模型种子波动，不是三个独立数据集。这里没有发布 IQ、样本级预测、真值、权重、checkpoint、原始日志或本机私有配置。

## 基底与比较口径

历史方法锚点为原 F＋跨 RX 同 TX 对比学习，三种子 clean 均值 85.113889%，执行提交 4df7b94d，原 run 为 20261016-phase1-F-supcon-e50-clean-manysig-m9-r02。它是方法来源，不与本机变体直接相减冒充严格配对。

本机严格匹配 BASE 为 84.7617 ± 0.3604%，模型种子固定 2026092701/02/03；每个变体只与同 seed 的本机 BASE 比。详见[基底约束](../../goals/F_CROSSRX50_WEAK_TX_20261019.md)和[匹配基底及首轮因子实验](../../../automation_reports/CV-SincNet/20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01/results.md)。

固定原三路 SIG_TIME、统一 Sinc、RDrop alpha=1、普通增强、LS0.01、旧门控＋ownEMA＋cos，每模型从零 50 轮＝33L＋17U。默认分类头与骨干联合训练，冻结后直接预测，没有另训头。无星地增强，只测 clean。L/U/V 为 6300/56700/27000，U 的 TX 真值始终隐藏；源 RX 为 1/3/4/6/8、日期 1/2/3；目标 168000 物理样本、6 TX×28000，7 RX×4 日期。完整清单、划分 seed 392005 和数据权限见分析 JSON 及各 experiment.json。

弱 TX14-7、TX20-19 来自合法源域风险诊断。方法与扩种子安排在相应目标访问前固定；本页只整理既有冻结产物，不进行目标反馈调参、择优重排或选择性重跑。已暴露的探索基准不被称为新独立盲测、Phase2 或卫星部署结果。

## 已完成的三种子组

表中差值均为变体−同种子 BASE，再在三个种子上求均值。弱两类均差为两个 TX recall 配对差的平均。

| 条件 | clean 均值 ± 样本 SD（%） | 整体均差 | TX14-7 均差 | TX20-19 均差 | 弱两类均差 | 弱均差正种子 |
|---|---:|---:|---:|---:|---:|---:|
| BASE | 84.7617 ± 0.3604 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0/3 |
| BOTH | 84.7887 ± 0.3510 | +0.0270 | -1.0595 | +1.1548 | +0.0476 | 1/3 |
| MEM4 | 84.6506 ± 0.6916 | -0.1111 | -0.6952 | -0.0107 | -0.3530 | 2/3 |
| CENTER_GRAD | 84.7157 ± 0.5431 | -0.0460 | -1.9560 | +0.2821 | -0.8369 | 0/3 |
| DCL | 84.8514 ± 0.8815 | +0.0897 | -0.5798 | -0.7857 | -0.6827 | 1/3 |
| STAT_GRAD | 84.6276 ± 0.7672 | -0.1341 | -2.2226 | -0.3024 | -1.2625 | 0/3 |
| NORM_ONLY | 84.8002 ± 0.4138 | +0.0385 | -0.6690 | +0.3095 | -0.1798 | 1/3 |

BOTH 的微小整体均差没有同时改善两个弱 TX；DCL 和 NORM_ONLY 也出现整体略涨、弱类均值下降。MEM4 首种子的收益没有稳定扩展到三种子。STAT_GRAD、CENTER_GRAD 的 TX14-7 明显退化，不能以其他 TX 的收益抵消后宣布弱发射机目标完成。各组的逐 TX、Macro-F1、最差 RX/day、样本 SD 和成本均在 CSV/JSON 中，原负结果报告保留。

最新统计梯度实验的预登记主判断要求：两个弱 TX 各均差>0且各至少2/3正差、弱均差3/3为正、整体均差≥0、其他各 TX 均差≥−0.25pp、最差 RX 均差≥−0.5pp。STAT/CENTER 未通过。NORM 是归因控制，其三种子也未形成稳定弱类收益。

## 最新九行：首种子信号与三种子结论

STAT_GRAD 保留均值与标准差的真实导数；NORM_ONLY 保持原停止统计量的梯度方向，将描述子梯度范数匹配到同一描述子的完整导数范数；CENTER_GRAD 只开放均值导数，标准差继续停止梯度。三者不改变同输入下的原描述子前向值，不增加分类头、网络前向或参数。范数控制不代表整网 AdamW 位移相同；CENTER−STAT 同时改变方向和大小，不构成尺度单因素因果证明。

| 条件 | 模型 seed | clean（%） | 整体配对差 | TX14-7 配对差 | TX20-19 配对差 | 弱两类配对差 |
|---|---:|---:|---:|---:|---:|---:|
| CENTER_GRAD | 2026092701 | 85.2821 | +0.1429 | -0.7893 | +0.5179 | -0.1357 |
| CENTER_GRAD | 2026092702 | 84.1994 | -0.2220 | -3.0179 | -0.7893 | -1.9036 |
| CENTER_GRAD | 2026092703 | 84.6655 | -0.0589 | -2.0607 | +1.1179 | -0.4714 |
| STAT_GRAD | 2026092701 | 85.4351 | +0.2958 | +0.0250 | -0.0357 | -0.0054 |
| NORM_ONLY | 2026092701 | 85.2488 | +0.1095 | +0.3786 | +0.3250 | +0.3518 |
| STAT_GRAD | 2026092702 | 83.9083 | -0.5131 | -3.2750 | -1.7357 | -2.5054 |
| NORM_ONLY | 2026092702 | 84.4333 | +0.0119 | -0.8071 | -0.0750 | -0.4411 |
| STAT_GRAD | 2026092703 | 84.5393 | -0.1851 | -3.4179 | +0.8643 | -1.2768 |
| NORM_ONLY | 2026092703 | 84.7185 | -0.0060 | -1.5786 | +0.6786 | -0.4500 |

| 预声明次级比较 | 整体均差 ± 样本 SD | 弱两类均差 ± 样本 SD | 弱均差正种子 |
|---|---:|---:|---:|
| CENTER_GRAD − STAT_GRAD | 0.0881 ± 0.2245 | 0.4256 ± 0.4921 | 2/3 |
| CENTER_GRAD − NORM_ONLY | -0.0845 ± 0.1364 | -0.6571 ± 0.7354 | 0/3 |

CENTER 原自动汇总完成时，STAT/NORM 尚未齐全，因此原 result.json 的次级项保留 PENDING 快照；本次另附完整三种子补充比较，避免覆盖原时点证据。相对更差对照的改善不等于相对原 BASE 有效。

九行终态已核对实际 final50 checkpoint、默认头与部署特征参数更新、完整 50 轮/11100 次原生及辅助日志、9990 次辅助更新/39960 次逐层反向、预测 ID 覆盖、混淆矩阵/F1/配对算术、预测先于独立评分及退出状态。检查为只读 CPU，不重新运行模型或读取 IQ/真值。见 [STAT/NORM 核对](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-statGradient-e50-clean-local-m6-r01/completion_verification.json)和 [CENTER 核对](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-centerGradient-e50-clean-local-m3-r01/completion_verification.json)。

## 保留的单种子探索

这些是既有首种子或小矩阵结果，不能用其正差替代三种子确认，也未因为目标分数好而补做选择性重跑。MEM4 的首行与上方三种子组首行相同。源域是否晋级、停止原因和配方差异以原报告为准。

| 条件 | clean（%） | 整体配对差 | 弱两类配对差 | 原报告 |
|---|---:|---:|---:|---|
| RX_MARGIN | 85.4476 | +0.3083 | -0.0196 | [RXmargin](../../../automation_reports/CV-SincNet/20261019-phase1-FcrossRX-RXmargin-e50-clean-local-m1-r01/results.md) |
| WEIGHT | 85.1327 | -0.0065 | +0.0196 | [TXrisk-geometry-m4-r01](../../../automation_reports/CV-SincNet/20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01/results.md) |
| GEOMETRY | 85.5220 | +0.3827 | +0.1911 | [TXrisk-geometry-m4-r01](../../../automation_reports/CV-SincNet/20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01/results.md) |
| ANCHOR | 85.1083 | -0.0310 | -0.1268 | [crossBatchMemory](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-crossBatchMemory-e50-clean-local-m2-r01/results.md) |
| MEM4 | 85.3268 | +0.1875 | +0.4286 | [crossBatchMemory](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-crossBatchMemory-e50-clean-local-m2-r01/results.md) |
| Q_DCL | 85.3958 | +0.2565 | +0.5232 | [decoupledContrast](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-decoupledContrast-e50-clean-local-m2-r01/results.md) |
| SYM4 | 85.1214 | -0.0179 | +0.1000 | [symmetricMemory](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-symmetricMemory-e50-clean-local-m1-r01/results.md) |
| STEP_NORM | 85.3065 | +0.1673 | +0.0214 | [weakTX-actualStep](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-actualStep-e50-clean-local-m2-r01/results.md) |
| UPDATE_CONE | 85.0655 | -0.0738 | +0.0554 | [weakTX-actualStep](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-actualStep-e50-clean-local-m2-r01/results.md) |
| NORM_L_ID | 85.1875 | +0.0482 | +0.2518 | [weakTX-blockPC](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-blockPC-e50-clean-local-m2-r01/results.md) |
| BLOCK_PAIR | 84.1435 | -0.9958 | -2.7196 | [weakTX-blockPC](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-blockPC-e50-clean-local-m2-r01/results.md) |
| RANDOM_WEAK | 84.9762 | -0.1631 | -0.5107 | [weakTX-challenge](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-challenge-e50-clean-local-m2-r01/results.md) |
| SALIENCY_WEAK | 84.8756 | -0.2637 | -0.9839 | [weakTX-challenge](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-challenge-e50-clean-local-m2-r01/results.md) |
| SELF_GAP | 85.1339 | -0.0054 | -0.2946 | [weakTX-crossGap](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-crossGap-e50-clean-local-m2-r01/results.md) |
| CROSS_GAP | 84.7702 | -0.3690 | -0.9750 | [weakTX-crossGap](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-crossGap-e50-clean-local-m2-r01/results.md) |
| EXTRA_RX_CE | 85.3143 | +0.1750 | +0.0286 | [weakTX-metaRX](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-metaRX-e50-clean-local-m2-r01/results.md) |
| META_RX | 85.3089 | +0.1696 | +0.1339 | [weakTX-metaRX](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-metaRX-e50-clean-local-m2-r01/results.md) |
| MEAN_WEAK_RX | 84.2988 | -0.8405 | -2.7804 | [weakTX-tailRisk](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-tailRisk-e50-clean-local-m2-r01/results.md) |
| TAIL_WEAK_RX | 84.3595 | -0.7798 | -2.5571 | [weakTX-tailRisk](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-tailRisk-e50-clean-local-m2-r01/results.md) |

## 从首次弱发射机诊断开始的工作索引

起点包括源风险定标、E20 本机迁移的三次技术失败和 r04 完成记录。E20 r04 实际使用 sat CE0.68，四组仅 clean 测试；它们作为独立历史负结果保留，不能混入原 F50 无星地增强的严格比较。之后研究依次覆盖加权/joint160 因子、RX 匹配间隔、特征挑战、跨 RX 元训练、均值/尾部风险、模块梯度、实际更新约束、独立样本类差、类内精度诊断、DCL、同 RX 负样本诊断、跨批次记忆与统计梯度。源域诊断和未训练/未扩种子条件不会伪造目标指标。

| 实验/诊断 | 类型 | 最终记录状态 | 本导出目标行数 |
|---|---|---|---:|
| [F＋跨 RX：三模型完整源V发射机与接收机风险诊断](../../../automation_reports/CV-SincNet/20261019-diagnostic-FcrossRX-TXrisk-sourceV-m3-r01/report.md) | SOURCE_ONLY_DIAGNOSTIC | ANALYZED | 0 |
| [原F跨RX50：六模型student与ownEMA的隐藏标签源U置信度诊断](../../../automation_reports/CV-SincNet/20261019-diagnostic-FcrossRX-Uconfidence-source-m6-local-r01/report.md) | SOURCE_ONLY_DIAGNOSTIC | ANALYZED | 0 |
| [本机原F跨RX50：三模型源V类内类间与跨RX特征几何诊断](../../../automation_reports/CV-SincNet/20261019-diagnostic-FcrossRX-feature-geometry-sourceV-m3-local-r01/report.md) | SOURCE_ONLY_DIAGNOSTIC | ANALYZED | 0 |
| [F弱发射机四组冻结模型：仅clean默认头评估](../../../automation_reports/CV-SincNet/20261019-eval-FcrossRX-TXgeometry-clean-local-m4-r01/report.md) | FROZEN_EVALUATION | ANALYZED | 4 |
| [原F跨RX50：跨RX难正对与同RX近异类的特征间隔](../../../automation_reports/CV-SincNet/20261019-phase1-FcrossRX-RXmargin-e50-clean-local-m1-r01/report.md) | TRAINING | ANALYZED | 1 |
| [本机 F＋跨 RX：弱发射机加权与 joint160 类内类间约束消融](../../../automation_reports/CV-SincNet/20261019-phase1-FcrossRX-TXgeometry-e20-local-m4-r01/report.md) | TRAINING | FAILED | 0 |
| [本机 F＋跨 RX：弱发射机加权与 joint160 类内类间约束消融（本机环境技术修复）](../../../automation_reports/CV-SincNet/20261019-phase1-FcrossRX-TXgeometry-e20-local-m4-r02/report.md) | TRAINING | FAILED | 0 |
| [本机 F＋跨 RX：弱发射机加权与 joint160 特征消融（无控制台技术修复）](../../../automation_reports/CV-SincNet/20261019-phase1-FcrossRX-TXgeometry-e20-local-m4-r03/report.md) | TRAINING | FAILED | 0 |
| [本机 F＋跨 RX：弱 TX 加权与 joint160 消融（两级无窗口执行）](../../../automation_reports/CV-SincNet/20261019-phase1-FcrossRX-TXgeometry-e20-local-m4-r04/report.md) | TRAINING | ANALYZED | 0 |
| [原F跨RX50弱发射机：源风险加权与最终特征几何](../../../automation_reports/CV-SincNet/20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01/report.md) | TRAINING | ANALYZED | 8 |
| [F跨RX源特征：同RX负样本替换与数量匹配诊断](../../../automation_reports/CV-SincNet/20261020-diagnostic-FcrossRX-RXnegative-sourceL-m3-local-r01/report.md) | SOURCE_ONLY_DIAGNOSTIC | ANALYZED | 0 |
| [F跨RX源特征：分离样本的类内精度几何诊断](../../../automation_reports/CV-SincNet/20261020-diagnostic-FcrossRX-source-withinTX-precision-m3-local-r01/report.md) | SOURCE_ONLY_DIAGNOSTIC | ANALYZED | 0 |
| [F源域三种子：弱TX跨RX判别方向诊断](../../../automation_reports/CV-SincNet/20261020-diagnostic-FcrossRX-weakPair-direction-sourceV-m3-local-r01/report.md) | SOURCE_ONLY_DIAGNOSTIC | ANALYZED | 0 |
| [原 F 三种子：弱 TX 与实际部署特征梯度诊断](../../../automation_reports/CV-SincNet/20261020-diagnostic-FcrossRX-weakTX-gradient-sourceL-m3-local-r01/report.md) | SOURCE_ONLY_DIAGNOSTIC | ANALYZED | 0 |
| [F跨RX：MEM4固定三种子补充](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-MEM4-threeSeed-clean-local-m2-r01/report.md) | TRAINING | ANALYZED | 3 |
| [F跨RX：保留标准差停止梯度的均值导数三种子](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-centerGradient-e50-clean-local-m3-r01/report.md) | TRAINING | ANALYZED | 3 |
| [F跨RX：当前锚点梯度与短期跨批次特征记忆](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-crossBatchMemory-e50-clean-local-m2-r01/report.md) | TRAINING | ANALYZED | 2 |
| [F跨RX50：解除正样本竞争与正负耦合](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-decoupledContrast-e50-clean-local-m2-r01/report.md) | TRAINING | ANALYZED | 4 |
| [F跨RX：批统计真实梯度与范数控制，固定三种子](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-statGradient-e50-clean-local-m6-r01/report.md) | TRAINING | ANALYZED | 6 |
| [F跨RX：保留当前批次参照梯度的短期记忆](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-symmetricMemory-e50-clean-local-m1-r01/report.md) | TRAINING | ANALYZED | 1 |
| [F跨RX50弱发射机：真实AdamW特征位移约束与存储范数对照](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-actualStep-e50-clean-local-m2-r01/report.md) | TRAINING | ANALYZED | 2 |
| [F跨RX50弱发射机：模块梯度协调与范数对照](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-blockPC-e50-clean-local-m2-r01/report.md) | TRAINING | ANALYZED | 2 |
| [F跨RX50弱发射机：浅中层随机与显著通道挑战](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-challenge-e50-clean-local-m2-r01/report.md) | TRAINING | ANALYZED | 2 |
| [F跨RX50弱TX：同RX/day两组独立样本的类差分离](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-crossGap-e50-clean-local-m2-r01/report.md) | TRAINING | ANALYZED | 2 |
| [F跨RX50弱发射机：可微临时骨干更新与曝光匹配CE](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-metaRX-e50-clean-local-m2-r01/report.md) | TRAINING | ANALYZED | 2 |
| [F跨RX50弱发射机：分组特征尾部风险与均值对照](../../../automation_reports/CV-SincNet/20261020-phase1-FcrossRX-weakTX-tailRisk-e50-clean-local-m2-r01/report.md) | TRAINING | ANALYZED | 2 |
| [F跨RX统计梯度：保留健康训练的调度恢复](../../../automation_reports/CV-SincNet/20261020-recovery-FcrossRX-statGradient-controller-local-r01/report.md) | CONTROL_RECOVERY | ANALYZED | 0 |

每个原目录均有 experiment.json、report.md 和证据，implementation_refs 在 analysis_data.json 中指向实际模块/配置；历史 code.commit 尚为占位文字的条目原样标注，不猜测其执行身份。101 条提交索引补充代码演变轨迹，具体实际执行身份仍以各原 release/完成证据为准。

## 交付边界

统计梯度原调度器因 Windows 进度文件替换冲突退出；控制恢复使用新 run/root，原健康训练未重启，失败记录保留，原科学提交不变。最终六行完整完成并核对，见[恢复报告](../../../automation_reports/CV-SincNet/20261020-recovery-FcrossRX-statGradient-controller-local-r01/report.md)。

本次同步没有启动新训练、改变测试范围或选择模型。结果记录完成与性能目标达成分别判断：本批产物已完整，弱发射机优化目标仍未达成。成本是实际并行环境下的观测值，不能据此单独认定算法加速。
