# WiSig 域泛化论文筛选与已有复现核查

核查日期：2026-10-19。范围是训练时不使用目标域数据的 RFFI 域泛化，以及已有 Phase1 对比方法能否声称达到原论文效果。

**IB-RFF 与 SDG（Avoiding Shortcuts）的作者代码复现均已完成并达到预登记日志参照。**IB-RFF四折均值93.91%，作者日志93.57%；SDG目标三日均值93.2444%，逐日准确率及200轮公开loss打印值均与作者日志一致。TFMix保留为候补，预算仍不完整。历史三个冻结模型也已补齐目标评分：ASKNet93.83%、WaveMLP90.44%、DIFL82.28%，独立复算一致。严格论文表验收仍缺证据，RIEI-FD、DRIFT的CVS结果也不能替代原论文协议。

## 1 初筛候选与当前选择

| 项目 | IB-RFF | TFMix |
|---|---|---|
| 论文 | Information Bottleneck-Driven Cross-Receiver RF Fingerprinting for Physical-Layer Security | TFMix: A Robust Time-Frequency Mixing Approach for Domain Generalization in Specific Emitter Identification |
| 出版信息 | IEEE TIFS，2026，21：8008–8021 | IEEE TCCN，2025，11(6)：3981–3992 |
| DOI | [10.1109/TIFS.2026.3723081](https://doi.org/10.1109/TIFS.2026.3723081) | [10.1109/TCCN.2025.3541723](https://doi.org/10.1109/TCCN.2025.3541723) |
| 作者代码 | [BeechburgPieStar/IB-RFF](https://github.com/BeechburgPieStar/IB-RFF) | [frownean/TFMix](https://github.com/frownean/TFMix) |
| 本次固定提交 | `5a03a04da86c4bbf5cf90f41a55c268bf2152203` | `67e96a5103059223dbcb60556a23073c9dc5582b` |
| WiSig 支持 | ManySig、ManyRx；CR 和 CRD | 已公开入口使用 ManySig，跨接收机 |
| 方法 | FAN 骨干与 HSIC 信息瓶颈，两阶段训练，无需 RX 标签 | 时间域 Mixup、频域幅度插值和相位扰动、复数 CNN |
| 代码许可口径 | README 声明源码 MIT；权重另限学术用途 | 作者自定义非商业许可，宜称“公开作者代码” |
| 优先理由 | 模型、两阶段脚本、数据加载器和作者日志齐全，适合先做作者代码复现 | 与 CVS 的时频鲁棒机制互补，已有 ManySig 原始 pkl 可接入 |

IB-RFF 的出版信息和方法由 [IEEE 官方页面](https://ieeexplore.ieee.org/abstract/document/11653216)及[IEEE Signal Processing Society](https://signalprocessingsociety.org/publications-resources/ieee-transactions-information-forensics-and-security/2026/09/information)交叉确认。TFMix 书目信息由 Crossref DOI 记录核对，数据和训练细节来自固定作者代码。没有以引用次数或未经核对的分区信息作为选择依据。

### IB-RFF 的可复现性与缺口

作者代码固定 ManySig 的 6 TX、12 RX；每个 TX/RX/day 取前 100 包，非均衡 IQ，样本平均功率归一化。训练日为作者编号 day1/2，源域按 70%/30% 划分训练和验证；12 RX 分四折，每折 9 个源 RX、3 个目标 RX。CR 测同日未见 RX，CRD 测 day3/4 的未见 RX。

`run_all.sh` 给出两阶段各最多 200 轮、Adam、学习率 0.001、batch 128；按源验证 loss 降学习率和早停。ManySig 的 patch size 为 32。第二阶段从同折第一阶段开始，重建优化器；必须保留原分类头与骨干联合训练，不能用另训头代替。

固定脚本对 CR/CRD、各折预置不同 alpha。可把这些值作为**作者已公开配方**固定复现；其历史选择依据尚未核实，不能把该实验称为新的独立盲测，也不能在本次目标成绩出来后重新选 alpha。

实际入口还需要数据路径和独占输出目录、当前 PyTorch 调度器参数兼容，以及完整对象 checkpoint 的兼容读回。`only_test` 当前仍会构建训练数据，需在本地工作副本拆开预测入口。归一化函数对零功率输入产生 NaN，正式适配应明确拒绝无效记录并保留物理 ID，不静默丢弃或替换样本。

### TFMix 的可复现性与缺口

实际 `dates` 值是 RX 坐标 `1-1/1-19/14-7/18-2`，不是日期。加载器固定 ManySig 的第一个 day、非均衡记录、每 TX/RX 前 500 包，以峰值功率归一化。四个 RX 留一测试；每源 RX 先留 10% 测试，再从其余部分取 30% 验证，实际训练/验证比例为总池的 63%/27%。目标只使用其独立 10% 测试部分，不能擅自改成全部目标包。

作者训练包括每源域 batch 32，三域拼批；Mixup beta 参数 0.7，幅度插值上限 0.9，相位扰动 0.01；Adam 0.001，指数衰减 0.95，源验证早停 patience 20。

**公开 `Config.epochs` 只有 1。**该值不能作为达到论文效果的预算。入口虽保存最佳验证权重，最终测试却直接使用循环结束时的模型，没有加载所保存的最佳权重；究竟采用哪种选模口径，需与论文核对。硬编码输入路径、未创建输出目录、CUDA 专用随机置换和训练 accuracy 分母也需作兼容或统计修复。不能在未核准预算时静默改成 200 轮并称“原论文设置”。

## 2 已有复现是否达到论文效果

| 方法与具体记录 | 已核实的结果 | 与论文的关键差异 | 判定 |
|---|---|---|---|
| RIEI-FD-CE，`20260927-phase1-baselines-final-clean-satellite-m5-r01` | 4 个新模型种子 clean 67.29% ± 1.49% | CVS 少标签划分、源日/RX、星地拼接训练及第 200 轮权重，与论文原表不同；历史引用版本混乱 | 不能判为已达到，也不能据此判原论文方法无效 |
| DRIFT-CE，同一最终评估 run | 4 个新模型种子 clean 63.94% ± 10.54% | 5 个源 RX、多个源日、7% 有标签和星地拼接；与 DRIFT Day1 全监督、指定 RX 组合和 5 种子协议不同 | 当前 CVS 移植表现较弱，原论文效果未验证 |
| DIFL，父run `20260930-phase1-three-original-manysig-m3-r01`，新补测 `20261019-phase1-existing-original-eval-m3-r01` | 教师/学生各100轮的冻结原头；equalized day1目标准确率82.2778%、Macro-F1 82.5758% | 仅一个9:3 RX划分/seed；fc1特征取点尚未与全文确认 | 目标评分VERIFIED；论文效果未验证 |
| ASKNet，同一父run和补测run | 实际41轮；raw CRD round0准确率93.8333%、Macro-F1 93.9307%；作者本折日志93.47% | 仅一折一seed，没有完整原表矩阵 | 本折接近作者日志；完整论文效果未验证 |
| WaveMLP，同一父run和补测run | 实际39轮；raw CRD round0准确率90.4444%、Macro-F1 90.4943% | 不能用本折直接对比论文ManySig四折均值93.13%；ManyRx未测 | 目标评分VERIFIED；论文效果未验证 |

RIEI-FD 和 DRIFT 的上述均值与标准差来自[既有独立评分汇总](COMPARISON_RESULTS_20260927.md)，是模型种子间统计。不是原论文指标的同条件复算，不计算跨协议的“复现差距”。本次还读回这两个 CE 首种子的远端 `resolved_config.json`，确认实际使用 RX 索引 `1,3,4,6,8`、源 day 索引 `1,2,3`、`L_s/U_s/V=0.07/0.63/0.30`、200 轮、batch 128，并从第 80 轮开始 clean/星地拼接 CE。该核对不冒充全部 40 行配置复审。

后三项初次核查确为 `SOURCE_TRAINING_COMPLETE_AWAITING_SOURCE_REVIEW`、`target_scored=false`；后续在新的独占补测run中完成全部预测及独立评分，旧source-only产物不改写。已取回10800条冻结预测与真值侧文件，独立重算准确率、F1、混淆矩阵及逐RX/day，逐值一致。[补测结果与边界](EXISTING_ORIGINAL_MODELS_RESULTS_20261019.md)单列DIFL的不同任务；旧登记的RUNNING及源V分数不作为目标证据。

### 原论文验收参照与版本问题

[RIEI 扩展论文 arXiv:2411.03636](https://arxiv.org/html/2411.03636)的 Table III 指定每行训练/测试 RX，并采用最后 5 个 epoch 的统计。例如 `1-1,8-8 → 1-19` 为 79.43% ± 1.66%；这不能替代其他 RX 组合的验收值。历史项目还引用 SPAWC 2023 的另一标题及 last10 规则，必须先固定版本再选验收表。

[DRIFT arXiv:2510.09405](https://arxiv.org/html/2510.09405)的 Table I 采用 Day1、指定源 RX 组合、每 TX/RX 800 个训练包、未见 RX 每 TX/RX 200 个测试包，并报告 5 个随机种子的最终轮均值。例如源 `1-1,14-7` 对应 68.66%。对比本项目分数前，必须先恢复同一行的数据和统计口径。

旧[论文代码审计](../baselines/PAPER_CODE_AUDIT.md)把 `arXiv:2309.02801v2` 写为 RIEI 来源，但[该编号实际对应无人机轨迹重建论文](https://arxiv.org/abs/2309.02801)。这是确定的引用错误，历史审计中的“完全一致”标签不能作为已达论文效果的证据。保留原文，并在文件开头追加本次更正。

DIFL 的旧修复将返回特征从 256 维 fc2 前移至已有 512 维 fc1，保留分类 logits 和参数形状，解决空 CORAL 分支。该修复有技术证据，但[修复记录](../paper_reproduction/phase1_external_20260930/REPAIRS_V2.md)明确保留“尚未核实论文精确位置”的边界，因此不能把修复后跑通等同于完整论文复现。

## 3 实验验收方式

先完成作者协议复现，再另建 CVS 匹配对比。作者协议允许其原有源域全监督数据、网络和训练预算；CVS 对比仍遵守当前数据契约和标签权限。E20 是 CVS 探索预算，不能用来代替外部论文的原始预算。两层结果分表，不用作者全监督分数装作 CVS 少标签成绩。

1. 在首次目标预测前固定论文版本、验收表/行、TX/RX/day、raw/equalized、每单元包数、物理 ID、归一化、完整训练和模型选择规则。未明确的细节单独标为实现选择。
2. IB-RFF 已按固定公开配方完成 ManySig 四折；第二篇改为 SDG 的作者 WiSig day1→day2/3/4 单源域泛化，固定200轮。TFMix 仍以全文预算核准为发布前提，不以目标成绩选择性扩行或换参数。
3. 从零训练。IB-RFF 第二阶段只能继承本次同折、同训练契约的第一阶段；DIFL 教师同理。公开权重只保留，不作为可证明干净泛化的初始化。
4. 训练冻结后使用 checkpoint 内原分类头预测。全部预登记预测固定后再由独立 scorer 连接 truth；原论文复现先按原表 clean 条件评分，CVS 匹配对比再按其自身预登记场景执行。
5. 每行保存准确率、Macro-F1、样本数、seed、物理 split、checkpoint、成本和失败项。四个 RX 折不能称为四个模型种子；源 V 分数不能替代目标成绩。
6. “达到论文效果”须是同条件统计比较。建议首次预测前固定工程容差，例如完整表平均准确率与论文相差不超过 1 个百分点，同时公开逐行差值和方法对自身 ERM 的改善；该容差是待固定的项目验收定义，不是论文原文，也不是领域通用标准。论文采用 epoch 窗口时另报该窗口，不能冒充独立种子标准差。

目前尚未取得两篇新候选的完整实验表；TFMix 的真实轮数与最终权重规则仍是影响正确性的缺项。用户确认没有全文。本次继续检索了 IEEE、作者仓库、公开网页和 OpenAlex，仍未获得可访问全文；这不等于证明公开全文不存在。不能以另一篇 WaveMLP 论文中重跑 TFMix 的 80.95% 作为 TFMix 原论文的复现目标，也不能以作者仓库日志当成本项目实验结果。

IB-RFF 可先独立开展明确标注的“作者代码与公开日志复现”：固定提交的四折作者日志给出 ManySig CR/CRD 算术平均 94.62%/93.57%，ManyRx 为 90.68%/89.1875%。这些值是本次从作者日志中的 `Test Accuracy` 复算的单 seed 四折均值，不是已核对的论文表值，也不是本项目新结果；可用于技术对照，不能代替论文效果验收。若 TFMix 的预算无法核准，候补是 [Avoiding Shortcuts 的作者代码](https://github.com/BeechburgPieStar/SDG-for-Channel-Robust-SEI)：README 明确给出 WiSig 200 轮、模型 S、主增强深度 2、辅助深度 1 和两个对比损失权重 1/100；候补还需核对其 WiSig 子集、原文和目标日志，不能凭 README 即认定论文效果可复现。WiSR 是 CSI 手势识别，不能作为 RFFI/WiSig 论文凑数。

## 4 本次交付与当前状态

已完成文献和公开代码筛选、两份作者源码的固定提交下载及独立远端 OID 对照、现有五项方法的证据核查。两份源码共 14 个 Python 文件解析通过；IB-RFF 和 TFMix 在合成 CPU 输入上均输出 `[8,6]`，损失/梯度有限且参数实际更新。IB-RFF 的原 HSIC 项参与反向计算。当前 PyTorch 不支持作者调度器的 `verbose` 参数，零功率输入 NaN 也已复现。合成检查通过不表示原生完整 CLI、真实数据流程或论文准确率通过。

IB-RFF 初次检查因缺少 timm 而失败；本次只在任务目录补充隔离依赖后重试，原失败证据保留。原作者源码未修改，未加载随仓库下载的权重。

原始只读回收证据与合成检查产物保留在 `local_artifacts/rffi_dg_research_20261019/`，包括 `existing_remote_evidence.json` 和 `synthetic_smoke*.json`；数据、权重和原始日志不进入 Git。下载的源码位于该目录的 `upstream/IB-RFF` 与 `upstream/TFMix`。

初次筛选阶段没有启动真实训练；后续已完成下述 IB-RFF 四折。整体目标尚未完成，TFMix 不按其 1 轮默认值发布“论文复现”，旧训练和旧预测保持。

## 5 IB-RFF 四折真实复现

[完整结果报告](IBRFF_AUTHOR_MANYSiG_RESULTS_20261019.md)：ManySig clean CRD、4折、model seed2023、每折3个未见RX的day3/4、共14400个不重复目标包。原分类头、逐折独立从零两阶段训练，冻结预测后独立truth-last评分。本地对全部冻结prediction重新计算，与远端一致，状态VERIFIED。

四折准确率95.2778%、89.6389%、97.3611%、93.3611%，均值93.9097%；作者日志均值93.57%，高0.3397个百分点，达到预先固定的92.57%工程下限。折均Macro-F1为93.9002%。这是单seed四折作者代码日志复现，原论文表仍未核准；不能当作CVS弱标签或卫星指标，也不能把折数当作模型seed数。复用导出IQ在归一化前已转float32，未宣称从原始pickle到训练张量全流程逐位一致。

r01在首个source smoke因NumPy2导出模块名与作者NumPy1.24不兼容而失败，未正式训练或访问target。旧产物保留，r02只修复反序列化模块名；本地同一源块6TX数组逐值一致回归及定点复审通过。代码与配置commit为`aaebd70187598165c7d937bfdf7de575a2723c1c`。

## 6 第二篇改用 SDG

论文 [Avoiding Shortcuts](https://doi.org/10.1109/TWC.2025.3528568)，IEEE TWC2025，24(4)：3163–3176，作者[公开代码与数据链接](https://github.com/BeechburgPieStar/SDG-for-Channel-Robust-SEI)齐全。固定提交`0090f4ba1d5cf76a4200a5be4830a9b08eb39699`的`2TWC_Version`，采用作者公开的4个RX1-1日期pickle、6TX、每类1000包；保留float64逐样本归一化。作者处理IQ的均衡状态未说明，直接保留作者文件。

这是day1训练、day2/3/4测试的单源跨天DG，非IB-RFF的跨接收机任务，不横向混排准确率。公开命令固定S模型、200轮、主/辅助增强深度2/1及类别/域损失权重1/100；原训练、验证、增强、对比损失、分类头及torchsummary前向均直接保留。源划分3780/1620/600与作者函数逐数组一致的合成检查通过，全部7种增强及原训练更新检查通过，独立审查无P0/P1。

真实run `20261019-phase1-sdg-author-wisig-s2023-r01` 已完成200轮，源NLL选择epoch191；冻结18600条预测后独立评分。day2/3/4准确率99.2167%、83.2667%、97.25%，目标均值93.2444%，与作者日志逐日一致；源holdout99.5%单列，目标合并Macro-F1为93.1334%。达到提前固定的92.2444%工程下限。[完整结果](SDG_AUTHOR_WISIG_RESULTS_20261019.md)保存day3弱类与全部200轮分析，不据此调参/重跑。

已独立复算全部预测、真值/物理ID连接、4个日期混淆矩阵和源NLL选模轮次；结果一致。全部200轮训练loss在作者6位小数精度一致，验证NLL在4位小数精度一致，stdout无致命异常。实际训练/验证累计1511.8349s，不包括发布、数据读取和评分。

## 7 严格论文效果的剩余证据缺口

本次在浏览器直接核查IEEE官方正文：IB-RFF文档11653216、SDG文档10847785均显示继续阅读需要登录且文档要求订阅。IB-RFF官方Figures页面也明确要求登录才能查看，未取得实验表。公共网页、作者仓库、机构公开页面及本地文件名查找仍未定位可核准全文；不把“没有找到”写成证明全文不存在。

因此当前可证明两篇作者代码/数据例程达到其公开日志效果；严格“达到论文表效果”仍保持未验证。TFMix预算与最终选模口径未核准，DIFL精确特征位置和完整匹配矩阵、RIEI-FD/DRIFT原协议对照也不能由现有CVS分数替代。整体目标保持进行，不以日志参照偷偷缩小原目标。
