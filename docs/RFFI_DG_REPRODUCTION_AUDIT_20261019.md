# WiSig 域泛化论文筛选与已有复现核查

核查日期：2026-10-19。范围是训练时不使用目标域数据的 RFFI 域泛化，以及已有 Phase1 对比方法能否声称达到原论文效果。

**新增复现优先选择 IB-RFF 和 TFMix。已有 RIEI-FD、DRIFT、DIFL、ASKNet、WaveMLP 均没有足以认定“达到原论文效果”的匹配证据。**其中 RIEI-FD、DRIFT 已有 CVS 条件下的目标测试结果；另外三项的原实现试跑已完成源训练，但目标尚未评分。两类情况不能混称论文复现成功。

## 1 两篇优先候选

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
| DIFL，`20260930-phase1-three-original-manysig-m3-r01` | 教师和学生各 100 轮；checkpoint 存在；`target_scored=false` | 仅一折一 seed；取特征位置修复尚未与论文逐项确认 | 源训练完成，论文效果未验证 |
| ASKNet，同上原实现 run | 早停于第 41 轮；checkpoint 存在；`target_scored=false` | 仅一折一 seed，没有原表目标成绩 | 源训练完成，论文效果未验证 |
| WaveMLP，同上原实现 run | 早停于第 39 轮；checkpoint 存在；`target_scored=false` | 仅一折一 seed；未完成 ManySig/ManyRx 原表评估 | 源训练完成，论文效果未验证 |

RIEI-FD 和 DRIFT 的上述均值与标准差来自[既有独立评分汇总](COMPARISON_RESULTS_20260927.md)，是模型种子间统计。不是原论文指标的同条件复算，不计算跨协议的“复现差距”。本次还读回这两个 CE 首种子的远端 `resolved_config.json`，确认实际使用 RX 索引 `1,3,4,6,8`、源 day 索引 `1,2,3`、`L_s/U_s/V=0.07/0.63/0.30`、200 轮、batch 128，并从第 80 轮开始 clean/星地拼接 CE。该核对不冒充全部 40 行配置复审。

后三项于本次 N607 只读核查中均返回 `SOURCE_TRAINING_COMPLETE_AWAITING_SOURCE_REVIEW`，完成记录明确写有 `target_scored=false`，且权重实际存在。旧登记的 RUNNING 不能当作实时状态。读取完整可用 epoch 元数据用于核对轮数，没有依据 loss 或源 V 分数推断泛化能力。

### 原论文验收参照与版本问题

[RIEI 扩展论文 arXiv:2411.03636](https://arxiv.org/html/2411.03636)的 Table III 指定每行训练/测试 RX，并采用最后 5 个 epoch 的统计。例如 `1-1,8-8 → 1-19` 为 79.43% ± 1.66%；这不能替代其他 RX 组合的验收值。历史项目还引用 SPAWC 2023 的另一标题及 last10 规则，必须先固定版本再选验收表。

[DRIFT arXiv:2510.09405](https://arxiv.org/html/2510.09405)的 Table I 采用 Day1、指定源 RX 组合、每 TX/RX 800 个训练包、未见 RX 每 TX/RX 200 个测试包，并报告 5 个随机种子的最终轮均值。例如源 `1-1,14-7` 对应 68.66%。对比本项目分数前，必须先恢复同一行的数据和统计口径。

旧[论文代码审计](../baselines/PAPER_CODE_AUDIT.md)把 `arXiv:2309.02801v2` 写为 RIEI 来源，但[该编号实际对应无人机轨迹重建论文](https://arxiv.org/abs/2309.02801)。这是确定的引用错误，历史审计中的“完全一致”标签不能作为已达论文效果的证据。保留原文，并在文件开头追加本次更正。

DIFL 的旧修复将返回特征从 256 维 fc2 前移至已有 512 维 fc1，保留分类 logits 和参数形状，解决空 CORAL 分支。该修复有技术证据，但[修复记录](../paper_reproduction/phase1_external_20260930/REPAIRS_V2.md)明确保留“尚未核实论文精确位置”的边界，因此不能把修复后跑通等同于完整论文复现。

## 3 实验验收方式

先完成作者协议复现，再另建 CVS 匹配对比。作者协议允许其原有源域全监督数据、网络和训练预算；CVS 对比仍遵守当前数据契约和标签权限。E20 是 CVS 探索预算，不能用来代替外部论文的原始预算。两层结果分表，不用作者全监督分数装作 CVS 少标签成绩。

1. 在首次目标预测前固定论文版本、验收表/行、TX/RX/day、raw/equalized、每单元包数、物理 ID、归一化、完整训练和模型选择规则。未明确的细节单独标为实现选择。
2. IB-RFF 先做 ManySig round0 的两个阶段；TFMix 先做固定 `1-1` 留出行和同骨干无混合对照，训练预算以全文核准为前提。单行技术与源侧机制检查合格后再按预登记条件扩至原表，不以看到的目标成绩选择性扩行或换参数。
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

本次没有启动新的真实 WiSig 训练或目标评分，未改变旧训练或旧预测。整体目标尚未完成；已完成的是候选筛选、代码可运行性检查和历史证据核查。后续可先做 IB-RFF 作者代码复现，同时核准论文预算与验收表；TFMix 不按其 1 轮默认值发布“论文复现”。真实训练仍须按既有最小流程登记、适配、启动和评分。
