# LoRa／WiSig 共用 DURA 与实增益研发记录

## 目标与边界

目标是一个两数据集共用核心、允许各自输入适配和合法独立权重的恢复方法。正式验收分别比较同包输入直通、当前 DURA、合法匹配实增益和新方法；clean、六种 practical、逐 RX/TX/day、配对改对改错及不确定性均按用户目标报告。两数据集分别完成全部正式预测及独立评分，且六景等权均值、低城区、clean 均达到各自可用对照的较高值后，目标才能完成。

本轮固定源 V 小筛查：LoRa `b200_1` 的 60 条物理包、WiSig RX1 的 360 条；每条均评估 clean 和 `practical_low_urban`。固定同包信道观测、冻结原识别器及默认头；先固定完整预测，再独立连接源 V 真值。LoRa 对所有可用对照的两个固定半区各净增至少 1 包、clean 净损失不超过 1 包；WiSig 各半区净增至少 2 包、clean 净损失不超过 2 包。源 V 已被以下多轮研发反复曝光，属于开发筛查数据。历史 LoRa Full 测试也已有曝光背景；目标 IQ、目标真值和目标统计未用于本轮训练或筛查。

LoRa 实增益原版的历史 Full：六景均值 82.2967%、低城区 62.1167%、clean 95.5467%。其权重及 E104 指导、伪标签训练祖先保留。原固定 DURA v1 是 WiSig 契约 R69；LoRa 筛查中 `R69 适配`是 LoRa 合法源训练的同架构独立权重，不能当作原 WiSig checkpoint。WiSig 实增益为该数据集独立合法源训练，未加载 LoRa 权重。当前尚缺对应目标正式测试中的新候选及必要匹配对照。

## 固定低城区源 V 筛查结果

下表单位为正确包数；LoRa 分母 60，WiSig 分母 360。LoRa 原输入/R17/实增益/R69 适配依次为 30/15/34/33；WiSig 原输入/R17/实增益/DURA v1 为 200/332/190/313。R17 是用户指定优先研究路线之一，R69 只作为已固定或合法适配的对照。

| 共用版本 | LoRa | WiSig | 状态与证据 |
| --- | ---: | ---: | --- |
| R17 + 实增益物理残差 | 15 | 233 | 未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-r17-realgain-s2026112893-r01/report.md) |
| 双先验逐频实增益 | 18 | 157 | 未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-twoprior-realgain-s2026112901-r01/report.md) |
| 逐频凸组合 | 18 | 145 | 未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-convex-realgain-s2026112905-r01/report.md) |
| 源波形误差整包门控 | 15 | 190 | 未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-selector-realgain-s2026112909-r01/report.md) |
| 双扰动一致性整包门控 | 17 | 190 | 未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-consistency-selector-s2026112913-r01/report.md) |
| 信号统计矩门控 | 15 | 190 | 未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-moment-selector-s2026112917-r01/report.md) |
| 源对比表征门控 | 15 | 190 | 未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-contrastive-selector-s2026112921-r01/report.md) |
| 冻结识别器置信度路由 | 26 | 333 | 未晋级；识别器参与逐包推理，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-confidence-route-s2026112925-r01/report.md) |
| 原输入识别一致性路由 | 28 | 297 | 未晋级；识别器参与逐包推理，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-agreement-route-s2026112929-r01/report.md) |
| DURA 前向残差路由 | 18 | 332 | 未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-physics-route-s2026112933-r01/report.md) |
| 源教师指导整包门控 | 33 | 190 | 未晋级；同预算 F-free 对照为 15/190，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-teacher-selector-s2026112937-r01/report.md) |
| 源教师指导包级连续融合 | 25 | 220 | 未晋级；同预算 F-free 对照为 15/199，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-teacher-mixture-s2026112941-r01/report.md) |
| 源教师指导逐频实增益，从实增益起步 | 30 | 193 | 未晋级；同预算 F-free 对照为 16/144，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-teacher-direct-s2026112945-r01/report.md) |
| 源教师指导逐频实增益，从 R17 起步 | 31 | 178 | 未晋级；同 seed F-free 对照为 15/162，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-r17-teacher-direct-s2026112949-r01/report.md) |
| 源教师指导 1% 有界逐频实增益 | 33 | 333 | 未晋级；同 seed F-free 对照为 34/332，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-bounded-teacher-s2026112957-r01/report.md)。共用网络与上限，独立权重的起点由已曝光源 V 决定：LoRa 实增益、WiSig R17 |
| 源教师指导 2% 有界逐频实增益 | 34 | 333 | 未晋级；同 seed F-free 对照为 34/332，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-bounded2-teacher-s2026112957-r01/report.md)。LoRa 相对实增益净 0 包，WiSig 相对 R17 净 +1 包，固定半区均未达门槛；2% 根据已曝光的 1% 源筛查结果提出 |
| 源教师及锚点分布保持 | 34 | 333 | 未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-anchor-consistency-s2026112957-r01/report.md)。相对 LoRa 实增益净 0、WiSig R17 净 +1，两个固定半区均未达门槛；同 seed 2% 教师版对照分别净 0、0 |
| 5% 修正的源教师及锚点分布保持 | 33 | 331 | 未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-anchor5-consistency-s2026112957-r01/report.md)。相对 LoRa 实增益及 WiSig R17 均净 -1 包；2% 锚点保持版尚未出结果前登记 |
| 源伪标签保守专家路由 | 34 | 332 | 未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-risk-router-s2026112957-r01/report.md)。LoRa 和 WiSig 相对各自强锚点均改对 0、改错 0；推理仅使用当前包与两个专家 |
| 源伪标签纠错加权专家路由 | 34 | 190 | 未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-risk8-router-s2026112957-r01/report.md)。LoRa 相对实增益净 0；WiSig 相对 R17 改对 9、改错 151，净 -142；正例权重 8，负例 1 |
| 冻结识别器置信度辅助路由 | 34 | 332 | 两侧均未晋级；相对各自强锚点改对 0、改错 0。LoRa 首次加载失败后通过[技术补跑](../../automation_reports/CV-SincNet/20261128-phase1-shared-confidence-lora-retry-s2026112957-r01/report.md)完成筛查。训练及推理使用同数据集冻结 F，推理每包额外两次 F 前向 |
| 大容量共享逐频实增益 | 31 | 331 | 两侧未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-wide-gain-s2026112957-r01/report.md)。LoRa 相对实增益改对 0 改错 3；WiSig 相对 R17 改对 1 改错 2。网络 467,842 参数，clean 保持 |
| 源 U practical 模拟配对训练 | 34 | 334 | 两侧未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-practical-pairs-s2026112957-r01/report.md)。LoRa 相对实增益改对 0、改错 0；WiSig 相对 R17 改对 5、改错 3、半区 +3/-1，区间 -4 至 +8 |
| 4096 包源 U practical 模拟配对训练 | 34 | 333 | 两侧未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-practical4k-s2026112957-r01/report.md)。LoRa 相对实增益净 0；WiSig 相对 R17 改对 2、改错 1、半区 +1/0，区间 -2 至 +4 |
| 低城区加权源 U practical 模拟训练 | 35 | 332 | 两侧未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-practical-urban-s2026112957-r01/report.md)。LoRa 相对实增益改对 1、改错 0、半区 +1/0，区间 0 至 +3；WiSig 相对 R17 改对 2、改错 2、半区 -1/+1，区间 -4 至 +4 |
| 20% 修正的低城区加权源 U 模拟 | 35 | 332 | 两侧未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-practical-urban20-s2026112957-r01/report.md)。相对同 seed 5% 加权版，两个数据集 clean 与低城区改对 0、改错 0；正误一致不代表 IQ 完全相同 |
| 共享专家互谱相位实增益 | 35 | 332 | 两侧未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-phase-gain-s2026112957-r01/report.md)。相对无互谱同源训练版，两个数据集 clean 和低城区的正确包集合一致，改对 0、改错 0 |
| practical 源域无新增教师匹配对照 | 32 | 330 | 两侧未晋级，[报告](../../automation_reports/CV-SincNet/20261128-phase1-shared-practical-mse-s2026112957-r01/report.md)。同包教师版相对本对照 LoRa 改对 4 改错 1、净 +3（区间 -1 至 +7），WiSig 改对 5 改错 3、净 +2（区间 -3 至 +8）；LoRa 实增益父专家保留历史教师祖先 |
| 共享逐包专家对齐实增益 | 待完成 | 待完成 | `20261128-phase1-shared-aligned-gain-s2026112957-r01` 正在训练；共同解析复数标量仅用于专家差分对齐，保留逐频实增益、原始专家特征、DURA 物理残差和 5% cap |
| 源教师一致性门控锚点损失 | 待完成 | 待完成 | `20261128-phase1-shared-agreement-loss-s2026112957-r01` 正在训练；仅当源 U clean 与锚点教师预测一致时施加锚点 KL，保留 clean 教师与波形约束，推理无需 F |

真实预测、逐对照 clean、改对改错、分半结果和配对区间见各报告及其 `screen-score.json`，不得从表中推断正式测试分数。示例：教师包级连续融合在 WiSig 类别 3 为 10/55，R17 为 46/55，存在明确单类退化。源 U 教师训练得到用户单独授权，方法属于 F-guided，祖先、伪标签和额外成本见各报告。无教师训练控制只表示新增模块不读取教师，LoRa 实增益父专家仍保留既有 E104 指导祖先。

## 仍需完成

R17 原始和逐包归一化路径的独立诊断在上述两个数据集、两景固定源 V 样本上预测逐包一致；这不证明波形逐值一致或其他条件一致。有界版本先按已固定源 V 门槛筛查，再对通过版本预登记额外源样本复现。通过后才冻结结构、权重、推理规则及正式接收机矩阵；补齐未有匹配预测的对照、全景预测、独立 truth-last、逐 RX/TX/day、paired CI、总参数、训练／显存及实测端到端时延。LoRa 和 WiSig 分别判断，缺项如实保留。
