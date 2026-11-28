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
| 源教师及锚点分布保持 | 待完成 | 待完成 | `20261128-phase1-shared-anchor-consistency-s2026112957-r01` 正在两数据集源 U 训练；同 2% 网络、锚点和 seed，新增受扰源包上的锚点识别分布保持损失 |
| 5% 修正的源教师及锚点分布保持 | 待完成 | 待完成 | `20261128-phase1-shared-anchor5-consistency-s2026112957-r01` 已发布；采用共同锚点分布保持目标和 5% 逐包 RMS 修正上限，2% 锚点保持版尚未出结果前登记 |

真实预测、逐对照 clean、改对改错、分半结果和配对区间见各报告及其 `screen-score.json`，不得从表中推断正式测试分数。示例：教师包级连续融合在 WiSig 类别 3 为 10/55，R17 为 46/55，存在明确单类退化。源 U 教师训练得到用户单独授权，方法属于 F-guided，祖先、伪标签和额外成本见各报告。无教师训练控制只表示新增模块不读取教师，LoRa 实增益父专家仍保留既有 E104 指导祖先。

## 仍需完成

R17 原始和逐包归一化路径的独立诊断在上述两个数据集、两景固定源 V 样本上预测逐包一致；这不证明波形逐值一致或其他条件一致。有界版本先按已固定源 V 门槛筛查，再对通过版本预登记额外源样本复现。通过后才冻结结构、权重、推理规则及正式接收机矩阵；补齐未有匹配预测的对照、全景预测、独立 truth-last、逐 RX/TX/day、paired CI、总参数、训练／显存及实测端到端时延。LoRa 和 WiSig 分别判断，缺项如实保留。
