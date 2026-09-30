# 全面验收与原论文WiSig实验条件

2026-09-30。用户要求先全面验收，只有原论文使用WiSig时才启动原论文口径复现实验。本次不把WiSig适配结果改称原论文复现。

## 原论文数据集核实

|方法|原论文实验数据|证据|是否满足WiSig原论文实验条件|
|---|---|---|---|
|CSIL|ADS-B|本地9页全文，第6页Evaluation dataset；https://arxiv.org/abs/2105.06381|否|
|LoRa_RFFI|自采60台LoRa设备/USRP N210|本地12页全文实验部分；https://arxiv.org/abs/2107.02867|否|
|MoPC-HR|AIS-100与ADS-B|本地15页全文，第1页摘要及实验部分；https://github.com/xmuLdz/MoPC-HR|否|
|ISSL|公开摘要列出电台信号与ADS-B|https://www.researchgate.net/publication/393038502_Innovative_Specific_Emitter_Identification_With_Self-Supervised_and_Incremental_Learning ，对应 DOI 10.1109/TCCN.2025.3583118；全文未取得|未满足：无WiSig实验依据，全文仍待核实|

三个可得全文均无WiSig字样，且实验部分明确了上述数据。ISSL的相关引用/后续论文中出现WiSig不能作为该篇ISSL使用WiSig的证据。不因数据集同属RFFI，就用WiSig替换原数据宣称原论文复现。

全文解析的定位证据：`local_artifacts/newclass_registration_20260930/paper_dataset_evidence.json`，仅用于核实数据集。本次没有启动原论文完整预算实验；其条件尚未满足。

## 已完成测试与修复

新增22项行为/数值/负测与已有CSIL、MoPC-HR的18项测试通过。覆盖输入维度/角色/物理ID、query真值不影响训练和预测输入、源代码固定、作者裁剪/归一化、阶段梯度与key网络模式、连续扩头、triplet公式及梯度、SAME padding、两种信号长度的模型前向、原长度预处理数值一致性、预测阶段与类别范围、评分K口径、已有输出防覆盖，以及CSIL/MoPC-HR的核心方法与协议。

本次修复均未替换作者算法：

- checkpoint的`training_ids_ref`从不存在的`.npz.json`修正为实际`.json`契约路径；r01旧checkpoint原样保留，不改写。
- 运行入口检查boolean角色mask、唯一物理ID、输入shape/finite和训练标签schema。
- scorer检查完成状态、完整预测阶段集合、旧类/全注册类预测范围与ID；K和query数量从实际数据推导。
- KD验证probe隔离随机数状态，避免验收检查改变后续训练随机序列；SSL日志记录实际optimizer学习率。
- checkpoint验收由“重载可前向”加强为“重载后的完整预测与已固定预测逐条相同”，并防止覆盖既有审计文件。

独立P0/P1审查通过。真实回归预先固定每阶段2epoch：ISSL base/SSL/transfer/incremental对照均2epoch，LoRa triplet2epoch；从零训练，不加载r01权重，不根据旧评分调参或选模。新run与输出均为r02，不覆盖r01。单日/单RX旧3类新3类、K128诊断，未叠加LEO，非正式CVS性能矩阵。

## 算法等价性边界

ISSL原作者函数数值验证确认：KD标量与温度交叉熵数值一致，但切断学生梯度；SSL保留缺少zero_grad而跨步累计梯度、queue reshape而非transpose。测试通过表示忠实确认了这些行为，**不表示这些问题符合论文算法**。全文尚未取得，不能宣称ISSL论文公式全面验收通过。

LoRa使用第三方Torch模型及WiSig短IQ适配。作者原长度预处理数值对照通过；Torch与TF2.1的优化器/归一化epsilon细节尚未完成全部训练数值等价证明。作者Keras原版环境及原数据均未做完整复现，不包含在本次通过范围。

CSIL、MoPC-HR已有模块测试与历史CVS产物核验有效；本次不重复启动既有历史矩阵，且不据历史完成状态宣称当前checkpoint来源已全部合规复核。

## 真实回归记录

- [ISSL r02](../../automation_reports/CV-SincNet/20260930-diagnostic-issl-wisig-s392005-r02/report.md)
- [LoRa r02](../../automation_reports/CV-SincNet/20260930-diagnostic-lora-wisig-s392005-r02/report.md)

运行/产物结论以两份报告与独立读回证据为准。2epoch通过不能证明论文完整训练预算或所有硬件/数据集上的长期稳定性。
