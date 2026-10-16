# raw／EQ 接收预处理配对审计：当前入口关闭

2026-10-17。状态：**UNVERIFIED_SAME_PACKET_PAIRING_CURRENT_INPUT_CLOSED**。本次是代码及既有元数据核实，没有启动新实验。

基线保持 F＋跨 RX 同 TX 对比学习、三路 SIG_TIME、统一 Sinc、真实 joint160、NoExplicitCFO、默认 cosine 头正常联合训练、从头 E20 和原 U 规则。原有效源折为 fitL5040／U45360／V21600（RX 3、4、6、8）；RX1 的 L1260 仅作既定最终源读出。不能与完整 L6300 或 L_train5580 契约混用。联合目标仍为 ACTIVE。

## 实际完成与结论

读取了 WiSig 官方 raw→EQ→PKL→ManySig 构造代码，并以只读方式核对当前实际训练依赖中的加载器、原 source_contract.json、已有 preparation_complete.json，以及已保存的本轮源折契约和 resolved args。没有反序列化新的 IQ 容器或 checkpoint，也没有按 IQ 相似度猜配对。

证实的是：**现有构造流程不保证 raw／EQ 同下标对应同一原始包，当前已保存元数据也没有补足该映射。** 不能证实原 fitL5040 的 raw 对应项仍为同一物理 L 包，所以不发布此增强。实际 ManySig 哪些行错配仍然未知；本审计没有测出实际错配率，也不声称全部错配。

上轮有界非恒定相位三匹配 E20 已完成并关闭：B10 clean 为 80.079365%，相对 raw 对照下降 0.555556 个百分点，相对 ordinary 对照下降 1.349206 个百分点；low urban 为 27.698413%。这些是 sourceRX1-held／seed01 的源域反证结果，不能解释为正式目标成绩。完整报告及负结果继续保留：[相位 E20 报告](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-boundedPhase-sourceRX1-e20-s2026092701-m3-r01/report.md)。

## 官方构造代码的直接依据

| 环节 | 代码事实 | 对配对的影响 |
|---|---|---|
| [均衡数据构造](https://github.com/WiSig-dataset/wisig-process-raw/blob/master/matlab_equalization/equalize_dataset.m#L22) | 逐包均衡后只追加成功输出，保存的 packet_log 不含原 pkt_i 映射 | 一次跳包便可使后续 EQ 下标相对 raw 偏移 |
| [单包处理](https://github.com/WiSig-dataset/wisig-process-raw/blob/master/matlab_equalization/equalize_channel.m#L3) | 检测及长度检查可返回空；处理还包括重采样、定时、频偏处理及均衡 | 不能假定所有包都保留，也不能预设 raw 视图只是轻度均衡变化 |
| [PKL 构造](https://github.com/WiSig-dataset/wisig-process-raw/blob/master/python_pkl_creation/py/build_dataset_multirx.py#L24) | 将各 packet_log 压成连续数据行，只保存 data 和 node_list | 不恢复已丢失的逐包对应关系 |
| [compact 构造](https://github.com/WiSig-dataset/wisig-subset-creation/blob/master/data_utilities.py#L124) | raw 与 EQ 分别加载并各取相同长度的前缀 | 相同前缀长度不能替代按原始包 ID 连接 |
| [ManySig 配方](https://github.com/WiSig-dataset/wisig-subset-creation/blob/master/py/002_FullWiSig_create_compact_datasets.py#L44) | max_sig=1000，保存两种表示 | 有两种表示不等于逐行配对 |

[官方概述](https://cores.ee.ucla.edu/wisig/overview/)对 Full WiSig 的总体说明不能替代具体 compact 载体的包映射。跨 RX 也不因此成为同步逐包配对。

无需 IQ 的反例已执行验证：原顺序为 p0、p1、p2、p3；p1 均衡失败后 EQ 顺序为 p0、p2、p3。两侧各取前 3 项时长度相同，却有 2 个同下标对不是同一包。这只是“流程不保证配对”的反例，不是实际数据统计。

## 当前契约核实

实际原生加载器使用 WiSigIndex(tx_i, rx_i, day_i, eq_i, sig_i)，按每个表示自己的数据行取值；没有原始 packet-ID 的跨表示连接。当前实际参数为 wisig_equalized=1。原角色契约、准备记录和有效源折契约记录了既有角色与不透明物理 ID，均未提供 raw↔EQ 对应表。准备记录内历史模型描述只作为已存在元数据读取，本次没有加载其权重或采用其训练方案。

因此，不能把换 eq_i 后的相同行号自动认定为原 L 的另一视图。即使同 TX／RX／day，也可能引入不同物理包；若直接把两种表示各自划分或共同训练，还会改变数据契约。原 VALIDATED_ONCE 载体继续复用，本次没有重新验证未变的数据。

## 关闭范围与成本

关闭当前 ManySig raw／EQ 同下标增强入口；不改为 equalized=both，不增加 L，不从 U／V／RX1／target 或同 TX 相邻包补齐，也不以模型相似度反推物理配对。这一结论不否定“若有可靠原包映射，多预处理表示可能有价值”的一般假设。

本次新增真实 IQ 读取／checkpoint 加载／NN 前向／优化更新／教师前向／头拟合均为 **0**；target 访问为 0，远端写入为 0。只读远端结果与本地已保存契约相互核对，读取状态 VERIFIED，配对可用性 UNKNOWN，实验未启动。

聚合证据见 [审计 JSON](cvs_paired_preprocessing_audit_20261017.json)。下一唯一候选见 [采样抖动来源假设](cvs_sampling_jitter_source_basis_20261017.md)，目前只有代数检验，没有源 IQ、身份保护或训练收益证据。已关闭的相位、AWGN、时移等固定路线保持关闭。
