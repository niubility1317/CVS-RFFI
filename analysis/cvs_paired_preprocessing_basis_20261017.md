# 下一唯一假设：同一物理包的接收预处理变化

2026-10-17。只提出下一项合法来源核实，尚无新IQ读取、模型、登记或训练。F＋跨RX同TX／三SIG_TIME／统一Sinc／joint160／NoExplicitCFO／默认头正常联合训练／E20保持，联合目标ACTIVE。

## 已完成的证据

有界非恒定相位G0的身份保护通过，E20三匹配训练却关闭：sourceRX1-held／seed01，B10 clean80.079365%，相对raw80.634921%下降0.555556pp，相对ordinary81.428571%下降1.349206pp；clean macro-F1也下降。lowurban27.698413%，相对raw−0.158730pp；suburban距自身clean约40至43pp。完整26460预测／13320更新轨迹／568320相位观察和成本复算一致，17469956NN，0target。固定sigma.006／peak.1rad／全覆盖路线不调幅、系数、RX、seed或预算救跑。

## 为什么换到这个问题

下一唯一假设：已有数据中同一RF包的不同接收预处理表示，可能比凭经验附加随机相位更适合构造有依据的观察变化。当前这只是研究动机，不是已证明的可用A_g、RX替换、source身份保持或泛化收益。

[WiSig官方说明](https://cores.ee.ucla.edu/wisig/overview/)称Full WiSig提供未处理前导的256点和同一256点的均衡版本。它也指出RX缺少时间同步；不能跨RX把相同sig_i当逐包配对。更重要的是，这段说明不能单独证明当前ManySig紧凑包的raw/eq数组经过子集抽样后仍逐行对应。

当前仓库 code/dataset_wisig.py 的载体为data[tx_i][rx_i][day_i][eq_i][sig_i]，equalized_list选择版本。loader按每个eq数组分别建立索引，__getitem__独立裁剪/RMS后返回；没有显式raw-eq物理pair-ID或重新排序映射校验。因此，存在eq维度不是配对证据，相同下标也不能绕过真实物理角色权限。

## 最小下一步与放弃条件

先只读原始compact构造代码、已经授权的source来源/物理ID元数据，核实：(1) raw和eq的抽样是否共享原packet索引；(2) 原fitL5040对应项是否一一同物理包；(3) 是否只涉及现有L，没有借用U/V/heldRX1/target或扩大有标签物理数量。此步骤0新IQ／checkpoint／NN／更新，不先反序列化不明全量数据来猜配对，不从其他RX或同TX相邻包补缺。

若不能证明这三个事实，放弃此入口，不把同sig_i当证据、不新增L、不擅自更换训练契约。若能证明，才另行固定唯一view定义、输入变化范围与同预算控制；它仍须实际默认头识别和E20比较，不能把配对正确或源健康等同泛化有效。此时不预设raw视图必然轻度，也不把当前EQ1训练头在另一表示上的错误直接等同身份标签被改变。

没有新loss/分支/头或新的审批、hash、receipt链。科学数据与物理角色核实属于既有输入合法性；不是重新验证未变的整个VALIDATED_ONCE载体。完整source/target基准、测试IQ与practical保持原协议，不能在不存在可用raw配对时静默改为equalized=both重训。
