# 真实同包连续窗口：仅完成可用性代码核对

2026-10-17。状态 NO_TRACEABLE_LONG_CONTEXT_CURRENT_INPUT_CLOSED；未登记或启动新训练，0新增IQ容器读取、checkpoint、NN、更新、teacher或target访问。当前不把这一入口称为有效Ag。

## 已核实与尚未核实

实际冻结训练后端commit 0ae066adb10a001e834c8c679c5b7b81c20b3ac8，SSDG/train_ssdg.py调用WiSigCompactDataset，实际args为wisig_out_len=256、wisig_equalized=1；调用指定crop_mode=center、normalize=True，数据类center默认False。数据类先读取同一个index对应的原数组，只有原长度T不同于256时才调用_pad_or_crop_2t，随后RMS归一化。T>256时可中心裁剪，T<256时会补零。

这只证明加载器能处理长向量，**不能证明当前合法L记录实际长于256**。本次未打开新的IQ容器、未读取其他eq表示，也没有把通用代码的默认值当作当前数据形状。当前role/IDs训练序列只保存eq/rx/day/sig/base，不保存原长度或帧参考偏移；sig_i相邻及按sig排序不能证明两包IQ时间连续，不允许拼接相邻包制造上下文。

## 唯一待检验问题与最小动作

合法fitL5040的同一个EQ1物理记录是否保留了更长的真实连续IQ，以及网络固定参考区域能否仍覆盖真实合法的参考片段？先复用已有source preparation元数据与原数据构造说明核对形状、原包与窗口位置；没有这些证据时不写增强或开E20。当前没有身份保持、跨RX clean或As组合收益证据。

若只有独立256点数组，或者长上下文不能追溯到同一原物理L／参考片段，就关闭当前输入入口。不以循环roll、插值／分数延迟、补零、重复采样、拼接相邻sig_i或换raw/EQ同下标替代真实上下文，也不扩大L、借U/V/heldRX1/target作供体。此前循环移位／分数延迟／timegrid及raw-EQ配对失败保持关闭；本问题仅检查实际观察是否存在，不通过改变这些已失败算子重开路线。

若可用性得到证明，才另行固定合法L默认头身份保持检查与后续匹配E20设计。可用性本身不支持Ag有效，不能与另一实验的增益相加。模型保持F＋跨RX同TX、三SIG_TIME／统一Sinc／joint160／NoExplicitCFO／默认cosine头联合训练／原U。训练数据契约及预算不因本代码核对而改变。

## 与当前已关闭试验的关系

逐包六scene均衡三匹配E20已完整结束：相对随机As clean＋0.238095pp、lowurban＋0.158730pp、urban均值＋0.502646pp，未达固定后两项各＋1pp条件，候选关闭，无扩seed或调参救跑。当前新问题不改变该判定。联合目标ACTIVE，Ag、稳定As、完整四格交互、跨seed与正式truth-last仍缺失。


## 现有输入入口关闭

已补查官方构造与本地既有元记录。[官方概述](https://cores.ee.ucla.edu/wisig/overview/)说明Full WiSig的识别样本取前导前256点，ManySig从Full WiSig选取。[PKL构造](https://github.com/WiSig-dataset/wisig-process-raw/blob/master/python_pkl_creation/py/build_dataset_multirx.py)固定pkt_len=256并取slice(0,256)，输出数组逐包为256×2，只保存data/node_list。[compact构造](https://github.com/WiSig-dataset/wisig-subset-creation/blob/master/data_utilities.py)在包轴上取前max_sig个记录，并不恢复包外样本；[ManySig配方](https://github.com/WiSig-dataset/wisig-subset-creation/blob/master/py/002_FullWiSig_create_compact_datasets.py)使用该构造。

原source carrier与preparation_complete元记录只提供角色、物理ID、split和历史准备信息，没有长IQ、原始capture偏移或参考位置映射。历史准备中模型／预算字段只读来辨认元记录范围，不采用其权重、特征或训练配方。泛用数据类可以裁剪T>256并不构成当前存在长上下文的证据。

据此关闭当前受信输入上的“真实同包相邻窗口”入口。不下载新raw、不扩大L、不拼接邻包或跨EQ猜配对。此结论是官方构造与现存映射不支持该入口，不是逐包扫描本地PKL数组所得的长度统计；本次未反序列化新IQ，不能声称逐项实测了全部记录形状。

成本：0新IQ容器反序列化／checkpoint／NN／优化更新／teacher／头拟合／target访问。只读本地已有JSON和冻结代码、官方说明；没有远端写入或新训练。下一独立依据问题见[cvs_frequency_phase_observability_basis_20261017.md](cvs_frequency_phase_observability_basis_20261017.md)。
