# PAMF未知身份结构冻结诊断结果

Run `20261030-phase3-star-open-feature-diagnostic-s2026092701-r01`，执行code `55977eec2cf0c2bd09d7c6856179f919e9f0923d`，VERIFIED / ANALYZED_SOURCE_DIAGNOSTIC。四模型七种表征导出和独立CPU评分全部完成，目标IQ/模型更新/G前向均0。**未发现M开放特征整体塌缩，但它包含的RX组间相对方差明显高于原fused；该结果不能证明某一个分支造成了实际拒识失败。** 完整Phase3仍ACTIVE，几何诊断不晋级。

## 数据和方法

只读已有source novel3150条固定G10 MIX01 P1，19-15/19-16/7-1各1050；Main RX1/3/4/6/8、2021-03-08/15/23，每TX/RX/day70条。设备未参与当前F/OE拟合，但已反复开发评分，sourceRX和日期也被祖先使用；不称新设备盲测、未见RX或未来匿名注册。未新读L/V/OE/原Us/target U/Q/正式RX24-18，不新造信道，不读clean。

原F0为Tri-SIG-TIME SSL E50/11100/default6head；S0/S1/M均为原source-onlyr02/code1029bc4f4ab86022c75ae0f3332b5dfafc1fe909的固定2220终点。全部权重冻结，默认头未改，不据本结果换主、调阈值或续训。导出同一次前向的unit fused160/open64；F0只有fused160。四actualcheckpoint合成G10首步smoke先于source cache打开，参数/buffers保持。

全部四NPZ/七表征合法且固定后，CPU评分端先为所有表征固定四个候选条件的近邻，再读取唯一source真值。距离为unit或真实zero向量的余弦，排除自身，exact ties按opaqueID稳定次序。候选条件是任意其他记录、不同RX、不同day、RX/day同时不同；TX/K没有参与距离、候选过滤或邻居选择。真实TX只用于判断固定近邻是否sameTX及描述统计。

这是同池参考的事后检索，不是部署分类器，不等于无标签分类或未来A_N。所有3150条均有候选，宏均值按三个真实TX等权；没有用真实K聚类，缺候选规范仍为N/A。七个结果全部报告，不排序替换主M。

## 实际几何指标

| 表征 | 任意其他1NN | 跨RX | 跨day | 同时跨RX/day | 有效秩 | TX组间/总方差 | RX组间/总方差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F0.fused160 | 90.8254% | 78.7937% | 82.9206% | 76.6032% | 3.5322 | 42.1118% | 2.8875% |
| S0.fused160 | 90.5714% | 78.0635% | 84.2540% | 76.6349% | 4.4805 | 43.4786% | 2.8937% |
| S0.open64 | 88.1270% | 72.4762% | 74.3810% | 68.1587% | 3.3540 | 4.6530% | 22.8880% |
| S1.fused160 | 90.8889% | 78.3492% | 84.4444% | 76.8254% | 4.4768 | 43.8330% | 2.8876% |
| S1.open64 | 91.4286% | 78.9524% | 85.0794% | 77.5873% | 10.2473 | 21.9579% | 13.9779% |
| M.fused160 | 90.6667% | 78.0635% | 84.7937% | 76.8254% | 4.4202 | 44.0348% | 2.8495% |
| M.open64 | 89.9683% | 77.4603% | 83.9365% | 76.9841% | 8.4215 | 19.1491% | 16.2868% |

M.open64同时跨RX/day为2425/3150、76.984127%；F0为2413/3150、76.603175%，只多12个命中。M.fused160为2420/3150、76.825397%；S1.open64为2444/3150、77.587302%。这些小幅同池描述差异不代表显著或普遍改善，也不能从最高控制结果重新选择主方案。

M.open64 RX组间比例16.286785%，原F0为2.887514%；TX组间比例19.149066%，原F0为42.111771%。S1.open64对应RX13.977899%/TX21.957925%；S0.open64为RX22.887990%/TX4.652980%。成员梯度联合训练较detach控制恢复了一部分未知身份结构，但M轮换集合没有进一步改善这些指标。

M有效秩8.4215、原F0为3.5322，不能把“秩更大”当“身份更好”。同时跨RX/day的sameTX−differentTX余弦gap，F0为0.143873、M.open64为0.204139、S1.open64为0.244173。gap与方差比例的尺度不同，不强行把所有统计解释成单一好坏排序；RX/TX/day单因素比例也不代表可加或因果分解。完整谱、各组比例、pair计数和检索分层见private score/results.json及analysis.json。

## 跨RX/day逐未知设备

| 表征 | TX | 数量 | 同设备近邻命中 | 命中率 |
| --- | --- | ---: | ---: | ---: |
| F0.fused160 | 19-15 | 1050 | 663 | 63.1429% |
| F0.fused160 | 19-16 | 1050 | 951 | 90.5714% |
| F0.fused160 | 7-1 | 1050 | 799 | 76.0952% |
| S0.fused160 | 19-15 | 1050 | 665 | 63.3333% |
| S0.fused160 | 19-16 | 1050 | 962 | 91.6190% |
| S0.fused160 | 7-1 | 1050 | 787 | 74.9524% |
| S0.open64 | 19-15 | 1050 | 595 | 56.6667% |
| S0.open64 | 19-16 | 1050 | 792 | 75.4286% |
| S0.open64 | 7-1 | 1050 | 760 | 72.3810% |
| S1.fused160 | 19-15 | 1050 | 670 | 63.8095% |
| S1.fused160 | 19-16 | 1050 | 957 | 91.1429% |
| S1.fused160 | 7-1 | 1050 | 793 | 75.5238% |
| S1.open64 | 19-15 | 1050 | 673 | 64.0952% |
| S1.open64 | 19-16 | 1050 | 922 | 87.8095% |
| S1.open64 | 7-1 | 1050 | 849 | 80.8571% |
| M.fused160 | 19-15 | 1050 | 667 | 63.5238% |
| M.fused160 | 19-16 | 1050 | 960 | 91.4286% |
| M.fused160 | 7-1 | 1050 | 793 | 75.5238% |
| M.open64 | 19-15 | 1050 | 650 | 61.9048% |
| M.open64 | 19-16 | 1050 | 935 | 89.0476% |
| M.open64 | 7-1 | 1050 | 840 | 80.0000% |

## 实际代码边界与下一步

只读代码审计确认原F0的参考分支不是现成的t2/PA/SIG信道逆。原融合为：时频base_norm160＋带learned gain的PA160＋tanh(response_gain)乘reference projection160＋已经限幅的SIG160。reference内部的参考匹配/CFO局部补偿只用于形成描述子，没有回写主干IQ或对t2/PA/SIG做参考条件化；NoExplicitCFO仅屏蔽坐标27，内部局部补偿保留。

当前open输入t2已经过GN/ReLU/pool，PA已过pa_norm，SIG也已过LN和真实增益，不能称“未归一化raw分支”。但新输入没有完整时频融合160、reference贡献和原PA融合增益，因而没有直接继承原最终加性组合。现有合法F0没有可直接取出的canonical/identity-corrected分支；未启用CRRA/NMFDU不能算当前能力。

上述边界来自frozen_sources/BASE的cvs_residual_identity/model.py、cvs_reference_identity/model.py及cvs_source_identity_readout/input_policy.py，SIG来自frozen_sources/AF的cvs_rdrop_sinc_units/x3_features.py，当前捕获来自star_open_world_model.install。AF是历史SIG/Sinc配置矩阵名称，不是一个额外参考校正模块。审计未改代码或重跑模型。

下一独立候选首先检验是否应继承原归一化分支贡献/融合尺度，并把参考条件化作为明确的新机制，而非假设已有RX逆。必须保留匹配控制、源身份保护和独立未来Q；不能根据这三个已暴露源设备选阈值、回收query训练，或把普通fused→MLP/SupCon/Gram的已做负路线换名重跑。本轮只是识别了可检验的输入和域信息问题，尚未证明新修正有效。

## 执行、验证与成本

新增4×3150=12600实物feature-forward、8合成烟测，G/optimizer/模型/阈值更新0，只有3150个既有物理记录。当前诊断驻留三完整native容器及F0共3090682参数，包含未用域分支；不是部署压缩结果。四私有feature NPZ合计12151092B，不部署、不Git。pipeline 12.234221s，export job 7.329332s，CPU score job 4.820889s，不代表B1时延。

11项聚焦检查分开完成（export3、launcher3、scorer5/9sub），6AST/runtime接口、一次直接P0/P1review和一次N607 preflight通过。423文件/data0归档SHA485210d86b1130ea0d28a4724d3a99b86a66056dfae4a97caa7f24d3a91fc652，远端一次比较/compile通过。actualcheckpoint source smoke、四模型state不变、七表征/全部邻居先于truth的完成标记齐全。

driver3583272的实际PID/CWD/argv已独立live核实；export3583366/GPU4与CPU score3583550仅jobs、命令和完整stdout支持，短进程未捕获独立live GPU/CPU快照，未为补遥测重跑。全部三个stdout已完整读取、无Traceback，最终live/ownedGPU空，其他用户操作0。source/target性能门槛没有降低，无新target U/Q/formal访问，AN/HOW/UDA/REG均N/A，完整目标ACTIVE。
