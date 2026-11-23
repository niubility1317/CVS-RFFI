# WiSig 短训练前导双路线工程对照

- run_id：`20261123-phase1-wisig-stf-engineering-source-s2026092701-r01`
- group_id：`wisig-observable-satellite-recovery`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在固定的 1,800 个源 V 物理包上，比较粗细载频校正、正则化均衡、估计残差功率和保留原载频偏移的相对作用。两条路线共享算法与固定规则。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 可用接收量与固定读出

ManySig 主比较使用 25 MS/s、256 点、逐包 RMS 归一化的 WiSig equalized=1 IQ。发布方以更长的输入做包检测、粗细载频偏移校正、L-LTF 信道与噪声估计和 MMSE 均衡，再加回载频偏移。现有窗口包含约 10.24 μs 样本，长训练字段的两段完整重复和同带宽空闲样本均不在评估输入内。当前代码使用索引 80 至 159 的四个名义 L-STF 短周期。已经处理过的 WiSig 参考、RX/TX 的频响和追加传播作用都会进入测得响应。[WiSig 原论文](https://arxiv.org/html/2112.15363) · [作者均衡实现](https://github.com/WiSig-dataset/wisig-process-raw/tree/main/matlab_equalization)

| 量 | 来源与可用时点 | 方法与误差 |
| --- | --- | --- |
| 粗、细 CFO | 当前包收到第 160 个样本后 | 20 点间隔的复相关给粗频偏，60 点间隔提高相位积累，使用粗估计解缠。粗估计无歧义区间为 ±625 kHz，低相干或时变通道会放大误差。 |
| `Â` | 当前 IQ 与 Wi-Fi 公共 L-STF 12 个占用频点 | 同包四周期频偏校正后求均值，除以已知导频复幅。读出只作用于已观测的 12 个频点，保持其他频点，在窗口两端渐变。保留通道、接收机、发射机和定时的混合响应。 |
| `σ̂²` | 当前包两组已对齐周期差 | 复差能量的一半给时域有效残差功率。在单周期归一化 DFT 坐标按每导频功率换算为正则增量。频偏复用和低相干影响估计，干扰、硬件失配与相位漂移计入残差。 |
| 置信度及表观采样时钟偏差 | 同包四个短周期 | 两个时间间隔中较低的相关系数控制均衡强度。对前后两组的跨频相位差拟合斜率并同时输出拟合误差，短基线上的相位变化可由时变通道造成。本轮仅诊断采样时钟，时域窗口保持原坐标。 |

六个工程读出为直通、粗 CFO、细 CFO、细 CFO 加相对岭逆、同一逆加入残差功率换算后的 Wiener 正则、Wiener 后加回估计 CFO。R69 用 `InferenceEngine(reuse=True, graph=True, native=False)` 运行。粗细校正项测试频偏校正本身对识别的影响，两个逆的差只改变残差正则项。加回 CFO 的行验证保留原接收相位斜率的影响。同一固定种子 F50 用作源域读出，对两路线均按同一规则运行，未产生恢复权重。正式 WiSig 主比较沿用其另行核实的原 Tri-SIG-TIME SSL 检查点。

估计器在已知复多径、正负 CFO 及有限噪声的独立合成波形上验证。设备识别的源域证据以全部冻结预测后的独立评分为准。长间隔 CFO 使用已知重复结构，与 OFDM 重复前导同步同类。正则化校正只在当前实际有信号支撑的频点上评估，与完整 L-LTF 接收机相比使用更短窗口和不同噪声测量。 [Moose 频偏估计](https://doi.org/10.1109/26.328961) · [Edfors 等，低秩 LMMSE](https://doi.org/10.1109/VETEC.1996.501446)

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
