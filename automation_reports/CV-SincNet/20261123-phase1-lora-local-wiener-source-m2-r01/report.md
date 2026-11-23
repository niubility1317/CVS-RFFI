# LoRa 局部复协方差 Wiener 源域对照

- run_id：`20261123-phase1-lora-local-wiener-source-m2-r01`
- group_id：`lora-observable-satellite-recovery`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

核查同包短时谱复协方差能否提高相同剂量 Wiener 去噪对冻结设备识别的收益。分别评估 full 和 residual，独立源域 truth-last。

部署输入为当前收到的六条 256 点 LoRa 上扫频窗和公开基准 chirp。用原前导追踪器估计各条实收窗的等效频率斜率和复载波，并在临时归一坐标内分解公共与独有成分。32 点复 STFT 使用 Hann 窗与 8 点步长。每个 4 点频率组的经验二阶矩使用邻近五帧和当前包的相应成分估计。噪声代理为去均值重复结构在偏离载波至少 96 kHz 的外侧频点功率中位数，经复高斯中位数与重复数量校正。它可能含非重复信号、跟踪误差和相关噪声。接收后可计算局部矩阵，无需包外观测、信道参数或额外导频。

两匹配半量读出的唯一恢复差别是逐频点保留二阶矩对角项，或在完整复协方差的特征基内收缩。所有读出保持每条 chirp 的追踪复载波，回写原相位、时间和幅度坐标。门限仅取 2,500 个源 U 包的噪声相对功率最大值乘 1.05，两路线共享阈值和推理规则。使用源 U 原先分区种子 2026112991。源 V 面板种子为 2026112992，五台接收机各 180 个包。这里无新训练或参数。固定预测后，独立评分核查准确率、净纠错、逐 RX 和物理包分层区间。

Hu 与 Loizou 的子空间增强对信号和噪声协方差联合对角化，用于有色噪声下的语音增强。Wang 等用多麦克风及学习掩码估计空间协方差，并用识别词错率评估。这里使用单接收链的协议重复窗和相邻复频点，用包内观测估计各成分的二阶矩，输出仍为复 IQ，评估冻结设备识别。LoRa 原始实现的去扫频和同步提供协议参考。本轮保留已有接收坐标，去扫频仅用于估计和局部恢复。 [Hu 与 Loizou 原论文](https://citeseerx.ist.psu.edu/document?doi=7dddfada0c1ed2c525bad5ef1b65253d387992cf&repid=rep1&type=pdf) · [Wang 等原论文](https://arxiv.org/abs/1707.00201) · [Tapparel 等原论文](https://arxiv.org/abs/2002.08208) · [LoRa 作者实现](https://github.com/tapparelj/gr-lora_sdr/blob/master/lib/frame_sync_impl.cc)

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
