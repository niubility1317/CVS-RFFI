# WiSig 重复前导频块复数恢复与幅相读出

- run_id：`20261123-phase1-wisig-repeat-block-source-m3-r01`
- group_id：`wisig-observable-satellite-recovery`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

使用实收短前导预测复谱，比较粗恢复是否遮挡同频块以及只修正幅度、相位或联合恢复的冻结源识别变化。

输入是原有 25 MS/s、256 点实收复 IQ。第 40 至 159 点按 20 点短训练周期估计相邻重复的相位斜率、相干度和相干均值。相位旋转仅用于合并重复，之后换回原相位坐标。粗恢复在四组相邻两点的交错频块上推断复谱。遮挡臂从其可见频点重算重复信息，全谱臂用相同结构和块标记但保留中心频点。细化读取原观测、粗预测、复残差和重复信息，幅度受同包方差代理限制。输出仍位于原接收时标与幅度坐标。幅相读出在原窗口的复 DFT 上分别固定原相位或原幅度。精确零系数采用可定义的相位回退。

门依据当前短前导重复差分有效方差相对信号功率，以及名义过渡附近的 20 点重复误差。方差代理含干扰、失配和估计误差。二者的 U 源阈值分别取 4,096 个 clean 包中的最大值乘 1.05 与最小值乘 0.95。方差比逾上限或过渡重复误差低于下限才激活，两路线和两训练臂均使用同一判定。实际训练和推理从同一接收估计函数得到重复和不确定性。两组各 240,868 参数、1,600 步，匹配独立通用幅相、时移、短回波与着色扰动。联合使用源 IQ 重建、同包一致性及 0.20 弱复谱差异约束。拟合与校准阶段仅读合法源 U，闭合后才读固定识别器并评估源 V。

WiSig 发布方的均衡实现使用更长捕获中的已知训练字段与信道均衡。本实验在既有截断窗内提取短前导重复，在原接收坐标检验冻结指纹识别。Noise2Self 用独立受扰维度的遮挡风险，本实验的粗预测遮挡同一频块，监督来自合法源 IQ 和通用扰动，细化允许接入当前频点。遮挡结构沿用 LoRa 实验的交错划分，重复前端按 Wi-Fi 协议重算。 [WiSig 原论文](https://arxiv.org/abs/2112.15363) · [WiSig 作者实现](https://github.com/WiSig-dataset/wisig-process-raw/blob/master/matlab_equalization/equalize_channel.m) · [Noise2Self 原论文](https://proceedings.mlr.press/v97/batson19a.html) · [Noise2Self 作者实现](https://github.com/czbiohub-sf/noise2self) · [LoRa 同类消融](../20261123-phase1-lora-spectral-block-source-m4-r01/report.md)

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
