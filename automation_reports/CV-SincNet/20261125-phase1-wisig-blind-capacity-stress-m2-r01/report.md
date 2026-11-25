# WiSig 主对比的默认与增强实用星地信道盲恢复

- run_id：`20261125-phase1-wisig-blind-capacity-stress-m2-r01`
- group_id：`wisig-blind-capacity-stress`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

将 LoRa Full 六景领先的扩容 DURA 半量版适配到 WiSig 256 点窗。固定两组从零重训，仅改变模拟信道强度，交换默认和增强测试，对比相同物理包的原始接收与冻结 R69。

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

## 预登记执行细节

LoRa 结构来源为完成三种子独立评分的扩容 DURA 半量版，Full 六景 56.4759%，低城区 43.4222%。WiSig 使用单个 256 点 IQ 窗和 WiSig 短训练字段的 IQ 差分估计，保留扩容网络、两阶段 MAP 和半量读出。用本地单窗与同窗复制对照核实单窗实现的一致性。LoRa 权重及 LoRa 数据均未用于 WiSig 训练。 [LoRa 正式结果](../20261124-phase1-lora-dura-capacity-target-m8-r02/report.md)

默认组复用原始 practical 源 IQ，增强组使用完全配对的实用信道配置，将参考 SNR 下调 6 dB、残余相位跟踪误差标准差乘 1.5。强度组使用相同的物理包、场景、路线、随机流、初始化和 40,000 次更新。每次取 16 包，四份固定训练模拟抽样轮换。预计原始 IQ 约占 15%，其余两路线各约 42.5%。生成边界只落盘受损 IQ。训练使用合法源 U 受损 IQ、配对原始 IQ 和原始样本标志。推理仅接收实收窗口，窗口内估计信道作用和有效噪声。冻结 R69 的历史祖先曾使用模拟器作用和噪声监督。

正式比较沿用 WiSig Main split 的 126,000 个 cross_rx、42,000 个 cross_both、Full/Residual 两条路线和三个冻结识别器。默认与增强测试均覆盖 clean 和六景，两组训练权重在两种测试强度上交叉读出。默认的直通及 R69 复用固定历史决策，增强强度重新计算。全八个盲预测分片写定后才执行独立 CPU 真值评分。主要证据为增强六景上训练增强组减训练默认组的同包变化，以及增强组相对直通和 R69 的收益。另报告低城区、默认六景和 clean 的变化，并按识别器种子、接收机与划分检查。冻结推理报告训练耗时、峰值显存和 B1 时间。

本地 9 项聚焦检查通过，覆盖单窗数值一致性、两路线采样、训练梯度、冻结预测及同包身份校验。N607 普通账号直连、磁盘和显存预检通过，两条源路线的默认模拟 IQ 均逐值复算一致。独立 P0/P1 复核发现训练进程接收到完整发布配置，已改为仅提供两条源侧路径，未发现其他 P0/P1。冒烟与正式完成状态以上线后的独立读回为准。
