# LoRa 分块复谱盲先验与观测残差匹配消融

- run_id：`20261123-phase1-lora-spectral-block-source-m4-r01`
- group_id：`lora-observable-satellite-recovery`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

检验频域相同分块在所有 chirp 上遮挡后学习的复数先验，能否提高可部署 LoRa 恢复的同包冻结识别。

从当前实收六条 chirp 估计去扫频斜率、复载波和每条 RMS，在固定坐标内对 256 点复频谱分成四组交错的相邻两频点块。两臂用相同的块标记和两级实卷积处理复数通道。频块遮挡组预测一组时，在本包所有 chirp 上同时遮去该组。全谱对照保留待预测频点。留给粗预测的公共均值与离散程度都由它可见的频点组成。盲性在给定已追踪频谱的坐标中定义。原始 IQ 参与确定该坐标，最终细化仍使用中心复观测。

粗读出使用固定初步残差量。细化同时读取当前频点、粗复先验、包内重复偏差、追踪相干度和外带估计噪声功率，输出由观测方差代理调制的实数更新量与有界复数修正。估计量可能受硬件弱分量、干扰和跟踪误差影响。回写保留每条 chirp 的观测复载波与原时间、幅度坐标。两臂各 34,693 个参数、同一次从零训练 1,600 步。两路线复用候选的同一权重、阈值和推理规则。

两臂用同样的源 U IQ、成对通用扰动与 0.35 弱复谱差异保真权重。扰动含连续变化的独立与跨 chirp 相关加性成分、一阶着色、平滑时变噪声包络和短复回波。训练只读取波形输出与真实源 IQ，联合优化粗恢复、观测约束细化和源差异传递。500 个源 U 留出始终独立。两臂在同一源 U 池固定旁路门。F、V、TX 真值、目标与真实 practical 参数均不参与拟合和冻结。

Noise2Self 的论文及代码用遮挡位的噪声独立性从单个噪声观测估计风险。Laine 等从盲点先验与噪声模型推导结合中心观测的后验去噪。本实验采用实际源复 IQ 配对监督、跨 chirp 同频块遮挡与接收端方差代理细化，评估任务是冻结设备识别。没有假定实用信道的复频点噪声相互独立。已有隔离诊断仅用于定位 Wiener 识别损失，任何精确噪声未进入本实验。 [Noise2Self 原论文](https://proceedings.mlr.press/v97/batson19a.html) · [作者实现](https://github.com/czbiohub-sf/noise2self) · [Laine 等原论文](https://proceedings.neurips.cc/paper/2019/hash/2119b8d43eafcf353e07d7cb5554170b-Abstract.html) · [作者实现](https://github.com/NVlabs/selfsupervised-denoising) · [先前隔离归因](../20261123-diagnostic-lora-noise-floor-source-m2-r01/report.md)

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
