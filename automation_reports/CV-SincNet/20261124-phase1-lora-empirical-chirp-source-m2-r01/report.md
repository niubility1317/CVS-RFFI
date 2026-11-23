# LoRa 经验复谱库与包内两组独立匹配

- run_id：`20261124-phase1-lora-empirical-chirp-source-m2-r01`
- group_id：`lora-observable-satellite-recovery`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

检验合法源 U 复谱模板、跨 chirp 留出查询和两半组的预测一致性对双路线同包冻结识别的影响。

将 5,000 个无标签源 U 的六条实收 chirp 按现有前导追踪建立去扫频、旋转和 RMS 坐标，把 256 点复谱作为模板。模板值无设备标签和接收机条件。匹配使用去载波后 1 至 32 kHz 的复数均值。在同一库中选取 16 个最近模板，用外带实收谱功率给距离权重设温度。实际用到的噪声代理还可能包含硬件分量、失配和相关干扰。

全包对照用全部六条 chirp 匹配，交叉读出用另一半三条选择当前半组的模板。模板均值和加权离散程度与当前包方差代理共同给出后验修正。一致性读出按两组模板预测的逐频复数差异，相对于模板离散程度和实收方差，压低不一致修正。四项均保留每条 chirp 的实收复载波，回到原接收时间、相位和幅度坐标。两路线共用同一个源 U 库和分离 2,500 个 clean U 的旁路阈值。此工程对照无优化器和新神经网络参数，库的存储单独计量。

非局部均值由相似观测块的距离推导加权平均。Noise2Noise 利用独立受扰对来训练恢复。这里使用实际无标签源 IQ 模板，在当前包两组协议重复上独立查询，用复数预测的相互一致性约束对原实收 IQ 的修正。没有假定 practical 两组噪声统计上独立。用同一冻结识别器比较直通、成熟 DURA 和四项工程读出。 [非局部均值原论文](https://doi.org/10.1109/CVPR.2005.38) · [作者公开实现](https://www.ipol.im/pub/art/2011/bcm_nlm/) · [Noise2Noise 原论文](https://proceedings.mlr.press/v80/lehtinen18a.html)

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
