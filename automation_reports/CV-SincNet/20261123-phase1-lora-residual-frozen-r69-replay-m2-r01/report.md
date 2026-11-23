# LoRa residual 原冻结 R69 三种子配对补齐

- run_id：`20261123-phase1-lora-residual-frozen-r69-replay-m2-r01`
- group_id：`lora-observable-satellite-recovery`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

用原 residual practical R69 2026092702 权重及原半量规则，补齐 Tri-set 01、03 的同包决策，并复用 02 的既有冻结预测形成三种子匹配基线。

恢复器复用原 residual 256 kHz 训练合同和 stage2 检查点，投影后剂量固定 0.5。三个识别种子共用同一份冻结 G，且只使用各原 Tri-set 的联合训练默认头。本补测复用原 residual 15 台接收机、同序 30,000 个物理包和原七个视图。全量冻结之前运行端只处理接收 IQ。新增两个种子的每个分片只运行一次恢复器，随后两识别头分别计算。由于 full 的半量历史权重与恢复版本已固定，full 原结果保留原实验身份。

四张较空闲 RTX 3090 承载互不重叠的接收机分片。源 L 上的真实 R69 和两个 F 无目标 smoke 通过才启动目标读出。新增矩阵的 420 个文件与原 seed02 文件全部通过一致性检查后，CPU 独立打开原 IEEE DataPort 本地测试归档。按接收机分层重抽物理包，三次对同包的种子判断作为同一统计簇，记录 1,600 次配对区间及逐景、逐 RX、逐种子纠错和改错。[原 LoRa 数据发布](https://doi.org/10.21227/d6vx-r538)

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
