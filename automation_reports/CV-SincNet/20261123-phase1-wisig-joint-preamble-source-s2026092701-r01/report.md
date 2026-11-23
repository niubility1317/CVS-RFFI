# WiSig 短长前导共享响应源域对照

- run_id：`20261123-phase1-wisig-joint-preamble-source-s2026092701-r01`
- group_id：`wisig-observable-satellite-recovery`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在两条登记路线的同包源 V 上，用冻结 Main split Tri-SIG-TIME 比较直通、原 R69、过渡检测及短长前导联合共享响应。

联合方法以实际收到的 256 点 IQ 和公开 Wi-Fi 前导为输入。L-STF 的 20 点重复给出粗频偏，短训练字段与过渡区内的固定接收坐标共同拟合一个复增益或三抽头复响应。候选时移以协议标称过渡点为中心，范围为两个短训练周期。拟合分数为经正则化后的归一化残差减小量，置信度取候选与标称区间的分数差。移位后逐点复制有效观测，仅对缺失边界按拟合响应外推，补回原相位坐标。

两路线用同一套规则和校准阈值。门限来自 4,096 个源 U 包的置信度 99.9% 分位。前导重复残差按实测有效残差方差记录，包含可能的 CFO、硬件差异和估计误差。它参与诊断，联合拟合本轮采用固定区间权重和抽头正则强度。可训练参数和优化器步数均为零。

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
