# WiSig 源域配对时标及窗口缺失归因

- run_id：`20261123-diagnostic-wisig-source-paired-time-edge-s2026092701-r01`
- group_id：`wisig-observable-satellite-recovery`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在隔离诊断中使用同物理包配对 clean，分别定位时标估计、缺失前后缀和整体复增益在冻结源识别中的作用。

固定 1,800 个源 V 包、同一 F50 和两路七景。各包在 ±64 点有效重叠范围内，选取使受损与配对 clean 复相关平方最大的时移。第一个归因读出保留收到的全部有效重叠，缺失边界仍由既有工程规则填充。第二个读出只将缺失边界替换为配对 clean。第三个还在重叠段拟合一个复系数，把观察到的整体增益、相位差纳入归因。三项都仅用于隔离诊断。完整源预测写入后，再由独立 CPU 评分器读取源 V 标签。本运行没有训练、校准、模型选择或目标访问。

接收端时标、直通和 R69 使用同源面板已冻结的预测，依物理 ID 对齐。时移只是源配对相关估算值。运行没有读取模拟器真实时移、信道、噪声，也不产生可供候选使用的数据或参数。代码模块与候选推理入口保持隔离。

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
