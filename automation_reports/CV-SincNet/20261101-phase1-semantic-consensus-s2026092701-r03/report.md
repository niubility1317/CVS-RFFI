# Phase1 类别一致性恢复网络

- run_id：`20261101-phase1-semantic-consensus-s2026092701-r03`
- group_id：`phase1-semantic-consensus`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

源域训练多候选特征类别预测器，只采用与高置信类别一致的恢复信号，固定后进行 Phase1 地面开发测试。

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

## 源域依据及固定规则

物理信道模拟源记录训练 698–256–6 类别推断网络；只选择冻结地面基座对其输出类别一致的专家信号，优先特征、边界、原始，否则保留 DURA。源留出每个视图有 1596 条有效记录，其中 398 条 L_s 真标签、1198 条高置信 U_s clean 教师伪标签；低郊区、中城市和低城市视图分别净修复 6、5、21 条，干净视图不变。源记录可能与先前探针重叠，基础专家训练见过源记录；模拟不是已验证真实卫星链路。

源验证固定 epoch 20、类别置信度至少 0.9，权重 SHA-256 为 `75dcc3b3c606867db4dd21860687ed25d40bdbcf551411f2db073ecfacfa33b6`。源重放见相邻 `20261101-diagnostic-phase1-source-semantic-consensus-s2026092701-r01` 报告。此前已观察 Phase1 测试结果，本轮测试不是独立盲测。

首次发布包因公共加载模块来源不符而被预检拦截。第二次运行的输出编号不符，停止时仅干净视图完成，未评分。r03 修复固定编号，并在发布预检核对代码和登记编号。失败状态和证据见 r02 报告。

## 正式预测前独立核验

真实冻结 Tri-SIG-TIME SSL、DURA、边界和特征专家及固定类别一致性网络，在 1640 条源留出物理记录的四个视图上逐条比较：正式推理实现的原始决策、候选选择和输出决策与独立源域重放完全一致；重放计数与源训练记录一致。预检未读取测试 IQ 或真值。脚本与机器可读证据保存在本目录。

## GPU 启动

GPU 4 的 worker 346451 已启动；真实冻结地面基座和专家的 128 条源记录 GPU 检查通过。独立 CPU 运行的专家选择计数与 GPU 检查一致，均为 `[0,128,0,0]`；这只核对计数，不声称逐条 CPU/GPU 选择完全一致。固定参数 SHA 与登记一致，七视图预测进行中。
