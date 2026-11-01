# Phase1 多候选修复损伤效用恢复网络

- run_id：`20261101-phase1-multi-utility-router-s2026092701-r01`
- group_id：`phase1-multi-utility-router`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在物理信道模拟源记录上分别估计原始与特征候选相对 DURA 的修复和损伤概率，固定后进行 Phase1 地面开发测试。

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

使用物理信道模拟源记录训练 698–128–4 网络，分别估计切换至原始信号或特征恢复信号的修复和损伤概率。源留出 1605 条有效记录，低郊区、中城市和低城市视图分别净修复 2、2、5 条，干净视图不变；目标由 L_s 真标签和高置信 U_s clean 教师伪标签混合构成，源记录可能与早期源探针重叠，基础专家训练见过源记录。模拟不是已验证的真实卫星链路。

源验证固定 epoch 10、净效用门限严格大于 0.08；权重 SHA-256 为 `b9201737fa428b6d1518ecbebd255505c6f8b3c26d821fa33d642269dcc64e21`。源重放见相邻 `20261101-diagnostic-phase1-source-multi-utility-s2026092701-r01` 报告。此前已观察 Phase1 测试专家上界，本轮测试不是独立盲测。

## 正式预测前独立核验

真实冻结 Tri-SIG-TIME SSL、DURA、边界和特征专家及固定效用网络，在 1640 条源留出物理记录的四个视图上逐条比较：正式推理实现的原始决策、候选选择和输出决策与独立源域重放完全一致；重放计数也与源训练记录一致。预检未读取测试 IQ 或真值。脚本与机器可读证据保存在本目录。

## GPU 启动

GPU 2 的 worker 323639 已启动；真实冻结地面基座和专家的 128 条源记录 GPU 检查通过。独立 CPU 运行的专家选择计数与 GPU 检查一致，均为 `[0,127,0,1]`；这只核对计数，不声称逐条 CPU/GPU 选择完全一致。固定参数 SHA 与登记一致，七视图预测进行中。
