# Phase1 物理信道丰富特征四专家恢复网络

- run_id：`20261101-phase1-physical-rich-router-s2026092701-r01`
- group_id：`phase1-physical-rich-router`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

独立源记录经物理信道模拟，以基座中间特征、频谱与专家概率训练选择器，固定后进行 Phase1 地面开发测试。

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

使用新源物理记录独立生成的实用链路模拟视图训练 378–128–4 选择器；冻结 Tri-SIG-TIME SSL、DURA 和另外两个专家。源留出 796 条有效记录，低郊区、中城市和低城市视图分别净修复 1、4、8 条，干净视图不变。目标由 L_s 真标签和高置信 U_s clean 教师伪标签混合构成；基础专家训练见过源记录。模拟信道不代表已验证的真实卫星链路。

固定 epoch 2、概率差门限 0.6；源选择器权重文件 SHA-256 为 `4e90a750e3d3b6f1f70e45d2b14e0a2053b3fc0d1c9068f9edf3ac55184eca16`。源重放见相邻 `20261101-diagnostic-phase1-source-rich-physical-router-s2026092701-r01` 报告。此前已观察 Phase1 测试专家上界，因此本轮测试不是独立盲测。

## 正式预测前独立核验

真实冻结 Tri-SIG-TIME SSL、DURA、边界和特征专家及固定选择器，在 820 条源留出物理记录的四个视图上逐条比较：正式推理实现的原始决策、专家选择和输出决策均与独立源域重放一致；重放计数也与源训练记录一致。预检未读取测试 IQ 或真值。脚本与机器可读证据保存在本目录。

## GPU 启动

GPU 4 的 worker 286825 已启动；真实冻结地面基座和专家的 128 条源记录 GPU 检查通过。独立 CPU 运行的专家选择计数与 GPU 检查一致，均为 `[0,128,0,0]`；这只核对计数，不声称逐条 CPU/GPU 选择完全一致。固定参数 SHA 与登记一致，七视图预测进行中。
