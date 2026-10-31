# Tri-SIG-TIME SSL 引导的生成轨迹星地恢复

- run_id：`20261101-phase1-trisigtime-source-rollout-E30-s2026092701-r01`
- group_id：`phase1-trisigtime-flow`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

只用 Phase1 源域训练数据，从接收 IQ 出发完整训练两条四步恢复轨迹，冻结 Tri-SIG-TIME SSL 并对 Phase1 测试集进行七视图预测。

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

## 已核实状态（2026-11-01）

代码版本 `952721591136a0c6b4e51c52541fcb72ff6e0b13` 已发布至 N607，远端归档 SHA-256 与本地一致。控制器 PID 2172525，实际工作目录为预登记 release，命令和进程已读回核实。

`smoke.json` 为 PASS：加载真实 Tri-SIG-TIME SSL E50／11100，校验源域 L_s／U_s 物理 ID 一致；两条源样本的 IQ 与冻结基座特征梯度范数分别为 4.387862 和 1.634125。基座权重保持冻结，临时 checkpoint 严格重载一致；未读 TX 标签或测试 IQ。

所有显卡均被其他计算占用，控制器等待一张空闲且剩余显存不少于 12000 MB 的显卡。30 轮训练、七视图预测和独立评分尚未开始；当前没有本候选性能结果。预测全部完成后自动进行 CPU truth-last 评分。

2026-11-01 更新：只停止了本 run 等待显卡的控制器，独立读回确认尚未提交 GPU worker、尚未开始训练；smoke 证据保留。后继 r02 按项目同账号单卡最多两个计算进程的容量规则运行，科学方法不变。
