# 跨接收机身份参照三变体单种子源留出验证

- run_id：`20261026-phase1-PAMF-RXReference-sourceHold-m3-r01`
- group_id：`phase1-pamf-receiver-reference-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（独立核实三个源训练worker已实际更新；不代表目标达成）

## 目的与对照

同seed01原E30+20至E50，raw/同G/双G身份参照三变体；新增分支排除RX1，完整源V八视图与既有PLControl配对，源阈值选择后另行扩种子。

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

## 发布前范围

详见 [最小预登记](preregistration.md)。三个方法变体仅有一个student seed2026092701；不是三种子测试。固定原E30+20、共同源剂量0.07145844915553692及默认头联合训练，完整源V八视图逐行自动评估；源条件通过后未来三种子矩阵另行登记。本run不访问目标数据，不替换默认模型。两项新检查与compile通过，一次独立P0/P1审查PASS；真实权重联合backward smoke为launcher第一步。

## 三条源验证训练已实际更新

VERIFIED_RUNNING，科学代码 `37dc44e3dd52854a415ca5f7663669b417171691`。三个变体均已进入真实更新，实际物理GPU为0/0/1，只有student seed2026092701。真实E30/defaulthead/L128-U256联合backward smoke通过；独立PID/CWD/argv/GPU和step日志增长核实。当前可用完整新增step记录分别为[261, 262, 278]，不是E50完成或新目标结果。原L768包、AsKL0、隐藏U、共同升温权重和零G/参照更新的最新记录符合配置。见 [启动证据](launch_evidence.json) 和 [启动源进度](startup_source_progress.csv)。逐行训练后自动完成完整源V八视图；本run不包含目标访问或三种子扩展。

## 运行阶段完整日志前缀核查

独立检查实际initial checkpoint及全部当前完整step记录，核对原默认头6×160、own E30父链配置、L768/AsKL0/CRIC.1、完整U曝光、排除RX1的新增接受计数、共同权重和非有限/真值标志。三个变体在相同步数的新增接受流完全一致，每完整epoch新增接受41939条；全U仍进入原raw PL主项。G与参照未更新，无新目标读取。

完整可用记录数、观测UTC时间与每行进度见 [阶段源执行](available_source_execution.csv) 和 [阶段证据](available_source_evidence.json)，快照可随观察更新。已缓冲而未形成完整JSON行的尾部不计入；这不是全部E50完成或目标收益。loss仅作源训练健康观察，不改变已冻结的源选择规则、训练预算或候选顺序。
