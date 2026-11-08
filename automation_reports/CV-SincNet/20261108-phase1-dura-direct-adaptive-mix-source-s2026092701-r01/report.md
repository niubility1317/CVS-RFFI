# DURA 与独立网络自适应组合训练

- run_id：`20261108-phase1-dura-direct-adaptive-mix-source-s2026092701-r01`
- group_id：`phase1-dura-direct-adaptive-mix-source`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定两个源合法恢复器，用父模型未见源 IQ 训练逐记录组合权重；训练后自动七景测试。

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

## 启动证据（2026-11-08）

N607 后台队列 PID 1040322，控制器 PID 1040338，GPU5 训练进程 PID 1040341；启动前显存余量 18501 MiB。发布代码 commit `708a3d8a9b1ce57907de5a26f9b6bbf6ba9553ab`；传输包本地与远端 SHA-256 一致，远端编译通过。六项针对性检查通过，独立 P0/P1 检查通过。训练使用 17632 条父模型未见的合法源信号，每条三种既有模拟视图、20 轮；恢复器训练不使用分类器、身份真值、验证或测试数据。检查点固定最后一轮，不依据测试选模。训练冻结后自动完成 Phase1 七视图预测，全部预测固定后独立评分。历史测试已被观察，结果不是独立盲测。所有启动均隐藏窗口。

真实源域 32 条样本检查已通过：远端 `source_train.log` 显示 `SIGNAL_MIX_GATE_SOURCE_SMOKE_PASS 32`；训练进程命令行和 GPU5 占用已核实。

## 训练冻结与测试衔接

GPU 训练按固定预算完成 20 轮，`source_summary.json` 为 SOURCE_TRAINED；17632 条源训练记录，三组模拟视图，训练损失首轮 0.018702、末轮 0.017960。最终权重固定，112 条合法源样本的分类链路检查通过，控制器已进入 RUNNING_GROUND_PREDICT；七景结果和评分尚未完成。
