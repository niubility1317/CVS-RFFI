# R42 直接注册：共享核与 CUDA 加速无中断接管

- run_id：`20261102-phase2-r42-directc-fast-m4-r02`
- group_id：`phase2-r42-direct-c-zero-target-old`；类别：`cvs`；阶段：`Phase2-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Fixed direct-C MIX01 16 KN; identical source and task definitions; inherit finished/in-flight predictions; share CPU kernels and use FP64 CUDA for large support.

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

## 已核实的接管

N607 controller 1526811、独立 readout 1528741；旧 controller 已退出，原 16 个完整单元复用，8 个旧 worker 自然运行并计入限额，首批 11 个新 worker 已启动。4 个实际源模型无 target/query 数值检查 PASS，最大分数差 2.22e-16。全部预测闭合后独立评分；原生 CIL 只复用 raw/raw。

本地合成数据单环节测试显示共享核与大支持 CUDA 可减少部分判决耗时，GPU 共享及负载变化使其不能作为本实验整体速度；实际吞吐待新任务闭合后核算。
