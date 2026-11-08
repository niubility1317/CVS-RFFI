# 大型时频网络源域 GPU 分层耗时诊断

- run_id：`20261108-phase1-cross-expert-layer-timing-s2026092701-r01`
- group_id：`phase1-cross-expert-layer-timing`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定非零时频模型与源输入，利用原始前向的逐模块钩子测量卷积、归一化和激活耗时并核对完整输出字节一致。

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

## 逐层 GPU 计时

执行代码 commit `705c835111fde7e7a7902106ed8b1889b94de8b3`。GPU2 启动前空闲 18,501 MiB，另有两个其他 owner 计算进程；队列 PID 1296966、GPU worker PID 1297122 和实际工作目录已核实。五项相关检查及独立 P0/P1 审查通过；发布包两端一致并在远端编译通过。

固定 128 条合法源记录，以确定性 FP32、TF32 关闭和固定种子构造诊断网络；时频输出头以标准差 0.001 的非零正态值初始化，确保分支影响输出。对 Conv1d、GroupNorm、GELU 模块注册不替换输出的计时钩子，逐模块同步，两次带钩子的完整输出均与原 forward 做字节比较。钩子会改变执行时序且 GPU 与他人共享，不代表未经插桩的完整训练速度；不读取 F/TX/V/目标，不更新或保存模型。

## 计时结果

VERIFIED / ANALYZED：两个带钩子前向均与原模型输出逐字节一致。9 个扩张卷积合计 37.915 / 48.329 秒，19 个普通卷积合计 0.0152 / 0.0134 秒；结果受 GPU 共享和逐层同步影响，详见 [analysis.md](analysis.md)。
