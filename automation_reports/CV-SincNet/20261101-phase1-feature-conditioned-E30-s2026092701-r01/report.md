# Phase1 冻结基座特征条件半监督恢复网络

- run_id：`20261101-phase1-feature-conditioned-E30-s2026092701-r01`
- group_id：`phase1-teacher-feature-conditioned-restoration`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结 Tri-SIG-TIME SSL 和原 DURA，以基座特征条件化受限 IQ 校正；源训练使用合法 L_s 标签和高置信 U_s clean 伪目标。

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

## 已核实排队状态

源码固定为 `c14ed7fa2f76ae362dff5fcb12a2b2b499ba3175`，发布归档 SHA-256 为 `6eb508ddb59a76163a51757ccd6e534a0c5a75764606ed05afa438c210bf4c62`。独立复核未发现 P0/P1；相关测试 7 项通过。

真实 Phase1 Tri-SIG-TIME SSL 和 DURA 权重的 CPU 检查通过：实际使用两条 L_s 记录，均为基座错误样本；梯度非零，冻结模型未变，临时权重重载一致。源档案物化 512 条 IQ（L128/U384），合法 L_s 标签表物化 6300 条；未读取 U_s 标签、V 或测试 IQ。这仅证明训练链路可运行，不是性能结果。

控制器 PID 2349731 已独立核验，正在遵守共享资源限制等待 GPU；GPU 批次检查和正式 E30 训练尚未启动。候选构思前已有 Phase1 开发测试结果被观察，后续同集不是独立盲测。

## GPU 初始化失败

控制器提交 GPU 1、worker PID 2359903；worker 在 GPU B128 检查加载冻结 Tri-SIG-TIME 时，PyTorch CUDA 初始化报 `CUDA unknown error`，尚未开始正式训练或测试。控制器和 worker 已退出；`complete.json` 与完整 `logs/worker.log` 保留，失败为技术故障。

独立设备查询发现 GPU 7（PCI `0000:C2:00.0`）的设备句柄返回 `Unknown Error`，`nvidia-smi -L` 非零退出且设备清单不完整。这是与 CUDA 初始化失败同时出现的设备层异常；现有证据尚不足以证明二者的因果关系。边界学生的原 worker 仍存活，不中断该进程。恢复完整设备清单及确定初始化能力前不重启或增加 GPU 任务。
