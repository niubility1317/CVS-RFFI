# Phase1 受限边界校正学生恢复网络

- run_id：`20261101-phase1-boundary-student-E30-s2026092701-r01`
- group_id：`phase1-bounded-boundary-student`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结原 DURA 和 Tri-SIG-TIME SSL；仅在源训练中蒸馏特权输入梯度反事实，推理由受限学生从接收和 DURA IQ 预测校正。

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

## 已核实启动状态

源码固定为 `49a6c379b83447d0efc36e8df1f83356e15b7561`，发布归档两端 SHA-256 一致；独立代码复核未发现 P0/P1，局部测试覆盖幅度上限、零校正初始化、特权目标训练与冻结网络行为。

实际 Phase1 Tri-SIG-TIME SSL 和 DURA 权重的 CPU 两条 L_s 样本检查通过，两个样本均为 clean 教师与 DURA 不一致的源记录；临时学生权重经严格重载。GPU 5 上实际 128 条 L_s 的检查也通过，其中 7 条教师判断不一致，特权 oracle 达到 7 条 clean 教师类别。源档案物化 512 条 IQ（L128/U384），实际检查选择 L_s。两套旧网络保持冻结，无 TX 标签、V 或测试 IQ 读取。该检查仅验证训练链路，不能当作可部署学生性能。

临时学生权重随后丢弃；正式 E30 源训练从固定种子重新初始化。控制器 PID 2332039、GPU worker PID 2332138 和显卡归属已独立读回；预登记七视图预测和 truth-last 评分将在固定最终权重后执行。候选构思前已有 Phase1 开发测试结果被观察，后续同集不是独立盲测。

## 训练中 CUDA 失败

GPU 5 的 worker PID 2332138 在第 9 轮期间报 `CUDA error: unknown error` 并退出。已完成 8 轮、3944 次更新；日志、失败状态和错误栈均保留。独立读回确认控制器和 worker 均已退出，`flow/final.pt` 和 `training_complete.json` 不存在，也没有固定预测或性能结果。本次不具备恢复训练的最终权重。

同时期同机另一实验的最小 `torch.cuda.init()` 检查也失败，设备管理工具无法取得 GPU 7 的句柄；这些现象提示设备层异常，但尚不能证明具体因果。未重置设备或重启其他作业。环境及完整设备清单恢复前不新开 GPU 任务；如之后重跑，须使用新的独立 run 并重新执行固定预算。
