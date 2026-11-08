# DURA 源接收机均衡训练

- run_id：`20261108-phase1-dura-large-context-receiver-source-s2026092701-r01`
- group_id：`phase1-dura-large-context-receiver`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

归一化 DURA 全局修正，源训练按公开接收机频率反比均衡，不在推理时使用 RX。

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

## 预登记补充

源接收机 1、3、4、6、8 的 32768 个合法源训练样本数分别为 8262、8182、5458、5341、5525。每个接收机总训练权重相同，同一源记录的三个模拟视图使用相同权重；推理不读取接收机标识。训练不读取 TX 标签、识别器、V 或目标数据。原 DURA 固定，独立初始化的归一化修正网络有 14793858 个可训练参数；固定 20 轮、每轮 3072 步，最终轮冻结。关联只支持源域观察，不证明因果关系。

11 项针对性测试、真实 DURA 和源 IQ 的 6 步 GPU 冒烟验证以及一次独立 P0/P1 审查通过。GPU2 以总计算进程少于 3 且空闲显存不少于 4096 MiB 为排队启动条件，不干扰其他任务。七个场景的预测全部完成后才独立评分，目标结果不回流训练与选择。传输包本地远端一次比较及远端编译通过。

## 启动核实（2026-11-08）

N607 后台队列 PID 722107，工作目录及命令已独立核实。当前日志 `GPU_QUEUE_WAITING 2`：GPU2 有 3 个已有计算进程，因此等待空位；尚未开始训练。

## 后台训练启动核实（2026-11-08）

GPU2 达到排队容量条件后自动启动：控制器 PID 730927，训练 PID 731113，训练进程已出现在 GPU2 计算进程列表，显存约 2074 MiB。真实 DURA 与源 IQ 的 64 条、三个视图的启动 smoke 完成；首轮训练仍在进行，尚无正式性能结果。

## 七场景正式结果

独立评分已完成。六个受损场景均低于 DURA，低城区 62.8327% 对 68.8012%；每景、净纠错和 TX/RX/day 见 [七景分析](../20261108-phase1-dura-large-context-receiver-ground-test-s2026092701-r01/analysis.md)。
