# 导师补实验：四种完整半监督对比方法

- run_id：`20261124-phase1-pamf-advisor-external-m96-r01`
- group_id：`phase1-pamf-advisor-external-full-pl`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 登记状态：ANALYZED（N607 完成 VERIFIED）

## 目的与对照

24 个数据设置 × RIEI、DRIFT、IB-RFF、SDG，沿用完整方法加普通 PL 和三场景源增强。与自适应 Tri 在相同角色及测试样本配对。

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

## 配方与矩阵

RIEI、DRIFT、IB-RFF 和 SDG 各覆盖 24 个设置，共 96 个从零训练模型。采用已验证的 Full＋普通 PL 与三种 practical 源训练增强，保留各方法原优化器、批量、轮数、原生损失和源验证规则。模型种子为 2026092701。与自适应 Tri 对齐 L、U、V 的物理样本、目标样本、划分和七景视图。训练标签预算为 630、3150、6300 条（L 与 V 合计），源数量覆盖 1 至 6 源 16 个组合，时间覆盖四折留日和 D0、D1 至 D2 的向前测试。

基线使用独立 N607 运行目录。每卡至多一个本 run GPU worker，入队时核实至少 3400 MiB 空闲。既有 D 配方的四种 F0 运行 CUDA 峰值分配为 77 至 601 MiB。预测完成后交给独立评分进程。提交的控制器也允许原自适应 Tri 队列先派发具备数据依赖的训练，训练用 8000 MiB 空闲阈值，预测保留 11000 MiB，已有 worker 继续运行。

## 启动记录

N607 启动 VERIFIED。真实权重无查询冒烟通过，四种方法均已开始训练。首批 8 行各自在 GPU 0–7 上运行，PID、工作目录、命令行、CUDA 和实际训练配置已核对。

运行代码提交为 `cd788bad712a0721e8522098c59765e291a17cdd`。首批对应 630 和 3150 条预算，每种方法各两行。其余 88 行随后由调度器派发。

## 已完成结果

N607 完成 VERIFIED。96 行完整训练、预测和独立评分。全部设置与自适应 Tri 结果对齐相同的源角色和目标物理 ID。 [详细结果](../20261124-phase1-pamf-advisor-adaptive-m48-r01/results.md) · [独立完成证据](../20261124-phase1-pamf-advisor-adaptive-m48-r01/completion_verification.json)
