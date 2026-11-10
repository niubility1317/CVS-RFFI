# 完整 PAMF 多分支标签与伪标签关系优化单种子验证

- run_id：`20261110-phase1-pamf-branch-relations-f0-s2026092701-m4-r01`
- group_id：`phase1-pamf-branch-relations`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按 events.jsonl 及独立证据更新）

## 目的与对照

旧版含参考响应三视图 PAMF，单种子四组检验 L、U 分支关系损失；普通 PL 与监督对照复用。

`sup_l` 在原监督训练中加入 L 分支关系；`pl_l`、`pl_u`、`pl_lu` 在原普通 PL 上分别加入 L、U、L＋U 分支关系。各分支独立计算同身份跨接收机对比损失，再等权平均；L 使用真实标签，U 只使用置信度至少 0.95 的伪标签。L 和 U 的损失系数均为 0.05，不增加推理分支。四组和系数在本次预测前固定；普通 PL 阈值 0.95、权重 1。

本轮为单种子探索，已有对照测试结果已公开，不能视作盲测或多种子结论。后续扩展的源域筛查预设为：`pl_lu` 的固定最终 checkpoint 在同一源域 V 上的准确率比原 F21 最终 checkpoint 低不超过 0.2 个百分点，且 L、U 关系损失均实际出现有效锚点；未达到时不直接扩大本方案。源域诊断不读取目标真值；所有预登记试验照常完成固定预测和独立评分，测试结果不用于调参、挑 checkpoint 或重跑。

正式测试包含原有三个人群上的 clean、六种 practical，分别报告 raw、DURA-L、DURA-LU；另外按项目协议，对 168000 个目标物理样本测试三种 weak 场景的 raw 输出。旧监督及普通 PL 的七景结果直接复用，并对同一冻结权重补算 weak 场景。所有预测固定后才在独立 CPU 评分阶段读取真值。

除与同种子监督／普通 PL 匹配比较外，预先列出旧版完整自然顺序 REPL 和固定重排完整 SSL 的已有七景结果作为参考：前者比较相同自然顺序，后者在相同 `cross_rx`／`cross_both` 人群仅比较已有 raw 场景。固定重排对照同时改变时间顺序与训练方法，不能将差异全部归因于半监督训练机制；两种参考均为已暴露历史结果。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

2026-11-10 已启动：代码 commit `beb681dc0458585c29fc1d66404bb96e4988057f`，单次发布包传输及远端编译通过；四臂真实 E20 无查询 smoke 全部通过，之后 GPU 0、1 各运行两组。已独立核对进程工作目录、命令、GPU 和日志增长；四组均完成续训首个 epoch、222 次更新。该状态不代表后续训练、预测或评分已经完成。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
