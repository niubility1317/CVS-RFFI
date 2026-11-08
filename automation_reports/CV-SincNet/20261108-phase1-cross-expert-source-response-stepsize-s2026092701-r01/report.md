# 时频网络源域响应单步学习率对照

- run_id：`20261108-phase1-cross-expert-source-response-stepsize-s2026092701-r01`
- group_id：`phase1-cross-expert-source-response-stepsize`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

基于先前源域单步损失上升，探索两个更小学习率下波形和响应项的输出头单步行为。

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

## 源域单步探索

先前 3e-4 的源域输出头单步使两项损失上升；本次固定 3e-5、3e-6 两个更小学习率，各比较仅波形与波形加 0.01 幅度三次响应。四个条件分别从相同种子新建模型和 AdamW，只更新两个输出头一次，不保存模型。32 条源记录用于内存单步，另 32 条不同源记录用于观察；都来自已探索的同源样本。

CPU 两线程诊断不替代正式 GPU 训练，不读取在训 checkpoint、F、TX 标签、V 或目标。代码 commit `b2a2e140b4d0123b84f11b2cf3976695e7d27429`；九项针对性测试、独立 P0/P1 审查、传输和远端编译通过。后台 PID 1184389 已观察，真实源 smoke 及首条件完成。

两种学习率、两种损失的四个单步条件已完成。与前次 3e-4 结果对照见 [analysis.md](analysis.md)；不影响在训模型。
