# reference_response：跨TX共同位移补偿×分支贡献预测，CE拼接四seed消融

- run_id：`20261008-phase1-receiver-residual-manysig-m16-r02`
- group_id：`cvs-cross-tx-displacement-branch-contribution`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

R2修复CPU头初始化影响CUDA dropout RNG的技术问题；保持固定强骨干、最基础CE+原生拼接、四组四seed从零E200及全部测试；R1保留为含随机混杂的诊断，不热改健康任务。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 修复与验证

R1发布后发现新增头初始化重置CUDA dropout RNG，保留为含随机性混杂的诊断；不停止或热改健康任务。R2使用独立包`experiments/cvs_receiver_residual_v2`、独立release和输出，从零执行相同四组四seed，不继承R1权重。仅修复CPU私有生成器，算法、损失、训练预算均不变，设计在R1任何target读取前固定。

已验证四组CPU与CUDA随机数状态均保留，CUDA全机制smoke通过（初始骨干/logits逐位等价、分支和、非零有限辅助梯度、逐样本推理、checkpoint、原生拼接边界）。新增参数为0/5578/2173/7751。实体配置与登记验证通过，复用同次独立P0/P1核心审查，并针对初始化修复作实测验证。

完整机制与固定系数见[DESIGN.md](../../../experiments/cvs_receiver_residual_v2/DESIGN.md)。每GPU最多2个总任务，R2只使用剩余合法名额；不改R1或其他健康任务。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
