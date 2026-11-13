# 源细微变化响应保持恢复器，从零单阶段30轮

- run_id：`20261107-phase1-Restore-ResponseJointPrior-E30-s2026092701-r01`
- group_id：`source-response-joint-prior-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同JointPrior结构和E0，增加固定1%源IQ变化的配对响应损失；训练用等价低秩反向，部署保持原dense后验。

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

## 响应保持版真实训练启动

VERIFIED / RUNNING。新响应保持版已在GPU6按原授权准入，实际PID/CWD/科学命令/GPU与源检查日志增长核实。真实原F0/source2、响应项单独到均值/协方差两头的梯度、物理隔离、低秩训练/原dense部署一致性及严格E0重载均通过。1471463参数与原JointPrior E0保持，增加.01响应损失，固定1%源RMS四方向逐记录/逐轮轮换；原L_s/U_s、从零单阶段30/7410和自动Main七景不变。训练后验实际调用翻倍并明确计数，F0在线行数不增加；微批实测提速不冒充完整训练提速。尚无本版Main成绩。 启动证据

## 源训练技术中止，产物保留

FAILED / PRESERVED。已完成9轮、2223步，第10轮训练中触发“源目标非有限或后验相对残差超过1e-8”的联合检查并退出；原异常没有区分具体触发项，原因尚未确定。E09技术恢复点已保存，未产生final30，也没有提交Main预测或读取目标真值；未自动重启。已完成轮次的平均损失、响应误差和平均残差有限，但平均值不能排除单样本极值。现有其他健康任务保持。 中止证据

## 第 9 轮恢复点只读核查

VERIFIED：E09 与原实验配置、代码身份和初始化状态结构一致；145 个模型张量及 139 组优化器状态有限，优化器步数均为 2223；CPU/CUDA 随机状态和第 10 轮 63000 条源顺序已保存。未读取 IQ 或目标、未更新模型。此核查不证明可逐位重放，也不自动恢复训练；须先定位原联合数值检查的实际触发项。恢复点审计
