# 扩张卷积等价分解的 GPU 数值与速度对照

- run_id：`20261108-phase1-polyphase-conv-benchmark-s2026092701-r01`
- group_id：`phase1-polyphase-conv-benchmark`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

先验明真实源 DURA 后，用固定随机张量对照原扩张卷积与多相普通卷积的值、输入及参数梯度和前后向耗时。

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

## GPU 卷积对照

固定代码 commit `a8f9abb7eef0d4279d2195cbad4505aeb1dc71f8`；CPU 双精度五项输出及输入/权重/偏置梯度检查和独立 P0/P1 审查通过。N607 发布包两端一致、远端编译通过；GPU2 启动前空闲 18,501 MiB，另有两个其他 owner 计算进程，队列 PID 1308635、GPU worker PID 1308651 和真实 DURA 源样本 smoke 已核实。

数学分解将长度 256 的每条信号按扩张率拆成不同余数序列，分别用不扩张卷积处理并还原顺序。GPU 对照使用固定随机张量，尺寸 batch128、192 输入及输出通道、长度256、核5，测试扩张率 2/4/8/16/32；两种路径采用相同输入、权重和上游梯度，测量前后向耗时及输出、输入梯度、权重梯度、偏置梯度的逐字节和浮点误差。每种条件热身，正反顺序各比较一次。训练机共享 GPU，单算子短程测试不能证明整个模型加速或训练与测试结果不变；新实现尚未接入任何在训网络。
