# LoRa 与 WiSig 共同恢复的源域 practical 模拟配对筛查

- run_id：`20261128-phase1-shared-practical-pairs-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；WiSig 已进入训练，LoRa 正生成源 U 模拟池。

## 目的与对照

两数据集各自源 U 抽 1024 包，利用独立源训练随机流生成 clean 和六景 practical 模拟配对，训练同一实增益与 DURA 物理残差恢复规则。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 49efe714e29b1513202add888c116dee5cd7949d；N607 控制器 3953233，LoRa 3953439/GPU 2，WiSig 3953440/GPU 3。源模拟前 16 包上的真实专家、教师及有限非零梯度 smoke 均通过；PID/CWD/GPU 和日志增长核实。预检、单包 SHA、远端编译和独立 P0/P1 审查通过。

## 结果与覆盖

每侧从合法源 U 固定抽 1024 个物理包，128 clean，896 包近似等量覆盖六种已登记 practical 模拟配置。使用源训练专用 namespace 和独立随机种子，未读取目标信号、目标统计、V 或源 U 真实 TX。训练 clean 和受扰信号先用模拟器定义归一化，再共同除以当前受扰包 RMS；两侧共用 5% 逐频网络、损失和训练预算，LoRa 实增益及 WiSig R17 锚点由已曝光源 V 选定。冻结教师仅训练时使用，模拟 CPU 成本及专家／教师祖先成本另计；筛查尚未完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
