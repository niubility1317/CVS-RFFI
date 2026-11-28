# LoRa 与 WiSig 源域 practical 配对训练的无新增教师匹配对照

- run_id：`20261128-phase1-shared-practical-mse-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；WiSig 源域训练与 LoRa 源模拟池生成进行中。

## 目的与对照

以相同 4096 个源 U 模拟包、seed 和 1000 步，仅关闭冻结识别器教师损失，检验指导训练的实际影响。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 d0977fba91ceae71494cb2d154f7cc35166e0318；N607 控制器 4019329，LoRa 4019351/GPU 6，WiSig 4019352/GPU 4。两侧前 16 包真实恢复专家与波形梯度 smoke 通过，PID/CWD/GPU 和日志增长已核实；预检、归档 SHA、远端编译与独立 P0/P1 审查通过。

## 结果与覆盖

匹配 5% 低城区加权教师版的共同网络、源 U 4096 包及模拟、seed、锚点、1000 步和非教师损失；本次新增训练只用 0.2 波形误差加 1.5 clean 锚点保持，不加载/调用识别器。LoRa 实增益父专家仍有 E104 指导及伪标签祖先，WiSig R17 为 F-free；两数据集筛查仍使用各自冻结默认 F。源 V 已曝光并参与锚点选择，无 U 真 TX、V 或目标参与训练；筛查待完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
