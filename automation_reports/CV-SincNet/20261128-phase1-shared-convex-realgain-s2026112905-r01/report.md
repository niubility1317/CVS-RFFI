# LoRa 与 WiSig 共用有界频域融合筛查

- run_id：`20261128-phase1-shared-convex-realgain-s2026112905-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；两数据集源训练已在 N607 启动。

## 目的与对照

固定两数据集相同的逐频点凸组合：实增益和 R17 输出按 [0,1] 实权重插值，DURA 物理回投影仅作为门控特征；各数据集独立合法源训练。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `52665d5177362af508ad98c124d7779c9be950f5`；独立 P0/P1 审查未发现直接缺陷，9 项本地测试通过。N607 归档传输校验和远端编译通过，控制器 PID 3664854；LoRa worker PID 3664977 / GPU 7、WiSig worker PID 3664978 / GPU 2。已核实 CWD、命令、真实专家权重首步、初始频谱中点及有限梯度；两行日志已分别增长至 250 和 400 步。新 run 不使用上一版工作中的 GPU 0/5，其他 owner 进程保留。

## 结果与覆盖

本候选尚无筛查或正式成绩。两数据集共用同一个有界实增益融合核心与训练目标，两个冻结专家的输入适配、来源及各自权重不同；每个非 DC 频点的两个专家权重在 0 至 1 之间，DC 采用实增益专家。LoRa 实增益专家继承经授权的 E104 指导和 U 伪标签训练；WiSig 专家 F-free。新头只读合法源 U IQ，不读 F、TX、V 或目标。既有源 V 两景面板多轮曝光；逐景、同包门槛和对照见 `experiment.json`。LoRa 可用同口径对照为 LoRa 自训 R69 结构适配，原 WiSig 固定 DURA v1 checkpoint 身份单列。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
