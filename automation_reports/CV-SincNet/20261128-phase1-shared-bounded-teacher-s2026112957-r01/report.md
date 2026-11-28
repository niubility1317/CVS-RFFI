# LoRa 与 WiSig 共用有界教师逐频恢复筛查

- run_id：`20261128-phase1-shared-bounded-teacher-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；两侧共同有界网络正在源域教师指导下训练。

## 目的与对照

共同逐频实增益网络融合实增益、R17 和信道残差，修正相对锚点逐包 RMS 不超过 1%；两侧各选来源合规、已曝光源 V 表现更好的专家作起始权重。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `3c6eeae5250ac76de6fc09e1a553d25b67596caa`；控制器 PID 3836395，LoRa worker PID 3836469 / GPU 3，WiSig worker PID 3836470 / GPU 6。N607 预检、归档 SHA、远端编译、PID/CWD/GPU、真实专家及教师首步和有限非零梯度已核实；独立 P0/P1 审查通过，15 项本地测试通过，其他 owner 进程保留。

## 结果与覆盖

筛查和正式结果尚未完成。两侧共用 SharedBoundedGain、逐频实增益和物理残差、固定 1% 包级修正 RMS 上限、教师 KL/波形/clean 保持损失和推理规则；独立训练权重的固定 skip 不同：LoRa 从实增益起步，WiSig 从 R17 起步。该选择使用已反复曝光的同一源 V 面板上两专家既有成绩，因此本轮筛查具有选择偏倚，不视为盲测；若通过仍须在预登记额外源数据复现。用户已授权源 U 冻结教师指导，LoRa E104、WiSig Tri-SIG-TIME SSL F0 保持冻结；不使用 U TX、V 或目标训练。LoRa 原 W24 scipy 识别器前向保持，反向使用 WideTracker 替代梯度。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
