# LoRa 与 WiSig 共用 5% 锚点分布保持教师恢复筛查

- run_id：`20261128-phase1-shared-anchor5-consistency-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；两侧源域训练进行中。

## 目的与对照

在共同恢复的锚点分布保持训练中设定 5% 逐包修正上限，并进行同包筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 0b72ff2f617f2057af3e9c2b68a4cb59eca9a23c；N607 控制器 3877422，LoRa 3877434/GPU 3，WiSig 3877435/GPU 2。两侧真实权重首步和有限非零梯度通过；PID、CWD、GPU 已核实。预检、归档 SHA、远端编译及一次独立 P0/P1 审查通过。

## 结果与覆盖

共同网络与损失、5% 训练和推理上限；LoRa 实增益及 WiSig R17 锚点和各自权重。两侧基于用户授权的源 U 冻结教师训练，目标包括 clean 和锚点分布 KL；LoRa W24 E104 原 scipy 前向、WideTracker 近似反向，WiSig Tri-SIG-TIME SSL F0。源 V 已多轮曝光并用于设计；2% 锚点保持版尚无结果时预登记本版。预定 LoRa/WiSig 教师样本前向 12000/72000，祖先和反向成本另计；训练不读 U TX、V 或目标。筛查待完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
