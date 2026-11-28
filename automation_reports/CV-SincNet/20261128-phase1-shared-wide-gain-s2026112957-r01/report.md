# LoRa 与 WiSig 共用大容量逐频实增益与 DURA 残差筛查

- run_id：`20261128-phase1-shared-wide-gain-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；两侧源域训练进行中。

## 目的与对照

128 通道扩张卷积共同预测逐频实增益，组合 DURA 物理残差，使用 5% 修正上限及冻结教师和锚点保持损失。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 f3047d963cd70989b9a17bef13ba143db6a6529e；N607 控制器 3934603，LoRa 3934615/GPU 2，WiSig 3934616/GPU 3。两侧真实专家和教师首步、有限非零梯度、PID/CWD/GPU 已核实。预检、归档 SHA、远端编译及独立 P0/P1 审查通过。

## 结果与覆盖

两数据集共用 128 通道四层扩张卷积逐频实增益网络、DURA 物理残差、5% 修正上限与损失，分别训练权重；LoRa 实增益、WiSig R17 锚点由已曝光源 V 选择。训练使用用户授权的各数据集源 U 冻结教师，LoRa W24 E104 原 scipy 前向和替代反向，WiSig Tri-SIG-TIME SSL F0，预定教师样本前向 12000/72000 次；推理无教师。原专家和教师祖先成本另计，不读 U 真实 TX、V 或目标训练；筛查待完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
