# 冻结特征恢复：星地六场景双评价线

状态：RUNNING / VERIFIED。用户最新要求将评价分为恢复 IQ 与 clean 的差距、接入基座后的目标测试表现，并明确 low-urban 达到 70% 以上、其余环境接近 clean。本轮据此完成既有六种 practical 目标评估，已完成的 clean 对照原样保留，不重复训练或预测 clean。

## 固定模型、目标与比较

同预算 waveform_control 和 frozenF_feature 均使用源域已冻结且通过原规则的末轮 checkpoint。原三套 F50（CROSS_RX-s2026092701/02/03）及其默认头全部冻结，分别报告每套结果与均值、样本标准差，不集成投票。恢复器仅一个训练种子，不称为三个恢复器种子。模型、权重、阈值、预算和源规则不因已公开的 clean 目标结果改变。

目标为每种场景同一 168000 条物理样本；六种为 practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。复用原 VALIDATED_ONCE formal 接收 IQ、父 R03 同行输出及物理 ID，不重新生成信道。

用户明确的 low-urban 门槛为三套原 F 读出准确率均值至少 70%。Agent 将“其他环境接近 clean”暂按各自三 F 均值与同一 F 的原 IQ clean 准确率差距不超过 2 个百分点解释，同时保留所有逐 seed 实际差值供用户判断。clean 已独立复现 85.113889%；特征恢复 clean 为 84.915873%，已有 0.198016 个百分点损伤，不隐去此结果，也不预先宣称无损或目标达成。

## 两条评价线

第一条：受损 IQ、父 R03、当前恢复 IQ 分别与同一物理 clean 比较。报告未作增益、相位或时延对齐的 full 256、prefix 232、tail 24 NMSE，以及 log FFT power MSE 的均值、中位数、P90。尾部缺失单列，不通过截短窗口掩盖不可恢复部分。

第二条：冻结原 F 默认头对当前受损 IQ、父 R03、当前恢复 IQ 逐包 argmax；报告真实 TX accuracy、Macro-F1、修复／损坏／净修复、逐 TX/RX/day 混淆矩阵。逐场景比较原受损 IQ、父 R03、匹配波形控制，并报告与同一 F 的 clean 准确率差距。clean 只是评分参考，绝不进入星地恢复或分类推断。

两组每一 row 的六种场景、全部三套 F 决策及恢复 IQ 完整固定后，独立无 CUDA CPU scorer 才读取 clean IQ 与目标 TX/RX/day truth。一个 run 最多一条 CPU scorer，不输入真实 batch 计数、quota 或全局重分配，不做目标适配、选模或参数更新。

## 执行与停止规则

N607 普通账号，GPU0 控制组、GPU1 特征组，每卡一条本 run worker，容量计入其他 owner、总 owner 少于 2 且空闲显存至少 12 GB。GPU2/3 的健康 owner 保持。CPU 每进程 2 线程，预测 batch 256。首步真实两组末轮权重、三套原默认头和 2 条 L_s no-query smoke，PASS 后立即继续。

一个不可覆盖输出 root；数据句柄／模型来源／commit 不符、输出冲突、非有限、预算或预测完整性失败即保留产物并停止，不自动重试，不停止其他健康进程。10 项聚焦测试覆盖六场景及三 F 完整性、提前 truth／clean、状态更新、ID 错位、非法类、IQ 缺失和未对齐全窗／尾部误差；沿用该固定候选已完成的一次独立 P0/P1 审查，不加重复 runner review。

预检确认原 formal 七视图、父预测、已完成 clean 对照和源资格齐全，新 run/release 不存在、GPU0/1 空闲。单次发布归档比较、远端 compile、启动后 PID/CWD/命令/GPU/日志增长必须保存。具体命令、CWD、环境、路径、矩阵、输出见 [experiment.json](experiment.json)。


## 启动核验

方法 commit：`096d6e9ce8028f25a98a1afc97d2a343d9270ca0`。一次归档 SHA 比较、远端 compile/import、真实两组末轮权重＋三套原默认头的 2 条 L_s smoke 通过。独立后态核实控制器及两 worker PID/CWD/完整命令、物理 GPU0/1 与两次日志增长，六种 practical 目标预测实际推进。尚未完成全部目标预测，不宣称性能达标。
