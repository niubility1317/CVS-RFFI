# Tri-SIG-TIME 源域 AdamW 更新方向诊断

## 固定源域诊断方案

不读取目标、V 或 U 的真实标签。十类与六源各取原论文 run 同契约、从零训练的 E20 检查点，seed `2026092701`；每个设置固定取源 L/U 的第 0、1、2 批，在每个 E20 副本上分别执行一次完整 Tri 源前向与梯度保护。普通 L 输入按原 continuation 的第 13 有标签 epoch 增强日程处理；hook 继续接收原始 L 输入。每个批次以相同模型及梯度构造独立 L-only 与受保护 L+U 参数容器，各执行一次真实 CENTER_UNITS、物理 clip5 与 AdamW 首步，记录三参数组的 L/U 梯度及相对 L-only 的实际增量下降方向。

CPU 2 线程、无 GPU、无持久模型更新；共 6 个源 batch 和 12 个可丢弃影子优化器步。六个观测共享模型和数据划分，不是独立种子。优化器动量为新建的首步状态，不代表历史训练轨迹；相对 L-only 的局部一阶负点积也不是实际损失上升或 clean 准确率变化。已有目标曾曝光，禁止利用目标结果改本诊断。

以独立 code commit 发布至 N607 独立 release 和 run root。发布前运行定向协议负例、影子优化器数值及观察器测试；每个真实 E20 副本先执行无 query checkpoint smoke，随后立即进行登记源 batch 诊断。检查原保护 backward 与捕获梯度一致，角色、身份、数值或方向桥接错误即终止该诊断，保留日志，不影响其他任务。

独立 P0/P1 审查发现初版遗漏有标签普通增强；已根据实际训练的 E20 父配置补齐，针对该项的独立定点复核通过。代码没有把目标结果用作训练或候选选择。

- run_id：`20261116-diagnostic-pamf-tri-adamw-e20-s2026092701-r01`
- group_id：`phase1-pamf-repl-weight-clean`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

十类和六源各三批次，比较同一 E20 的 L-only 与原保护 L+U 实际首步。

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
