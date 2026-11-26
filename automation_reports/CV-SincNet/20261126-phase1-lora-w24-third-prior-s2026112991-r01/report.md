# LoRa W24 两步与三步独立先验匹配对照

- run_id：`20261126-phase1-lora-w24-third-prior-s2026112991-r01`
- group_id：`phase1-lora-w24-restoration-performance`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

从同一源码与随机状态从零训练。第三步复用同包估计作用和方差，增加一个独立先验及 2 次 CG。

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


## 预登记假设与读出

两步对照复用原 W24 DURA 的两级 prior 与每级两次 warm-start CG。三步臂额外复制第二级先验的初始参数，作为独立可训练的第三级。前两组参数初始值、同条源 IQ 扰动流、源角色、损失和 1,800 步预算保持匹配。训练中复用同包作用与噪声方差估计，并以第三级输出替代最终输出计算原定损失。此对照检验额外展开级及对应容量的联合效应。

两组从零训练，仅用合法源 U_s 的 20,000 条拟合包及 500 条独立留出包。无校准 IQ。分类器只在两组权重固定后加载。冻结后在两条路线的同一批 300 个源验证包评估直通、长训半量、两步及三步的半量和全量。两条路线的三步半量均须满足六景超过三个对照、低城区至少持平三个对照、clean 至少持平直通。通过时自动执行既定 Full 和 Residual 的 30,000 包、七视图冻结预测及独立评分。

## 发布与运行核验

2026-11-26 HKT，代码 `b3f9825c5ca73a752a911bb4e718a13a7ff952da` 已与远端 Git 分支读回一致。本地 11 项聚焦检查通过。独立 P0/P1 审查发现评分状态读取可因短时 SSH 超时而退出，已修复为在原截止时间内重查同一运行。登记的 release SHA 对比及远端编译通过。

N607 控制器 的 argv 和 release CWD 已核实。两步在物理 GPU 5、三步在物理 GPU 6；训练 worker 和对应 fit 输入、原训练归档 CWD、GPU 实际占用、日志增长均已读回。两组真实源 IQ 一步检查通过，分别确认 2 和 3 个 prior 有有限非零梯度，均为 0 次 TX／W24／目标读取，0 条校准 IQ。独立本机评分进程等待同一运行状态，未打开测试标签。正式评分仍待冻结与源资格。
