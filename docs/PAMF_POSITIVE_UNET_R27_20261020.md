# 后续正收益 U-Net 的正式补测 R27

用户最新限定：只跑有正收益的 U-Net，指 R23 这类后续探索中此前只给出源域的恢复器。R26 全量补测已撤下；其控制器和两条生成式 worker 经逐 PID/CWD/argv 核对后停止，独立确认进程退出且 CUDA 无对应 PID，13 个完整正式结果与两个中途产物全部保留。未改写旧源筛选或旧输出。

R27 只测试后续以 U-Net 为恢复先验的管线。以同一源域 1800 个物理样本、原 F50/defaulthead 的六个受损场景平均准确率，相对预声明的对应控制组严格正增益作为准入；clean 准确率最多下降0.1个百分点。逐场景有涨有跌如实列出，源收益是单 seed 点估计，不冒充显著性或全场景改善，也不要求重写历史完整 source qualification 为通过。源对照与全部数值固定于 design.py；运行时读回原源汇总与相同物理 ID 面板，不读取目标成绩作筛选。

共12行：8行补测、4行复用已有完整正式结果。新增按用户点名优先运行 R23 prior_only、joint_coarse，然后 R18 terminal_dc、current_anchor，R19 time_frequency，R17 unroll_shared，R15 unet_control，R12 pair_control。复用 R10 native_map_prior、R16 unet_control、R17 unroll_unshared、R19 time_time。R23 两行相对各自对照的源六场景均值增益约0.0370、0.0648个百分点，属于微小点估计，low urban、mid urban 的涨跌必须同时报告。TCN、attention、扩散、流，以及相对声明控制没有源均值正增益的后续 U-Net 行不启动新测试。已有负结果保留为历史，不删除。

没有新增训练、head 拟合或状态更新。每个恢复器沿用原严格 loader 与最终 E20/4940 checkpoint、合法源 L6300/U56700 和完整继承；训练0TX/F/target。先读取原源证据冻结子集，然后真实 checkpoint＋2个 L_s 无 query smoke，通过即进入预测。恢复器输出单一 IQ，原 F50 seed2026092701 与原联合训练默认头一致派生 IQ/FFT/PA/统计等分支。

测试原 R05 固定168000条 received IQ 与 clean＋六 practical，R09同F50原输入作配对基线。每行七场景完整有限预测与物理ID固定后，独立CPU scorer首次接truth；逐行自动评分，报告全部郊区／urban、Macro-F1、修复／损伤／净修复、TX/RX/day、波形/FFT、配对clean特征及成本。low urban≥70%，其他五受损场景各≥79.67619047619047%；restored clean损伤另报。原clean是参考而非理论上限。原F50本身有源域监督，整条链不称完全无监督。

N607普通账户，新不可覆盖root `20261020-phase1-restoration-positive-unet-F50-s2026092701-m12-r27`，GPU pool0/1/5/7，最多两个本任务worker、每卡一个、所有owner上限两个且空闲≥12GB，共享capacity4锁持有到CUDA PID可见。精确代码commit、命令、环境/CWD、路径、GPU、期望产物见登记。一归档SHA对照、远端compile和一次启动PID/CWD/argv/GPU/log增长检查。技术失败保留健康worker与输出、停止派发pending、不自动重试；只停止明确属于本run的进程。目标为已暴露仿真工程基准，不宣称新盲测／在轨有效性，不将目标分数反馈训练、结构、源子集、seed或选择性重跑。
