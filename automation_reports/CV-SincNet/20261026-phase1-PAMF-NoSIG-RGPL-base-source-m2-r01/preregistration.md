# NoSIG 完整监督／普通 PL 源基线

PLANNED。按用户“后续去SIG_TIME”及已固定[RGPL设计](../../../docs/goals/PAMF_NOSIG_RGPL_DESIGN_20261026.md)，先建立新结构匹配控制和冻结E30图教师。只有学生seed2026092701，两条控制，不是新方法／三种子目标结果。

NoSIG-PL复用m201原固定从零`F0-A1B1-NO_SIGNATURE-F21-s2026092701/segment20`，新10轮至E30/6660供后续图教师，再20轮至E50/11100。NoSIG-Sup独立从零20/10/20至同预算E50/11100，不继承任何U训练权重，U loader实际关闭。两条原完整T1 L监督、默认160维六类头联合训练保持，只有整个sequence_pool换NoSequence，原时频/PA/reference保留；不是NO_TIME／L-only CE结构控制。

原F0 L6300/U56700/V27000与VALIDATED_ONCE角色/物理顺序不变。NoSIG-PL的raw selfPL .95/weight1自E14启用；nativeU/EMA/entropy/REPL/style/额外半监督loss关闭。原L768、ordinary/raw RDrop、As两次真实CE且KL0、LS.01、浅中层ORD跨RX.1按原日程。监督行不读U IQ/标签；共同smoke中PL行会读隐藏U，不把整个进程说成U读取0。默认头不另训。

已读回实际NoSIG E20权重：sequence_pool state entries0，原头6×160，scratch/ancestors[]/原epoch20和4440更新、实际run/row/nativeUoff/EMAoff/teacher空/baseline空相符。源contract与原F0一致。该只读张量核查无模型前向；launcher仍先做真实NoSIG E20/defaulthead/G L2 smoke，再分别PL L128/U256和Sup L128/U-off联合backward，原L768/AsKL0/有限主头梯度/无signature参数/零optimizer和query。PASS立即两条训练。

每row固定final后自动V27000八视图raw/Graw/lowSubH/G/midUrbanH/G/lowUrbanH/G，逐TX/RX/day/held/trained；F/G eval参数及全部registered buffers不变。G仍同一原F0 G-LU，reuse=True/graph=True/native=False、FP32/TF32off，不更新；Sup+G-LU不称全系统L-only。无目标句柄/预测/真值访问，不以source或target选epoch，不自动启动图候选。Graph质量规则属于下一独立run；本run两个基线全部完成，不按低分取消。

N607普通账户、既有Python/Torch2.1，每worker CPU2线程，无安装。CWD PROJECT/releases/20261026-phase1-PAMF-NoSIG-RGPL-base-source-release-r01；命令PYTHON -u -m experiments.pamf_channel_pl.nosig_rgpl_base dispatch --config该release/runtime.json。不可覆盖root PROJECT/runs/20261026-phase1-PAMF-NoSIG-RGPL-base-source-m2-r01。实际代码commit记发布runtime/submit和产物；一个archive一次传输比较、remote compile、真实smoke和独立PID/CWD/argv/GPU/step增长。

GPU0..7全局CUDA+pending<2、free≥10000MiB，最多两条本run train_eval，保留其他owner（包括NoSIG纯CE E200，配方/预算与本run不同，不重复接管）。首技术/权限/NoSIGidentity/非有限/来源/预算/smoke故障停新派发，健康已启动任务完成；保留全部产物，不自动重提／覆盖。

3项聚焦检查/local compile、唯一独立P0/P1 PASS。新增F训练80轮/17760更新，另共享历史NoSIG E20；两模型累计各50，唯一累计100F轮。无新G训练、图训练或头拟合。代码/配置/公共汇总入Git，IQ/权重/runtime/逐包输出/日志私有。原三种子rawclean/raw六景/同G六景联合目标未达成，不因架构切换降低。
