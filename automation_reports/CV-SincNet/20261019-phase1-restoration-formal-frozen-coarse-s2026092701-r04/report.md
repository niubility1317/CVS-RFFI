# 复时频粗恢复器的完整正式观测结果

状态：ANALYZED / VERIFIED。原 168000 条物理样本的 clean＋六 practical 预测全部固定后，独立 CPU scorer 完成评分，原进程自然退出。恢复器、原 cleanF/AsF 和默认分类头全部冻结，训练/适配/头拟合均为 0。R05 确定性修正仍按其原冻结计划独立运行，不因本行结果改型或调参。

实际方法代码 `a173cc5002262f243abeb5a904433fcac42ee1cf`。完整矩阵和输入见 [experiment.json](experiment.json)，机器可读分数见 [summary.json](results/summary.json)。本行模型固定由 R02 源波形 A、源身份 B 选中且通过源资格；没有根据目标分数选择网络。

|视图|原 AsF（%）|接粗恢复（%）|变化（百分点）|修复/损坏/净修复|原 cleanF（%）|接粗恢复（%）|
|---|---:|---:|---:|---:|---:|---:|
|clean|85.0720|84.5315|-0.5405|691/1599/-908|81.6185|81.0863|
|practical_high|81.0405|79.2185|-1.8220|8783/11844/-3061|54.2125|74.7363|
|practical_mid|78.3411|78.0387|-0.3024|12151/12659/-508|51.5804|73.5256|
|practical_low_suburban|75.6173|76.7060|+1.0887|15242/13413/+1829|49.1673|72.0583|
|practical_high_urban|77.4101|77.9167|+0.5065|13125/12274/+851|51.5411|73.3667|
|practical_mid_urban|62.6149|70.9435|+8.3286|28935/14943/+13992|40.5881|66.5321|
|practical_low_urban|51.1780|63.6137|+12.4357|37803/16911/+20892|32.7887|59.3935|

低仰角城区 AsF 的 7 个 RX 聚合均改善，但 TX 14-7 为 38.9286%→31.3357%，TX 8-20 为 80.9750%→79.3107%。clean 损失为 AsF -0.5405、cleanF -0.5321 个百分点；AsF practical_high、mid 同样下降。此结果确认该固定面板上存在平均身份读出收益，不能声称所有 TX 细节忠实恢复或全场景无损。

恢复训练以原 6300 L_s 与 56700 U_s clean IQ 合成配对，TX 标签读取为 0；属于无 TX 标签的合成自监督，仍需源 clean 样本。源波形 A 不读 F/TX；单独的源身份 B 使用 source V 标签辅助验收。R 历史成本 E20/4940 更新；原 cleanF E200/10000 和 AsF scratch E20＋自身 E10/6660 更新另计。统一恢复 IQ 送原 PAMF，FFT/PA/时频等分支由原实现派生。

输入复用原完整正式 IQ 字节、原 opaque IDs，VALIDATED_ONCE、source_target_disjoint。receiver_seed2027/原namespace/fc2450MHz 与1200诊断面板会话不同，但仍为同一原生工程信道引擎，不称独立物理族或在轨验证。七视图相关，不能当成多个模型 seed。单 R seed 与两个固定 F 不足以完成最终身份证明。

实际源2样本复合 checkpoint/no-query smoke PASS；单归档传输比较、远端 compile/import、独立 PID/CWD/argv/GPU4/日志增长已核实。全七视图 prediction count=168000、TX_truth_reads=0、state_updates=0，之后独立 scorer 才接 truth；完整 stdout 无 traceback/OOM/运行错误。数据、权重、原日志和机器状态不提交。

全部原坐标 IQ/FFT/频带/PA 等波形指标、scalar-aligned 单独指标、两 F 的 RX/TX 分层与混淆矩阵保留在 summary。辅助 paired-clean feature margin 仅使用固定512参考记录，全168000 query参与；不是完整最近其他TX检索胜率，不与1200面板完整池百分比直接比较，不输入模型或选模。
