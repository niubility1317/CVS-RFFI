# F＋跨 RX 配对星地：五源折确认结果

源确认 **FAIL**。八个新 scratch E20 加已冻结 RX1 参考，一个模型 seed、五个源 RX 留出折；0 target。

|held RX|C0 clean|C1 clean|clean Δ/pp|C0 low|C1 low|low Δ/pp|urban Δ/pp|源规则|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|1|83.4444%|86.1667%|+2.7222|25.6852%|38.4074%|+12.7222|+17.7469|PASS|
|3|54.6296%|57.7222%|+3.0926|26.0741%|34.2037%|+8.1296|+13.4074|PASS|
|4|83.2963%|87.6667%|+4.3704|23.8333%|46.0741%|+22.2407|+32.2099|PASS|
|6|81.1296%|77.4815%|-3.6481|27.5926%|47.7963%|+20.2037|+26.6420|FAIL|
|8|61.1296%|59.0000%|-2.1296|29.5926%|36.7037%|+7.1111|+10.1914|FAIL|

固定失败源折：[6, 8]。五折平均 clean Δ=+0.881481 pp、low Δ=+14.081481 pp、urban Δ=+20.039506 pp；最差 clean 损失=3.648148 pp。所有折必须满足原规则，平均阳性不覆盖失败折。

|场景|C0 源RX均值|C1 源RX均值|Δ/pp|C0 最差RX/准确率|C1 最差RX/准确率|
|---|---:|---:|---:|---|---|
|clean|72.7259%|73.6074%|+0.8815|RX3/54.6296%|RX3/57.7222%|
|practical_high|36.6185%|64.6741%|+28.0556|RX3/30.0370%|RX3/49.2037%|
|practical_mid|35.1111%|62.1593%|+27.0481|RX4/29.6852%|RX3/48.1296%|
|practical_low_suburban|33.8852%|59.3222%|+25.4370|RX4/28.7963%|RX3/46.2407%|
|practical_high_urban|35.0963%|61.7000%|+26.6037|RX3/29.2407%|RX3/47.7963%|
|practical_mid_urban|30.1481%|49.5815%|+19.4333|RX4/26.1852%|RX3/40.9259%|
|practical_low_urban|26.5556%|40.6370%|+14.0815|RX4/23.8333%|RX3/34.2037%|

新增160模型轮／35520更新／28632144网络包，复用RX1参考40模型轮／8880更新／7158036网络包单列。父权重成本0，各模型从零E20；本次八行确认矩阵墙钟4309.823827s，不是每对独立墙钟，不能把四个分折报告的该数值相加。

所有新行各完整20轮／4440步／native CSV／stdout／SupCon与F物理clip、同折全5994条L/U顺序、同流源physics和默认头预测独立复算。逐折训练曲线、状态/质量/TX/day分层、修复/新增错误和固定C0 clean正确集合保持率保存在RX3/RX4/RX6/RX8目录。RX1使用已完整独立核实的冻结参考，不重新训练或读其checkpoint。

均值是五个源RX等权平均，F1也是逐源RX宏F1均值，最差RX单列。五折共享同一model seed和同一原源数据池，训练集重叠；不能给出跨modelseed复现或把五折当成五个独立seed。全source27000个V包各在其留出模型上评估，绝不把seen V当held V。

未修改CVS、默认头或正式practical IQ/信道/SNR。没有新Ag、额外对齐/课程/U增强。此处不能证明独立Ag有效，也不等于正式clean+6practical或lowurban>60%、suburban距自身clean≤2pp的联合目标。预算耗尽不等于收敛。任一失败折保留，不换RX/seed或调整配方救援，不用目标反馈选择。


实际状态：**VERIFIED / ANALYZED，科学判定 FAIL_SOURCE_CONFIRMATION**。八 worker 和 dispatcher 全部自然结束，无技术 failure、重启或热改。既有一次限定 P0/P1、真实 own scratch checkpoint no-query smoke、一次 archive 本远对比、remote compile 和 PID/CWD/argv/GPU/log-growth 证据保留。技术成功与科学失败分别登记。

实际模型固定为 F＋跨 RX 同 TX SupCon：三路 SIG_TIME、CENTER_25K、CENTER_UNITS、25 MHz／IQ256／equalized1／native RMS／NoExplicitCFO、原 joint160 与默认 cosine30 联合训练。time_down/t2.pool 上的 L-only SupCon 温度 0.2、最大权重 0.1、warm5/ramp10，只读共同第一对的已有前向；没有新部署参数、分类头或诊断头。默认头、三个 SIG 投影、Sinc 正常更新，4440 步均执行 F 物理 clip／学习率映射。

每折实际 fit 是其余四个原 source RX，L5040／隐藏 U45360／seen V21600；held RX 的 L/U 均不训练，held V5400 仅 FINAL20 使用。两个 arm 各 scratch E20＝13L＋7U、4440 更新，父权重／EMA／optimizer／统计继承成本 0；不是完整源 L6300，也不是内部 5580/720 契约。原 U／own EMA／旧门控与 cosine 日程保持，U TX 真值始终隐藏。模型 seed 仅 2026092701，split seed 392005。

C0＝0.5 RD(共同旧 ordinary IQ)＋0.5 RD(同一 IQ)；C1＝0.5 RD(共同旧 ordinary IQ)＋0.5 RD(practical(raw))。每 RD 都是同一已生成 IQ 两次 dropout，CE label_smoothing=0.01、对称 KL alpha=1；四 L 网络前向匹配。原 ordinary 保留为共同条件，不称其为新 Ag 的已验证收益。As 直接作用原 IQ，避免 ordinary PA/AWGN/多径后再叠完整 practical；没有新增课程、全特征对齐或 U 增强。

实际引擎导入 release 的 experiments/adv3b02_xuc/code/leo_practical，非根 code。全部六场景保持 residual/post_sync、EQ off（G=1）、reference_snr_db=30、quality_adaptation=true／quality_reference_snr_db=20、phase_tracking=true、IQ compensation=true、原 AGC／reflect 边界。原传播／遮挡／噪声／接收动作依赖完整保留，added-noise SNR 不等于总 SNR。输入沿用 native RMS 和原 simulator normalize_input，不额外提高测试 SNR 或移除困难样本。

训练 As 先验均匀 low_suburban／mid_urban／low_urban，每步从原 128 包产生完整 residual，epoch/step/view 刷新；硬件 session 按 RX/day/eq 稳定且不由 TX 定义。每个 C1 恰好 4440 次／568320 包 As，C0 调用 0；TX×fit RX×训练场景和遮挡状态有效覆盖。全部合法有标签 As 包进入 CE，0 学生高置信筛除。训练与验证随机流分离；同折两模型的冻结验证物理 ID／信道元数据逐条相同。

首折源规则原样用于全部五折：urban≥+3 pp、low≥+5 pp、urban 宏 F1 不降、held clean 与 seen V 各损失≤0.5 pp、完整有限更新和实际覆盖健康；五折全部 PASS 才能通过。RX6 与 RX8 clean 分别下降 3.648148、2.129630 pp，均值阳性不能覆盖失败折。本固定配对 As 配方关闭独立晋级；不换 RX/seed、不延长 E20、不扫描强度或改阈值救援，不启动其 full-source／多 seed／target 扩展。

clean 负结果不是少量随机错误：RX6 的 TX index2 下降 29.555556 pp（修复5／新增错误271），TX index1 下降6.777778 pp；RX8 的 TX index4 下降22.777778 pp（修复4／新增错误209），TX index3 下降10.888889 pp。TX index 是原网络 0..5 标签索引，不另猜物理 TX 编号。RX6 day1/day2 分别下降3.833333/7 pp；RX8 day1/day2 分别下降5.222222/1.166667 pp。其他正负 TX/day 及状态／质量分层全部保留，不删除失败类别。

从五折等权均值看，C1 clean73.607407%、high64.674074%、mid62.159259%、low_suburban59.322222%；suburban 与自身 clean 仍相差8.933333、11.448148、14.285185 pp。RX6/RX8 high 差距较小伴随 clean 下降，不能称联合改善。C1 最差 low 源 RX3 为34.203704%，五折 low 均低于60%。完整 per-RX 宏 F1、修复／新增错误、默认头间隔、固定 C0 clean 正确集合保持率及原融合参数轨迹见实际表图；融合参数变化是观测相关性，不是单因素因果证明。

独立读取完整八行各20轮／4440步／native CSV／stdout，检查实际 SupCon 和 F 物理 clip、每同折全部5994条 L/U 批次顺序完全相同，再由保存的默认头预测复算七视图准确率／宏 F1／confusion。新行共有302400个 source 冻结预测，RX1 参考75600个单列；读取不重开 IQ、checkpoint 或网络，不重测旧行。五源训练池重叠，不是五个独立模型 seed，也不是正式 target truth-last 结果。

新增八模型各 E20，共160模型轮／35520更新／28632144网络包；其中L18186240，其余为原 U、own EMA、source eval、held clean／六 practical 和 own smoke，逐行实测分解保存在 measured_costs.csv。复用 RX1 的40模型轮／8880更新／7158036包单列。八行整个矩阵墙钟4309.823826789856 s（约71.83分钟），不能累加四个分折报告里的同一矩阵时间；合计十个独立 E20 也不是一个 E200 父模型。E20 结束时 C1 第二视图 CE 仍约0.55至0.68；预算结束不等于收敛。离线完整分析与绘图 0 模型更新／网络前向，CPU分析总墙钟未统一计时。

独立观测配对实验已在本五折最终结果前冻结并发布，其 SAME_RX 控制和 CROSS_RX 候选按原方案继续，不受这里的得分调整。它只检验真实有标签观测的训练组织，不冒称同一 IQ 的数字 Ag。真正独立 Ag、合格 As、B00/B10/B01/B11 交互、正式七视图以及跨模型 seed 复现仍未完成；最终目标保持 active。

实际训练代码 b3fe1c7b2e04d3abada07e79335508be2af01882；复用 RX1 代码8235c71cc4a1619eade9def1371e1bdfc31b9eae。精确命令、环境/CWD、路径、GPU及停止规则见 experiment.json／preregistration.md；运行根保持不可变。

离线复算命令（0 IQ/CP/network，不重测）：

`F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_f_crossrx_folds20.harvest --output E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final --state E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/terminal_readback.json --reference-summary E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/final/analysis_summary.json --cached`

实际完整表图路径（已逐张检查17图）：

- [paired_source_RX.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/paired_source_RX.csv)
- [all_source_scene_results.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/all_source_scene_results.csv)
- [source_scene_means.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/source_scene_means.csv)
- [source_fold_confirmation.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/source_fold_confirmation.png)
- [RX3/complete_training_curves.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX3/complete_training_curves.png)
- [RX3/source_scene_results.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX3/source_scene_results.png)
- [RX3/low_urban_strata.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX3/low_urban_strata.png)
- [RX3/fusion_and_margins.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX3/fusion_and_margins.png)
- [RX4/complete_training_curves.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX4/complete_training_curves.png)
- [RX4/source_scene_results.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX4/source_scene_results.png)
- [RX4/low_urban_strata.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX4/low_urban_strata.png)
- [RX4/fusion_and_margins.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX4/fusion_and_margins.png)
- [RX6/complete_training_curves.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX6/complete_training_curves.png)
- [RX6/source_scene_results.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX6/source_scene_results.png)
- [RX6/low_urban_strata.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX6/low_urban_strata.png)
- [RX6/fusion_and_margins.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX6/fusion_and_margins.png)
- [RX8/complete_training_curves.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX8/complete_training_curves.png)
- [RX8/source_scene_results.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX8/source_scene_results.png)
- [RX8/low_urban_strata.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX8/low_urban_strata.png)
- [RX8/fusion_and_margins.png](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX8/fusion_and_margins.png)
- [RX3 全部分层、保持率、成本与曲线表](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX3/report.md)
- [RX4 全部分层、保持率、成本与曲线表](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX4/report.md)
- [RX6 全部分层、保持率、成本与曲线表](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX6/report.md)
- [RX8 全部分层、保持率、成本与曲线表](E:/type10-7/local_artifacts/cvs_FcrossRX_folds20_20261016/final/RX8/report.md)
