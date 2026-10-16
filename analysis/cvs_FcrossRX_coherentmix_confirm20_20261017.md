# 相干 IQ 插值：seed01 独立确认与两 seed 完整源域结果

seed01 确认门槛 FAIL；关闭本固定相干插值配方，不以改幅度、参考窗、RX、seed 或预算救援。

实际状态 ANALYZED，独立读回 VERIFIED。固定 F＋跨 RX 同 TX SupCon、SIG_TIME_X3、统一 Sinc CENTER_UNITS、NoExplicitCFO、joint160、默认 cosine30，骨干与默认头正常联合训练。两条新模型均从零 E20，13 轮有标签＋7 轮原半监督；U 真值隐藏，无新 As/对齐损失/分类头。

训练 commit：2609d1ec4f62ccace9694db9ce7bf5fe53739964；独立读回实现：379f4bc44da3dbb0e9abfb4f6615d828c4b8b2d6。历史 seed02 的 r01（0b692362a）在读 CP/IQ 前因 operator.py 遮蔽标准库失败，原产物保留、0 NN/更新；该历史故障与本轮 seed01 run 身份区分。当前复用已修复的 iqmix.py，科学算子未改变。

本轮检验的唯一假设：同 TX、同 day 的真实 IQ 经固定参考窗复标量对齐与轻度混合后，跨 RX 伙伴能提高 clean 跨 RX 身份识别，并超过同 RX 伙伴的收益。插值保留 anchor 能量、相对 L2 变化不超过 10%；数值界不等于身份保持证明，不宣称数字变换替换了真实 RX。

fit RX3/4/6/8：L5040、隐藏 U45360、V21600；held source RX1 原 L1260 只在 FINAL20 读出。split392005，model/global ordinary seed2026092701；伙伴私有 seed20261016961。所有训练角色 ID、共同 L/U 顺序均与复用 raw 对照完全一致。两条新行与 raw 均 8 次 L 前向，总 pair 系数 1.25、共同 raw 绝对权重 .75、ordinary .25；仅第四对输入不同，不稀释共同 base。

以下均为源域 held RX1、L1260、单 seed、冻结学生默认头结果，不是正式目标集成绩。每格为准确率 / 宏 F1（%）。

|视图|复用 raw|同 RX 混合|跨 RX 混合|跨 RX−raw（pp）|跨 RX−同 RX（pp）|
|---|---:|---:|---:|---:|---:|
|clean|80.48 / 80.10|81.35 / 80.96|79.76 / 79.32|-0.71|-1.59|
|high suburban|38.73 / 28.05|39.76 / 29.28|40.08 / 29.00|+1.35|+0.32|
|mid suburban|39.37 / 29.31|41.19 / 30.57|40.63 / 30.50|+1.27|-0.56|
|low suburban|36.19 / 28.30|36.98 / 28.99|36.83 / 28.72|+0.63|-0.16|
|high urban|38.49 / 29.17|39.76 / 30.03|39.44 / 29.73|+0.95|-0.32|
|mid urban|32.46 / 26.84|32.14 / 26.53|32.62 / 27.10|+0.16|+0.48|
|low urban|27.62 / 24.45|27.78 / 24.30|27.54 / 23.98|-0.08|-0.24|

预登记资格：clean 相对 raw≥+1pp、相对同 RX≥+.5pp；clean 宏 F1 非负、fit V 下降≤.5pp，完整预算/数据覆盖/默认头联合更新与共同 base≤.5。单独 Ag 的各 practical 变化完整报告但不阻断独立 clean 资格；不能据此跳过四格联合或跨 seed 验证。

未满足项：reused_D_raw_clean、reused_D_raw_clean_macroF1、SAME_RX_clean、SAME_RX_clean_macroF1。

|比较|clean 增益|clean 宏 F1 增益|urban 均值增益|low urban 增益|fit V 增益|
|---|---:|---:|---:|---:|---:|
|CROSS_RX−D_raw|-0.714|-0.788|+0.344|-0.079|+0.005|
|CROSS_RX−SAME_RX|-1.587|-1.641|-0.026|-0.238|+0.005|

clean 净减 9 个正确判决：修复 12 个原错误，同时新错 21 个。TX1/5 改善，TX0/2/3 下降、TX4 持平；day1 持平，day2/3 下降。上述增益均为百分点。完整逐 TX/day 修复与新错在 source_strata.csv；没有删除负类或困难样本。

|模型|high suburban 距 clean|mid suburban 距 clean|low suburban 距 clean|
|---|---:|---:|---:|
|D_raw|41.746|41.111|44.286|
|SAME_RX|41.587|40.159|44.365|
|CROSS_RX|39.683|39.127|42.937|

源域探索目标仍是 low urban>60%、每个 suburban 距 clean≤2pp，同时 clean 相对同预算参照有正收益；差距不得靠降低 clean 缩小。上述结果不能替代正式目标和联合验证。

固定 raw clean 正确集合 n=1014；该集合上的 low urban 保持率：D_raw 27.712%、SAME_RX 27.613%、CROSS_RX 27.318%。新行默认头逐包间隔统一除以 30；原 raw 未保存逐包间隔/physics，缺项明确保留，不补造。两条新行实际 physics 完全相同；参照按原 code/config/命名随机流与物理 ID 对应。

|low urban 状态|n|raw|同 RX|跨 RX|跨 RX−raw（pp）|
|---|---:|---:|---:|---:|---:|
|LOS|384|41.15|41.41|43.23|+2.08|
|PARTIAL|162|34.57|33.33|33.33|-1.23|
|BLOCKED|714|18.77|19.19|17.79|-0.98|
|locked|467|41.54|41.54|42.83|+1.28|
|degraded|228|21.93|20.61|21.05|-0.88|
|unlocked|565|18.41|19.29|17.52|-0.88|

分层仅描述既有记录的相关表现，不将 LOS、锁定或质量分箱视为单损伤因果结论。完整状态、接收质量、新增噪声 SNR、残余 CFO、残余核、AGC 与边界统计在私有 mechanism_analysis.json。

|训练行|epoch / 更新|L 前向包数|总 NN 包数|native 训练秒数|共同 base 最终值|
|---|---:|---:|---:|---:|---:|
|D_raw|20 / 4440|4546560|5823318|1492.91|0.31082|
|SAME_RX|20 / 4440|4546560|5823318|1511.41|0.31152|
|CROSS_RX|20 / 4440|4546560|5823318|1551.36|0.31126|

新增 40 model-epoch / 8,880 更新，总 NN 11646638（含 dispatcher 合成 smoke 2 包）；raw 既有 20 epoch / 4,440 更新 / 5823318 NN 单列，不是初始化。每条新增 568,320 次 donor 曝光/插值、bank 5,040 次 IQ 读取，可见物理 L ID 仍 5,040；不将伙伴曝光计为独立新样本。分析与绘图 0 NN/更新；历史 seed02 r01 技术失败 0 NN/更新，本轮无技术失败重跑。原 teacher 计数继承的 2 包 offset 已按原规则扣除。E20 耗尽不等于收敛。

SAME_RX：完整伙伴单元 72；相对 L2 中位数 0.100000；cap 55.462%；fallback 0；全部 5,040 个合法伙伴均覆盖，无 U/V/held/teacher 调用。
CROSS_RX：完整伙伴单元 216；相对 L2 中位数 0.100000；cap 87.837%；fallback 0；全部 5,040 个合法伙伴均覆盖，无 U/V/held/teacher 调用。

实际测试引擎来自冻结 release 的 experiments/adv3b02_xuc/code/leo_practical；25 MHz、IQ256、residual/post_sync、EQ 关闭 G=1、reference_snr_db30、接收质量自适应/相位与 IQ 补偿按原配置。6 个 practical、SNR、完整困难样本与命名随机流未改，新增噪声 SNR 不等于未知总 SNR，EQ 关闭不等于已经均衡。原 synthetic_reflect、滤波数值延迟与 AGC 记录原样保留。

验证：沿用未改算子/模型的历史 20 个负例、3,090 合成 NN/3 更新、16 个权限/阈值负例及公共损失/所有参数梯度一致性证据，不重复计算。本轮新增 25 个 profile/入口/实际 seed 负例 PASS；两个新模型与同 seed raw 后端初始状态逐项相同，本地 0 IQ/CP/NN/更新；一次限定 P0/P1 PASS；一次 N607 预检、归档比较/compile、真实 CP no-query smoke PASS；独立 PID/CWD/argv/GPU/日志增长与自然终态验证。完整逐包预测、20 轮与 4,440 步均已解析，未以抽样代替。

复现命令见 preregistration_report.md 和 experiment.json；实现位于 experiments/cvs_f_coherentmix20。当前原始日志、预测、pair traces、图与统计放在 E:\type10-7\local_artifacts\cvs_FcrossRX_coherentmix_confirm_20261017\final，不提交 IQ/权重/原始物理元组。

图：[source_accuracy](E:/type10-7/local_artifacts/cvs_FcrossRX_coherentmix_confirm_20261017/final/figures/source_accuracy.png)、[training_curves](E:/type10-7/local_artifacts/cvs_FcrossRX_coherentmix_confirm_20261017/final/figures/training_curves.png)、[urban_strata](E:/type10-7/local_artifacts/cvs_FcrossRX_coherentmix_confirm_20261017/final/figures/urban_strata.png)、[fixed_clean_retention](E:/type10-7/local_artifacts/cvs_FcrossRX_coherentmix_confirm_20261017/final/figures/fixed_clean_retention.png)、[error_redistribution](E:/type10-7/local_artifacts/cvs_FcrossRX_coherentmix_confirm_20261017/final/figures/error_redistribution.png)、[IQ_mixing_and_coverage](E:/type10-7/local_artifacts/cvs_FcrossRX_coherentmix_confirm_20261017/final/figures/IQ_mixing_and_coverage.png)。

未完成：其他 held 源 RX 的证据、合格 Ag 的四格 B00/B10/B01/B11 与交互、正式目标 clean＋六种 practical 和最终联合验证。当前目标保持 active，未宣称 low urban>60%、clean/星地双突破或跨 seed 有效。

两 seed 的完整确认

固定配方未能在两个预登记 seed 同时通过原门槛。本轮未通过固定跨 seed 资格，关闭固定配方，不用 seed03、RX 切换、强度、参考窗或预算救援。

|seed|raw clean|SAME clean|CROSS clean|CROSS−raw clean pp|CROSS−SAME clean pp|CROSS−raw urban pp|CROSS−raw low pp|原资格|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|2026092701|80.476|81.349|79.762|-0.714|-1.587|+0.344|-0.079|FAIL|
|2026092702|81.190|81.270|83.492|+2.302|+2.222|-0.979|-0.238|PASS|

两 seed 配对变化的均值 ± 样本标准差（pp，n=2；不作为新晋级门槛）：

|视图|CROSS−raw 准确率|CROSS−SAME 准确率|
|---|---:|---:|
|clean|+0.794 ± 2.133|+0.317 ± 2.694|
|practical_high|+0.635 ± 1.010|-0.119 ± 0.617|
|practical_mid|+0.516 ± 1.066|-0.516 ± 0.056|
|practical_low_suburban|+0.397 ± 0.337|-0.278 ± 0.168|
|practical_high_urban|-0.317 ± 1.796|-0.833 ± 0.730|
|practical_mid_urban|-0.476 ± 0.898|-0.556 ± 1.459|
|practical_low_urban|-0.159 ± 0.112|+0.000 ± 0.337|

实际 source 角色、held 物理 ID/标签以及六场景逐包 physics 在两个 seed 间相同；native args 差异为 ['candidate_id', 'output_dir', 'run_id', 'seed']。模型/global ordinary seed 不同，不能归因于单纯初始化。相同1260个物理样本的重复预测不作为2520独立样本，不给伪重复置信区间。

两轮合计新训练 80 model-E / 17760 更新 / 23293276 NN；两个 raw 参照历史 40 model-E / 8880 更新 / 11646636 NN 单列，无权重继承。本轮并行墙钟约 1577.8 秒（约1秒精度），全部本run进程自然退出。分析/绘图0NN/更新。

[两 seed 图](E:/type10-7/local_artifacts/cvs_FcrossRX_coherentmix_confirm_20261017/final/figures/crossseed_comparison.png)；完整42行逐seed×arm×view在 E:\type10-7\local_artifacts\cvs_FcrossRX_coherentmix_confirm_20261017\final\crossseed_scores.csv。

下一步以本次实际 PASS/FAIL 结论推进，不将 clean 收益与旧 As 增益相加预测联合成绩。当前仍无 B00/B10/B01/B11 交互或正式目标结果，low>60%与suburban接近clean未得到证明，目标保持active。
