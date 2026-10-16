# 同 TX 真实 IQ 有界相干插值：完整源域结果

源域首轮门槛 PASS；只支持独立源域确认，尚不能称为跨 seed 有效或联合目标完成。

实际状态 ANALYZED，独立读回 VERIFIED。固定 F＋跨 RX 同 TX SupCon、SIG_TIME_X3、统一 Sinc CENTER_UNITS、NoExplicitCFO、joint160、默认 cosine30，骨干与默认头正常联合训练。两条新模型均从零 E20，13 轮有标签＋7 轮原半监督；U 真值隐藏，无新 As/对齐损失/分类头。

训练 commit：fdd6724a67c1e983b1e9a130a958f994541c2f04；独立读回实现：137b7f73a1bc553a4b32825ef6af40e6aa460555。r01 的 0b692362a 发布在读 CP/IQ 前因 operator.py 遮蔽标准库而失败，0 NN/更新；原发布保留。r02 将同一代码体改名 iqmix.py 并补测直接入口，科学配置未改变。

本轮检验的唯一假设：同 TX、同 day 的真实 IQ 经固定参考窗复标量对齐与轻度混合后，跨 RX 伙伴能提高 clean 跨 RX 身份识别，并超过同 RX 伙伴的收益。插值保留 anchor 能量、相对 L2 变化不超过 10%；数值界不等于身份保持证明，不宣称数字变换替换了真实 RX。

fit RX3/4/6/8：L5040、隐藏 U45360、V21600；held source RX1 原 L1260 只在 FINAL20 读出。split392005，model/global ordinary seed2026092702；伙伴私有 seed20261016961。所有训练角色 ID、共同 L/U 顺序均与复用 raw 对照完全一致。两条新行与 raw 均 8 次 L 前向，总 pair 系数 1.25、共同 raw 绝对权重 .75、ordinary .25；仅第四对输入不同，不稀释共同 base。

以下均为源域 held RX1、L1260、单 seed、冻结学生默认头结果，不是正式目标集成绩。每格为准确率 / 宏 F1（%）。

|视图|复用 raw|同 RX 混合|跨 RX 混合|跨 RX−raw（pp）|跨 RX−同 RX（pp）|
|---|---:|---:|---:|---:|---:|
|clean|81.19 / 81.08|81.27 / 81.02|83.49 / 83.39|+2.30|+2.22|
|high suburban|37.70 / 31.71|38.17 / 32.10|37.62 / 32.09|-0.08|-0.56|
|mid suburban|37.14 / 31.60|37.38 / 31.61|36.90 / 31.99|-0.24|-0.48|
|low suburban|34.92 / 29.87|35.48 / 30.20|35.08 / 30.67|+0.16|-0.40|
|high urban|38.49 / 33.81|38.25 / 33.27|36.90 / 32.35|-1.59|-1.35|
|mid urban|30.16 / 26.47|30.63 / 26.87|29.05 / 25.61|-1.11|-1.59|
|low urban|27.38 / 24.32|26.90 / 23.68|27.14 / 24.36|-0.24|+0.24|

预登记资格：clean 相对 raw≥+1pp、相对同 RX≥+.5pp；clean 宏 F1 非负、fit V 下降≤.5pp，完整预算/数据覆盖/默认头联合更新与共同 base≤.5。单独 Ag 的各 practical 变化完整报告但不阻断独立 clean 资格；不能据此跳过四格联合或跨 seed 验证。

未满足项：无；仅本次源域首轮门槛。

|比较|clean 增益|clean 宏 F1 增益|urban 均值增益|low urban 增益|fit V 增益|
|---|---:|---:|---:|---:|---:|
|CROSS_RX−D_raw|+2.302|+2.309|-0.979|-0.238|-0.051|
|CROSS_RX−SAME_RX|+2.222|+2.366|-0.899|+0.238|-0.019|

clean 净增 29 个正确判决：修复 49 个原错误，同时新错 20 个。收益主要来自 TX1/2/5，TX0/3 下降；day3 clean 略降，未隐藏这些负项。上述增益均为百分点。完整逐 TX/day 修复与新错在 source_strata.csv；没有删除负类或困难样本。

|模型|high suburban 距 clean|mid suburban 距 clean|low suburban 距 clean|
|---|---:|---:|---:|
|D_raw|43.492|44.048|46.270|
|SAME_RX|43.095|43.889|45.794|
|CROSS_RX|45.873|46.587|48.413|

源域探索目标仍是 low urban>60%、每个 suburban 距 clean≤2pp，同时 clean 相对同预算参照有正收益；差距不得靠降低 clean 缩小。上述结果不能替代正式目标和联合验证。

固定 raw clean 正确集合 n=1023；该集合上的 low urban 保持率：D_raw 30.303%、SAME_RX 29.619%、CROSS_RX 29.912%。新行默认头逐包间隔统一除以 30；原 raw 未保存逐包间隔/physics，缺项明确保留，不补造。两条新行实际 physics 完全相同；参照按原 code/config/命名随机流与物理 ID 对应。

|low urban 状态|n|raw|同 RX|跨 RX|跨 RX−raw（pp）|
|---|---:|---:|---:|---:|---:|
|LOS|384|40.10|39.32|40.89|+0.78|
|PARTIAL|162|31.48|32.72|32.72|+1.23|
|BLOCKED|714|19.61|18.91|18.49|-1.12|
|locked|467|40.26|39.83|41.11|+0.86|
|degraded|228|19.74|19.74|20.61|+0.88|
|unlocked|565|19.82|19.12|18.23|-1.59|

low urban 的小幅正项出现在 LOS（+0.78pp）、PARTIAL（+1.23pp）、locked（+0.86pp）、degraded（+0.88pp）；BLOCKED（−1.12pp）和 unlocked（−1.59pp）下降，总体未缓解困难 urban。

分层仅描述既有记录的相关表现，不将 LOS、锁定或质量分箱视为单损伤因果结论。完整状态、接收质量、新增噪声 SNR、残余 CFO、残余核、AGC 与边界统计在私有 mechanism_analysis.json。

|训练行|epoch / 更新|L 前向包数|总 NN 包数|native 训练秒数|共同 base 最终值|
|---|---:|---:|---:|---:|---:|
|D_raw|20 / 4440|4546560|5823318|1432.42|0.33728|
|SAME_RX|20 / 4440|4546560|5823318|1519.79|0.33822|
|CROSS_RX|20 / 4440|4546560|5823318|1455.68|0.33667|

两条新行并行墙钟约 1541.8 秒（Linux 启动时间约 1 秒精度），含各自行末读出；所有本 run 进程自然退出。

新增 40 model-epoch / 8,880 更新，总 NN 11646638（含 dispatcher 合成 smoke 2 包）；raw 既有 20 epoch / 4,440 更新 / 5823318 NN 单列，不是初始化。每条新增 568,320 次 donor 曝光/插值、bank 5,040 次 IQ 读取，可见物理 L ID 仍 5,040；不将伙伴曝光计为独立新样本。分析与绘图 0 NN/更新，r01 技术失败 0 NN/更新。原 teacher 计数继承的 2 包 offset 已按原规则扣除。E20 耗尽不等于收敛。

SAME_RX：完整伙伴单元 72；相对 L2 中位数 0.100000；cap 55.365%；fallback 0；全部 5,040 个合法伙伴均覆盖，无 U/V/held/teacher 调用。
CROSS_RX：完整伙伴单元 216；相对 L2 中位数 0.100000；cap 87.802%；fallback 0；全部 5,040 个合法伙伴均覆盖，无 U/V/held/teacher 调用。

归因限制：SAME/CROSS 共享公式、系数、上限、前向与伙伴曝光，但实际相对 L2 均值分别为 0.07360 和 0.09492，未逐包匹配扰动强度。当前支持固定跨 RX 混合配方的 clean 收益，不能将其唯一归因于跨 RX 内容而非实际变化分布。

实际测试引擎来自冻结 release 的 experiments/adv3b02_xuc/code/leo_practical；25 MHz、IQ256、residual/post_sync、EQ 关闭 G=1、reference_snr_db30、接收质量自适应/相位与 IQ 补偿按原配置。6 个 practical、SNR、完整困难样本与命名随机流未改，新增噪声 SNR 不等于未知总 SNR，EQ 关闭不等于已经均衡。原 synthetic_reflect、滤波数值延迟与 AGC 记录原样保留。

验证：本地 20 个算子/模型负例、3,090 合成 NN/3 更新；16 个权限/阈值负例；原始公共损失及所有参数梯度相同、默认头联合更新；直接脚本入口 RED→GREEN；一次限定 P0/P1＋启动修复定向读回 PASS；一次 N607 预检、归档比较/compile、真实 CP no-query smoke PASS；独立 PID/CWD/argv/GPU/日志增长与自然终态验证。完整逐包预测、20 轮与 4,440 步均已解析，未以抽样代替。

复现命令见 preregistration_report.md 和 experiment.json；实现位于 experiments/cvs_f_coherentmix20。当前原始日志、预测、pair traces、图与统计放在 E:\type10-7\local_artifacts\cvs_FcrossRX_coherentmix20_20261016\final，不提交 IQ/权重/原始物理元组。

图：[source_accuracy](E:/type10-7/local_artifacts/cvs_FcrossRX_coherentmix20_20261016/final/figures_v2/source_accuracy.png)、[training_curves](E:/type10-7/local_artifacts/cvs_FcrossRX_coherentmix20_20261016/final/figures_v2/training_curves.png)、[urban_strata](E:/type10-7/local_artifacts/cvs_FcrossRX_coherentmix20_20261016/final/figures_v2/urban_strata.png)、[fixed_clean_retention](E:/type10-7/local_artifacts/cvs_FcrossRX_coherentmix20_20261016/final/figures_v2/fixed_clean_retention.png)、[error_redistribution](E:/type10-7/local_artifacts/cvs_FcrossRX_coherentmix20_20261016/final/figures_v2/error_redistribution.png)、[IQ_mixing_and_coverage](E:/type10-7/local_artifacts/cvs_FcrossRX_coherentmix20_20261016/final/figures_v2/IQ_mixing_and_coverage.png)。

未完成：独立源域/跨 seed 复现、合格 Ag 的四格 B00/B10/B01/B11 与交互、正式目标 clean＋六种 practical 和最终联合验证。当前目标保持 active，未宣称 low urban>60%、clean/星地双突破或跨 seed 有效。

下一步唯一假设：固定配方的 clean 独立收益能跨模型/ordinary seed 复现。最小确认矩阵为 seed2026092701 的同源 RX1、同角色 raw（既有只读复用）／SAME_RX／CROSS_RX，两条新 scratch E20，原算子、系数、10%界、参考窗、U 规则、默认头和七视图均不变。仍要求 CROSS 相对 raw clean≥+1pp、相对 SAME≥+.5pp，宏 F1非负、fit V 下降≤.5pp和原健康条件；失败关闭当前固定配方，不用 seed03、RX 切换、幅度或预算救援。此处是后续固定设计，尚未登记或启动，不计为已完成的跨 seed 证据。
