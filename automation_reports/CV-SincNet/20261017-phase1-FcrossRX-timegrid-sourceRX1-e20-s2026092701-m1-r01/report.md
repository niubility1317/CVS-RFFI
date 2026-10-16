# 有界包内时间栅格收缩：完整源域结果

实际状态 **ANALYZED／VERIFIED；FAIL，固定候选关闭**。1个scratch E20、1个modelseed、held source RX1 L1260；0 As训练、0正式target。

本轮假设：轻度改变包内采样时间尺度，是否能提高固定F＋跨RX同TX SupCon的clean未见源RX识别。仅新训TIMEGRID E20，匹配既有D_raw八L前向结果；不继承权重。原共同raw.75／ordinary.25／SupCon完整保留，第四对.25RD(raw)改为.25RD(timegrid(raw))，同IQ两次dropout；无新loss/头/部署参数/U增强。

|场景|D_raw准确率|TIMEGRID准确率|差（百分点）|D_raw宏F1|TIMEGRID宏F1|
|---|---:|---:|---:|---:|---:|
|clean|80.4762%|80.4762%|+0.0000|80.1032%|80.0853%|
|practical_high|38.7302%|39.3651%|+0.6349|28.0528%|28.6548%|
|practical_mid|39.3651%|39.2857%|-0.0794|29.3130%|29.4625%|
|practical_low_suburban|36.1905%|36.3492%|+0.1587|28.3013%|28.3208%|
|practical_high_urban|38.4921%|37.9365%|-0.5556|29.1707%|28.7589%|
|practical_mid_urban|32.4603%|32.4603%|+0.0000|26.8440%|26.8836%|
|practical_low_urban|27.6190%|27.6984%|+0.0794|24.4474%|24.4333%|

事前固定判据：clean≥+1pp、宏F1不降、fitV下降≤.5pp和完整健康/预算/序列/覆盖。独立结果{"clean": false, "macroF1": false, "seenV": true, "complete_health": true}；clean宏F1差-0.0179pp、fitV差+0.0046pp、urban均值差-0.1587pp。

关闭固定timegrid；不改rho/插值/概率/权重/RX/seed/epoch/门槛挽救。全部practical正负项保留；独立clean资格不能代替星地保持。

|模型|clean|high suburban距clean|mid suburban距clean|low suburban距clean|
|---|---:|---:|---:|---:|
|D_raw|80.4762%|41.7460pp|41.1111pp|44.2857pp|
|TIMEGRID|80.4762%|41.1111pp|41.1905pp|44.1270pp|

suburban距clean≤2pp是探索目标，不通过降低clean缩差；源RX1结果不是正式low urban>60%或联合突破。

|low urban分层|n|D_raw|TIMEGRID|差（百分点）|
|---|---:|---:|---:|---:|
|state=LOS|384|41.1458%|41.4062%|+0.2604|
|lock=locked|467|41.5418%|41.7559%|+0.2141|
|state=BLOCKED|714|18.7675%|18.9076%|+0.1401|
|lock=degraded|228|21.9298%|21.4912%|-0.4386|
|lock=unlocked|565|18.4071%|18.5841%|+0.1770|
|state=PARTIAL|162|34.5679%|33.9506%|-0.6173|

分层是固定命名物理观测下的条件表现，不是单损伤因果份额；BLOCKED 与 unlocked 允许重叠。本轮没有一类 urban 退化显示足以支持晋级的收益。

|场景|TIMEGRID 默认头平均间隔（除以 30）|
|---|---:|
|clean|0.110378|
|practical_high|-0.037900|
|practical_mid|-0.036495|
|practical_low_suburban|-0.039810|
|practical_high_urban|-0.038786|
|practical_mid_urban|-0.044944|
|practical_low_urban|-0.048562|

间隔为真实类 logit 减最大非真实类 logit，并统一除以固定 cosine 尺度 30。旧对照缺少逐包 practical 间隔，本轮不声称间隔改善或门控改善，也不补做推理以填补历史缺项。

|TX index|clean差（百分点）|low urban差（百分点）|
|---|---:|---:|
|0|-0.9524|-0.4762|
|1|+0.0000|+0.0000|
|2|-0.9524|+2.3810|
|3|-0.9524|-0.4762|
|4|+0.0000|-0.4762|
|5|+2.8571|-0.4762|

|场景|修复原错误|新增错误|净正确变化|
|---|---:|---:|---:|
|clean|6|6|+0|
|practical_high|14|6|+8|
|practical_mid|10|11|-1|
|practical_low_suburban|7|5|+2|
|practical_high_urban|3|10|-7|
|practical_mid_urban|9|9|+0|
|practical_low_urban|10|9|+1|

固定D_raw clean正确集合n=1014，low保持率27.7120%→27.8107%。新模型逐包间隔统一除原cosine30尺度。旧D_raw逐包practical margin/physics缺项不补造；源分层使用新行保存、同config/命名流/物理ID对应的physics。

全20epoch/4440更新、5994原L/U批次与D_raw逐条相同；完整568320变换记录按每步128原始L物理ID匹配，独立L5040、72个TX×RX×day、240个epoch/session，私有rho逐项复放。变化统计为保存trace核对，未重读IQ或将重复视图计成新样本。实际变换：{"views": 568320, "physical_unique": 5040, "sessions": 240, "coverage_cells": 72, "max_relative_L2": 0.030263095961959117, "power_ratio_min": 0.9595339228160166, "power_ratio_max": 0.9994060151773194, "construct_seconds": 20.993281056173146}

完整解析双方 20 轮 native CSV、JSONL、stdout，以及各 4,440 条步骤、SupCon 和 Sinc 单位记录。两行均无有效路径的非有限值、零梯度更新或跳步；禁用路径的 null/NaN 明确保留为不可用值。SupCon 非零步骤均为 3,330，原 warm 5／ramp 10 与 13L＋7U 不变，U 真值可用数为 0。梯度裁剪前均值为 D_raw 0.948666、TIMEGRID 0.944761，最大值分别为 10.752827 和 10.744086，超过 5 的步骤分别为 32 和 31。每行完整 stdout 均为 800 行、20 个结束标记、0 条关键错误；各 260 处 NaN 提及均来自未启用指标。E20 预算耗尽不等于收敛。

新增E20/4440更新/5823318 NN，dispatcher真实CP noquery兼容smoke另2，总新NN=5823320；L4546560，原U/EMA/V/held费用完整保留，0新增U或teacher视图。历史D_raw E20/4440/5823318 NN单列，父权重继承成本0。新native训练及V 1443.5856s，旧对照1492.9099s；时间栅格构造20.9933s（含同步/trace，不含骨干前向）。新 worker 启动至最终结果墙钟为 1,463.2444 s，至 dispatcher 完成为 1,466.3174 s；进程启动时刻的系统 boot 精度为 1 s。详细分项见 cost.json。离线分析绘图0NN/IQ/CP。

F三路/SIG_TIME_X3/统一Sinc CENTER_UNITS/NoExplicitCFO/真实joint160/默认cosine30结构保持，骨干和默认头原流程联合更新。时间栅格只在原256点内部作凸插值；rho来自预定数学变化界，按RX/day/eq共享且跨epoch刷新，无padding/wrap/crop/新RMS/AGC/噪声/多径/PA/DAC。已知插值逆不是未知TX硬件反演；有界可逆不等于身份保持或泛化有效。插值同时改变相邻样点相关性和微弱频率响应，不能把结果唯一归因为 RX 时钟不变性。完整公式、证明、默认增强审计、参数和科学停止条件保留[发布前预登记](E:/type10-7/automation_reports/CV-SincNet/20261017-phase1-FcrossRX-timegrid-sourceRX1-e20-s2026092701-m1-r01/preregistration.md)。

源合同为原VALIDATED_ONCE split392005，fitRX3/4/6/8、day1/2/3、六TX，L5040/U45360(hiddenTX)/V21600；heldRX1 L1260 FINAL-only。不是fullL6300或内部5580/720。原fold准备1404额外L读取（held1260/unused144）保留并丢弃unused，无新bank。V仅健康，0初始化/resume/teacher/EMA/optimizer继承、0目标训练、选模或评价。

真实leo_practical来自固定release的experiments/adv3b02_xuc/code/leo_practical；25MHz/IQ256/fc2462MHz、residual/post_sync、EQoff即G1、reference30/qualityreference20、原接收质量/相位跟踪/IQ补偿/reflect保持。新旧source_eval/RX1/final20和eval20261016103相同；新增噪声SNR不是总SNR。没有改变正式测试IQ、SNR、信道或困难包。

验证PASS：23协议负例/2060合成NN/2合成更新/0真实IQ或CP；新旧206初态、共同损失及全部共同参数梯度逐位相同，8前向/同IQ RDrop/默认头联合更新/导出160相同。六类oracle/4负例和旧对照完整记录离线核对PASS。唯一独立P0/P1 PASS；一次普通账户预检/归档比较/remotecompile/真实CP noquery smoke2NN在任何源IQ前PASS，随后立即继续；PID/CWD/argv/GPU/log增长和自然终态独立VERIFIED。未改健康旧任务，无额外审批/hash链。

训练代码fee77907549a89722dd20736498308fa640dd5f3；离线reader b34ca8a2c392df3418d7ebf377c874fe408114b6，若有后续离线技术修复另记，不冒充训练代码。最终交付commit在git_delivery.json；命令/environment/CWD/GPU/paths/stop rule见experiment.json。完整产物在E:/type10-7/local_artifacts/cvs_FcrossRX_timegrid20_20261017/final；不提交IQ、权重、原始日志或物理ID。

已实际生成以下5张图的PNG/PDF共10文件。视觉核对状态在analysis_execution.json，不以生成成功代替视觉检查。

- [source_scene_results](E:/type10-7/local_artifacts/cvs_FcrossRX_timegrid20_20261017/final/figures/source_scene_results.png)

- [complete_training_curves](E:/type10-7/local_artifacts/cvs_FcrossRX_timegrid20_20261017/final/figures/complete_training_curves.png)

- [low_urban_strata](E:/type10-7/local_artifacts/cvs_FcrossRX_timegrid20_20261017/final/figures/low_urban_strata.png)

- [fixed_clean_retention_and_errors](E:/type10-7/local_artifacts/cvs_FcrossRX_timegrid20_20261017/final/figures/fixed_clean_retention_and_errors.png)

- [grid_exposure_and_original_U](E:/type10-7/local_artifacts/cvs_FcrossRX_timegrid20_20261017/final/figures/grid_exposure_and_original_U.png)

未完成：Ag的独立跨seed/跨源RX确认、Ag/As四格因子及交互项、冻结联合模型正式clean＋六practical truth-last、low urban>60和suburban距clean≤2pp。当前目标ACTIVE。
