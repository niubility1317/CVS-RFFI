# 高置信硬伪标签监督扩展：移除 U 信道增强的匹配对照

状态：VERIFIED_QUEUED。160 项相关检查通过；一次独立 P0/P1 审查发现前序已启动未注册 CUDA 的 PID 漏计，已修复并经唯一定向复查关闭。真实 F21 E30 source-only smoke 待首卡位。本方法尚未训练，未读取新版目标结果。

用户要求从普通 PL 的高纯度伪标签出发，提高权重并叠加监督机制。已发布 HPL2 与 HPLT1U，当前健康训练保持。本独立匹配对照只检验 U 信道强视图的影响，来自冻结源域诊断，未使用新 HPL 或 UVB 目标成绩。

## 源域依据

完整源 V 每 seed 27000 条，实际普通 PL F21 E30、当前学生 eval、阈值 0.95 的接受后正确率为 99.8868%/99.8366%/99.9027%，覆盖 94.9037%/95.1815%/95.1296%。源 V 是带标签诊断，U_s 真值继续隐藏；这些数字不能称为全部 U 的实测精度。

新增同记录诊断使用各 seed 的 V720 与 U960 条，比较同一条 IQ 的 raw、普通噪声、原生 Practical 视图；无 GPU worker、模型/教师/原型/归一化更新、U 真值或 target 读取。V 中被 raw 门控接受的 674/681/678 条，原标签正确 674/678/677，信道后教师正确 523/518/524。accepted hard CE 的普通视图均值约 0.0021 至 0.0026，信道视图约 0.56 至 0.57；隐藏标签 U 的对应值约 0.0021 至 0.0025 和 0.59。数值只说明冻结教师面对信道视图更困难，不等同训练中 LS/RDrop 总损失、参数梯度比例，也不证明正确 raw 标签失效或移除该视图会提高目标性能。详见 source_view_reliability.json。

## 唯一改变与固定训练

HPL_NO_UAS 从原 F21 同 seed E30 独立追加 20 轮至 E50/11100，累计 13L＋37LU。HPLT1U 是匹配比较项，不是权重父模型。保留原 L 六前向与信道 As、默认头联合训练；U 的当前学生硬标签、阈值 0.95、权重 2、LS0.01、ordinary/raw RDrop1、As 两次 CE 且 KL0、accepted-only 浅中层跨 RX 对比全部保持。ordinary/mixed/raw 各两次 U 学生前向，目标仍为 2×[0.5 ordinary＋0.25 mixed-ordinary＋0.25 As＋0.25 raw＋0.1 CRIC]/1.25。

唯一变化：交替半批 U 的 mixed As 输入从原生 Practical 输出变为完全相同的 raw IQ，跳过 U 信道生成。原 L 信道生成继续，As CE 分支仍执行两次，训练剂量与六次 U 前向保持。没有 EMA、原型、新分类头或 G 参与训练。父 U 曝光 963900，新 20 轮 1134000，累计 2097900；累计学生 U 视图 7767900，新 raw-As 567000、physical-U 生成及学生视图均 0。记录实际接受数、浅中层锚点、KL、增量与累计成本。

## 预登记矩阵与执行

三个 seed 2026092701/02/03 固定物理 GPU5/6/7，一 seed 一卡。新任务等待同 seed HPL2 与 UVB 全部 14 项 GPU 预测完成、HPLT1U 已派发，再占用释放的一个卡位；每卡最多两条本任务 worker、含其他 owner 总数最多三条、空闲显存至少 10000 MiB。smoke 和正式派发都把前序 RUNNING HPLT1U 的 PID 纳入注册 PID 与当前子进程 PID 并集，防止 CUDA 初始化窗口漏计。不停止、重启或热改健康任务。

执行入口 control/hard_pl_no_uas_worker，完整命令、CWD、环境和路径见 experiment.json。不可覆盖 PROJECT/releases/20261024-phase1-PAMF-HardPL-noUAs-m3-r01 和 PROJECT/runs/20261024-phase1-PAMF-HardPL-noUAs-m3-r01；本地 Windows Git 承载，N607 普通账户固定 CVS-RFFI 环境，FP32/TF32off。一次代码 commit、一次发布 archive 的本地远端 SHA 比较、一次远端 compile。首卡 source-only smoke 使用真实原 F21 E30，验证 U As 输入等于 raw、U Practical0、原 L Practical64、U1536 学生视图、浅中层与默认头真实有限梯度、As KL0；PASS 即正式重载对应父权重继续。两个相同技术失败停止新派发，健康进程和产物保留；不按性能停止、调参、重排或选择性重跑。

每 row 固定最终 E50 checkpoint 后自动 raw PAMF 与同一冻结 DURA v1.0 各 clean＋六 practical，每景 168000 条，共 42 个预测 cell。每 row 全部预测固定后独立 truth-last；单 CPU scorer 接在整个 HardPL6 终态之后。完整矩阵后报告每景/seed/TX/RX/day、净纠错、F20/F21 配对差与成本；两组完整矩阵均评分后再报告 HPLT1U 单因素配对差。DURA 仅冻结工程参考，G＋F20 不称全流程仅 L。

主目标仍是 raw clean 同时超过 F20/F21 至少 1 个百分点，六景均值同时严格超过 1.5 个百分点；DURA 不能替代主目标。当前 goal ACTIVE，尚无达标方法。历史目标暴露如实披露，不宣称全新独立盲测。

预期产物：state/smoke、source E50 checkpoint 和 hard_pl_state.json、完整训练记录、source V 诊断、source_execution_check、14 项预测、独立 scores 与完整汇总。启动后独立核实 PID/CWD/命令/实际 GPU/日志增长；排队或本地通过不等于训练成功或目标提升。


## 发布后实际排队证据

执行 commit `5b2578b2d8f01258c491d0064a13ef614c7e0c21`；一次发布 archive 本地／远端 SHA 比较与远端 compile PASS。独立读回实际 CPU 控制器 PID／CWD／完整命令、存活和同版本 state，三 row 均 QUEUED、实际 GPU 子进程 0。首 GPU5 的真实原 F21 E30 source smoke 待同 seed HPL2／UVB 全部 GPU 预测完成、HPLT1U 已派发后执行，PASS 立即继续；前序启动中 PID 计入容量。现有健康训练和其他 owner 未修改。排队不视为训练成功或目标收益，详见 queue_verification.json。


## 实际源检查与三卡训练

VERIFIED_NO_UAS_RUNNING：真实原 F21 E30 source-only smoke PASS，U As 输入逐值等于 raw，U 信道生成 0、原 L 信道生成 64；L768／U256、学生 U1536、raw-As128、门控 0.95、权重 2、无 EMA、As KL0、浅中层真实梯度与默认头梯度均核实。probe 丢弃，正式三 seed 各重载原同 seed E30，GPU5／6／7 已有真实 PID／CWD／完整命令／GPU 和全部已刷盘记录增长证据，前序 HPLT1U 保留。执行 commit `5b2578b2d8f01258c491d0064a13ef614c7e0c21`，详见 source_activation.json；训练前缀不视为源训练完成或目标收益。


## 三模型源训练完整，正式评测继续

VERIFIED_SOURCE_FINAL：HPL_NO_UAS 三 seed 均新增 20 轮／4440 步至 E50／11100，全部 13320 条新增闭合记录逐条核对。门控 0.95、权重 2、无 EMA／原型／U 真值／G 训练或另训头。每模型累计 U2097900、学生 U7767900，新增 raw U As567000、physical U 生成与学生物理视图均 0；As KL0、RDrop、跨 RX 锚点与 CE 归一化 1.25 按预定执行，原 L 信道保持。实际接受量和父／增量／累计经过时间见 source_training_complete.json，共享 GPU 墙钟不视为独占 GPU 小时。

冻结学生已自动接续 raw 与同一 DURA 各七景完整预测，全部决策固定后独立 truth-last；不读取部分目标成绩，不宣称目标达成。


## HPL_NO_UAS 冻结 E50 的伪标签源域健康检查

三个固定最终学生各覆盖完整源 V27000 条，原默认头 eval 硬 argmax；无 GPU worker、参数／教师／原型／归一化更新、U 真值或目标读取，不改变健康训练或门控。此项只报告该方法的源域健康，不把源精度当作 U 实测正确率、完整本批测试成绩或目标达成。

| seed | E30 接受精度（%） | E50 接受精度（%） | E30 覆盖（%） | E50 覆盖（%） | E50 接受错误数 |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 99.886825 | 99.876132 | 94.903704 | 95.681481 | 32 |
| 2026092702 | 99.836570 | 99.891658 | 95.181481 | 95.718519 | 28 |
| 2026092703 | 99.902667 | 99.945722 | 95.129630 | 95.529630 | 14 |

上表门控均为 0.95；0.98 对照及完整 TX／RX／day 分层保留在 source_pseudo_quality_E50_HPL_NO_UAS.json。不依据微小源域变化选 checkpoint 或修改当前配方，完整预登记 raw／DURA 目标矩阵继续。


## 完整目标测试结果

ANALYZED：各方法三个seed，共3模型、42项raw／固定DURA七景决策均完成独立truth-last评分，六个匹配监督／普通PL对照模型完整复用。逐景／seed／SD／Macro-F1／RX／TX／day、修复损伤及曝光／成本见target_results.md和附属完整表。原raw双门槛尚未满足，研发目标仍ACTIVE。

HPL_NO_UAS / raw clean：85.423015873% ± 0.312398699%（三个模型 seed 样本 SD）。
HPL_NO_UAS / DURA-LU clean：85.493849206% ± 0.316359015%（三个模型 seed 样本 SD）。
HPL_NO_UAS / raw six_mean：70.657242063% ± 0.485256892%（三个模型 seed 样本 SD）。
HPL_NO_UAS / DURA-LU six_mean：77.985648148% ± 0.267683849%（三个模型 seed 样本 SD）。

不依据本结果热改健康旧任务或替换父来源。
