# 原普通 IQ 观察的独立刷新：源域最小检验

状态：PREREGISTERED，尚无本轮成绩。上一目标轮完成了P12两组E20及完整复算，属于progress；RXGROUP clean仅＋0.1587pp，未达固定＋1pp，候选关闭。本轮不复活该分组、已失败的PA/shift/RMS/AWGN/phase或ANTI噪声候选。

起点提交 b927f01718eb8b4777ec4748f5e27dbb63f657b2；实际训练代码由本次release_commit.txt记录，发布后回填。实现 experiments/cvs_f_viewrefresh20。

## 唯一假设与现有证据

当前固定 F＋跨RX同TX：3SIG_TIME／统一Sinc CENTER_UNITS／NoExplicitCFO／joint160／默认cosine30，骨干及默认头正常联合训练。原6L base为raw .75＋normal ordinary .25＋既有ordinary-only浅中层SupCon，RDrop每一对使用同一个已生成IQ。额外重复一次旧ordinary的历史C_ordinary仍重用同一IQ，没有检验第二份独立抽样的IQ。普通增强冻结源fit约92%的识别率也表明它可能抹除身份证据，不能预先声称标签保持或Ag有效。

本轮唯一问题：相同输入扰动边际、相同额外监督/前向成本下，每步新增一份独立的既有ordinary观察是否比重复观察提供更好的跨RX clean泛化。只复用原anti-shortcut→general-channel阶段，保留原概率、范围、顺序、边界与原E13截断日程；新增观察止于PA/DAC之前，不新增PA/DAC/class-signature／硬件效应，不反演TX。原normal以及其中原PA规则保持。

[已完成AWGN控制报告](../20261017-phase1-F-crossRX-isolatedAWGN-sourceRX1-e20-s2026092701-m3-r01/report.md)记录同seed同配方raw历史与当前控制有13个clean判决差异，数值来源尚未独立识别；因此本轮三组控制全部新跑，不用历史分数代替新控制。不是将那个未选ordinary控制晋级，本轮另有独立刷新与复用的明确差异。P12改变跨RX同批相关性；本轮保持整批原抽样，只改变第四对是否与第一对共用IQ条件。

## 固定矩阵、监督与成本

| row | 第四对输入 | 新训练 |
|---|---|---|
| RAW-RX1-s2026092701 | 同一raw IQ | scratch E20／4440更新 |
| REPLAY-RX1-s2026092701 | 原normal的PA/DAC前缓存 | scratch E20／4440更新 |
| REFRESH-RX1-s2026092701 | 从同一raw独立抽原anti＋channel | scratch E20／4440更新 |

三组均保留原完整base，并添加 .25 RD(view)，RD为平滑CE均值＋同IQ RDrop1 KL。每对CE按128个物理样本平均，原raw.75／ordinary.25的绝对监督权重保留，总pair权重1.25；相对占比变化不隐瞒。RAW的总raw权重为1.00，另两组为.75。共8次L前向，原SupCon只用原两次ordinary maps；新增一对不添一致性、特征对齐、projection或头。既有U／ownEMA／旧门控／AdamW lr.0002、WD.0001、cosine／LS.01／13L＋7U保持，后7轮继续L并启用原U。

所有组每步均构造fresh shadow，只将登记的IQ交给第四对。私有seed20261017961＋step，不依赖TX/RX/day/质量/损失，恢复原CPU／Python／当前CUDA流；各对内部两次网络前向仍用同一IQ。只对显式真实L启用，U包括非负伪标签原生直通。Runtime不进入EMA/reader深拷贝，不增加新U/教师视图；无新的RX硬件参数，因此不制造TX定义的session。

三组各20epoch／4440更新／4546560 L NN，其中额外1136640；每组568320份原normal和568320份fresh shadow（后两者均含RAW控制的IQ构造成本）。原U学生/教师、V、初始化及held8820按实际计量，预计每row5823318 NN；dispatcher兼容smoke另2。总60 model-epochs／13320更新，不是单模型E60。全新scratch，没有父模型或优化器状态；历史B00 E20／4440仅严格载入作2合成包兼容smoke后丢弃，其历史成本分列。

## 数据、固定评价和晋级条件

复用既有VALIDATED_ONCE source carrier及split392005；fit RX3/4/6/8×day1/2/3×6TX，L5040/U45360（TX真值隐藏）/V21600。held源RX1原L1260仅FINAL20读出；角色物理ID及原缓存合法性沿现有source fold验证，不重建数据。不混同完整L6300或L_train5580/L_dev720；V只健康诊断，无状态更新。

IQ2×256／fs25MHz／原RMS位置、原网络内部必要逐包补偿不改。每row完成立即自动默认头clean＋六practical源读出，保持原named stream／nominal信道／residual／post_sync／EQoffG1／reference30／质量适配／相位与IQ补偿／reflect边界；保留所有困难样本。当前不读正式target/query/truth，不减弱正式测试。全部预测完成后独立重算七视图accuracy、macroF1、TX/day及state/lock/quality/addednoiseSNR/CFO/kernel/AGC/boundary分层、默认cos30间隔、固定RAW clean正确集合保持率、repair/break、完整曲线及成本。新增噪声SNR不是总SNR，quality为原白输入功率代理。

主要条件在真实数据运行前固定：REFRESH相对RAW及REPLAY均clean≥＋1pp、clean macroF1不下降、seenV下降≤.5pp；三组native最终L CE≤.5。要求每组完整20epoch／4440更新／4546560LNN／3330有效SupCon／4440生成与消费／568320fresh包／5040唯一L／72个TX×RX×day格，以及finite、U隐藏、0As／头拟合／目标访问／外部权重。六种practical均须报告，但不作为本次独立clean资格的门槛。

任一条件FAIL，关闭固定独立ordinary观察，不改算子子集、强度、概率、系数、RX、seed、阈值或预算救援。PASS也只支持另行登记第二seed源确认，不代表已验证Ag、Ag×As或正式目标lowurban>60%。完整因子矩阵仍须有经验证的Ag，并独立报告交互项。

## 执行、验证与停止

owner codex/root/FcrossRX-viewrefresh-20261017，普通N607 CVS-RFFI Python／原fullFP32／CPU2；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_viewrefresh_sourceRX1_E20_20261017_r01。全部owner每GPU最多3进程、发布容量free≥12000MiB，沿既有共享锁，不改健康旧任务。

命令：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_viewrefresh_sourceRX1_E20_20261017_r01/experiments/cvs_f_viewrefresh20/runner.py

run root：/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-viewrefresh-sourceRX1-e20-s2026092701-m3-r01

日志：/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-viewrefresh-sourceRX1-e20-s2026092701-m3-r01/logs/<row>.stdout.log

模型及teacher从零；固定不可变backend7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b。实际__file__、args、resolved、角色、默认头/Sinc及信道数值由run记录。每组独立FINAL20，不按V或中间成绩选epoch，结果不回流另一健康组。

系统停止仅限本run的角色／来源／预算／非有限／输出碰撞／无法启动或无法生成合法预测；保留partial，无自动重试，不停止健康其他任务。低分只属科学FAIL。

本地native20日程检查、13协议负例通过：原normal IQ与CPU/Python末随机状态逐位一致；新IQ在15/20配置变化；RAW/REPLAY取值精确；U非负伪标签直通；runtime deepcopy为None。主合成normal10496包＋fresh7680包，0NN／更新／真实IQ。

真实CVS合成4组检查通过：206初始state一致，RAW8与原D_raw完整loss／全部参数梯度／末RNG逐位一致；REFRESH相对REPLAY有68个参数梯度改变；原六次前向及同IQ对保持，实际8L；默认头联合更新／export160／0新增参数。10配置负例通过。合成模型检查为覆盖有效路径，四组均强制同一个原生phase gate，仅属测试，不改变实际增强配置；4120合成NN／4合成optimizer更新，0真实IQ／checkpoint／target。输入变化及梯度变化均不能代替识别收益。

一次限定P0/P1只读审查PASS。一次本次N607只读preflight VERIFIED。实际发布采用一个Git代码/config提交、一份archive本远端SHA比较、一次remotecompile，真实同契约checkpoint smoke PASS立即继续；无额外审批、seal、逐成员hash或数据重验。

预期保留完整20轮、4440step、原L/U序列、refresh IQ统计与可见L标签/ID、native gates、默认头七视图预测/间隔/物理状态、独立全量复算、曲线/分层图/负结果、增量与历史成本及最终commit。尚无本轮成绩。
