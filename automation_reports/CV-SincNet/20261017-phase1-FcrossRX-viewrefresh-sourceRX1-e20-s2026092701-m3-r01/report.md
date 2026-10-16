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

启动核实：执行 commit c72c359581cee36e9fc70440844803288ba851ef 已 push 且独立 GitHub OID 相同。单 archive transfer VERIFIED／remote compile PASS；dispatcher123079、RAW123167/GPU3、REPLAY123245/GPU4、REFRESH123322/GPU5 的PID/CWD/argv/GPU/日志增长已独立核实。真实同契约B00 strict-load兼容smoke2合成NN PASS并立即继续scratch。当前RUNNING，无target访问；运行中代码与配置不改。离线评分18个协议负例PASS，0 IQ/CP/NN/更新。


---

# F＋跨 RX 同 TX：独立普通 IQ 观察与复用控制

三组 scratch E20 和全部七视图源域读出完成；固定独立普通 IQ 观察未通过预设条件，保留负结果并关闭。

训练代码 c72c359581cee36e9fc70440844803288ba851ef；离线复算 6443685b6bd14ed0d895be843d867db57f2fad2f（之后如有离线修复另记）。全部成绩来自 source held RX1 原 L1260／seed2026092701／FINAL20／默认 cosine 头，不是正式 target、跨 seed 或 Ag×As 结果。三组控制均为新训练，不用有未解释判决漂移的旧同 seed 控制代替。

## 逐 row／seed 七环境结果

| view | RAW % | REPLAY % | REFRESH % | REFRESH−RAW pp | REFRESH−REPLAY pp |
|---|---:|---:|---:|---:|---:|
| clean | 80.3175 | 81.4286 | 82.0635 | 1.7460 | 0.6349 |
| practical_high | 39.1270 | 36.6667 | 37.0635 | -2.0635 | 0.3968 |
| practical_mid | 39.6032 | 36.7460 | 37.2222 | -2.3810 | 0.4762 |
| practical_low_suburban | 36.2698 | 34.2063 | 34.2857 | -1.9841 | 0.0794 |
| practical_high_urban | 38.4921 | 36.5873 | 36.8254 | -1.6667 | 0.2381 |
| practical_mid_urban | 32.4603 | 31.0317 | 31.1111 | -1.3492 | 0.0794 |
| practical_low_urban | 27.9365 | 27.3016 | 27.2222 | -0.7143 | -0.0794 |

| view | RAW macro F1 % | REPLAY macro F1 % | REFRESH macro F1 % |
|---|---:|---:|---:|
| clean | 79.9414 | 81.0801 | 81.7507 |
| practical_high | 28.3045 | 28.2520 | 29.2606 |
| practical_mid | 29.5865 | 29.0866 | 29.8858 |
| practical_low_suburban | 28.3534 | 28.6393 | 28.5131 |
| practical_high_urban | 29.0724 | 29.0923 | 29.4290 |
| practical_mid_urban | 27.0797 | 26.8646 | 26.9864 |
| practical_low_urban | 24.6109 | 24.4648 | 24.5581 |

| row | urban mean % | low urban % | high/mid/low suburban 与自身 clean 的差 pp |
|---|---:|---:|---|
| RAW | 32.9630 | 27.9365 | 41.1905, 40.7143, 44.0476 |
| REPLAY | 31.6402 | 27.3016 | 44.7619, 44.6825, 47.2222 |
| REFRESH | 31.7196 | 27.2222 | 45.0000, 44.8413, 47.7778 |

## 固定条件与收益归因

原完整 base 保留：raw.75／原 ordinary.25／原 SupCon；三组同加 .25 RD(第四视图)，总 pair 权重1.25、每个 L 包8次前向。RAW 第四视图是 raw，REPLAY 是第一份原 ordinary 在 PA/DAC 前的缓存，REFRESH 是从同一 raw 独立抽取的原 anti-shortcut→general-channel。独立观察使用原算子、概率、范围、顺序、边界和 E13 截断日程，不新增 PA/DAC；共用 base 内原 PA 仍在。各对内部同 IQ RDrop，新增对没有 SupCon／跨视图对齐／新头或部署参数。原 U／EMA／旧门控保持。

raw 的原绝对.75权重保留，但总量1.25后相对占比变化已披露；RAW加额外raw后总raw1.00，另两组.75。所有组均构造相同私有 fresh shadow、恢复原 RNG；本轮区分额外监督预算与独立观察，不能将相对旧6L的任何增益直接归于新Ag。

固定要求：REFRESH 相对 RAW 和 REPLAY 均 clean≥＋1 pp、macro F1≥0 pp、seen V下降≤.5 pp；三组 final native L CE≤.5。完整 E20／4440更新／4546560 L NN／3330 SupCon steps／4440生成和消费／568320 fresh包／5040唯一L／72格、合法角色及有限更新必须满足。全部六 practical 必报，但不参与本次独立clean资格判断。

实际条件：{"RAW_clean": true, "RAW_macroF1": true, "RAW_seenV": true, "REPLAY_clean": false, "REPLAY_macroF1": true, "REPLAY_seenV": true, "healthy": true}。
相对 RAW：urban均值 -1.2434 pp；clean macro F1 1.8094 pp；seen V 0.0741 pp。
相对 REPLAY：urban均值 0.0794 pp；clean macro F1 0.6706 pp；seen V 0.0000 pp。

FAIL 关闭固定独立 ordinary 观察；不调整子集、强度、概率、权重、RX、seed、阈值或预算救援，不把控制组事后晋级。

## low urban 分层与默认头间隔

| 分层 | n | RAW % | REPLAY % | REFRESH % | REFRESH−RAW pp | REFRESH−REPLAY pp | REFRESH margin/30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| state:LOS | 384 | 41.9271 | 40.3646 | 39.8438 | -2.0833 | -0.5208 | -0.0311 |
| state:BLOCKED | 714 | 18.9076 | 19.8880 | 19.7479 | 0.8403 | -0.1401 | -0.0581 |
| state:PARTIAL | 162 | 34.5679 | 29.0123 | 30.2469 | -4.3210 | 1.2346 | -0.0322 |
| lock:locked | 467 | 42.1842 | 39.8287 | 40.0428 | -2.1413 | 0.2141 | -0.0295 |
| lock:degraded | 228 | 21.4912 | 20.1754 | 19.7368 | -1.7544 | -0.4386 | -0.0540 |
| lock:unlocked | 565 | 18.7611 | 19.8230 | 19.6460 | 0.8850 | -0.1770 | -0.0576 |
| quality:[20,30) | 396 | 42.6768 | 40.6566 | 40.4040 | -2.2727 | -0.2525 | -0.0301 |
| quality:[5,10) | 157 | 19.7452 | 21.0191 | 21.0191 | 1.2739 | 0.0000 | -0.0597 |
| quality:[0,5) | 385 | 19.4805 | 20.2597 | 20.0000 | 0.5195 | -0.2597 | -0.0561 |
| quality:<0 | 180 | 17.2222 | 18.8889 | 18.8889 | 1.6667 | 0.0000 | -0.0608 |
| quality:[10,20) | 142 | 32.3944 | 26.7606 | 27.4648 | -4.9296 | 0.7042 | -0.0338 |

这是固定同批源 IQ／同物理状态的表现定位，不能唯一归因于噪声、多径、残余 CFO、RX IQ 或内部门控。六 practical 的实际传播、状态与补偿逐包完全相同；EQ关闭时G=1，residual不等于已均衡；quality为原白输入功率代理，新增噪声SNR不等于总SNR。没有重新生成弱测试或删除困难样本。本轮不训练As，未检验与As的组合交互。

## 固定 RAW clean 正确集合与错误变化

集合 n=1012，所有视图复用，不按 REFRESH 重选。

| view | RAW 保持率 % | REPLAY 保持率 % | REFRESH 保持率 % | 对 RAW 修复/损坏 | 对 REPLAY 修复/损坏 |
|---|---:|---:|---:|---|---|
| clean | 100.0000 | 99.3083 | 99.4071 | 28/6 | 15/7 |
| practical_high | 36.9565 | 34.7826 | 35.1779 | 26/52 | 25/20 |
| practical_mid | 37.3518 | 36.2648 | 36.4625 | 24/54 | 23/17 |
| practical_low_suburban | 35.0791 | 34.2885 | 34.0909 | 26/51 | 24/23 |
| practical_high_urban | 36.8577 | 35.7708 | 36.0672 | 17/38 | 18/15 |
| practical_mid_urban | 31.2253 | 29.9407 | 29.9407 | 33/50 | 20/19 |
| practical_low_urban | 27.6680 | 27.3715 | 27.3715 | 37/46 | 18/19 |

## 全量执行、覆盖与实际成本

| row | updates | L NN packets | 总 NN packets | fresh 构造/审计 s | native 训练 s | 到最终结果 s |
|---|---:|---:|---:|---:|---:|---:|
| RAW | 4440 | 4546560 | 5823318 | 3.1111 | 1416.1719 | 1435.1568 |
| REPLAY | 4440 | 4546560 | 5823318 | 3.1454 | 1413.6264 | 1432.3398 |
| REFRESH | 4440 | 4546560 | 5823318 | 3.3004 | 1534.9699 | 1553.9418 |

总计60 model-epochs／13320更新／17469956 NN包（含dispatcher真实CP兼容smoke2）。每个学生／ownEMA／optimizer均scratch，父训练成本0；旧B00 E20／4440更新只载入后作2个合成包smoke再丢弃，没有继承。NN扣除EMA复制时继承的2个初始化计数，保留实际初始化前向，held每组8820单列。矩阵首worker启动至dispatcher完成 1563.0698 s，进程start ticks及产物mtime独立核对，近似1 s精度。预算耗尽不等于收敛。

本地合成native runtime：10496原normal＋7680 fresh包、13负例、0NN／更新；真实CVS合成：4120NN／4更新、206初始state、RAW与原8L完整loss/grad/RNG逐位一致、10配置负例；离线18协议负例0NN。一次P0/P1审查PASS、N607单preflight、单archive传输及remotecompile通过。合成梯度变化不是识别收益。

| row | 最终原生 L CE | grad norm 均值/最大 | fresh 非同一 IQ 的 step 数 | 唯一 L/覆盖格 |
|---|---:|---|---:|---|
| RAW | 0.2641 | 0.9464/10.7528 | 2916/4440 | 5040/72 |
| REPLAY | 0.2469 | 1.1333/10.7528 | 2916/4440 | 5040/72 |
| REFRESH | 0.2403 | 1.0755/10.7528 | 2916/4440 | 5040/72 |

全部20轮／4440步／5994原L-U序列及全stdout复算，所有有效更新有限，U TX真值隐藏，没有新增U／teacher增强。原L/U序列与RAW逐一相同：{"REPLAY": true, "REFRESH": true}；原／fresh shadow全量保存统计相同：{"REPLAY": true, "REFRESH": true}。私有流只依赖固定seed＋step，不依赖TX/RX/day/quality/loss。

| REFRESH 原算子 | 原normal前PA触发% | fresh触发% | 原/fresh条件不同% |
|---|---:|---:|---:|
| p_dc_offset | 6.3063 | 6.3288 | 11.7793 |
| p_bandedge_taper | 4.6396 | 5.0901 | 9.2342 |
| p_time_shift | 6.3964 | 6.3288 | 11.9144 |
| p_amp_scale | 8.9414 | 8.7162 | 16.2162 |
| p_phase_rot | 9.0991 | 8.3784 | 15.5856 |
| p_cfo | 6.7568 | 6.3288 | 11.8243 |
| p_phase_noise | 5.4955 | 5.0901 | 10.0901 |
| p_multipath | 2.8378 | 3.8288 | 6.3514 |
| p_awgn | 7.6351 | 7.5901 | 13.9189 |

实际训练入口：原SSDG与DataAugmentation及leo_practical均来自已记录的adv3b02_xuc/code；新wrapper来自本run不可变release。完整actual args、resolved、默认头/Sinc、原RMS位置、普通增强实际配置与六practical配置随每row保留，默认值不能代替实际运行值。

## 图表、文件与边界

离线出图只读冻结汇总，0 IQ／CP／NN／更新。
- [source_scene_results PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/source_scene_results.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/source_scene_results.pdf)
- [complete_training_curves PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/complete_training_curves.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/complete_training_curves.pdf)
- [ordinary_observation_exposure PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/ordinary_observation_exposure.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/ordinary_observation_exposure.pdf)
- [low_urban_strata PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/low_urban_strata.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/low_urban_strata.pdf)
- [fixed_clean_retention_and_errors PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/fixed_clean_retention_and_errors.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/fixed_clean_retention_and_errors.pdf)

[独立汇总](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/analysis_summary.json)；完整training_curves.json、source_strata.json与逐row原记录同目录。raw IQ／权重未拉取或提交。run_id：20261017-phase1-FcrossRX-viewrefresh-sourceRX1-e20-s2026092701-m3-r01。

本轮只检验一个已暴露源 RX 和一个 model seed，无法报告最差目标 RX、正式默认头提升、正式clean代价或跨seed重现。固定正式 practical 条件保持，但尚无新的正式target结果；low urban>60%、suburban接近clean、新Ag确认和完整B00/B10/B01/B11交互均未完成。内部融合／门控权重未另作本轮读出，不能凭margin或梯度推断。活动目标继续。


NN计数分解（raw计数保留，EMA复制时继承的2个初始化eval计数在总成本中扣除；不是两次新前向）：

| row | 学生 train | 学生 eval | EMA train | EMA eval | held源评估 | 额外第四对L |
|---|---:|---:|---:|---:|---:|---:|
| RAW | 4942928 | 475202 | 0 | 396370 | 8820 | 1136640 |
| REPLAY | 4942928 | 475202 | 0 | 396370 | 8820 | 1136640 |
| REFRESH | 4942928 | 475202 | 0 | 396370 | 8820 | 1136640 |

完整曲线诊断：下列best seen V仅描述曲线，不改变FINAL20选模。

| row | 末5轮总L损失范围 | 末5轮seen V范围% | 末5轮V首末差pp | 诊断best V epoch |
|---|---|---|---:|---:|
| RAW | 0.3328 至 0.3644 | 98.0000 至 98.0370 | -0.0185 | 12 |
| REPLAY | 0.3710 至 0.4168 | 98.0509 至 98.0880 | 0.0324 | 20 |
| REFRESH | 0.3708 至 0.4085 | 98.0694 至 98.1111 | -0.0046 | 12 |

REFRESH相对两控制的最早训练分歧（完整step记录）：{"RAW": {"first_extra_CE_difference_step": 2, "first_total_L_difference_step": 3, "first_gradient_preclip_difference_step": 1}, "REPLAY": {"first_extra_CE_difference_step": 2, "first_total_L_difference_step": 3, "first_gradient_preclip_difference_step": 1}}。完整stdout中的warning行、禁用指标NaN提及数和epoch marker保留在analysis_summary.json；有效loss/gradient有限性另行核验，不把禁用项NaN视为训练崩溃。


## 实际执行补充审计

工作起点 commit b927f01718eb8b4777ec4748f5e27dbb63f657b2。本轮训练 commit c72c359581cee36e9fc70440844803288ba851ef；F／SIG_TIME_X3／统一 Sinc CENTER_UNITS／真实 joint160／NoExplicitCFO／defaultcos30与原CROSS_RX保持。没有父权重初始化；旧匹配B00仅用于兼容smoke，继承链与source角色已核对。

实际模块不是仓库顶层code，而是固定release中的experiments/adv3b02_xuc/code/SSDG/train_ssdg.py、DataAugmentation.py、leo_practical/channel.py及residual.py；完整__file__见每row actual_modules.json和refresh_actual_modules.json。25MHz、IQ2×256、原loader RMS位置保留；新增视图没有后置RMS、重叠完整practical或新增PA/DAC。原normal调用仍包含原PA，DAC与class_signature实际关闭。ordinary_augmentation_actual.json内no_second_pair_redraw指原有同IQ RDrop pair；本轮第四对是否刷新由refresh_steps与refresh_audit单独完整记录。

六种practical逐row实际配置与逐包传播／接收状态完全相同：residual／post_sync／EQ关闭G=1、reference_snr_db30、quality_adaptation／phase_tracking／IQ_compensation开启、AGC上限30dB、reflect边界。quality locked/unlocked阈值15/5dB、quality reference20dB、max residual scale10；接收机CFO std200Hz／residual CFO std100Hz、phase noise linewidth1Hz／tracking residual1deg，IQ注入上限0.2dB/1deg及估计误差0.05dB/0.2deg。EQ配置中的mmse／length129／delay64属于关闭功能，不能据此声称已均衡。观察原点／FIR群延迟与有效观察范围没有另做本轮移除控制；保留full physics中的kernel/boundary与原固定数值实现，不推断它们为唯一原因。

原普通增强base probabilities不是实际每步概率。E13截断日程下，实际p_dc_offset3.00%至7.5180%、taper2.50%至6.2650%、time_shift/CFO3.50%至8.7710%、amp/phase4.50%至11.2770%、phase_noise3.00%至7.5180%、multipath1.80%至4.5108%、AWGN4.00%至10.0240%；逐步值和触发次数全量保存。新观察只使用这些原native阶段。

首步三组记录的grad norm存在约1e-8量级差异，原／fresh保存IQ统计和L/U序列则全程相同。本轮没有在真实GPU训练宣称所有梯度逐位复现；此前逐位loss/grad/RNG一致是合成RAW控制检查的结论。默认原数值策略cudnn_deterministic=false，跨独立run的微小漂移来源未被本轮唯一定位，不能由一个source seed宣称重现。

最终图使用figures_v2，负增益标注留足边距；初版图保留。末段V汇总已纠正原生百分比单位（98.xx即%，不再乘100），这不改变固定判断、全部source预测或七视图分数。出图及离线修复均0IQ／CP／NN／更新；训练与预测没有重跑。


## 已执行命令

训练使用预登记的 N607 dispatcher，CWD 与不可变 release 路径见预登记；它先真实CP smoke，再独立启动三组，完成后逐row自动七视图源评价。以下离线命令的 CWD 为 E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014：

```text
F:/App/miniconda3/python.exe -m experiments.cvs_f_viewrefresh20.checks_analysis
F:/App/miniconda3/python.exe -m experiments.cvs_f_viewrefresh20.harvest --output E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final
F:/App/miniconda3/python.exe -m experiments.cvs_f_viewrefresh20.analyze --input E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final
F:/App/miniconda3/python.exe E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/plot_local_v2.py
```

既有 run/output 保持不可变；这些是实际执行记录，不表示对同一 root 重新启动训练。运行代码/config与离线实现分别提交，最终Git报告提交另记。
