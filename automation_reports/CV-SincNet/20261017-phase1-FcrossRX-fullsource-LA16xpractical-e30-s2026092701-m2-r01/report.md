# 全源 F＋跨 RX：LA16×真实 practical E30 四格结果

本轮是完整负结果。固定CVS/F、joint160、默认cosine头与原正式IQ条件下，LA16独立组B10的clean相对同预算B00下降0.882个百分点；联合组B11的七个环境均低于普通practical拼接B01。B11 low urban=52.852%，未超过60%；clean相对B00下降2.374个百分点；三个suburban距自身clean为2.324、4.555、6.830个百分点，均未达到首轮2pp探索目标。不能称为clean与星地共同突破。

本轮新训练B10/B11，各scratch E30=20L10U、6660更新；B00/B01只读复用同契约E30参照 `20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01`，不继承权重。最新用户E30授权覆盖目标文本旧E20；没有延长至E50/E200。唯一model seed为2026092701，不声称跨seed复现或新的独立盲测。

记x为同一次共享ordinary增强输出：B00=x/x，B10=x/LA16(x)，B01=x/As(x)，B11=x/As(LA16(x))。

| view | B00 | B10 | B01 | B11 |
|---|---:|---:|---:|---:|
| clean | 85.157 | 84.275 | 82.797 | 82.783 |
| practical_high | 46.513 | 48.373 | 81.469 | 80.459 |
| practical_mid | 44.678 | 46.311 | 79.041 | 78.228 |
| practical_low_suburban | 42.714 | 44.426 | 76.586 | 75.953 |
| practical_high_urban | 45.233 | 46.926 | 78.177 | 77.252 |
| practical_mid_urban | 37.583 | 38.593 | 64.161 | 63.640 |
| practical_low_urban | 31.986 | 32.774 | 53.138 | 52.852 |
| urban三场景均值 | 38.267 | 39.431 | 65.158 | 64.582 |

单位为准确率%。每个view固定168000个相同物理样本；每组全部7个prediction固定后，独立exec scorer才读取truth。B00/B10保留旧普通增强，但无As；不能把B10与B01的差直接解释成Ag效应。完整逐row/seed/view/RX/day/TX及macroF1在 [per_row_seed.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_LA16xpractical_E30_20261017/analysis_r01/per_row_seed.csv)。

| 条件差值 | clean(pp) | urban均值(pp) | low urban(pp) |
|---|---:|---:|---:|
| Ag独立 B10−B00 | -0.882 | 1.164 | 0.788 |
| 联合净值 B11−B00 | -2.374 | 26.314 | 20.867 |
| Ag上加As B11−B10 | -1.492 | 25.151 | 20.079 |
| As上加Ag B11−B01 | -0.014 | -0.577 | -0.285 |
| 交互 B11−B10−B01＋B00 | 0.867 | -1.740 | -1.073 |

真实As仍有明显整体适应收益：B01−B00 urban均值+26.891pp、low+21.152pp，clean−2.360pp；B11−B10 low+20.079pp。但LA16在已有As上使urban均值−0.577pp、low−0.285pp，未保留其无As时的微小受损视图增量。clean交互项为正仅表示两个负效应不是简单相加，不代表联合clean超过B00。

| arm/view | macroF1(%) | 最差RX准确率(%) | 默认头平均真类top-competitor间隔/30 |
|---|---:|---:|---:|
| B00/clean | 84.892 | 74.433 | 0.147373 |
| B00/practical_mid_urban | 34.858 | 31.529 | -0.026694 |
| B00/practical_low_urban | 30.218 | 27.700 | -0.036008 |
| B10/clean | 83.962 | 72.317 | 0.148706 |
| B10/practical_mid_urban | 36.205 | 29.846 | -0.022564 |
| B10/practical_low_urban | 31.047 | 26.787 | -0.033501 |
| B01/clean | 82.378 | 75.013 | 0.129577 |
| B01/practical_mid_urban | 63.650 | 57.417 | 0.057697 |
| B01/practical_low_urban | 52.497 | 47.996 | 0.032490 |
| B11/clean | 82.314 | 74.575 | 0.129242 |
| B11/practical_mid_urban | 63.126 | 56.275 | 0.057830 |
| B11/practical_low_urban | 52.241 | 47.404 | 0.032870 |

B11 low urban相对B01修复2930个样本、破坏3409个，净少479个正确；clean修复1620、破坏1644，净少24个。固定B00 clean正确集合143063个样本上，low保持率B01=58.020592%、B11=57.854232%；mid为70.494118%→70.140428%。B11 low平均间隔/30略增0.032490→0.032870，却伴随准确率下降，不能把均值间隔改善或错误重分配当作识别突破。

源判据在读取B11正式分数前已固定失败，记录于提交 `39ed0bd6f766971217891bdfe7ea5b7828d4e922`。B10完整V clean相对B00+0.033333pp，仅通过健康条件；B11相对B01−0.011111pp，健康通过。B11在两draw×mid/low urban上的源差值为[−0.888889,+0.444444]及[+0.388889,−0.611111]pp，0/4达到预登记每格+1pp，故不扩seed。SOURCE_QUALIFIED_READOUT只表示读出完成。旧RX1/E20 LA16 standalone FAIL保留；旧源clean+2.555556pp没有被当成当前全源Ag收益，未改窗口/深度/阈值/seed/预算救回。

源机制使用同一原V1800、两次原P24信道随机实现、三场景full reflect缓存，CPU2冻结默认头，无新信道和更新；加载V27000和未使用L6300，U0，无额外L_dev或留出RX。下表每场景3600次观测，包含同一1800物理包的两draw，不能冒充独立模型seed。

| 源场景/分层 | n | B00(%) | B01(%) | B11(%) |
|---|---:|---:|---:|---:|
| practical_mid_urban/ALL | 3600 | 46.333 | 76.306 | 76.056 |
| practical_mid_urban/BLOCKED/unlocked | 678 | 20.796 | 38.496 | 38.053 |
| practical_mid_urban/<3dB signal/added-noise | 383 | 18.277 | 34.987 | 35.770 |
| practical_low_urban/ALL | 3600 | 38.111 | 62.556 | 62.472 |
| practical_low_urban/BLOCKED/unlocked | 1565 | 22.300 | 34.952 | 34.569 |
| practical_low_urban/<3dB signal/added-noise | 1127 | 20.408 | 31.500 | 31.322 |

新增LA16未改善low urban的BLOCKED/unlocked与<3dB新增噪声比分层。As本身在这些源困难状态上明显好于B00，LA16没有进一步缓解。mid的<3dB组有+0.783pp局部变化，但总体与主要遮挡失锁组下降；不择取局部分组声称成功。正式测试没有逐包state/quality标签，不能把源状态结果冒称正式LOS/lock归因，也不能据分层准确率分离噪声、多径、CFO的因果贡献。完整源分层见 [source_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_LA16xpractical_E30_20261017/analysis_r01/source_strata.csv)。

独立复算核对4组120个epoch、26640步、504个正式混淆矩阵及1112个源混淆矩阵；无NN、IQ或CP读取、无训练更新。5对训练序列每对8880条全部一致。四组全部30轮ordinary摘要完全一致。B10/B11的6660次Ag随机位置、物理ID/TX及数值统计完全一致，0差异；各852480包，全部6300个L物理ID/241起点保留。两As组各852480包，均覆盖90个TX×RX×scene和90个TX×RX×state组合；LOS/PARTIAL/BLOCKED=427829/134726/289925，locked/degraded/unlocked=511572/142274/198634，两组相同。

实际native practical为residual/post_sync，EQ关闭G=1，reference_snr_db30、fs25MHz、fc2.45GHz，接收质量适配、原相位跟踪/IQ补偿/原生AGC开启，session按RX/day、不按TX。残余核长度62、FIR数值延迟24样本，signal/noise边界都为原synthetic_reflect，AGC限幅各79次；正式IQ/SNR/边界保持原文件。质量采用white_input_expected_power_proxy，均值16.001663dB；实际signal/added-noise均值B01/B11为16.030427/16.030341dB，二者均不是总SNR。逐包实际比代理可低约20.9dB，不能用均值一致断言状态代理完全准确。Ag使进入As前平均RMS从0.981203变为0.968623，但上述状态、lock、CFO和残余核分布未改变，不能将其描述为弱化As。能量有效样本数是集中度指标，不是去边界后的真实可观测长度。

静态融合审计仅读取4个已核实同契约FINAL30的state_dict，CP读取4、NN/IQ/优化/另训头均0，耗时1.823630秒，见 [parameter_audit.json](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_LA16xpractical_E30_20261017/parameter_audit.json)。B11−B01的PA/reference/signature静态增益平均绝对差分别0.002470/0.002531/0.000438；这不是逐包门控激活。默认cosine尺度固定30，head.gain是物理融合向量，不是logit温度。B11主干总梯度E1/E30为2.781472/1.423834，默认头weight从本组初值移动L2=1.163345，Sinc与三SIG投影均更新；未冻结骨干或另训头。inactive梯度冲突字段为0不证明无冲突。逐包融合激活未保存，不能补造门控因果机制。

训练曲线显示B11最终主/次视图CE=0.348435/0.803420（B01=0.346313/0.785204），未显示额外LA16减轻次视图困难；E30 U旧门控接受880/56700，B01为878/56700。U真值隐藏，接受率不是伪标签精度。当前只说明固定预算结果，预算结束不等于收敛。

固定F为SIG_TIME_X3＋统一Sinc CENTER_25K/CENTER_UNITS＋原跨RX同TX SupCon，900722参数/206状态、joint160、原cosine默认头、NoExplicitCFO并保留内部补偿，随骨干联合训练。保留原RDrop alpha1、label_smoothing0.01、AdamW lr2e-4/min1e-6/weight_decay1e-4/cosine30/globalclip5；SupCon仅clean半部time_down/t2.pool，temperature0.2、max0.1、warm5/ramp10、无额外前向。旧U规则/ownEMA固定，新LA16/As仅L。原L6300/U56700隐藏/V27000、RX1/3/4/6/8、day1/2/3、6TX、7/63/30/split392005保持，无留出训练域。原equalized1记录256点完整float32 RMS(center=false,eps1e-12)先于普通增强；As后无新增RMS。真实导入来自固定release的experiments/adv3b02_xuc/code/leo_practical，非顶层code默认实现。共享ordinary详细实际概率、PA/DAC/class-signature/CFO和重复AWGN/多径审计见 [ordinary_audit.md](ordinary_audit.md)。

LA16固定width16、完整起点0至240、sin²窗、depth≤0.5、relativeL2≤0.1，无新增RMS/噪声/相位/索引变化。顺序为ordinary→LA16→一次完整practical，primary保留ordinary；两部分CE/RDrop KL各归一化0.5，两次dropout前向复用同一生成IQ。增强前后参数结构完全相同，L每物理样本4NN。model seed2026092701，split/data/eval392005；ordinary私有20261016051、Ag20261017111+step、As view202610172201、RX硬件2026092701，随机流相互分离。

| arm | 本组训练epoch/更新 | 原训练及源评价NN | 源墙钟(s) | 固定7view预测NN/墙钟(s) | 额外冻结源读出NN/墙钟(s) |
|---|---:|---:|---:|---:|---:|
| B00（历史只读参照） | 30/6660 | 5407922 | 1540.012 | 1176002/47.866 | 0/0.000 |
| B10（本轮新增） | 30/6660 | 5407922 | 1604.665 | 1176002/49.322 | 12602/8.107 |
| B01（历史只读参照） | 30/6660 | 5407922 | 3026.315 | 1176002/48.894 | 0/0.000 |
| B11（本轮新增） | 30/6660 | 5407922 | 2964.630 | 1176002/50.657 | 12602/7.604 |

每组L NN3409920；student train/eval NN3976920/864002，teacher train/eval NN0/567000。新增模型60epoch/13320更新，历史参照60epoch单列，无训练父权重成本；ownEMA来自本组scratch。B00/B01历史冻结源机制各23402NN，本轮只读复用；首步E20真实检查点仅兼容smoke的2synthetic NN，丢弃后scratch，历史E20/4440不计为训练父模型。独立图表分析0NN，参数审计另0NN/4CP读；模型训练、额外前向与只读成本没有混算。分析本身未直接记录总墙钟，保留启动/结束日志，不冒充精确测量。

实际实现文件为 `experiments/cvs_fullsource_factorial27/{design,attenuation,observer,model,source,qualification,dispatch,evaluate,preflight,publish,runner,channel,report,checks,__init__}.py`；独立分析为 `analysis/cvs_factorial_E30_results_20261017.py`，复用已有 `analysis/cvs_fullsource_E30_results_20261017.py`。关闭源失败路线的策略记录为 `configs/phase1/F_crossrx_e30_baseline.json`。实际训练代码提交 `bfbef1d853ffe2eca5d3094b60b53e2ff8a5f7e8`，分析代码提交 `fa7c7494722df03bcef2413a3d0a613c0b220e91`；报告/配置收尾提交在Git历史，不冒充训练执行版本。

实际完整启动命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_LA16xpractical_E30_20261017_r01/experiments/cvs_fullsource_factorial27/runner.py
```

CWD为 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_LA16xpractical_E30_20261017_r01`；原不可覆盖run根 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-LA16xpractical-e30-s2026092701-m2-r01`。普通N607账户、两组分别GPU0/1，共享容量锁每GPU最多3个所有owner实验进程、free≥12000MB；未停止/重启/热改健康旧任务。各row训练后自动exec预测、独立exec评分，均自然退出；复算PID504365也正常结束。原source/actual_native_args.json、resolved_config、module_files、ordinary/Ag/physics/SupCon/Sinc完整日志、final_ssdg.pth、source_frozen、prediction与scoring仍保存在各row原root。数据、权重、原始日志及生成图表不进Git。

执行过的本地测试命令：

```text
E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_LA16xpractical_E30_20261017/local_checks.py
```

4项PASS：24配置负例；241完整窗口位置/零输入/private RNG/非法step；256synthetic原生物理调用；512synthetic完整F前向及默认头/Sinc/三SIG梯度，真实IQ/CP/优化更新0。另包compileall/diff检查、独立四格算术/缺arm/非法概率/混淆重数检查PASS；一次有界独立P0/P1审查PASS。首步真实同契约CP strictload no-query smoke PASS（2synthetic/真实IQ与更新0）；单包传输比较、远端编译、PID/CWD/argv/GPU/log增长均VERIFIED。完整实训active损失有限、跳过更新0；日志中NaN为未启用诊断字段，不能据此伪造训练失败。

实际已生成并检查的7张PNG及各PDF在 `E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_LA16xpractical_E30_20261017/analysis_r01`：formal_accuracy、training_curves、receiver_accuracy、source_added_noise_accuracy、source_state_lock_accuracy、training_support_B01、training_support_B11。原远端路径见 [figures.json](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_LA16xpractical_E30_20261017/analysis_r01/figures.json)，数值与成本见 [analysis_summary.json](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_LA16xpractical_E30_20261017/analysis_r01/analysis_summary.json)，原预登记保留于 [preregistration.md](preregistration.md)，逐row矩阵唯一维护于 [experiment.json](experiment.json)。

本轮固定四格实验与交付已完成，但总体目标未完成：没有有效Ag的clean正收益，没有low urban>60%，没有suburban三场景距clean≤2pp，也没有跨seed复现。source失败的LA16组合关闭，不调窗口/深度/概率/阈值/RX/seed/预算重复救回。后续候选需要独立的合法源证据；共享ordinary的稀有大幅值与重复损伤是已记录的源审计线索，尚未做组件因果定位，不能直接据此宣称PA导致失败或启动未经验证的硬件增强。
