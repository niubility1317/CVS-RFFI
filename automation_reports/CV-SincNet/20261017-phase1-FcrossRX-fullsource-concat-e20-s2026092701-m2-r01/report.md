# 完整源域 F＋跨 RX 同 TX：E20 普通拼接星地参照

状态：两行均已 ANALYZED；预固定 RAW 对照也已完成，三行完整源域/同seed比较与378组指标复算齐全。联合目标未完成。唯一矩阵与逐行配置维护在 experiment.json；当前模型 seed2026092701 的两行参照 B00/B01。用户2026-10-17明确要求不再留训练域，主线回到真实 practical 增强与普通拼接直接比较。

完整原角色：RX1/3/4/6/8，day1/2/3，6TX，split392005；原 L6300 / hidden U56700 / V27000，7/63/30。实际原契约及兼容 CP 契约 IDs 相等已只读确认，不重建数据。新source.py在构建后检查全部原角色ID精确一致，不调用任何fold.prepare、不带fold参数、不切L_dev。V仅原源健康诊断；不训练V，不解封U TX。

固定网络：F（三路SIG_TIME、统一Sinc CENTER_25K/CENTER_UNITS）、原真实joint160、原cos30默认头、NoExplicitCFO，900722参数/206state tensors。正常联合训练，无新分支/头/投影器/部署参数。保留RDrop1、label_smoothing.01、原13L+7U/旧门控/ownEMA、原cosine/AdamW峰值.0002最低.000001/weight_decay.0001/physicalglobalclip5。每模型scratch20轮、222步/轮、4440更新；无训练父模型。

B00使用原native concat_full但p=0，普通增强IQ精确复制成两半；B01为原native concat_full,p=1，ordinary(x)与practical(ordinary(x))。两行均每物理L四次网络样本量（两次256拼接前向），总2273280L NN/row，不将额外前向收益算作信道收益。B00是匹配成本的无As参照，不能冒称原单视图/两前向F性能。原native的普通增强先于practical，其AWGN/多径等可能重复叠加；本轮保留这一真实对照行为并记录，不静默改成practical(raw)。

各半分别计算原smoothed CE/RDrop并各占0.5，与native全256平均CE严格等价；两次RDrop复用同一已生成IQ，不重抽信道/普通增强。真实L标签监督全部卫星视图，不设学生置信门控。跨RX同TXSupCon只读两次clean半批的time_down/t2.pool；前5轮0、10轮ramp到.1，温度.2，沿既有损失公式，无额外前向。原U仅原规则，没有新卫星SSL。

星地训练保持原native完整practical：low_suburban/mid_urban/low_urban三场景原先验逐batch抽取，物理流跨epoch/view刷新；RX硬件session按RX/day稳定，不依TX。residual/post_sync、EQ实际关闭(G=1)、reference_snr_db30、25MHz/2.45GHz、quality adaptation/phase tracking/IQ compensation开启、reflect边界，星地后无外部RMS。actual_practical_first_batch.json/module_files.json记录真实配置与导入文件；新增噪声SNR不冒充总SNR。正式固定received IQ、测试SNR和六practical参数不变，不删困难样本。

每row FINAL20固定后自动exec预测原clean＋六practical，共168000×7判决；完整prediction固定后另exec隐藏CUDA的独立truth-last评分。没有source或target性能晋级门槛，不能用分数改变另一行、选checkpoint、重排候选或重跑。报告逐row/seed/view/RX/day/TX accuracy、macroF1、urban均值、最差RX、suburban与自身clean差；low>60%、clean净正、suburban差≤2pp仍为探索目标，无保证。只有两行，不能估计Ag×As交互或跨seed复现。

环境/CWD：普通N607 CVS-RFFI Python，/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_concat_E20_20261017_r01；每GPU最多3个总实验进程，计入其他owner且空闲≥12000MB。单launch owner/root、独占新root /home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-concat-e20-s2026092701-m2-r01；不操作健康旧任务。dispatcher/scorer不占GPU。精确入口：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_concat_E20_20261017_r01/experiments/cvs_fullsource_concat22/runner.py`。代码Git commit在发布前登记为实际实现提交。

检查：8项非法role/budget/hiddenU输入拒绝及全512logit梯度归一化PASS；真实完整F的256×2前向、两行206初始tensor相同、clean-only SupCon和默认头/三SIG/Sinc联合有限梯度及2次合成更新PASS（1028NN）。开发期另一次float精确相等断言失败消耗512NN/0更新，已改容差；另一次过时跟踪入口失败0NN。故本地总1540合成NN/2更新、0真实IQ/CP。已编译通过。

一次独立P0/P1审查PASS。曾发现旧RX6 bootstrap未暴露新依赖；已把既有F数学、SupCon纯数学、RecordedLoader/PrivateAugment放进新包，去除旧实验入口依赖，完成一次定向复核。未更改这些数学。实际远端可达native/leo/runtime/capacity/control/truth_arrays/score依赖已只读核实，证据在本地artifact root。一次N607预检PASS。

启动第一步仅使用已核实全源域CROSS_RX FINAL50 checkpoint做2个合成样本严格加载兼容检查。其原物理L/U/V、scratch来源、无祖先/另训头、FINAL50且冻结前无target接触已核实。历史E50/11100属于兼容参考成本，不是新训练父模型；不将E50权重载入新训练或EMA。然后立即进入两行scratch E20，无smoke token或额外审批。

停止：预算固定不早停选模；任一数据角色/标签权限/非有限loss或grad/预测/来源/技术错误保留产物，阻止未启动row；其他健康row继续完成，不自动重启，不按成绩补救。预算用尽不等于收敛。

预计产物见experiment.json：完整源契约、resolved config/args、普通增强配置与真实模块文件、native星地physics/覆盖、完整逐步/逐轮日志与曲线原表、FINAL20/source冻结、每row七视图prediction及独立score、结果CSV/成本/负结果。训练结束和本地检查通过均不等于实验完成。当前唯一问题是完整源域下普通拼接的实际urban增量与clean代价；没有新增Ag或联合成功结论。

实际实现代码 commit：`dc4e47fe0c2b3e0dcf73dda104cdbd4a7c814f5a`；已提交/推送，发布前使用这一固定实现。


## 实际发布与早期运行

代码 `dc4e47fe0c2b3e0dcf73dda104cdbd4a7c814f5a` 已推送，Git远端OID独立核对相同。一次归档SHA本地/远端比较VERIFIED，remote compile PASS；dispatcher已启动，真实匹配全源域CP无query兼容smoke PASS（2合成NN），两行均独立进程运行于GPU0/1。CWD/cmdline/PID/start_ticks/GPU/日志增长/角色已独立读回，完整role IDs与原L6300/U56700/V27000精确一致，未出现fold参数。运行证据与精确PID保存在本地artifact root。

早期完整可解压step记录：B00至少652次更新，B01至少269次更新；活跃loss/RDrop CE/KL有限，nonfinite loss/grad skip=0。B01首批含LOS46/PARTIAL11/BLOCKED71、locked54/degraded16/unlocked58，EQ实际off、G=1，reference30和post_sync/quality/phase/IQ补偿及reflect已核实。该早期状态不是完整预算、收敛或性能证据；持续沿同PID运行。

实际ordinary base配置也已保存：PA正常增强允许(p_pa=.14)、CFO(.35/最大.0004)、AWGN(.4/20至36dB)、多径(.18/2至4tap)、循环移位(.35/±32)等属于两行共有旧配方；class_signature关闭、p_dac=0。不将旧普通增强宣称已验证Ag/身份保持。本次native顺序会在practical前保留这些损伤，此为普通拼接参照的真实行为。运行时practical_buffer_input/output与execution_fast均false，按实际耗时报告，不追改健康运行来加速。

截至本次读回，未完成两行训练、正式七视图评分及跨seed确认，未宣称urban>60%或clean/星地双收益。


## B00 完整预算与独立评分

B00 已完成 E20/4440，固定七视图各168000样本，独立truth-last评分，126个混淆矩阵复算一致；B01仍健康训练，run未完成。

B00是同4L成本的clean重复参照，不是原两前向F；不能把额外前向收益当成As收益。固定FINAL20，未按源V最佳轮选择。最终源V98.118519%只作健康诊断。

| 视图 | Accuracy | Macro-F1 | 最差RX accuracy |
|---|---:|---:|---:|
| clean | 84.950595% | 84.668105% | 74.929167% |
| practical_high | 47.655952% | 42.971204% | 40.008333% |
| practical_mid | 45.714286% | 41.481029% | 38.733333% |
| practical_low_suburban | 43.495238% | 39.692490% | 36.962500% |
| practical_high_urban | 46.239881% | 41.981278% | 38.920833% |
| practical_mid_urban | 38.278571% | 35.544042% | 32.920833% |
| practical_low_urban | 32.222619% | 30.502712% | 28.120833% |

完整解析20轮/4440步、5994条loader记录，L暴露568320/U396900，唯一ID仍6300/56700；活跃目标有限且无跳步，完整stdout无错误标记。200次nan文本来自禁用的diagnostic/test占位。1704960个实际SupCon锚均有跨RX同TX正例。U选择比例不是伪标签正确率，真值始终隐藏。

父训练成本0；新增20epoch/4440update。source NN3661082（student train2670180/eval594002、ownEMA eval396900），L前向2273280，source墙钟1052.837400秒。固定测试NN1176002（含2合成smoke），墙钟48.098098秒。共享兼容smoke另2合成NN，历史E50只作兼容参考，不初始化。离线分析0NN/0IQ/0CP/0更新。

分析代码commit `936d96a695219f00d6eec677aa5d0bfb3c798465`；3项聚焦计算/边界测试与编译PASS，已push且独立远端OID相同。产物 `E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_concat_20261017/B00_analysis_r01`：summary、逐row/seed/RX/day/TX CSV、20轮曲线CSV、accuracy/training/RX图PNG/PDF；远端同row `/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-concat-e20-s2026092701-m2-r01/B00-s2026092701/analysis_r01`。固定测试未保存逐样本state/quality，不推造LOS/lock测试分层。

目前仅seed2026092701的B00，尚无As增量、Ag效用、urban>60%、suburban接近clean或跨seed复现结论。其余行保持原配置。


## 两行正式七视图结果

完整源域两行均已完成E20/4440、固定七视图各168000、独立truth-last共252组指标。普通As相对无As urban+25.250992pp、low+20.020833pp、clean-2.761905pp。单seed，未达clean双收益/low60/全suburban2pp，预固定RAW行仍运行，目标分数不回流。

| 视图 | B00无As | B01普通拼接As | B01−B00（pp） |
|---|---:|---:|---:|
| clean | 84.950595% | 82.188690% | -2.761905 |
| practical_high | 47.655952% | 80.405357% | 32.749405 |
| practical_mid | 45.714286% | 77.935714% | 32.221429 |
| practical_low_suburban | 43.495238% | 75.495238% | 32.000000 |
| practical_high_urban | 46.239881% | 77.063690% | 30.823810 |
| practical_mid_urban | 38.278571% | 63.186905% | 24.908333 |
| practical_low_urban | 32.222619% | 52.243452% | 20.020833 |

B01三郊区距自身clean差1.783333/4.252976/6.693452pp，只有high≤2pp。最差RX：clean73.0625%、midurban56.279167%、lowurban47.079167%；low宏F1为51.650902%，clean宏F1为81.705228%。urban均值由38.913690%至64.164683%。

两行均全新scratch20/4440，无父训练；B00/B01 source墙钟1052.837400/1999.742716秒。每行source NN3661082（L2273280），七视图prediction NN1176002。合计新增40modelE/8880更新、9674168网络样本量；共享兼容smoke另2合成NN。本地1540合成NN/2开发更新另列。

这说明真实practical训练有很大适应收益，同时clean退化2.761905pp，尚无联合突破。较小suburban差距中包含clean下降。无逐样本formal物理状态，不能归因为单一多径/CFO/门控因素。原RAW行在这些target分数读取前已锁定并启动，其他健康行/参数/预算/默认头不变。


## B01 完整日志及物理覆盖

完成B01全部20轮/4440步、5994条采样记录及568320条practical物理元数据解析；126正式混淆矩阵独立复算一致。全部90个TX×RX×scene及90个TX×RX×state组合均覆盖。0额外NN/IQ/CP/更新；曲线和RX图已取回核验，单seed/目标未达。

训练覆盖：LOS285384/PARTIAL89591/BLOCKED193345；locked341069/degraded95159/unlocked132092；AGC限幅61/568320。所有568320条均EQ关闭、reflect边界、总SNR未知。残余核长62、数值滤波延迟24样本；256点输入的实际有效观察区及边界对分类的因果影响仍未识别，不能由核长直接推断准确率损失。

实际signal_to_added_noise_db减white-input质量代理均值+0.029652dB，标准差0.482165dB；1%/99%分位-1.299783/+1.234072dB，极值-19.387220/+3.841197dB。该差异本身不证明锁定状态切换的因果影响，需固定接收状态的受控实验才能解释；本轮未冒称已完成。残余输出CFO的1%/99%分位-972.330797/+955.525805Hz，不据此预设主因。

E20平均clean CE0.272667、sat CE0.788595、RDrop KL0.040506、SupCon1.618511（权重.1），源V97.637037%。活跃目标均有限，非有限跳步0，全部1704960有效SupCon锚有正例。E20预算耗尽不等于卫星视图收敛。U最终选择673/56700，真值隐藏，不能报告伪标签precision。

分析实现 `4d451e6b09cab511b318b923b458d2c28ca0dd06`；真实产物 `E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_concat_20261017/B01_analysis_r01`，远端同row `/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-concat-e20-s2026092701-m2-r01/B01-s2026092701/analysis_r01`。曲线及RX热图已检查，标题排版已修复。0新增NN/真实IQ/CP读取/训练更新。两个row均ANALYZED，完整三行比较待RAW完成；不扩seed、不调整方案。


## 预固定 RAW 对照最终汇总

RAW已按原计划完成E20/七视图/独立评分，三行13320步/60轮/采样5994条每行均完整且顺序一致，无留域。RAW vs B01 urban+0.228175pp、low+0.235714pp、clean+0.514286pp；RAW仍比B00 clean低2.247619pp，low52.479167%，未达联合目标或跨seed确认。完整结果、逐RX/day/TX表、共同B00 clean正确集合保持率、间隔、修复/破坏计数、覆盖及训练曲线见相邻run `20261017-phase1-FcrossRX-fullsource-rawpractical-e20-s2026092701-m1-r01` 的report.md。真实图/CSV/summary `E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_rawpractical_20261017/comparison_analysis_r01`，远端 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-rawpractical-e20-s2026092701-m1-r01/comparison_analysis_r01`；全部378混淆矩阵复算VERIFIED。未重测、未从正式成绩调参，未启动新候选或扩seed。
