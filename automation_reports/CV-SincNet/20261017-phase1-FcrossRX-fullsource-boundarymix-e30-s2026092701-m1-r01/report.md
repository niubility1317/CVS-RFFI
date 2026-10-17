# 完整源域 F＋跨 RX：E30 practical 边界混合负结果

状态：ANALYZED。本行训练、原七环境预测、独立 truth-last 评分、完整日志分析、冻结源域机制诊断及独立复算全部完成。边界混合候选不成立为识别优化方案；总体 Ag＋As 联合目标保持 ACTIVE，未完成。唯一矩阵与数据/seed 记录在 experiment.json。

本行相对同预算 B01，urban 均值下降 1.112302 个百分点，low urban 下降 0.914881 个百分点，clean 下降 0.971429 个百分点，七个环境均下降。源域的边界压力敏感性确实降低，但原 full practical 的保持条件失败。按预登记源解释条件关闭该候选，不改比例/seed/阈值或扩大预算救跑；未扩模型 seed。

## 固定基线与真实执行

最新授权为从零 E30，覆盖目标文本中旧 E20。固定 F：三路 SIG_TIME_X3、CENTER_25K／CENTER_UNITS Sinc 修正、原内部门控及参考响应、真实 joint160、NoExplicitCFO、默认 cosine scale30；900722 参数、206 个 state 项。骨干和默认头正常联合训练，未增加部署分支、替换或另训分类头。gain 是特征融合系数，后续 joint 与类别权重均 L2 归一化；它不是可训练 logits 温度。

本轮仅检验 As 训练组织，Ag=false。B00/B01 复用已完成同 seed 的独立 scratch E30 终局产物，不作训练初始化；新增一行 BCTX-s2026092701。B00=ordinary(x)＋同 IQ 精确重复；B01=ordinary(x)＋全 native practical(ordinary(x))、训练 reflect；BCTX 仅将 L 卫星视图的 native 边界按奇 reflect／偶 edge 交替。无 B10/B11，不能计算或宣称 Ag/As 四格交互。B00 是共有旧普通增强的同4L控制，不是完全无增强或原2L控制。

全部原源 RX1/3/4/6/8、day1/2/3、6TX、L6300／隐藏 U56700／V27000、split392005 和7/63/30保持。没有额外 L_dev 或留出源 RX。训练逐物理ID核对原角色，目标 loader=0，U 标签未开放。原 IQ 长256、25MHz、equalized1；保持原中心裁剪/补齐、center=False、逐包复 RMS。旧普通增强后进入 practical 的原 unit-RMS 输入处理和 native AGC，无新增 As 后 RMS。

训练30轮=20轮仅L＋10轮L和原U，每轮222步，6660更新；原 ownEMA／旧门控／cosine日程保持。L 的 clean/satellite 两半 CE/KL 各0.5归一化，RDrop alpha1、label_smoothing=.01，两次前向复用同一已生成 IQ，共4L。clean 半上的原跨RX同TX SupCon 保持 temperature.2、max.1、warm5/ramp10，无额外前向；未增加新一致性、课程、对抗、Fishr/MixStyle或SSL。所有合法有标签卫星包均参与 CE。

旧普通增强实际保留，不能把它或 RDrop 称为新增 Ag 证据。运行快照 E1/E30：p_dac=0/0，p_pa=.014/.049，p_cfo=.035/.1225，p_awgn=.04/.14，p_multipath=.018/.063；这些是各快照时 inner 的当前概率，非累计均值。完整配置和30轮实际概率在 source/ordinary_base_config.json 与 mechanism_progress.json。三行共有此配方，不据本行结果改动健康旧任务。

practical 运行值：residual/post_sync，EQ=false、G=1，reference_snr_db30，fs25MHz/fc2.45GHz，quality adaptation／phase tracking／RX IQ compensation 配置开启，实际补偿仍随 locked/degraded/unlocked 状态执行。保留几何、遮挡、传播、噪声、质量和补偿依赖；不反演原录音未知TX硬件。RX硬件按RX/day/eq session固定，不由TX定义。训练场景 low_suburban/mid_urban/low_urban 原nominal均匀，原私有普通增强seed20261016051、native viewseed202610172201、RXseed2026092701不变，逐epoch/view刷新；正式测试数组/SNR/reflect及困难样本不变。

实际 leo_practical.channel/batch、cvsrffi.practical_adapter、DataAugmentation 和 SSDG.train_ssdg 的 __file__ 均指向既有不可变依赖 releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/，不是顶层 code/。完整绝对路径及 resolved/native args 已保存在运行 source/ 和本地 delivery_metadata.json。

每轮 reflect/edge 各111批；最终各426240卫星包，共852480。180个TX×源RX×场景×边界单元全部有支持，原90个TX×RX×场景及90个TX×RX×状态单元均覆盖。BCTX/B01/B00 三组两两比较各8880条 L/U 序列记录，0不一致。仅边界及其 native AGC 总响应改变，未将固定AGC控制波形作为训练数据。

## 正式逐 row／seed 结果

以下为同一模型seed2026092701、固定168000物理样本、checkpoint内默认头的准确率（%）。三行是方法对照，不是三个模型seed。每行七视图全部prediction固定后才由独立exec scorer接触truth；378个总体及RX/day/TX分层混淆矩阵由终局分析再次核对一致。

| 环境 | B00 | B01普通practical | BCTX边界混合 | BCTX−B01（百分点） |
|---|---:|---:|---:|---:|
| clean | 85.156548 | 82.797024 | 81.825595 | -0.971429 |
| practical_high | 46.513095 | 81.469048 | 80.166667 | -1.302381 |
| practical_mid | 44.677976 | 79.041071 | 77.848214 | -1.192857 |
| practical_low_suburban | 42.714286 | 76.586310 | 75.410119 | -1.176190 |
| practical_high_urban | 45.233333 | 78.176786 | 76.894643 | -1.282143 |
| practical_mid_urban | 37.582738 | 64.160714 | 63.020833 | -1.139881 |
| practical_low_urban | 31.985714 | 53.137500 | 52.222619 | -0.914881 |
| urban三场景均值 | 38.267262 | 65.158333 | 64.046032 | -1.112302 |

| 主要保持/识别指标（%） | B00 | B01 | BCTX |
|---|---:|---:|---:|
| low urban宏F1 | 30.218017 | 52.496507 | 51.180382 |
| low urban最差RX准确率 | 27.700000 | 47.995833 | 46.262500 |
| clean最差RX准确率 | 74.433333 | 75.012500 | 71.008333 |

BCTX相对B00 clean下降3.330952pp。其high/mid/low suburban距自身clean分别1.658929／3.977381／6.415476pp，未实现三个场景都接近clean的≤2pp探索目标，且不是通过保持clean达到的。原B01普通As相对B00仍是已验证主要收益：urban+26.891071pp、low+21.151786pp，clean−2.359524pp；不能把该共同收益归给新增边界混合。

相对B01，BCTX在low修复6742包、破坏8279包，净少1537；clean修复2971、破坏4603，净少1632。固定B00 clean正确143063包集合上，low保持率B01为58.020592%，BCTX为57.406877%。默认头真类最难竞争者间隔除以30：clean .129577→.115058，mid .057697→.054287，low .032490→.030409。没有用logit尺度变化冒充间隔收益。完整逐RX/day/TX、F1、最差RX、配对修复/破坏在 analysis_r01/ 的CSV及JSON，不以少量层代替全体结果。

## 冻结源域机制：敏感性下降未转化为识别保持

在正式成绩产生前已固定的补充机制计划：三个同契约 FINAL30 默认头先各做2包合成strict-load smoke，再打开原合法源缓存；只使用原V1800，每TX×RX×day单元20包，覆盖所有90单元。复用已完成P24的同一两信道draw、三场景full/edge成对IQ，不重抽信道。edge固定原接收状态、补偿与AGC，是受控边界效应，不是重算接收机的总效应。训练则使用native重算AGC，二者解释边界明确区分。

源缓存实际载入V27000和L6300（L不消费）、U0，源panel仍是原V1800而非新留出训练域。两个draw是同一模型seed下的信道重复，不是跨seed。每模型23402NN（其中2为smoke），合计70206；0训练/头拟合/teacher更新，所有206项state逐项不变，0目标IQ/成绩/truth访问。独立读回再次核对1179个源混淆矩阵及配对/固定clean集合计数一致。

以下源准确率仅用于机制诊断，绝不能当正式low urban达到60%。每个场景n=3600指V1800的两个draw，非3600独立物理包。

| 源条件 | B01 full | B01固定edge | BCTX full | BCTX固定edge |
|---|---:|---:|---:|---:|
| practical_low_suburban | 90.444444 | 78.750000 | 90.027778 | 88.944444 |
| practical_mid_urban | 76.305556 | 66.500000 | 75.111111 | 74.916667 |
| practical_low_urban | 62.555556 | 53.611111 | 62.055556 | 60.416667 |

源clean：B01 97.722222%，BCTX 97.888889%；源小面板上这3包差别没有证明跨RX clean泛化。正式clean反而下降0.971429pp。

| 源解释单元 | edge惩罚改善（pp） | 原full变化（pp） | 原定条件 |
|---|---:|---:|---|
| practical_mid_urban/draw0 | 9.944444 | -1.444444 | 失败 |
| practical_mid_urban/draw1 | 9.277778 | -0.944444 | 失败 |
| practical_low_urban/draw0 | 8.000000 | -0.555556 | 失败 |
| practical_low_urban/draw1 | 6.611111 | -0.444444 | 通过 |

预先固定条件要求四单元edge惩罚均改善≥3pp且各自full下降≤0.5pp；仅1/4通过，整体false。pooled mid边界损失由−9.805556pp降至−.194444pp，low由−8.944444pp降至−1.638889pp，但mid原full下降1.194444pp，low下降.5pp。不能通过只报受控edge收益、平均掉失败draw或放宽保持阈值宣称有效。

| 原full源分层 | n | B01 | BCTX |
|---|---:|---:|---:|
| practical_mid_urban/<3 | 383 | 34.986945 | 29.503916 |
| practical_mid_urban/8-15 | 293 | 62.457338 | 58.361775 |
| practical_mid_urban/BLOCKED/unlocked | 678 | 38.495575 | 34.365782 |
| practical_low_urban/<3 | 1127 | 31.499556 | 31.144632 |
| practical_low_urban/LOS/locked | 1060 | 96.037736 | 95.377358 |
| practical_low_urban/BLOCKED/unlocked | 1565 | 34.952077 | 34.504792 |

质量分层是signal_to_added_noise_db，不是总SNR。没有证据表明新增混合缓解了原低质量BLOCKED/unlocked困难；tiny strata（如1至2包）仅描述，不作泛化结论。正式数组没有逐包state/quality sidecar，未编造目标LOS/lock分层准确率。

## 分支、融合、梯度与实际物理统计

沿真实前向hook读取原time/frequency/PA/reference/signature。low full源观测clean→sat余弦B01→BCTX分别为 .968483→.981182、.984573→.985033、.861898→.853626、.338347→.337898、.371764→.345874。time更接近clean仍伴随识别下降，不能以特征接近替代身份收益。90组分支统计与原160维PA/reference/signature静态增益保存；没有虚构逐包门控，也未强迫全部embedding/分支相等。

原训练总/骨干梯度遥测E1约2.773117、E30约1.507146，6660步无非有限loss/grad跳步；默认头weight、Sinc坐标和三SIG_TIME投影均实际移动且保持可训练。未开启逐分支训练梯度/冲突遥测或梯度手术，因此日志中inactive conflict=0不是“无梯度冲突”的证明。本行未追加eval梯度前向。

训练全852480条物理记录已读取。LOS/PARTIAL/BLOCKED为427829/134726/289925，locked/degraded/unlocked为511572/142274/198634；这些潜在状态计数与B01相同。62tap残余核、数值延迟24；原检查确认195个不依赖人工前缀的内部点。BCTX AGC限幅80包（B01为79）。质量白输入代理均值16.001663dB，实际signal/added-noise均值15.919530dB，二者差均值−.082133dB；并非总SNR，原录音噪声未知。EQoff，未把均衡噪声着色机制套到G=1。本轮不改正式接收机或信道。

## 成本、验证与执行状态

| 行／seed2026092701 | 新增轮/更新 | source网络包 | source墙钟秒 | 正式预测网络包 | 预测墙钟秒 |
|---|---:|---:|---:|---:|---:|
| B00 | 30/6660 | 5407922 | 1540.012254 | 1176002 | 47.865980 |
| B01 | 30/6660 | 5407922 | 3026.314576 | 1176002 | 48.893565 |
| BCTX | 30/6660 | 5407922 | 2980.434024 | 1176002 | 58.672418 |

本轮新增BCTX，父训练0；复用历史B00/B01成本60轮/13320更新，新行30轮/6660，三行累计90轮/19980，不能冒充三seed。各行 source网络包5407922 = student_train3976920＋student_eval864002＋teacher_eval567000，teacher_train0；L网络包3409920，L/U loader实际852480/567000、唯一L/U6300/56700。每行预测1176002包含2合成包。launcher另有2包同契约E20真实CP兼容性smoke，随后丢弃，不是初始化或训练父权重。

冻结源诊断另计70206NN、CPU2墙钟44.693185秒；终局正式分析及源独立复算均0NN/IQ/CP/更新。分析/图表后处理未单独计墙钟，不虚构其精确运行时长。预算用尽不等于收敛；本轮没有延长或按测试分数重跑。

三行均完整30轮/6660步/6660条Sinc更新、5550步active SupCon、各8880条L/U序列；每行1198行训练stdout无错误标记。300处NaN文本来自inactive测试/域等诊断项，全部active CE/KL/SupCon及真实更新已逐步检查有限，0跳步。U truth_available始终0，门控选择率不当作伪标签精度。

本地实际验证：4项边界/协议测试PASS（8配置、6call、4namespace、7物理配置负例；48个合成native调用，0真实IQ/CP/NN/更新）；2包真实网络合成前向/反向验证900722参数、206项与B01初态一致，默认头/Sinc/三投影梯度有限，0优化更新。另7项完成后正式分析测试、3项源机制测试（13角色/控制负例）、4项源复算测试均PASS。首次直接-m缺少拆分legacy导入依赖，沿原已核实本地装配解决；本地Torch/NumPy导入警告保留，未使用ndarray桥接。compile、一次限定P0/P1审查、真实CP no-query smoke均PASS，无额外重复审查或hash链。

实际本地测试命令（CWD本工作树）：

```text
E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/local_checks.py
E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/local_model_checks.py
E:/type10-7/local_envs/ssr-gpu/python.exe -m unittest analysis/test_cvs_fullsource_E30_results_20261017.py analysis/test_cvs_boundary_E30_results_20261017.py
E:/type10-7/local_envs/ssr-gpu/python.exe -m unittest analysis/test_cvs_boundary_E30_source_20261017.py
E:/type10-7/local_envs/ssr-gpu/python.exe -m unittest analysis/test_cvs_boundary_E30_source_readout_20261017.py
```

普通N607预检VERIFIED，一不可覆盖发布归档的一次local/remote SHA比较VERIFIED、remotecompile PASS。唯一训练owner/root、GPU0、每GPU≤3含其他owner，未修改健康任务。训练dispatcher447326/worker447354及后处理475906/475907/477987均已自然退出，队列空、0failures；预测、评分、两类分析completion独立读回VERIFIED。

## 文件、命令与未完成项

训练实现 experiments/cvs_fullsource_boundary26/，代码commit 8cf7ceed44f9818390332cebd9f4c34c3b8b7967；复用不可变全源E30实现commit bafba40e365d0914a2e86f861fb55a49f1cd3997，不继承其权重。正式分析/配对脚本 analysis/cvs_boundary_E30_results_20261017.py（81e69b1befd0cb184056834ebfe8e4323876b5f9）；源只读脚本 analysis/cvs_boundary_E30_source_20261017.py（06bbb078d250eabfc4771c1162535e041d853ac6）；源独立复算/图表 analysis/cvs_boundary_E30_source_readout_20261017.py（c583c7154c4f98c12063137783bdedc57583afcd）。均已推送并核对远端OID；此前暂时连接失败已恢复，原提交未改写。

训练CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_boundarymix_E30_20261017_r01。实际原命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_boundarymix_E30_20261017_r01/experiments/cvs_fullsource_boundary26/runner.py
```

真实分析命令、PID/CWD/argv和代码commit在本地 terminal_analyses_launch.json、source_readout_launch.json，实际resolved/native args、__file__、规范化、边界first-batch和完整物理/序列/训练曲线在不可变远端run：/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-boundarymix-e30-s2026092701-m1-r01。本地完整配置与终局证据：E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/delivery_metadata.json。IQ/CP/原始训练输出不入Git。

本地实际产物：E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/analysis_r01/ 含逐row/seed/RX/day/TX CSV、全部90条训练曲线记录、analysis_summary、378矩阵复算证据和6张PNG/PDF；E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/source_readout_r01/ 含源分层CSV、summary、1179矩阵复算证据和4张PNG/PDF。10张PNG均已目视检查；PDF同源保存。实际图表链接：[正式七环境](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/analysis_r01/formal_accuracy.png)、[全部训练曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/analysis_r01/training_curves.png)、[逐RX](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/analysis_r01/receiver_accuracy.png)、[边界覆盖](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/analysis_r01/training_boundary_support.png)、[源边界敏感性](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/source_readout_r01/source_boundary_sensitivity.png)、[源质量分层](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/source_readout_r01/source_quality_accuracy.png)、[源状态分层](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/source_readout_r01/source_state_accuracy.png)、[真实静态增益](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_boundarymix_E30_20261017/source_readout_r01/source_static_gains.png)。

最终回答：本候选缓解的是人工边界压力的源敏感性，未缓解原正式urban退化；默认头相对普通As的low下降.914881pp、urban均值下降1.112302pp，clean额外付出.971429pp，相对同预算B00共付出3.330952pp。当前1个模型seed，不能判断跨seed复现，更不能声称可重复正收益。low>60%、clean净正收益、全部suburban接近clean、有效Ag以及B10/B11/四格交互均未完成。当前候选按源保持条件失败关闭；总体目标不缩减、不标complete，后续Ag必须先建立新的合法源域依据，不能从本次目标分数反向调参或选择性重跑。
