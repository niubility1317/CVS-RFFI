# F＋跨 RX 同 TX：轻度接收频偏独立视图 E20 源对照

状态：ANALYZED_SOURCE_ONLY／VERIFIED。两组 scratch E20、全部七个源视图及独立完整分析已完成。固定轻度频偏候选未通过源条件，关闭固定配方。前一固定常数增益候选因clean+0.555556pp未达+1pp、全部practical下降而关闭，不调幅度/seed/RX救跑。当前唯一科学假设：保持旧ordinary/PA、原U、raw固定监督和默认头，独立轻度接收频偏L视图能改善未见RX clean且保护六practical。

## 已有依据与固定矩阵

频偏范围只沿用actualordinary .0004×native E13 cap scale .2506003339349741，即±.00010024013357398965cycles/sample；未按source practical或target挑幅度。单独工程r02已完成真实匹配CP/162合成NN/0真实IQ，正负频偏有实际默认函数作用、逆变换恢复；只支持开展训练检验，不证明标签保持/身份收益。其r01 stdlib命名遮蔽技术失败已保留，实际0NN，不是科学负结果。

| row | raw监督 | 共有ordinary | 第三槽 | 预算与自动读出 |
|---|---|---|---|---|
| B00-RX1-s2026092701 | .5 | .25旧完整ordinary，PA保留 | .25 raw | scratchE20/4440，final20源clean＋6practical |
| B10-RX1-s2026092701 | .5 | 相同.25旧完整ordinary | .25 frequency(raw) | 同预算/前向/数据/初始化规则 |

两row固定fit RX3/4/6/8×day1/2/3×全部6TX，实际L5040/U45360/V21600；heldRX1原L1260仅final20读出，不参与训练/EMA/归一化或选checkpoint。不是完整L6300或L5580/Ldev720。复用已验证carrier/source fold及原角色，U真值隐藏；前13轮仅L，后7轮L＋原U，所有20轮L监督继续。V健康诊断不选模，checkpoint固定最后20轮，0target。

F三路SIG_TIME、统一Sinc、joint160、默认cosine头联合训练、NoExplicitCFO及内部参考逐包补偿保持，无新增部署分支/头/诊断分类器。三对RDrop alpha1/LS.01，每step6个L前向；每对复用同一生成IQ。固定raw.5和oldordinary.25监督，第三槽及对应前向控制显式登记。CROSS_RX仅旧ordinary两前向的time_down/t2.pool，temp.2、maxweight.1、E1至5零/E6至15ramp；raw/frequency/U不新增对比/一致性。

新频偏每包/step私有CPU seed20261016401+step对称uniform；x[n]exp(j2πfn)，n0至255，不居中，不改RMS/窗口/幅度，不加noise/multipath/PA/DAC/CFO身份坐标或As；全部128合法L参与监督，不做高置信过滤。NoExplicitCFO屏蔽coord27，内部参考补偿不删，不将频偏当TX身份或真实RX替换。原ordinary完整抽样/PA/原生min(epoch,13)日程、原U/ownEMA保留。

## 原定判断、成本与停止

全部同时通过才支持独立源确认：B10−B00 heldRX1 clean≥+1pp、clean MacroF1≥0、每个六practical准确率≥−.5pp、seenV≥−.5pp；两row完整E20/4440更新/3409920LNN、3330非零SupCon/13320层位前向项、最终LCE≤.5，B10频偏包观测568320/B00为0，功率最大相对误差≤1e−6、0As训练/头拟合/target。科学失败关闭固定频偏配方，不改频偏/seed/RX/阈值或追加预算救跑。单seed源PASS不是Ag跨域或联合目标完成。

各模型scratchE20，不加载训练父权重；唯一真实CP来源仅dispatcher no-query smoke2，为自有satmix25 B00/seed01/scratch20/4440/final20、无祖先/target，实际四RX角色已核实，strict metadata/payload/load复查；不用于worker初始化/teacher。ownEMA由当前scratch student产生。新run预期总40模型轮/8880更新；每组3409920LNN另实际原U/ownEMA/V/2scratchsmoke/8820held，先前同配方每组4686678NN但以实际计数为准，若一致含公共smoke2为9373358。B10另568320个频偏IQ生成，B00 raw复用，两rowNN视图数匹配。先前工程162NN、其local324NN及本实现两次check4624NN/4合成更新分开记录，不当成真实E20训练。

每row训练后即自动冻结/源clean＋6practical读出，保留全部场景accuracy/F1/CM及TX/day/repair-break/suburban-ownclean差距；全部预测完成后独立复算。hard predictions无logits，运行时meanCE只引用，不冒称独立CE复算。源筛选不构造target，完整源正式四因子/交互项和多seed联合目标仍缺。

唯一owner codex/root/F-CROSS_RX-frequency-E20-20261016，普通N607账号、CPUdispatcher2线程、worker沿用共享容量锁/每GPU最多3个全部owner进程/启动前≥12000MiB空闲，健康任务不改/不停/不重启。精确逐row配置、命令/环境/CWD/路径/六类seed见experiment.json。一个Git实际code/config commit、一个不可覆盖root/一个N607 preflight/一个archive单次SHA比对/远端compile；启动后独立PID/CWD/argv/GPU/log增长读回。

角色/来源/初始化/非有限/实际预算/无法产生合法预测为技术失败，保留产物、停止pending，不干预健康worker、无自动retry；科学低分不早停。首步实际CP no-query smoke2 PASS立即继续，无许可artifact/额外审批。代码/登记/小汇总Git交付，IQ/CP/逐包预测/原日志保留私有。13负例、同初始化、旧B00损失/头梯度精确控制、sixforward/sameIQ/ordinary-onlySupCon/defaulthead jointupdate/export160/privatefrequency/canonical算子与直接脚本回归PASS，一次限定P0/P1审查PASS；尚无实际训练或识别收益。


## 实际发布与启动读回

执行commit `bb38d5ef6dfcd3e60e7fb6d00c9c2fca976bfd66`；本地与远端分支OID独立一致。archive单次比对、远端compile及真实匹配CP no-query smoke2 PASS；dispatcher/两worker PID-CWD-argv-GPU与日志增长VERIFIED。实际seed2026092701、epochs/label/pseudo20/13/7、sourcefoldRX1、无baseline/teacher权重、Ag仅B10/0As均核对。

- B00-RX1-s2026092701：GPU1，完整step387/4440、E2/20，stdout11662字节。
- B10-RX1-s2026092701：GPU2，完整step372/4440、E2/20，stdout11663字节。

上述是运行进度，不是身份收益。独立全量分析已准备，只在全部声明预测完成后执行；不修改已发布代码，不据中间/目标分数调配方。联合目标仍未完成。


## 完整源结果与原定判断

两worker及dispatcher自然退出，completion齐全、无failure；未重启、热改或选择性重跑。每组完整E20/4440更新/3409920LNN、3330非零SupCon/13320层位前向项，全部17640默认头源决策、完整epoch/step与L/U物理序列及frequency值独立重算。meanCE只引用冻结运行记录，hard predictions不含logits，不冒称独立重算CE。

| 固定 held RX1 源视图 | B00 准确率 | B10 准确率 | 差值 | B00 Macro-F1 | B10 Macro-F1 | 修复/新错 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 81.746032% | 81.746032% | +0.000000pp | 81.404741% | 81.476500% | 9/9 |
| practical_high | 39.841270% | 35.952381% | -3.888889pp | 29.525164% | 27.007268% | 18/67 |
| practical_mid | 39.603175% | 37.063492% | -2.539683pp | 30.339465% | 29.072466% | 19/51 |
| practical_low_suburban | 35.714286% | 33.809524% | -1.904762pp | 28.416547% | 27.027594% | 21/45 |
| practical_high_urban | 39.126984% | 37.380952% | -1.746032pp | 30.292077% | 29.030755% | 21/43 |
| practical_mid_urban | 32.380952% | 30.952381% | -1.428571pp | 27.295015% | 26.162243% | 26/44 |
| practical_low_urban | 28.333333% | 28.253968% | -0.079365pp | 24.822521% | 24.833965% | 33/34 |

| suburban 与自身 clean 绝对差 | B00 | B10 |
|---|---:|---:|
| practical_high | 41.904762pp | 45.793651pp |
| practical_mid | 42.142857pp | 44.682540pp |
| practical_low_suburban | 46.031746pp | 47.936508pp |

原预登记条件：{"healthy": true, "clean": false, "clean_macroF1": true, "practical_protection": false, "seen_V": true}。必须全部同时通过；不删除任何practical保护失败项，不改变clean门槛。

- B00：最终LCE 0.277215628，seenV 98.050926%；实际NN 4686678，墙钟1273.478秒，完整stdout 800行；新增频偏包观测0，范围Nonecycles/sample，均值0.000000000、RMS0.000000000，恒功率关系最大相对误差0。
- B10：最终LCE 0.273580104，seenV 98.023148%；实际NN 4686678，墙钟1208.891秒，完整stdout 800行；新增频偏包观测568320，范围[-0.0001002401186269708, 0.00010023963113781065]cycles/sample，均值-0.000000202、RMS0.000057904，恒功率关系最大相对误差2.38418579e-07。

两个row的完整L/U物理序列一致，原ordinary/PA和13轮截断日程保留；U TX真值可用及非有限跳步均0。同IQ RDrop、六L前向、总损失权重逐step重算；0As训练、0另训头/部署分支/训练父权重。

真实成本为40模型轮/8880更新，各3409920LNN另实际原U/ownEMA/V/held；本E20总实际NN（含公共realCP smoke2）为 9373358。EMA继承计数2已扣除。先前独立输入工程162NN另行累计，不含在上述E20总数内。本实现两次局部检查4624合成NN/4合成更新、先前工程local324NN/0更新分别另列，均0真实IQ。没有target预测或调参；工程作用PASS不是身份保护证明，源V健康或LCE下降也不替代识别。

| 分层准确率差 | clean | high sub | mid sub | low sub | high urban | mid urban | low urban |
|---|---:|---:|---:|---:|---:|---:|---:|
| TX0 | +0.476190pp | +1.904762pp | +2.380952pp | +0.952381pp | +0.952381pp | -0.952381pp | -0.476190pp |
| TX1 | -0.476190pp | +0.000000pp | +0.476190pp | -0.476190pp | +0.476190pp | +0.476190pp | +1.904762pp |
| TX2 | +0.952381pp | -12.857143pp | -9.523810pp | -8.095238pp | -7.619048pp | -8.095238pp | -3.333333pp |
| TX3 | +0.952381pp | -15.238095pp | -10.952381pp | -8.571429pp | -7.142857pp | -2.857143pp | -4.285714pp |
| TX4 | +0.000000pp | +4.761905pp | +3.333333pp | +6.190476pp | +6.190476pp | +4.285714pp | +5.714286pp |
| TX5 | -1.904762pp | -1.904762pp | -0.952381pp | -1.428571pp | -3.333333pp | -1.428571pp | +0.000000pp |
| day1 | -1.428571pp | -3.809524pp | -0.238095pp | -1.190476pp | -1.666667pp | +0.000000pp | -1.428571pp |
| day2 | +0.952381pp | -6.428571pp | -4.761905pp | -2.619048pp | -3.809524pp | -2.619048pp | +0.476190pp |
| day3 | +0.476190pp | -1.428571pp | -2.619048pp | -1.904762pp | +0.238095pp | -1.666667pp | +0.714286pp |

固定轻度频偏训练候选失败关闭，不改range、seed、RX、阈值或追加预算救跑。实际函数对频偏敏感，没有转化为本次全部预定身份识别收益；不将工程敏感性、健康或局部正差冒称Ag有效。

正式完整源B00/B10/B01/B11及交互项、冻结后目标clean+六practical、多seed联合确认仍缺失；lowurban>60%和suburban接近clean的联合目标未完成。


## 结果解释与下一唯一假设

本次clean修复9包、新增错误9包，净0；lowurban修复33包、新增34包，净−1包。TX4六practical均有局部正差，但TX2/TX3在全部六practical下降，其中highsub分别−12.857143/−15.238095pp。这是本次错误的重新分配，不是联合收益。最终运行字段final_L_CE下降、seenV健康没有改变原判定。两组完整物理L/U序列相同、每组4686678实际NN、总9373358（含新smoke2），没有目标接触。

下一唯一拟检验：旧ordinary的循环移位是否造成可避免的身份错误。实际DataAugmentation._apply_time_shift用模256 gather，整batch事件门控后逐包抽integer shift；native13截断下max3至8，人工拼接缝位于边界外侧、没有落入80:160参考窗，但窗内内容对齐改变。代码现象不能证明身份受损；不预断固定窗已破坏。它区别于已失败的新增分数延时视图，也不重启频偏/相位/gain路线。

最小下一实验需独立登记：自有匹配B00/seed01/final20冻结CP，只读fit四RX的原L5040、全部6TX×3day；raw一次，实际native epoch配置E1至13各做（warmup强度可重复，逐epoch独立固定draw）fullordinary/同draw跳过shift两视图，共136080默认头决策＋2合成CPsmoke=136082NN、0更新/头拟合。E14至20实际重复E13强度，不外推。保留原生shift抽样和所有其他外生随机draw，canonical full输出及最终RNG必须一致；counterfactual仅返回shift前IQ，下游输入依赖的power/noise自然随其变化。保留所有物理样本，不改practical/test IQ，不读heldRX1/U/V/target。当前只是拟定，未实现、登记或启动。

固定放弃条件拟为：无法匹配canonical完整输出/原生RNG终态，则技术无效、无身份结论；实际非零shift观测上的净默认头修复不足2pp，或不足3/4 fitRX有正修复，则关闭该普通增强修正方向，不换draw/seed/RX救跑。完整池及实际非零shift子集都报告，不能隐藏未改善样本。源fit修复即使成立，也只支持另行预登记匹配E20；之前PA源修复没有转化为E20收益，不能重犯相同推断。普通增强修正尚不等于新Ag或联合目标。
