# F＋跨 RX 同 TX：旧循环移位旁路的匹配 E20 源域消融

状态：ANALYZED_SOURCE_ONLY／VERIFIED。两组scratchE20、全部七个源视图及独立完整复算已完成。固定循环移位旁路候选未通过源条件，关闭固定配方。唯一假设：在所有训练成本与原始监督份额匹配时，只旁路旧普通 IQ 循环移位能提升未见源 RX clean，同时保护六个 practical。这是旧普通配方修正，不是新的时延增强，也不是已成立的 A_g。

前项冻结源fit诊断已完整复算：3946个非零shift观测，150修复/16新错、净恢复+3.395844pp；全65520增强观测+0.204518pp，4个fitRX均为正。其模型已经见过fitL，不能由此推断训练后的跨RX收益。历史PA旁路曾有更大fit恢复却未转成E20 clean收益，本次仍须训练反证；微频移、常增益、PA旁路及分数时延的固定失败配方不救跑。

## 数据、基线与矩阵

唯一网络基线 F＋跨 RX 同 TX 对比学习：三路SIG_TIME、统一Sinc、真实joint160、NoExplicitCFO及必要的内部参考补偿、原cosine默认头随骨干联合训练。沿用源响应time_down/t2.pool的既有SupCon，temp0.2、weight0.1×clip((epoch−5)/10)，只用L同TX异RX正对；同TX同RX排除、异TX负对。无新增分支、损失、分类器或头拟合。

固定source RX1-held fold和modelseed2026092701，两组独立从零训练E20，各4440更新；无父权重、resume、外部teacher或未知祖先，EMA从自己的scratch学生复制。匹配旧B00只用于dispatcher的真实CP无query合成smoke2包，不载入训练。原source carrier/source fold角色复用已验证结论，无builder重验；实际训练L5040/U45360/V21600来自RX3/4/6/8×day1/2/3×6TX，heldRX1原L1260只在final20冻结后作源选择读出。U TX真值始终隐藏，V只作健康和已登记保护，不选epoch。不是完整L6300或L5580/Ldev720；无target访问或目标调参。

| row | 普通L增强 | raw份额 | ordinary份额 | 第三槽 | 训练 | 自动源读出 |
|---|---|---:|---:|---|---|---|
| B00-RX1-s2026092701 | 原生完整增强 | 0.5 | 0.25 | raw0.25 | scratchE20 | final20 clean＋六practical |
| B10-RX1-s2026092701 | 同抽样，只旁路循环移位输出 | 0.5 | 0.25 | raw0.25 | scratchE20 | final20 clean＋六practical |

每个物理L的三个视图各做同IQ两次随机子网络前向，共6次；RDrop1/LS0.01/SupCon、AdamW lr0.0002/weight_decay0.0001、既有CENTER_UNITS Sinc与cosine日程、原13L+7U、U/EMA/旧门控全部匹配。13L+7U表示后7轮仍训练L，同时启用原U；不隐式缩小clean监督份额。默认头正常联合更新，不冻结骨干后重训。

## 唯一增强差异与预算

原顺序DC/taper→circular shift→gain/phase/CFO/Wiener/multipath/AWGN→DAC→PA。NoExplicitCFO保留，DAC p=0、class_signature=false、PA normal原样开启。每步两arm各只调用一次原生normal augmentor，仅L128显式normal调用进入wrapper，U原生bound methods不受更改。两arm均执行真实_apply_time_shift gather及整数抽样；B00返回gather结果，B10返回shift前输入，再执行后续整条管线。噪声按各自信号功率自然重算，不强行匹配噪声振幅、不追加RMS、不做第二次星地信道。训练E1..20实际configure min(epoch,13)，最大shift样点3至8，E14..20保持E13强度，不能外推到20强度。

每次实际shift触发，两arm都观察同一CPU/CUDA整数draw：保存原native draw后的状态，短暂重放整数采样验证gather，然后恢复native末状态；额外scalar draw数独立计量，没有额外NN。完整记录每步shift整数、批gate、非零及IQ变动mask、实际PA strength/触发、输出与raw的MSE/功率、实际日程，用于独立检查两arm外生抽样是否匹配。不把568320个eligible ordinary包写成全部shift实际施加。

两模型共40modelepochs/8880更新。每arm 4440L步×128包×6=3409920L NN、568320普通增强生成，加实测原U学生/EMA、V、各自scratch smoke、held源8820(1260×7)。dispatcher另有匹配旧CP2合成前向；父训练没有复用作初始化。前向instrument计数去除EMA复制学生eval计数2的继承偏移，实际总数训练后复算。无As训练/额外头拟合/额外部署模块。局部验证13056合成增强、2312合成NN、2合成优化步、17负例、0真实IQ或target，成本与正式训练分开。

## 固定判断与产物

固定final20，不按V、held或任何target选择epoch。每row独立完成即自动源clean及六practical读出，不等另一row。源评估使用heldRX1全部L1260物理包和原固定实用信道配置/随机流，所有困难包保留。完整prediction固定后独立source scorer重新计算7视图accuracy/MacroF1/CM及repair/new break、TX/day分层；CE由运行时默认头报告，当前硬预测不含logits，因此不能声称独立重算CE。

晋级须全部满足：B10−B00 clean≥+1pp，clean Macro-F1≥0；每个六practical差≥−0.5pp；训练RX的V差≥−0.5pp；双方E20/4440步、原SupCon3330有效步、3409920L NN、每步一个ordinary调用、无As/头拟合/target、final L CE≤0.5且无非有限跳步。达不到即关闭固定shift-bypass配方，无seed/RX/draw/强度/阈值救跑。PASS仅支持独立源确认，不自动扩seed、全源训练或target，不宣称A_g/DG/As或联合目标成功。正式B00/B10/B01/B11、交互项、多seed、target clean＋六practical及lowurban>60%/suburban接近自身clean仍未完成。

唯一owner `codex/root/F-CROSS_RX-shiftbypass-E20-20261016`。普通N607用户，CPUdispatcher2、loader0，GPU按既有共享capacity lock选择，所有owner每GPU最多3进程且发布时至少12000MiB空闲，不停止或热改健康任务。CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_F_crossRX_shiftbypass_sourceRX1_e20_20261016_r01`。dispatcher命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u experiments/cvs_shift_bypass20/runner.py`；worker追加 `--worker <row_id>`。精确路径、角色、六类seed、输出和矩阵见experiment.json；代码身份以实际Git commit/release_commit/resolved为准，发布后回填。

已完成一次P0/P1审查PASS；本地normal输出、两arm CPU随机末状态、B00旧损失/头梯度逐位相同、B10独立same-draw control、6forward/sameIQ/默认头更新和export160验证PASS。一次N607 preflight，单release归档本远端SHA对比、remote compile；dispatcher第一步真实CP无query smoke，PASS立即继续。启动后独立PID/CWD/argv/GPU/log增长核实。协议/初始化/预算/输出碰撞/系统非有限或无法产生合法prediction属技术失败，保存partial并取消pending，健康worker自然完成，无自动重试；负分不停止、重启或改配方。

保存每row实际args/resolved/roles/init/frozen、私有checkpoint、完整stdout与20epoch日志、全部4440步native/view/Sincunit/shift轨迹和L/U物理序列、固定七视图prediction、result/completion或failure。训练后独立复算全部轨迹和17640源预测，给训练曲线/分层图/全部负结果；IQ、权重、逐包轨迹和日志不入Git，小型报告汇总入Git并推送。


## 实际发布与启动读回

执行commit `5476e2238d6d43408227decf916f7177a02f2129`，本地固定提交已核实；首次GitHub连接失败已恢复，使用系统既有代理完成push，远端分支OID独立读回与执行commit一致。归档单次比对、远端编译、匹配真实CP smoke PASS；dispatcher和两worker的PID/CWD/argv/GPU及日志增长VERIFIED。两行实际seed均2026092701，epochs/label/pseudo=20/13/7，source fold RX1，无baseline/teacher权重，实际shift旁路/Ag/As标志与登记一致。

- B00-RX1-s2026092701：GPU0，已观察完整step250/4440、E2/20；stdout11683字节。此为运行进度，不是最终性能。
- B10-RX1-s2026092701：GPU1，已观察完整step217/4440、E1/20；stdout6516字节。此为运行进度，不是最终性能。

独立全量分析程序已准备并编译，只有全部声明预测完成后才执行。不修改已发布训练代码，不重复提交同run，不基于中间或目标分数改变配方。联合目标仍未完成。


本地调用环境问题在正式发布前解决：直接使用原生 Windows PowerShell 5.1和可运行的既有SSH客户端，补齐当前子进程缺失的标准环境；未修改系统配置。此前失败均发生在本地连接检查或GitHub访问层，没有远端训练提交、CP前向或重启。实际release只提交一次，训练配置和代码5476e2238保持固定。最近独立读回两worker均进入E4，前3轮各666步的shift gate/非零计数一致；此为运行核验，尚无完整源七视图结论。


## 完整源结果与固定判断

两worker与dispatcher自然退出，completion齐全，无failure、重启、热改或选择性重跑。每组E20/4440更新/3409920LNN、3330有效SupCon步/13320层位前向项；全部17640默认头源决策、20epoch日志、4440步native/view/Sincunit/shift、L/U物理序列与实际13截断日程独立复算。meanCE引用运行时默认头记录，hard predictions不含logits，未声称独立重算CE。

| held源RX1视图 | B00准确率 | B10准确率 | 差值 | B00 Macro-F1 | B10 Macro-F1 | 修复/新错 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 81.666667% | 82.063492% | +0.396825pp | 81.326326% | 81.709708% | 8/3 |
| practical_high | 39.126984% | 38.333333% | -0.793651pp | 28.835668% | 27.457676% | 55/65 |
| practical_mid | 38.809524% | 37.063492% | -1.746032pp | 29.626437% | 26.467528% | 48/70 |
| practical_low_suburban | 35.317460% | 34.761905% | -0.555556pp | 27.941117% | 26.336153% | 45/52 |
| practical_high_urban | 38.492063% | 37.380952% | -1.111111pp | 29.503436% | 27.559561% | 61/75 |
| practical_mid_urban | 31.746032% | 31.825397% | +0.079365pp | 26.718790% | 25.708636% | 46/45 |
| practical_low_urban | 28.492063% | 27.222222% | -1.269841pp | 24.657720% | 22.885326% | 25/41 |

| suburban与自身clean绝对差 | B00 | B10 |
|---|---:|---:|
| practical_high | 42.539683pp | 43.730159pp |
| practical_mid | 42.857143pp | 45.000000pp |
| practical_low_suburban | 46.349206pp | 47.301587pp |

预登记条件：{"healthy": true, "clean": false, "clean_macroF1": true, "practical_protection": false, "seen_V": true}，所有条件同时成立才晋级。不能删去任一失败视图、降低clean或用健康检查代替识别结果。

- B00：finalLCE=0.277217768，seenV=98.041667%；实际NN=4686678，墙钟=1249.471秒，完整stdout=800行。ordinary eligible生成568320、调用4440，实际shift批gate284、非零shift33497、gather IQ变动33497；B10旁路返回全部ordinary调用的eligible包0，不把此数称实际施加shift。额外draw观测36352个scalar；ordinary真实输入MSE复算最大差2.38418579e-07。
- B10：finalLCE=0.263099430，seenV=98.037037%；实际NN=4686678，墙钟=1337.337秒，完整stdout=800行。ordinary eligible生成568320、调用4440，实际shift批gate284、非零shift33497、gather IQ变动33497；B10旁路返回全部ordinary调用的eligible包568320，不把此数称实际施加shift。额外draw观测36352个scalar；ordinary真实输入MSE复算最大差2.38418579e-07。

两组逐step整数shift、非零/变动mask、gate、有效概率/最大shift、PA触发和strength匹配：{"shifts": true, "nonzero_bits": true, "shift_changed_bits": true, "shift_gate_calls": true, "effective_p_time_shift": true, "effective_p_pa": true, "effective_max_time_shift": true, "PA_triggered_bits": true, "PA_strength": true}。原L/U物理序列完全相同，U真值可用和非有限跳步均0；同IQ RDrop对、六L前向和目标尺度逐步复算，无As训练、头拟合或新部署模块。

正式训练共40模型轮/8880更新，含dispatcher真实CP smoke2包，总实际NN=9373358。L/U学生、EMA、V/学生评估与held8820分别计量，EMA复制的eval计数偏移2已扣除。每arm568320native ordinary生成；局部合成13056增强/2312NN/2更新和先前冻结shift诊断136082NN/0更新单列，不冒充真实训练成本。无训练父权重或target访问。

已否定这项固定shift-bypass训练配方的晋级依据；冻结fit源的+3.395844pp恢复没有证明训练后的共同收益。关闭该配方，不扩seed，不换RX/draw/范围/阈值或追加训练救跑。clean、practical、类别和suburban差距按实际全表解释，不能通过错误重新分配宣称联合成功。

| 分层差值 | clean | high sub | mid sub | low sub | high urban | mid urban | low urban |
|---|---:|---:|---:|---:|---:|---:|---:|
| TX0 | +0.476190pp | -0.476190pp | -2.857143pp | -1.428571pp | -1.428571pp | -2.857143pp | -2.857143pp |
| TX1 | +0.000000pp | +0.000000pp | -0.476190pp | -0.476190pp | +0.000000pp | -0.952381pp | -0.476190pp |
| TX2 | +0.476190pp | +23.333333pp | +20.000000pp | +17.142857pp | +26.666667pp | +14.761905pp | +8.095238pp |
| TX3 | +0.476190pp | -22.857143pp | -21.428571pp | -14.285714pp | -26.190476pp | -11.428571pp | -12.380952pp |
| TX4 | +0.000000pp | -0.952381pp | -0.476190pp | +0.000000pp | -0.476190pp | +3.333333pp | +0.476190pp |
| TX5 | +0.952381pp | -3.809524pp | -5.238095pp | -4.285714pp | -5.238095pp | -2.380952pp | -0.476190pp |
| day1 | +0.000000pp | -0.952381pp | +0.000000pp | +1.190476pp | +0.000000pp | -1.190476pp | -2.857143pp |
| day2 | +0.000000pp | -3.571429pp | -4.523810pp | -4.285714pp | -3.333333pp | -1.190476pp | -1.190476pp |
| day3 | +1.190476pp | +2.142857pp | -0.714286pp | +1.428571pp | +0.000000pp | +2.619048pp | +0.238095pp |

[完整源汇总](source_analysis.json)包含每视图CM、指标和全部成本；完整曲线、七视图/差距图、TX/day分层图及全部逐包证据保留于私有 `local_artifacts/F_shift_bypass20_20261016/final`，已渲染核对。正式target clean＋六practical、四因子交互和多seed联合目标仍未完成。


识别收益的分配也不支持晋级：highurban中TX2为+26.666667pp、TX3为-26.190476pp；lowurban中分别为+8.095238pp和-12.380952pp。六practical的Macro-F1全部下降，包含midurban准确率微升但Macro-F1下降。不能把这类类别间错误转移称为更通用的身份提取。三个suburban与clean差距全部扩大。finalLCE由0.277218降到0.263099、V基本不变，并未改变科学FAIL。


## 下一唯一假设：保留全部 baseline raw 监督，再独立添加 As

[raw监督份额代码核对](../../../analysis/cvs_clean_supervision_mass_audit_20261016.md)确认旧六前向As比较同时把raw总权重0.75改为0.50；它不能解释因果，却使“原raw份额固定”的既有表述需要限制。这项新假设研究训练组织，保留完整practical传播、噪声、接收状态与补偿，不重开旧固定75/25替换配方，不再旁路PA或shift。

后续最小控制矩阵拟固定同RX1源折、seed01、每row从零E20，沿用原F＋跨RX同TX、默认头、U、PA和shift。三个row都做8次L前向，原6次基线与全部raw0.75/ordinary0.25监督保持；新增一对同IQ随机子网络前向，分别为：D00额外raw但监督权重0的成本控制；D_raw额外raw、权重0.25的重复监督/目标尺度控制；D_As额外A_s(raw)、权重0.25。D00目标总权重1.00，后两组1.25，不能重新按视图数平均而稀释原监督。原baseline损失及每物理样本归一化、额外前向对随机状态和BatchNorm的影响须在实际实现中明确控制并记录。额外0权重仅属于明确预登记的成本控制，不引入无用教师或部署分支。

0.25取自既有As监督系数，不从本次practical或目标分数选值。D_As所有合法L的卫星视图参与监督，不要求urban学生高置信。As只施加一次于raw，不把旧ordinary的AWGN/multipath再叠入As。原有RDrop仍是同IQ随机子网络配方，不能冒称IQ泛化增强。

训练前固定的探索条件拟为：相对D00，D_As clean和clean Macro-F1均不下降，urban均值至少+3pp、lowurban至少+5pp，每个practical和V不低于−0.5pp；相对D_raw，也要求urban均值至少+3pp、lowurban至少+5pp，并保护clean/Macro-F1/每practical/V不低于−0.5pp。所有row完整E20、原数据权限、实际8次L前向与监督成本、无非有限跳步等健康条件也须成立。任一失败即关闭固定候选，不换seed/RX/系数/场景概率或追加预算救跑。PASS仅支持独立源确认，不等于A_g或联合目标成立。

这只是下一候选的明确方案，尚未实现、登记或启动。训练命令、实际配置、BatchNorm/随机状态与8前向实现、本地验证及唯一P0/P1审查将在该新候选的正常最小流程内完成，不对当前已结束run追溯改配方。已重新读取另一对话，其当前实验为局部平滑IQ衰减；本方案研究监督组织，方向不同。正式target数据不参与设计。完整A_g/B00-B10-B01-B11、交互项、多seed和最终星地目标继续保留。


权重口径补充：拟议D_As在保留baseline各损失绝对系数后，总目标为1.25；归一化相对份额为raw0.60/ordinary0.20/As0.20，不能写成raw相对份额不变。D_raw的raw总绝对权重1.00、ordinary0.25，同样总1.25；D00为raw0.75/ordinary0.25、总1.00。这是显式的监督重加权与新增视图组织，不是新的物理增强算子。AdamW和梯度裁剪下，原raw绝对系数保留并不保证实际clean保护，必须由三个控制row的识别结果回答；旧固定替换配方仍保持负结果和关闭状态，不做系数扫描。
