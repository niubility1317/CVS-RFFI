# F＋跨 RX 同 TX：旧循环移位旁路的匹配 E20 源域消融

状态：RUNNING／VERIFIED。两组已从零启动，尚未完成训练或源评分。唯一假设：在所有训练成本与原始监督份额匹配时，只旁路旧普通 IQ 循环移位能提升未见源 RX clean，同时保护六个 practical。这是旧普通配方修正，不是新的时延增强，也不是已成立的 A_g。

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
