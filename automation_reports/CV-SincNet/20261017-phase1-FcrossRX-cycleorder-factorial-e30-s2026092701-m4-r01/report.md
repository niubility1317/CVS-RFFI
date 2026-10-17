# 固定周期IQ重排×真实practical：全源scratch E30四因子

状态：RUNNING，已核实四行真实训练；正式预测和评分尚未完成。父训练成本为0，四行均从头E30=20L+10U/222step每轮/6660更新，modelseed2026092701。用户最新E30覆盖旧E20/E50要求；F＋跨RX同TX对比、全部原RX1/3/4/6/8、L6300/U56700隐藏TX/V27000、split392005/7-63-30不变，不切L_dev、不留RX。每行NN/墙钟/教师等实际计数独立报告。

唯一假设：原80:160四个20点周期的phase-transported S4打乱能抑制周期位置偶然依赖，并在真实传播前保留可用身份信息。P30后置源FAIL保持；P31同算子前置源兼容4/4单元通过，原信道重放误差0，未证明泛化或训练收益。算子、窗口、相位估计、均匀24排列及seed202610173001全固定，不扫描强度/概率/窗口。

四行均为.5 ordinary(x) primary＋.5 secondary，分别B00=raw、B10=Ag(raw)、B01=As(raw)、B11=As(Ag(raw))。原普通增强私有流20261016051只作用primary；secondary使用同物理L的loader RMS raw IQ。raw是数据集原float32 RMS后波形，不是更换数据或反演TX。四个raw-secondary控制必须新训练，不能混用旧ordinary-secondary B00/B01冒充因果对照；旧普通拼接E30保留为直接外部参考。历史rawAsE20负结果不改写，当前原始输入行只是新固定Ag的匹配控制，不把更换预算称为旧候选复活。

Ag只用于合法L，不读TX来选排列；physical eq/RX/day/sig/base构成不透明ID，私有trainL/epoch*/step*/secondary命名流随view刷新；同包重复采样保留并用occurrence区分view，不删样本。两次RDrop使用同一已生成concat IQ；同生成physical输入在B10/B11对应。U原EMA/旧门控/原弱强增强保持，U TX仍隐藏，无新增UAg。单个As前向走原native residual/post_sync/noEQ(G=1)/ref30/fs25MHz/fc2.45GHz/reflect，receiver seed2026092701按RX/day/eq跨TX稳定，view seed202610172201。quality/补偿/噪声/传播/AGC依赖不拆；无额外postAs RMS，无第二次As。

实际部署F=SIG_TIME_X3＋CENTER_25K/CENTER_UNITS＋joint160/defaultcos30/NoExplicitCFO/原内部补偿，900722参数/206state；既有RDrop1/LS.01/SupCon只primary(time_down/t2.pool,temp.2,max.1,warm5/ramp10)联合训练。CE与原RDrop分别归一每半后.5相加；每行L网络包3409920，Ag无额外NN，B10/B11各852480额外IQ变换；原U学生、EMA和源V计数实测。无新损失、分类器、诊断头或部署结构。

训练先真实全源FINAL20兼容CP/2合成包no-query smoke，绝不加载为训练init；PASS立即进入scratch。epoch30固定后每行先真实ownFINAL30/2NN smoke，再固定V1800＋2draw×3原source cached full视图，共12602NN，0新信道/训练/teacher更新；保存读出后自动预测原clean＋全部六practical，每row全部7prediction固定后独立进程truth-last评分。任何源科学FAIL仍完成固定四行正式矩阵；不因部分row测试结果改变其他行。

唯一源扩种证据判据：B10完整Vclean≥B00+.5pp，B11完整Vclean≥B00，且B11相对B01在每个2draw×2urban均≥+1pp；1e−10pp数值容差。report先写source_matrix_decision，再读formal；FAIL关闭本候选，不调窗口/相位/排列群/概率/RX/seed/轮次救跑。不自动扩seed，单modelseed不证明重现。源V覆盖全RX并非独立目标跨RX证据，正式target只用于终态报告。

正式比较默认头accuracy、macroF1、最差RX、3urban均值、low/midurban、3suburban与自身clean差；四因子主效应、joint净收益和交互B11−B10−B01＋B00。探索目标为clean净正、low>60%、各suburban差≤2pp，目标不是保证，不回流调参/晋级/重跑。额外源CM/分类margin按cos尺度30、固定父clean正确集合保持率及state/quality分层在同row分析中保存；不新增模型前向或拟合头。最终训练曲线、负结果、缺项与成本全部保留。

唯一owner/root；普通N607，sharedlock下每GPU总experiment进程≤3且free≥12000MB含其他owner，2CPU线程；run/output /home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01 独占。命令/CWD/GPU/paths/stoprule见experiment.json。协议角色/非有限/步骤/实际结构/输入顺序/预测不完整/outputcollision等直接技术失败保留ownpartial，停止未发本run行，不动健康旧任务、不自动重试。已通过4个synthetic routing/protocol-negative tests和compile，0真实IQ/CP/NN；一次限定新增接线独立P0/P1静态审查PASS，未发现会使真实运行错误/越权/破坏/不可启动/非法预测的问题；无真实IQ/CP/target或远程调用。一次发布archive比较/remotecompile/启动后独立PID-CWD-argv-GPU-log增长检查。N607preflight VERIFIED：普通无特权、全源句柄一致、formalVALIDATED_ONCE、own新根不存在、8GPU可用容量将再由dispatcher准入。

当前无本轮训练或性能结果，原正式ordinary拼接E30 clean82.797024%/low53.137500%仍只是只读参考；源62%不能替代正式>60%。联合目标ACTIVE。

## 启动实证

实际代码 commit 0c47f4bcd49c5df43093579c7f33e6c35bc5245d，GitHub独立比较与远程branch相同。一次archive本地/远端SHA比较及remote compile PASS。dispatcher PID577854；4worker分别B00 PID577955/GPU1，B10 PID578438/GPU2，B01 PID578519/GPU3，B11 PID578604/GPU4。独立/proc的PID/start/CWD/argv/CUDA及nvidia-smi核验一致，log由6449字节增长至B00/B10约16760字节、B01/B11约11605字节。最后保存状态B00/B10 E2/444更新、B01/B11 E1/222更新，failure为空；此为进度快照，不是收敛或性能结果。

实际resolved结构900722参数、20label＋10pseudo、完整L6300/U56700hidden/V27000、held_RX=null、target_loaders0，训练模块实际来自不可变cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code，非顶层code。原子模块channel.py、batch.py、residual.py与实用adapter实际__file__均保存各row/source/module_files.json。Smoke兼容CP只2合成包/0真实IQ/0query/0训练初始化；各row另外严格保存ownscratch初始化。

当前无本轮正式准确率、F1或跨seed结论；未完成项为各row E30、12602NN源读出、7view预测固定、独立truth-last评分、完整独立统计/曲线/分层图/收益归因和跨seed验证。运行过程不会依target结果改变其他行或健康旧任务。

## 独立分析准备

分析实现为 analysis/cvs_cycleorder_factorial_E30_results_20261017.py，准备 commit f95ec6e3ebd45b3a71c674a9ba844766cc3b870d，对应3项synthetic protocol-negative/数值检查已PASS。分析覆盖四个新E30模型的全部结构化30epoch/6660step/普通增强/原生物理/SupCon/Sinc/完整stdout记录，不以tail推断训练表现；全矩阵prediction/scoring完成后才允许分析truth，独立源判据先复算落盘。四因子accuracy和macroF1交互、原普通secondary参考、固定P25 B00 clean正确集合及新matchedB00集合的保持率分别报告；不把两种输入控制混同。

成本单列每row source/teacher/EMA/额外L/U/预测/源读出，shared compatibility smoke2NN、P30/P31源前置诊断和只读P25训练成本，不冒充训练父权重继承。实际全部结果后才导出PNG/PDF/CSV/summary并核验图；当前尚未对本轮数据运行独立分析、尚无本轮正式准确率或可复现结论。
