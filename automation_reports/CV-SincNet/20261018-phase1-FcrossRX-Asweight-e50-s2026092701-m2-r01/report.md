# As 权重配方：源证据支持的匹配总 E50 验证

状态：VERIFIED_RUNNING，两条匹配续训已实际启动；性能结果尚未完成。run_id `20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01`。

唯一假设：已固定.375 As监督份额后，再追加20轮能否改善mid／low urban并保持clean。新增B00/B01两条均为各自weighted FINAL30 student＋ownEMA继续训练20轮，总50轮／11100更新，轨迹20＋10＋20，追加段resetAdamW lr2e−4／WD1e−4／cos20 min1e−6；不是连续cos50或追加50。

推进依据在读取新E30正式分数前已固定并保存：newAs−originalAs30源fullV clean−.059259pp（≥−.1），两draw mid／low均值＋.777778pp（≥＋.5），均PASS。匹配NoAs源clean差额与交互完整报告，源通过不等于正式阶段通过。此后E30正式clean失败照实保留，不能反过来取消、改权或选seed。源决定：[source_decision.json](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e30-s2026092701-m2-r01/source_decision.json>)。

每条严格加载自身weighted30父及EMA，固定父release `cvs_FcrossRX_AsWeight_E30_20261018_r01`／commit `3befd47c8041d81f9430eda53562d0821624ff04`，核实原own20从零祖先及同L6300／U56700隐藏TX／V27000／split392005／全5源RX和day，不留训练域、不分Ldev。原始IQ256、F＋CROSS_RX／三路SIG_TIME／统一Sinc／joint160／默认cos头联合训练及NoExplicitCFO保持。

不改IQ／物理信道／RNG／As64 mask／uniform三场／U规则／loss公式：.5ordinary128＋.125mixedordinary64＋.375selected64＋.25raw128，CE与对称KL相同组均值权重，总L1.25，6L前向，ORD128 CROSS_RX.1。NoAs真实selected为raw.clone；As为As(raw64)，不把NoAs误称ordinary。ordinary冻结原20参数的epoch13，SupCon+30、信道e31至50／step6661至11100。每新行4440更新／L3409920NN，As284160（B01）或0（B00）。原始样本监督尺度保持本候选既定值，不能声称clean份额没有转移。

八逻辑条件：original／weighted × NoAs／As × E30／E50。仅两weighted50新训练，其余六行冻结prediction／score／source audit直接复用。每新row固定FINAL50后自动source、clean＋六practical预测完成后独立truth-last评分；即使源科学失败也完成正式收尾。正式数据RX／day／物理ID／receivedIQ／SNR／通道／全部包保持。报告全七场accuracy、F1、最差RX、RX/day/TX、修复／新增错误、urban均值、suburban与自身clean差及weight×As×budget交互。

正式阶段判定weightedAs50−weightedNoAs50 clean≥−.1绝对pp，并报相对原As50／NoAs50与自身30的变化。未来固定02／03扩种支持条件：As50−ownAs30源fullVclean≥0pp且两draw mid／low均值≥＋.5pp。任一失败放弃固定配方扩种；不以正式分数改权、选种、epoch或救跑。不保证超过60%，单seed不宣称重复双提升。

唯一launch owner `codex/root/practical-Asweight50`。普通N607既有CVS-RFFI/PyTorch FP32／CPU2，无环境安装或admin；CWD `PROJECT/releases/cvs_FcrossRX_AsWeight_E50_20261018_r01`，命令 `CVS-RFFI/python -u experiments/cvs_practical_as_weight50/runner.py`，不可变输出 `PROJECT/runs/20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01`，logs在该root/logs。GPU由共享队列分配，每GPU≤3个全部owner实验进程、free≥12000MB；健康任务不动。实际code/config commit在发布时由release_commit.txt及submit记录HEAD，后续回填，不做自引用hash链。

必要核查：68聚焦负例、19py与两段remoteAST已PASS；唯一P0/P1审查PASS，P0=0／P1=0。launcher首步两个真实父checkpoint4NN及两实际6L1536NN共1540NN、0更新、无query，PASS即训练；一次N607 preflight／单release归档本远端SHA比较／remotecompile和启动后PID/CWD/argv/GPU/log增长。停止仅限本run非法数据／父/head错误／非有限／成本或系统技术错误，保留产物且不自动重试；科学负结果不是技术停跑理由。

成本分别记录父30／增量20／累计50、student／teacher、train／eval／source／formal／smoke与实际墙钟；不把并行时长相加当历时。预期所有source／prediction／独立score／日志／0NN分析／报告与登记闭合。精确矩阵、路径和输入见[experiment.json](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01/experiment.json>)。

实际发布代码 `64176e7ff1aa5cd2477a504f8b442bb7966079dc`，远端分支OID独立一致。单archive传输比较与19py远端编译VERIFIED。dispatcher 1276091；NoAs worker1276185／GPU0，As worker1276263／GPU1；PID／CWD／argv／CUDA／nvidia／log增长独立VERIFIED。首步真实双父4NN＋两行6L1536NN共1540NN smoke PASS、0更新、无query后正常训练。source roles EXACT_MATCH、ownweighted30EMA严格恢复，resetAdamWcos20空optimizer，实际physics同原row。首记录6L／As0或64／groupweights.125/.375／total误差<2e−8；ordinary来源仍原20参数epoch13。0NN分析1277501在WAIT_SOURCE，CWD／argv／空CUDA已核实，truth未开；analysis单archive比较VERIFIED／compile＋10checksPASS。新增段进度属于历史快照，不等于训练或评分完成。
