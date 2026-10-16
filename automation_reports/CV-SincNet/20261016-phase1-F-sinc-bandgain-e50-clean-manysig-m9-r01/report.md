# F 三路＋统一 Sinc＋RDrop：目标 clean 三种子优化

状态：VERIFIED_RUNNING。代码bc913013已推送核对远端，6个新F模型均已完成E1/50，3个历史F基线已完成核验复用；尚未产生新候选目标结果。本轮直接使用测试集目标域衡量性能；源域仅检查合法性、运行和特征更新，不作性能晋级门槛。

固定9行：FULL复用A–F r02的F-s2026092701/02/03完整final50及其clean默认头prediction/独立评分；NORM和BAND各新增相同3seed，从零50轮=33L+17U、每轮222步、每行11100更新。共6新模型/300模型轮/66600更新，历史F150模型轮单列。所有三seed在任何新target结果出现前固定并提交，不读新目标成绩决定扩种子，不按成绩停止、重排、改参数或重跑。

基座严格为F：三个独立局部投影及SIG_TIME描述子共同进入实际joint160，不投票、不加MEAN_VAR/COV。完整网络900722参数，部署f_theta232033，默认cosine头6×160/scale30共960参数，原联合训练；无换头、冻结后训头或RX诊断头。

新机制只改变Sinc的Adam自适应位移：FULL原统一步；BAND按w_i=max(初始有效带宽_i,Fs/K)/Fs逐通道缩放，Fs25MHz、K79，初始w约0.012658至0.166913；NORM用正全局倍率，使自身输入步的自适应L2等于BAND反事实L2。该公式在未训练初始化上固定，不扫描倍率。NORM不匹配不同轨迹或包含物理衰减的总位移；浮点加法舍入差单列。

保持p=p0+25000q、q初始0、lr_q=alphaFs/25000、eps_q=epsilon25000/Fs、物理Hz全局clip5；按原cosine alpha先且只先施加q-=alpha lambda(q+p0/s)，nativeqWD0，再Adam更新，最后只缩放自适应增量，origin物理衰减不缩放、不漏项、不双重施加。student/ownEMA/推理/导出同坐标，沿用实际叶版本的滤波缓存失效。

原完整源角色保持L6300/U56700隐藏/V27000，RX1/3/4/6/8、day1/2/3、原6TX、split392005，原7/63/30及物理role_ids不变，不使用sourcefold。不继承任何旧权重作为新模型初始化；所有父代码仅复用09554c3a的实现。原RAW/普通增强/LS0.01、RDropalpha1两随机前向共用同次实际增强IQ、旧batch_neighbor门控/ownEMA/cos及NoExplicitCFO保持；无星地训练增强，不解封U TX。每新模型963900隐藏U观察，共5783400。

逐行实际训练完成后，固定final50默认头先产生168000个目标clean预测；prediction完整后另起CUDA为空的CPU进程连接truth独立评分。自动执行训练→冻结→预测→评分→9行汇总，不停在源域或等待再次测试授权。FULL3行只复制并核对既有物理ID一致的prediction与18分层独立评分，不重复训练或预测。新目标预测1008000个，逻辑9行1512000个，共162分层记录。

主要比较NORM−FULL、BAND−FULL、BAND−NORM；报告逐seed准确率/Macro-F1/最差RX及day、配对修复和新增错误、差值均值/样本SD(ddof1)/正差数、参数和时间成本，全部负结果保留。现有F的84.270238%±0.275383个百分点只是匹配基线，不是新机制结果。实际有效必须以新目标均值及配对差判断；源V涨分或公式正确均不能代替目标收益。该基准已暴露，不宣称新的独立盲测、新TX、Phase2或卫星部署收益。

无源性能资格门槛，三seed均已固定。原未启动的F源资格方案20261016-phase1-F-triple-sinc-bandgain-sourcefold-s2026092701-m3-r01已标记SUPERSEDED_BEFORE_LAUNCH/0远端启动。已运行的单路源诊断20261016-phase1-sinc-bandgain-rdrop-sourcefold-s2026092701-m3-r01继续，不热改、停止或重训，且不替代F目标结论。

代码身份为发布Git提交；命令CVS-RFFI/bin/python -u experiments/cvs_f_band50/runner.py，CWD PROJECT/releases/cvs_F_sinc_bandgain50_clean_20261016_r01，root PROJECT/runs/20261016-phase1-F-sinc-bandgain-e50-clean-manysig-m9-r01，精确逐行配置见experiment.json。唯一launchowner，CPU dispatcher无CUDA；共享跨owner容量锁，每GPU含preCUDA最多3总进程且启动前≥12000MiB。

最小发布：一次archive传输比较、remotecompile、首先对真实已完成F checkpoint做无query smoke并立即继续；每新行自己的scratchcheckpoint另有原no-query smoke。一次独立下一真实运行P0/P1审查PASS。启动后独立PID/CWD/argv/GPU/log增长核验。技术停止仅数据/来源/角色、非有限、预算/输出/合法prediction问题；保留产物、停止新pending，健康worker继续，无分数停止或重复提交。

CPU/CUDA真实F检查PASS：各8个协议负例，三个独立SIG_TIME分支/默认头形状、物理origin衰减一次、同真实梯度下其他body/W首步一致，NORM局部范数正确。测试只用合成IQ，0源/目标IQ读取，0另训头。运行见证每新行保存E1/17/33/50各13步，Sinc梯度/位移、实际clip、EMA核和移除W的joint160导出均检查；无额外训练前向或更新。完整原始日志/权重/向量/预测私有保存，Git提交实现、配置和必要结果。

发布后核验：archive传输、remotecompile为VERIFIED，真实F checkpoint no-query preflight PASS；6个worker的PID/CWD/argv/GPU/log增长均已独立核验，实际900722参数、SIG_TIME_X3、50=33+17、fullsource6300/56700/27000且无sourcefold。每行E1有222实际RDrop/默认头联合更新。详见run_evidence.json。

首轮真实更新独立复算：6新模型各13步，共78步，物理origin衰减及自适应增量的最大误差均0Hz；NORM局部范数FP32舍入差最大0.000784Hz。只读取保存的更新见证，无额外模型前向或源/目标IQ读取；这是执行证据，尚非目标性能结果。
