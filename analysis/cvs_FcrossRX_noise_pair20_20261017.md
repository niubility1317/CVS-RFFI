# F＋跨 RX：正负噪声配对 As 的匹配 E20 源域训练

本轮唯一问题是：在相同 practical 边际分布、监督权重和前向预算下，正负新增噪声的相关性是否让训练后的默认分类头获得 urban 收益，同时保护 clean。上轮冻结eval CE诊断两seed方差下降29.946%／31.790%，只支持检验采样理由，不是训练或身份提升。固定相干IQ插值及旧fixed additive As的跨seed失败均保留，不改其系数、SNR、seed、RX或epoch救援。

固定底座为F＋跨RX同TX SupCon、SIG_TIME_X3、统一 Sinc CENTER_UNITS、NoExplicitCFO、joint160、默认cosine30；无新部署分支／分类头／诊断头。三行RAW10、INDEP、ANTI都用model/globalordinary seed2026092701独立从零E20／4440更新；原13轮L＋7轮旧门控半监督、ownEMA、AdamW和cosine保持，U真值隐藏。旧ordinary原顺序／PA／shift及实际原配置三组共有，As只作用raw，不与ordinary噪声／多径叠加。

共同原三对监督绝对权重不变：raw总.75、ordinary.25，原SupCon只用ordinary两次前向的浅中层特征。额外两份IQ各用原同IQ RDrop：R=两次default头CE均值（LS.01）＋同IQ对称KL（alpha1），再平均两份R并乘.25。不同噪声IQ之间没有KL／特征对齐。总pair权重1.25、五对共10次L前向／包，全部合法有标签As直接CE，不经置信度筛选。

|行|额外两份IQ|新训练预算|L前向包数|
|---|---|---:|---:|
|RAW10|raw、raw|E20／4440|5683200|
|INDEP|同物理信道的独立noise0／noise1|E20／4440|5683200|
|ANTI|同物理信道的noise0／−noise0|E20／4440|5683200|

INDEP与ANTI共用每包／epoch／step的noise0，传播、遮挡、几何、CFO、相位、RX硬件、补偿和接收状态相同，仅noise1相关性不同。每一份从原primitive noise入口经过原RX IQ／补偿／G，再独立计算原AGC，不平均IQ、不缩小噪声、不调整reference30。训练用六种practical nominal均匀先验，25MHz／256IQ／fc2462MHz／residual／post_sync／EQoff G=1／reflect，质量仍为原白输入功率代理；新增噪声SNR不冒称总SNR，数值延迟／反射边界保持。非噪声latent逐包检查相等，正负组在处理后检查噪声仍互为负值。硬件按RX/day/eq固定，session不由TX标签定义。所有场景和状态的先验支持保留、0拒绝或困难样本删除。

fitRX3/4/6/8的有效L5040／U45360hidden／V21600，held source RX1原L1260只在固定FINAL20读出。原物理split392005和角色handles／VALIDATED_ONCE复用；不混用5580／720内部折，也不扩大可见标签。native源数据先核对原carrier，再按已验证fold过滤，训练／教师／校准／持久特征cache全部排除heldRX1。三行所有L/U物理顺序记录并完整比较。

不继承旧学生、EMA、optimizer或教师状态。dispatcher首步只用已核实相同seed、相同有效LUV契约的既有scratchE20 D_raw8L权重做2包合成no-query结构兼容smoke；随后丢弃，三条新model都从零。该旧权重来源7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b、无祖先／目标选模／另训头，已核实来源结论复用。本轮RAW10是新10L匹配控制，旧raw8L不作为性能参照。

每行训练完成后固定末轮学生checkpoint，自动在原heldsourceRX1全部1260包上读出clean＋六种原practical，保存默认头预测、margin/30、物理ID和完整physics，再由离线独立reader重算accuracy／macroF1。源评价随机流20261016103、IQ／场景／SNR不改；训练channel20261017501、noise20261017502、硬件20261016102，训练按epoch／step刷新，和源评价／正式测试分离。此为最小source-only证伪矩阵，未通过固定源规则不扩seed或进入target，不把源成绩写成正式测试。

固定科学条件：ANTI相对RAW10的clean准确率和宏F1不降、urban三场景均值≥+3pp、low urban≥+5pp；相对INDEP的urban均值和low各≥+1pp，clean和宏F1下降≤.5pp。相对两参照的每个practical、fitV均下降≤.5pp，且完整预算／覆盖／default头联合更新、最终共同base≤.5。按全部条件共同判断，不以局部状态或某个positive对照替代。失败关闭固定采样训练理由，不调系数、SNR、seed、RX、抽样数或追加epoch救援；通过仅允许独立第二modelseed源确认，不证明Ag、四格交互、正式target或low>60。

报告所有row的clean／六practical绝对值、urban均值／low／mid、宏F1、每个suburban距自身clean、固定RAW10 clean正确集合的sat保持率、各RX/TX/day/state/lock/quality／signal_to_added_noise／CFO／残余核／AGC／边界及实际覆盖。目标仍要求clean正收益、low>60%、每个suburban距clean≤2pp（探索定义），不靠降低clean缩小差距；本轮不预告达成。

新成本60model-E／13320更新；每行5683200 LNN＋原U／ownEMA／V／8820held，总预计20879874 NN＋dispatcher2合成CPsmoke，实际由计数器重算。两As行各568320物理anchor呈现、1136640噪声观测，后者不是新增独立物理样本；额外4NN每anchor。无新U/teacher视图。没有训练父权重，兼容smoke的既有E20训练成本另列。墙钟待真实完成记录，E20耗尽不等于收敛。

本地40个配置／权限负例，3858合成真实网络前向／3更新通过：三组206个state键初态相同，共同loss与全部参数梯度精确一致，CPU/Python随机流末态相同，10前向输入和每对同IQ验证，noise0位相同，原默认头正常联合更新、export160精确。0真实IQ／CP／target访问。一次独立P0/P1静态审查PASS，未新增重复审查或报告审批。

一个不可覆盖run／release和唯一owner codex/root/FcrossRX-noisepair20-20261017；共享all-owner≤3/GPU、空闲显存≥12000MiB，CPU2／loader0。技术停止仅本run数据／来源／非有限／错预算／输出碰撞／无法合法预测等系统错误，保留全部partial，不自动重试，不误停其他健康实验；负成绩按科学规则完整结束并报告。

实际代码／配置在发布前由Git固定，精确SHA记录于canonical experiment.json与本报告；发布归档只比较一次本远端SHA、一次remote compile，首个真实CP no-query smoke PASS立即继续。启动后独立检查PID/CWD/argv/GPU/log增长与真实actual_modules／args，最终独立自然终态和成本。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_noisepair_sourceRX1_E20_20261017_r01/experiments/cvs_f_noise_pair20/runner.py`。环境/CWD：`ordinary N607 CVS-RFFI Python/Torch originalfullFP32`／`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_noisepair_sourceRX1_E20_20261017_r01`。远端root：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-noisepair-sourceRX1-e20-s2026092701-m3-r01`，本地root：`E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair20_20261017`；逐row config、六类seed、日志、输出、GPU政策、停止规则和预期artifact均在experiment.json。

REJECTED_EXTRA_GATE：不增加design批准、配置／报告hash、成员hash、smoke凭证、receipt链或额外review。当前整体目标ACTIVE；尚无合格新增Ag、Ag/As四格与交互或正式目标联合证明。

独立结果读取实现已就绪：完整20轮／4440步、全部L/U顺序、两条As共2273280条噪声记录、21个源域读出及固定资格规则均独立重算；不重新加载IQ、checkpoint或拟合分类头。20个reader负例与门槛边界测试通过，新增分析前向／更新／目标访问均为0。六组科学图仅使用实际重算输出，未完成前不生成结果或性能结论。

真实发布状态VERIFIED/RUNNING：一次归档传输比较、remote compile与真实CP合成no-query smoke通过；启动后已独立核实三行进程身份、原adv3b02_xuc实际导入、空初始化、E20/13+7、default头、原physics和持续日志增长。训练发布代码仍固定ee305facac7874bfc1340291ee5360a3665621d9；之后提交的离线reader不热改运行源码。本轮全部结果待自然完成后逐row独立重算，整体goal保持ACTIVE。

逐行部分完成记录（源域，非正式目标测试）

RAW10 已自然完成 scratchE20／4440 更新，固定FINAL20默认头联合训练；原进程已退出。独立读取七份既有预测，共8820个决策（每视图为相同1260物理样本），逐类混淆矩阵重算accuracy／macroF1均与原输出一致。重算0IQ读取／0NN／0更新；不加载checkpoint或拟合分类头。

|RAW10源held RX1视图|准确率%|宏F1%|
|---|---:|---:|
|clean|79.683|79.159|
|practical_high|37.857|27.086|
|practical_mid|38.175|28.078|
|practical_low_suburban|34.762|26.387|
|practical_high_urban|38.333|28.456|
|practical_mid_urban|31.190|25.681|
|practical_low_urban|27.698|24.137|

RAW10 urban三场景均值为 32.407%；这是等10L前向控制行的源域结果，不是本轮增强收益。实际NN前向包数 6959958，其中L前向 5683200；native记录1563.797秒，整矩阵并行墙钟仍待全部完成。全部训练步／序列与分层机制将在三行齐全后由同一独立reader完整重算。

两条As仍为同一健康进程正常训练，整体状态保持RUNNING。当前不能比较增强净收益、clean代价、跨seed重复性或判断固定资格；不改变其他行的系数、SNR、场景、seed、预算或最终选模规则。

一次非阻断的实际日志回读覆盖两As前3完整epoch／666步／85248个anchor，共340992条噪声记录：noise0与非噪声latent逐条相同，ANTI加噪功率逐包相同，INDEP各份功率不同；ANTI有85247个anchor的两份AGC增益不同，符合原AGC逐视图重算，不能把处理后整IQ当作严格相反。六种原场景均已出现。此为实际采样关系核验，不是身份识别提升。
