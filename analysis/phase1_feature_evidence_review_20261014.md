# Phase1特征证据复核与当前时序实验

当前结论：优先执行用户最新附件的“覆盖范围×分支内非线性”四条件实验。原TIME_MS50有平均收益但存在覆盖、表达能力、旁路、参数化和随机流混杂；真实训练核仍保留4个线性方向，因此应拆解机制，不能直接继续加核/加分支。本轮只用默认分类头joint训练，不单独拟合诊断头；最新附件优先于先前目标文件允许source refit的条款。84%是完整三seed默认头clean均值研究目标，尚未达到。

## 证据范围与两分支

主分支 codex/report-physical-ssl-clean-delivery-20261011，本次审阅起点48325acd1f7781411cba214269fdf5353a2d6e9e，上游已独立核对一致。另一分支 codex/phase1-feature-challenge-20261013 审阅快照1f8814c046a8c68eba8a334e66a0aceb76e491e0，同上游一致，仅只读参照，不整体合并/改动其代码和实验。

复用两分支完整已提交汇总、源实际配置核验、既有全日志/原始confusion审计。当前没有重扫历史全部stdout，因此全日志PASS引用原审计，不称本次重新全量扫描。新直接证据为12份历史初始/末轮Sinc参数及optimizer状态核对、三份TIME_MS50真实source checkpoint等效核/数学fold检查。旧source权重的只读尺度撤除在当前新run中自动执行，与新训练统计分开。

完整同seed表见phase1_feature_evidence_seed_table_20261014.csv。不能跨配方合并：A为33标签＋17伪标签，B为32＋18。A基线mean80.5268%，B基线80.6454%，不能因为名字或head相同当成同基线。两者都保留历史batch_neighbor gate、EMA、cos、50epoch/11100步；数据契约L6300/U56700隐藏TX/V27000固定物理ID一致，TX6、sourceRX1/3/4/6/8×3day，目标clean168000，NoExplicitCFO。source day显示0/1/2或1/2/3应区分索引/显示编号，不据此认定实际role改变。V只读，目标不参与训练/选择。

## 默认头完整结果

|实际配方|实验|条件|clean均值%|样本SD百分点|Macro-F1均值|最差RX均值%|
|---|---|---|---:|---:|---:|---:|
|A33/17|多尺度|OLD_GATE_COS50|80.5268|2.3776|0.800024|见同row分组记录|
|A33/17|多尺度|TIME_MS50|81.6905|0.6753|0.811841|见同row分组记录|
|A33/17|多尺度|PA_MS50|79.0032|1.3202|0.786648|见同row分组记录|
|A33/17|多尺度|JOINT_MS50|81.2895|0.7917|0.807658|见同row分组记录|
|A33/17|归一化/竞争|BASE_REF|80.5268|2.3776|0.800024|见同row分组记录|
|A33/17|归一化/竞争|ORDINARY|80.3075|1.8899|0.798284|见同row分组记录|
|A33/17|归一化/竞争|RESTORE|80.2532|2.0336|0.797431|见同row分组记录|
|A33/17|归一化/竞争|COMP|80.5663|1.6194|0.799939|见同row分组记录|
|A33/17|归一化/竞争|RESTORE_COMP|80.1298|1.5387|0.795943|见同row分组记录|
|B32/18|临时episodic|LEGACY50|80.6454|2.0918|0.801381|67.8611|
|B32/18|临时episodic|EPISODE_SAME_RX|80.6204|2.2619|0.801424|68.2847|
|B32/18|临时episodic|EPISODE_CROSS_RX|80.7494|1.3278|0.802724|69.4306|
|B32/18|频谱交互|LEGACY50|80.6454|2.0918|0.801381|67.8611|
|B32/18|频谱交互|POWER_RELATIONS|76.8591|2.3940|0.761218|62.2708|
|B32/18|频谱交互|COMPLEX_RELATIONS|79.3058|0.4940|0.787188|63.6903|

多尺度A的TIME相对原基线平均+1.1637个百分点、2/3 seed正向；PA −1.5236、0/3；JOINT +0.7627、1/3，相对TIME −0.4010。四模型都降低源末轮CE，TIME seed03源V更好而target差，故源拟合不能直接解释DG收益。身份参数含W基线177025、TIME237697、PA291585、JOINT352257；classifier W960是gW，norm/gain属于fθ。

归一化/competition A原始默认头相对基线 ORDINARY −0.2192、RESTORE −0.2736、COMP +0.0395、RESTORE_COMP −0.3970个百分点。已有普通容量控制和实际time/frequency/PA/reference sidepath，整体未形成稳定默认头提升，不把负结果改名再重跑。COMP的L几何RX-withinTX下降仅部分source证据，不证明纯RX shortcut。几何现有7row非完整15row，不能把effective rank约7.65直接解释为信息塌缩，更不能把卷积核秩与joint160特征秩混用。

频率上下文B：LOCAL_FREQ79.5353%、GLOBAL_FREQ79.4298%，相对同配方B基线都3/3负向（−1.1101/−1.2157个百分点）；仅导入恢复没有加训练。历史完整9模型450epoch/99900步审计U forward9185400、接受148082（1.6121%）、U真值可用0；低覆盖是需要诊断的现象，不能以覆盖低就放松原门控。

episodic B：CROSS_RX默认头+0.1040个百分点，仅1/3正向；同预算原头证据不足。历史prototype readout+1.2863且3/3正向，只能作为特征几何诊断，不替代实际默认头、不证明新TX。临时ridge系数无持久类W，gradient进入support/query实际joint160，但额外L曝光99900/model应披露。

cross-frequency B：complex相对power +2.4466个百分点且3/3正向，但complex相对原基线−1.3397、power−3.7863；只是matched机制相对证据，默认头尚无可靠增益。complex/power均新增45504参数，所有实际时频PA/reference路径保留。

## 历史readout诊断边界

保留已完成旧诊断，不再启动新head-only训练。A RESTORE历史sourceL cosine refit为10epoch AdamW lr.001 wd.0001、默认normalize eps1e−12、scale30；B episodic/cross-frequency为20epoch Adam lr.01 wd0、eps1e−4、scale30、Xavier seed+60013、500 W更新。它们不能合并。A基线refit78.5657%，B基线refit77.7637%，本身不构成θ好坏排序。原joint头11100更新且看L/U与refit仅L500更新的预算/信息差异，不能把原头—refit差全部归为W过拟合。

A leave-RX仅冻结θ重新拟合头，θ已看过全部source RX，属于读出头leave-RX压力，非真正held-out-RX骨干DG。所有旧refit只作为历史readout证据，本轮默认head禁止另训。已有目标错误“纠正/引入”逐物理ID配对完整跨分支表尚缺；本轮source V逐分支撤除将保存该类统计，目标group分数只在独立truth-last报告，不用于架构和下一候选选择。

## 新直接机制证据

TIME_MS50三个seed×t1/t2/t3真实训练后核：每输入通道的等效输出×偏移矩阵数值秩全部4，平均effective rank3.0682至3.1574；|τ|>2能量3.4190%至4.7954%。真实权重float64折叠max误差3.33e−15。说明多滤波基能力训练后没有退化成rank1，但远偏移能源较小也不代表无判别价值，均不能单独归因于TX信息。核可折叠的是preGN线性部分，GN的组内全序列依赖仍在。

原identity Sinc的12份真实初始/末轮权重和optimizer状态：24 band参数全部0变化、low仅1/24微变，10模型23核逐位不变、2模型24核逐位不变，最大kernel abs变化1.79e−7，实际moment非零/step11100。支持参数量纲/FP32更新问题的假设，尚不证明DG受它限制；当前四条件不修Sinc，避免混因。合成gradient smoke只能验证接线，不能替代这项真实数值证据。

## 可执行方向与当前取舍

1. 当前执行：时间覆盖×分支SiLU。作用于实际t1/t2/t3→fuse→默认joint160；四条件每新arm同81672时序参数、同对应初始tensor、CPU/CUDA随机流，BASE_REF固定旧基线只对照。回答在同多滤波基/投影下长覆盖和独立非线性的增量；失败解释为该fixed机制/预算不成立，不按target找最佳seed。
2. 后续source证据驱动：定位层位，先t1/t2多尺度＋原t3与全层/中层对照；t3承担47.2%新增参数且局部最长覆盖扩张最大。若source默认margin/撤除修正错误及参数使用支持，再做独立从零同预算小矩阵，别扫几十核。非线性正向需增加同类post-fusion额外SiLU控制，才能归因“位置”；普通卷积＋投影控制才进一步隔离滤波基和旁路。
3. 独立Sinc量纲假设：保持初始kernel和原head/门控，以固定二进制频率单位参数化改善真实filter更新，配同初始/同预算控制；不同时叠时间SiLU，不以现有target挑父权重。真实source filter不动是code/optimizer证据，不是结果已有效。
4. 机制诊断优先于改SSL：本轮记录epoch32/33/34/40/50 source及实际θ变化，全U RX/day接受、teacher weak/student strong一致度与预测类分布。不读U TX、不按targetgroup调gate。只能在允许sourceL做正确性代理；先看结构影响何时出现，再研究teacher反馈的单因素修改。

## 需求落实与状态

|要求|落实|当前证据/缺项|
|---|---|---|
|默认头joint、0另训head|native fitted RX/day/channel closure禁用；每row强制0head updates|本地和原native协议负例PASS|
|新增能力进入实际fθ/z|新t块安装在optimizer/EMA之前；保留PA/freq/ref/headnorm|实际全t参数数值更新、z变化PASS|
|W修改/删除不改z，pure export等于actual z|小范围复用另一分支纯特征接口，保留norm/gain|修改/删除W、strict checkpoint/teacher、actual native L/U路径PASS|
|匹配数据/预算/RNG|fixedroles、scratch50/33+17、native args，新增CPU-only独立初始化|CPU与实际CUDA RNG都逐armPASS；不声称GPU完全bitwise|
|无目标反馈|固定四条件和primary/源门槛；逐rowpredict固定后fresh CPU评分|自动管线/错误counts/IDs/extrahead拒绝PASS|
|已训练核、stage和scale证据|真实旧TIME kernel audit已VERIFIED；新/旧冻结V撤除自动接续|旧TIME三seed scale撤除与新source/target指标待真实运行返回|
|完整三seed/成本/负结果|12新scratch50＋3历史参考，600新增epoch＋150历史|未返回前不称结果完整或84%达标|

唯一新run为20261014-phase1-time-coverage-nonlinearity-oldgate-e50-clean-manysig-m15-r01，最小报告和15row单一矩阵见automation_reports。同GPU最多3总实验进程，保留所有健康旧任务。源gate通过后扩seed，不借已暴露target进行筛选或选择性重跑。本地checks/registry valid、唯一P0/P1及一次针对修复确认PASS；Git/release/PID独立证据由同run报告更新。未授权Phase2、head refit、kernel scan或同时Sinc变化。

2026-10-14发布更新：执行代码7b1e2fcbe1bf5d05f24bc91f86aa7d4b46d3ee95已commit/push并独立remoteOID一致；ordinary preflight、archive校验、remote compile、真实checkpoint noquery、实际PID/CWD/argv/GPU/log增长VERIFIED。第一seed四新条件健康训练，第1至2epoch；旧TIME seed01冻结source诊断已完成。剩余seed按固定source门槛自动扩展；目前不是完整三seed结果。

## 局部分布与路径实验：完整冻结结果

20261013-phase1-localdistribution-pathsignature-oldgate-e50-clean-manysig-m24-r01 已EXPERIMENT_COMPLETE，dispatcher及全部worker自然退出，独立进程/产物读回VERIFIED。实际训练commit c1bf3b74c7bc38b03e4e29931d9b90dc60f4a296。24行全部50epoch；21新scratch模型1050epoch/233100 optimizer及EMA更新，3历史基线150epoch/33300更新分开。4032000固定默认决定、432条原始CM/accuracy/F1独立重算一致。所有新增模型无另训头、U真值可用0、nonfinite/跳步0，完整stdout、50epoch、11100step逐行实际分析，未使用tail替代。

|条件|clean均值%|样本SD百分点|Macro-F1|每seed最差RX均值%|每seed最差day均值%|配对基线差百分点|正差seed|
|---|---:|---:|---:|---:|---:|---:|---:|
|BASE_REF|80.5268|2.3776|0.800024|67.9708|76.0556|—|—|
|MEAN|82.0734|0.8367|0.815588|68.4542|77.5865|1.5466|2/3|
|MEAN_VAR|82.4536|1.1572|0.819609|69.2875|78.7278|1.9268|3/3|
|NONLINEAR_MEAN|81.8990|1.3385|0.813819|68.6736|77.4286|1.3722|2/3|
|RFF|81.5742|1.3390|0.810201|67.6972|76.9659|1.0474|2/3|
|COV|82.5778|1.5350|0.820782|69.9583|78.3937|2.0510|3/3|
|SIG|82.8476|1.0954|0.822917|69.0986|79.0294|2.3208|3/3|
|SIG_TIME|83.0393|1.4557|0.825453|70.3431|79.3587|2.5125|3/3|

预登记co-primary分别报告：RFF81.5742%（相对BASE＋1.0474pp，2/3正），SIG_TIME83.0393%（＋2.5125pp，3/3正），均未达84%；SIG_TIME距目标0.9607pp。不能把单seed超过84%当达标。RFF低于MEAN/MEAN_VAR/NONLINEAR_MEAN，三种简单控制均3/3高于RFF，因此本轮不支持随机非线性分布读出比同参数简单池化更有价值。SIG_TIME相对COV＋0.4615pp、3/3正；相对SIG＋0.1917pp仅2/3正，仍是3模型seed证据，不泛化为新TX或普遍定理。

SIG_TIME相对同seed原基线纠正/引入错误：2026092701＝9483/4829，净4654；2026092702＝10400/4782，净5618；2026092703＝8259/5868，净2391。这些是预测全部冻结后独立CPU truth侧连接的报告，未反馈训练。完整逐RX/day/TX错误transition和分组表见本run analysis/paired_error_counts.json、group_metrics.csv。

机制与边界：21份真实initial/final模型的新增五类参数全部出现实际数值变化、固定omega/lift buffers不变；使用执行7b1e2fcbe1bf5d05f24bc91f86aa7d4b46d3ee95的纯特征接口严格加载原权重，actual joint160与feature-only一致，修改/删除W后z逐位不变。固定人工IQ下真实训练新分支Δz均值范数1.033至1.308，证明分支活跃，不是识别效果或新TX证据。真实source L几何21row和V27000/row顺序诊断已返回：RFF/COV置乱改变默认决定0，maxz差≤9.54e−7；SIG/SIG_TIME分别改变2647至4301、2062至2867个源V决定，说明有序分支在实际fθ中起作用。撤除/置乱会破坏共同适应，不能替代scratch消融。仅默认头收益，缺普通换头证据（当前用户禁止新增headfit），不宣称通用新类迁移。

公平性限制明确保留：原sequence初始化在CPU-only fork中调用torch.manual_seed，也改变CUDA RNG。7个新控制/候选之间新模块初始化与CUDA起点匹配，但对BASE_REF的新增参数和CUDA随机流仍混杂。不能将所有对BASE增益独立归因为descriptor。该健康run保持原注册计划、未重启/热改；当前TIME新矩阵已用CPU default_generator修正并验证CPU/CUDA RNG。参数匹配也不证明descriptor intrinsic rank相等。

源训练与成本：源码原loss_unlabeled是伪标签CE；CE权重0.16、熵权重0.01。原日志没有单独熵分量，表保留native伪标签CE和total−L差，不假造完整熵曲线。源CE下降/更新有限证据完整；对所有损失逐项重构仍有此观测缺项。每新模型U处理963900次物理观察（17×56700），全21模型20241900；weak teacher和strong student是两次前向，物理观察计一次。接受比例1.5829%至1.6301%，不能解读为精度。新模型source执行进程时间合计13.7878小时，含原验证/闭合且GPU并发负载不同，不是exclusive GPU时长或加速结论。source_curves.csv和静态source_training_curves.png保留全部50轮，33/17交界标明；不按50轮预算就断言已收敛。

![完整源训练曲线](../automation_reports/CV-SincNet/20261013-phase1-localdistribution-pathsignature-oldgate-e50-clean-manysig-m24-r01/analysis/source_training_curves.png)

## 另一分支的新完成证据与重叠发布

只读最新HEAD/upstream＝0b3c498b4695f8c6e435531b10b995a4323fc173。PA_CROSS_DELAY实际执行1f8814c046a8c68eba8a334e66a0aceb76e491e0，32＋18的9row已完整，3历史基线＋6新scratch50/300新增epoch/66600更新。baseline80.6454%，PA_CONV80.7738%，PA_CROSS_DELAY80.8133%；后者相对同容量CONV＋0.0395pp仅2/3正，不确认三阶机制收益。当前复用该owner完整audit/162CM报告，未称本次重新扫描其原始日志。其同源监督/伪标签配方与33＋17组分开。

另一owner已独立发布同构TIME：20261014-phase1-time-factorial-oldgate-cos-e50-clean-manysig-m15-r01，执行351e35cd0d298a6832c0865628bba28ab5a14cbe。该新发布在本次最新只读盘点才确认，存在研究问题重叠。它的CPU初始化offset62103、首seed门槛仅finalLCE≤.5、额外源V33/34共54000包及固定L720包/seed历史撤除；本run offset48213、加固定默认V总体/最差RX非劣门槛、5个阶段V共135000包及历史V27000撤除。两份各自内部4条件权重匹配，但不跨run合并成六个独立seed，也不从target择优。已发布健康任务全部保留，不再新增第三套同类矩阵；本对话后续只对本run完成闭环，并按明确配方单列另一owner证据。

当前最多两个动作：完成正在运行的TIME四条件及真实stage/参数/U反馈/尺度诊断；随后依据合法源证据明确是否需要post-fusion SiLU或层位对照，暂不发布新矩阵。Sinc坐标修复保留独立源证据，未混入当前运行。

补充机器表：phase1_feature_group_table_20261014.csv含所有已回收近期run/readout的1998条同row分组指标，配方分组和readout不合并；逐seed表补齐每seed Macro-F1、最差RX/day及三seed最差day均值。历史内容复用已验证CM汇总，未新增head拟合或历史原始日志重扫。
