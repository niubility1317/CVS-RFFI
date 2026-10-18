# P61：nominal practical 下单侧默认头间隔独立复验

状态：**ANALYZED／VERIFIED，missing=[]**。run=`20261018-phase1-FcrossRX-pairMargin-tail-e30-s2026092701-m1-r01`；实际训练、配置和分析代码 commit=`5bc8ccff40c7e1873c261eb37f8b9b444283871e`。唯一模型 seed=2026092701，split=392005。记录时间 2026-10-18T09:00:49.406Z。

单侧间隔在当前完整 practical、保留 clean 份额的 F＋跨 RX 配方上只有很小的独立增量：相对匹配 As30，正式 low urban +0.110714、mid urban +0.076786、urban 三均值 +0.079960 个百分点，clean −0.067857。相对 NoAs30，clean +0.025595 个百分点，满足本种子“下降≤0.1个绝对百分点”的阶段准确率约束；clean Macro-F1反而 −0.005988 个百分点，微小准确率正值不能证明稳定双提升。

**源判据在任何新正式分数读取前已经 FAIL，维持原判定，关闭本固定复验。** 没有扫 lambda/rho/q、追加种子、延长 E50 或选择性重跑。low urban=51.354762%，尚未达到60%，suburban与clean仍有明显差距；总目标未完成。

## 1. 唯一假设及真实执行基线

旧 C3 是在状态课程 C2 上增加margin，并采用旧single SIG_TIME监督组织，未独立证明margin有用。本次关键改变是去掉课程、使用当前F＋CROSS_RX的6L clean保留配方，单独检验同一约束；不是新IQ增强或新损失创新。原历史源证据和失败路线去重见 [预登记](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-pairMargin-tail-e30-s2026092701-m1-r01/report.md>)。旧C3−C2三seed Vclean均值−.012346pp，不当成已有阳性；旧720训练L诊断不冒充独立V。

唯一新增行AS_MARGIN严格继承P43 B01自身scratchAs20 student＋ownEMA，追加10轮，总E30／6660更新。原P43 B00 NoAs30、B01 As30 source／prediction／score全部复用，不重训练、重预测或重评分。训练前后来源均按同一物理角色核实：L6300／U56700（TX真值隐藏）／V27000，源RX1/3/4/6/8、day1/2/3、六TX；无源RX留出或L_dev，复用未变VALIDATED_ONCE，不加载E200或不匹配权重。

F／SIG_TIME_X3／统一Sinc／joint160／默认cosine scale30分类头、原内部门控和参考响应仍联合更新。每步ordinary128×2、mixed128×2、raw128×2，共768 L前向；原L loss scale1.25、label_smoothing=.01、ordinary128 CROSS_RX .1、旧U／EMA规则保持。全部真实标签As参与CE，原RDrop保持同一IQ双前向，附加约束不丢弃或置信度过滤任何As。

As64从raw进入完整practical，不先叠ordinary PA／通用AWGN／多径；三个训练场景low suburban／mid urban／low urban均匀各1/3。几何、遮挡、传播、质量、补偿与噪声依赖不变；硬件按RX/day session稳定，不由TX标签定义，信道按physical/epoch/step命名流刷新。

间隔项复用已有两次raw logits和两次同IQ satellite logits的均值。logits统一除30，clean正确参考stop-gradient；rho=.5、m_max=1、lambda=.5，urban-only q=sigmoid(clip((signal_to_added_noise_db−5)/5,−60,60))，显式None为0。q只影响额外单侧约束，不影响CE，不要求全部特征、分支或门控等于clean。新增前向、部署模块、参数和额外teacher均0，原EMA teacher成本仍计入。

严格加载ownEMA，重置AdamW／cos10／cache，lr .0002→.000001、WD .0001；ordinary参数用原scratch20在E13冻结的设置；累计epoch/step +20/+4440。不是恢复原优化器轨迹，父与增量成本单列。

实际__file__来自PROJECT/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/{channel,residual}.py，而非仅检查仓库根同名实现。实际channel配置与原As完全一致：fs25MHz、IQ256、原normalize_input逐包complexRMS center=false；训练fc2.462GHz、正式固定2.45GHz；residual/post_sync、EQ=false即G=1、refSNR30、质量适配／相位跟踪／IQ补偿开启、reflect。固定正式IQ／信道／SNR／困难样本没有更改。

## 2. 先保存源判定，再读取正式成绩

完整源决定见 [source_decision.json](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-pairMargin-tail-e30-s2026092701-m1-r01/source_decision.json>)，没有重算或用正式分数翻案。writer的formal_scores_read=false是其写入时的事实；最终分析随后依法读取正式结果。

| 固定源条件 | 实际值（绝对百分点） | 判定 |
| --- | ---: | --- |
| fullV clean−NoAs30 ≥−0.1 | −0.118519 | **FAIL** |
| pooled mid/low−As30 ≥+0.5 | +0.069444 | **FAIL** |
| low≥As30 | 0 | PASS |

| 源指标 | NoAs30 | As30 | AS_MARGIN |
| --- | ---: | ---: | ---: |
| fullV27000 clean % | 98.359259 | 98.255556 | 98.240741 |
| fullV最差RX % | 95.611111 | 95.333333 | 95.296296 |
| V1800面板clean % | 98.444444 | 98.333333 | 98.333333 |
| pooled mid urban % | 43.555556 | 72.805556 | 72.944444 |
| pooled low urban % | 35.611111 | 59.083333 | 59.083333 |

面板satellite每场景为同1800物理包×两个draw，共3600次观测，不是两个模型seed；fullV27000与面板clean不同。source仅三个训练场景，没有source high urban或source all7；fullV没有F1和独立错误复算，不用面板F1替代。

mid相对As30仅修复8／新增3条错误，low修复8／新增8；low两个draw分别−.222222／+.222222pp，正负抵消。V1800 clean与As30的预测完全相同，但fullV clean仍−.014815pp，不能用小面板掩盖下降。

## 3. 正式全部逐row／逐seed结果

每个视图168000包，三逻辑行全部七视图冻结并独立评分完整；CPU分析核实all3×all7后才唯一连接truth，378份混淆矩阵和63项分区复算一致。全部采用checkpoint内原默认头，无另拟合头。以下accuracy为%，变化为绝对百分点。

| 环境 | NoAs30 % | As30 % | AS_MARGIN % | 相对As30 Δpp | 相对NoAs30 Δpp |
| --- | --- | --- | --- | --- | --- |
| clean | 85.247619 | 85.341071 | 85.273214 | -0.067857 | +0.025595 |
| practical_high | 42.685119 | 81.405357 | 81.453571 | +0.048214 | +38.768452 |
| practical_mid | 40.995238 | 78.661310 | 78.716071 | +0.054762 | +37.720833 |
| practical_low_suburban | 39.472619 | 75.861905 | 75.953571 | +0.091667 | +36.480952 |
| practical_high_urban | 41.439881 | 77.767262 | 77.819643 | +0.052381 | +36.379762 |
| practical_mid_urban | 35.132738 | 62.806548 | 62.883333 | +0.076786 | +27.750595 |
| practical_low_urban | 30.139881 | 51.244048 | 51.354762 | +0.110714 | +21.214881 |
| urban 三场景均值 | 35.570833 | 63.939286 | 64.019246 | +0.079960 | +28.448413 |

相对NoAs30的巨大星地收益主要属于原完整As训练，不属于新增margin：原As30的urban均值63.939286%，AS_MARGIN64.019246%，独立增量只有.079960pp；不能把对无As的+28.448413pp全部归因于间隔项。

以下单元格依次为准确率／Macro-F1／最差RX准确率，单位%。NoAs30全部环境最差RX为14-7；As30与AS_MARGIN的clean最差为1-1，六practical均为14-7。

| 环境 | NoAs30：Acc／F1／最差RX % | As30 | AS_MARGIN |
| --- | --- | --- | --- |
| clean | 85.248／84.980／78.450 | 85.341／85.044／78.362 | 85.273／84.974／78.408 |
| practical_high | 42.685／39.311／34.304 | 81.405／81.084／71.908 | 81.454／81.133／72.071 |
| practical_mid | 40.995／37.955／33.521 | 78.661／78.360／70.213 | 78.716／78.415／70.367 |
| practical_low_suburban | 39.473／36.759／32.054 | 75.862／75.573／67.742 | 75.954／75.663／67.933 |
| practical_high_urban | 41.440／38.481／34.000 | 77.767／77.423／69.108 | 77.820／77.475／69.146 |
| practical_mid_urban | 35.133／33.447／29.496 | 62.807／62.450／56.458 | 62.883／62.526／56.592 |
| practical_low_urban | 30.140／29.185／25.750 | 51.244／50.887／46.379 | 51.355／50.990／46.575 |

AS_MARGIN三个suburban与自身clean差距为3.819643／6.557143／9.319643pp，均未接近建议的2pp探索目标。相对As30 clean下降也是差距缩小的一部分，不把它作为泛化收益。单seed、小于.1pp的clean上涨仅作初步观察。

## 4. 修复／新错、RX/date边界及机制

| 环境 | 对As30修复／新错 | 对NoAs30修复／新错 |
| --- | ---: | ---: |
| clean | 90／204 | 3825／3782 |
| high suburban | 257／176 | 68555／3424 |
| mid suburban | 297／205 | 67365／3994 |
| low suburban | 363／209 | 65754／4466 |
| high urban | 272／184 | 65242／4124 |
| mid urban | 396／267 | 53379／6758 |
| low urban | 511／325 | 44482／8841 |

每环境分母168000。urban三视图共504000次观测，不是504000个独立物理样本；对As30合计1179修复／776新错，净403。clean对NoAs30净增43条，但有3782条原正确样本变错，不能把微小净增当作身份保持普遍改善。全配对见 [formal_pair_summary.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/formal_pair_summary.csv>)／[formal_paired_effects.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/formal_paired_effects.csv>)。

对As30，clean在6/7 RX和4/4日期下降：RX8-8 −.204167、14-7 −.129167、20-1 −.095833pp；2021_03_08 −.116667pp。mid／low在所有7RX及4日期均有小幅正增，urban三均值没有RX／日期负层，但仍只有一个模型seed。对NoAs30，clean在RX1-1为−2.579167、20-1为−.775000pp；总体微升并不代表每RX保持。完整RX/day/TX表见 [formal_metrics.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/formal_metrics.csv>)。

固定NoAs30 clean正确参考集合为143216包，所有row共用；这是匹配参照集合，不冒称新行真实As20父的正式正确集合（该父未保存对应全部正式预测）。默认头间隔统一为真类对最强竞争类margin／30。

| 环境 | As30 margin/30 | 新 margin/30 | As30 固定集合保持率 % | 新保持率 % |
| --- | --- | --- | --- | --- |
| clean | 0.14612423 | 0.14591175 | 97.408809 | 97.359234 |
| practical_high | 0.09939187 | 0.10136674 | 90.110044 | 90.149145 |
| practical_mid | 0.08929700 | 0.09118091 | 86.772428 | 86.827589 |
| practical_low_suburban | 0.07950377 | 0.08129276 | 83.456457 | 83.565384 |
| practical_high_urban | 0.09072460 | 0.09256845 | 85.932438 | 85.986203 |
| practical_mid_urban | 0.05380980 | 0.05506497 | 68.782119 | 68.866607 |
| practical_low_urban | 0.02807553 | 0.02890718 | 55.632751 | 55.748659 |

间隔变化与识别变化分开：mid／low的margin/30分别+.00125517／+.00083166，但accuracy只+.076786／+.110714pp。没有独立margin参数梯度、逐分支／门控轨迹；全局梯度健康不能证明它改善了梯度冲突。未添加或拟合诊断头。

源状态关联也很小：mid PARTIAL/degraded n59 +1.694915pp来自仅修复1条；mid BLOCKED/degraded n656 +.152439（3修复／2新错）。low PARTIAL/locked n307 −.325733（0／1）、PARTIAL/degraded n253 −.790514（0／2），BLOCKED/degraded n414 +.241546（2／1）、BLOCKED/unlocked n1565 +.127796（6／4）。不能把小样本最大百分点称为稳定状态改善。

信号／新增噪声比不是总SNR：low<3dB n1127 +.266193（5／2），但3–8dB n794 −.251889（2／4），≥15dB n1376 −.072674（0／1）。空状态保留，PARTIAL/unlocked n1／2不支持结论。正式没有逐包信道状态标签，本次亦未进行固定／重算接收状态的物理移除反事实试验，因此没有证明独立缓解噪声、多径或CFO；只能报告上述源观测关联与正式识别小增量。源明细见 [source_strata.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/source_strata.csv>)／[source_paired_effects.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/source_paired_effects.csv>)。

## 5. 全量训练轨迹及覆盖

完整读取新增10epoch／2220步，不用日志抽样冒充全量。pair记录48840个数值字段、epoch230个字段非有限值0；原生梯度字段全部覆盖2220步，主干epoch梯度范围1.237141–1.713271。native loss .631091→.612290，范围.563050–.644721；fullV曲线97.711111%–98.274074%，最终98.240741%。预算用尽不等于收敛。

pair_raw均值.01546396637，weighted=.007731983185，2220步均正；q均值.4716491168，94555个urban样本、其中93670个clean正确。活跃竞争项249979／710400=35.188485%，在clean正确urban竞争项中为53.374399%。训练clean／satellite平均竞争类margin/30=.256202374／.136291357。约束确实激活，但不以其激活或间隔增长替代识别收益。

原L损失分解最大误差1.594424e−7，pair记录误差0；全部真实标签As CE保留。记录中的extra_CE是64As＋64ordinary的mixed128 CE，没有独立As-only CE曲线，不补造。

新增L前向1704960，生成As142080；原U输入567000、原规则接受8721。物理L/U序列4440条与As30差异0，ordinary MSE流差异0。覆盖CSV展开810格，其中515非空，总As142080；全部90个TX×RX×scene组合、15个RX/day session。locked／degraded／unlocked分别86115／23865／32100；没有As丢弃或置信度过滤。覆盖不等于所有理论状态均出现，空格保留。见 [training_coverage.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/training_coverage.csv>)、[new_tail_training.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/new_tail_training.json>)、[training_curves.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.csv>)、[pair_margin_steps.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/pair_margin_steps.csv>)。

## 6. 实际成本与执行验证

| 阶段 | 模型轮／更新 | native NN | 训练墙钟秒 |
| --- | ---: | ---: | ---: |
| 自身As20父 | 20／4440 | 4797722 | 1810.249311 |
| 新margin尾段 | 10／2220 | 3162962 | 1186.548547 |
| 候选累计训练 | 30／6660 | 7960684 | 2996.797858 |

新增native NN中student train2271960、student eval324002、teacher eval567000、teacher train0。NN是前向样本数，不是独立物理样本或batch数。新机制额外margin前向0、额外teacher0，原EMA teacher真实成本照计。

新增source读出12600NN／1.829463秒；正式预测1176002NN／48.443742秒；远端真实父smoke2NN／1CP／0IQ／0更新。因此**新增远端总4351566NN**，本地唯一合成768NN另计（脚本内1.8225325秒、进程3.988秒）。CPU分析22.385836秒，NN／IQ／CP／更新0。本次训练墙钟19.776分钟，不把前向相同理解为墙钟完全相同；并发、I/O与记录成本没有独立分解。

两个历史E30参考共60模型轮／13320更新，计入新尾段后去重总70模型轮／15540更新；共享As20父只计一次。已知去重累计远端22650138NN、训练墙钟6193.647844秒；历史共享smoke未分摊，不能称所有历史操作完整总额。参考重预测／重评分0。成本详见 [costs.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/costs.json>)。

实际代码/config/launcher/分析与定向检查均在5bc8ccff40c7e1873c261eb37f8b9b444283871e，启动记录commit cd02f6c60beca8561398cfcf349184a8583a7d48。修改范围为experiments/cvs_pair_margin61/**、analysis/cvs_pair_margin61_analysis_20261018.py和analysis/check_pair_margin61_analysis.py；没有改动其他健康实验或数据协议。

执行过27项协议负例及margin闭式／stopgrad／scale／q／梯度／captureRNG检查、4项流程检查；唯一实际F合成768NN；19py编译；分析12项纯toy1.020秒及2py编译。唯一P0/P1审查PASS，审查者仅重复相同12项CPU toy1.040秒、0NN/真实IQ/CP/truth，不重跑F；最后CLI实测localNN及commit字段只定向编译。

一次普通账号N607预检PASS；单release archive一次本远端SHA比较VERIFIED，21py远端编译PASS；真实自身父CP烟测PASS后直接训练。dispatcher1312196、worker1312286／GPU0，PID/CWD/argv/CUDA/nvidiaPID与日志增长独立核实，最终两进程自然退出，全部固定产物完整；没有重启、终止或热改健康任务。远端发布／运行／评分／分析均有独立后状态支持VERIFIED。

精确入口：

```text
CWD=PROJECT/releases/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01
CVS-RFFI/python -u experiments/cvs_pair_margin61/runner.py
自动CPU分析：analysis/cvs_pair_margin61_analysis_20261018.py --project PROJECT --out BASE/analysis_r01 --local-nn 768 --code-commit 5bc8ccff40c7e1873c261eb37f8b9b444283871e
输出：PROJECT/runs/20261018-phase1-FcrossRX-pairMargin-tail-e30-s2026092701-m1-r01
Python3.10.19 / Torch2.1.0+cu121 / NumPy2.2.5，原pinned runtime，FP32
```

## 7. 实际图表、证据及未完成项

五张图均实际生成并逐张查看，PNG及同名PDF均已保存：

- [完整训练曲线](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.png>)。
- [间隔激活、CE与训练间隔曲线](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/paired_margin_training.png>)。
- [mid urban源状态及新增噪声比分层](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/practical_mid_urban_source_states.png>)。
- [low urban源状态及新增噪声比分层](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/practical_low_urban_source_states.png>)。
- [全部正式环境比较](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/formal_accuracy.png>)。

可核查汇总：[complete.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/complete.json>)、[formal_summary.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/formal_summary.json>)、[source_summary.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01/retrieved_r01/analysis_r01/source_summary.json>)。聚合CSV／图／原预测／权重仍在各产物根；不提交IQ、权重、原日志、预测或私有运行细节。

本固定矩阵无技术缺项，不代表科学目标完成。只有一个模型seed，无法估计跨seed波动；两个信道draw不构成跨seed重现。未获得low>60%、可重复clean正收益、suburban接近clean或新增Ag的独立价值；没有证明某个独立物理损伤被margin缓解，真实As20父正式正确集合和完整gate/分支轨迹亦缺失。当前采用固定NoAs30参照集合，已明确口径。

最后回答：相对原As，默认头low仅提高.110714pp、urban均值+.079960pp，clean付出.067857pp代价；相对无As整体low+21.214881pp、urban+28.448413pp而clean+.025595pp，单seed阶段约束通过，但不能把整体As适应归因于margin或宣称稳定双提升。当前源判据失败，该复验关闭，不以正式微涨救活；后续新研究需重新核对另一聊天已完成/进行路线，不能将本损失加到其他方案后直接相加收益。
