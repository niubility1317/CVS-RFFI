# P71 实际结果：固定 As 总量的有界质量风险

完整训练、七环境 prediction、独立 truth-last 评分与完整日志分析均已完成：ANALYZED，missing=[]。仅模型 seed 2026092701；source 两个信道 draw 不是两个模型 seed。正式使用 final_ssdg.pth 的学生 model 及原默认 cosine 分类头，没有另训头。

M3 对精确匹配 M2：clean +0.106548、mid urban +0.236905、low urban +0.322024、urban 三场景均值 +0.209921 个百分点。对原 As30：clean +0.020238、mid +0.206548、low +0.398214、urban 均值 +0.187897 个百分点；部分 suburban／high urban 仍略低于原 As。对同预算 NoAs30：clean +0.113690、mid +27.880357、low +21.502381、urban 均值 +28.556349 个百分点。大部分星地收益来自已有完整 As，不能全部归给风险机制。

本种子正式 clean 无代价，满足下降≤0.1 个百分点阶段目标；low urban 为51.642262%，距60%仍差8.357738个百分点。没有跨模型 seed 重复证据；原 As 三种子 clean 波动的历史不能由本行取代，也不能将本行与其他 seed 拼成新方案均值。

源判定在 owner 读取新正式结果前已保存并关闭：全 V clean 对 NoAs −0.133333 个百分点，pooled mid／low 对 M2 −0.027778 个百分点，两项均 FAIL。本固定配方不扩 E50、不扫描 gamma/v/seed。正式单种子上涨不改变先前源判定。

| 环境 | NoAs30 | 原 As30 | M2 nominal | M3 bounded | M3−M2（百分点） |
|---|---:|---:|---:|---:|---:|
| clean | 85.247619 | 85.341071 | 85.254762 | 85.361310 | +0.106548 |
| practical_high | 42.685119 | 81.405357 | 81.274405 | 81.286310 | +0.011905 |
| practical_mid | 40.995238 | 78.661310 | 78.532143 | 78.572619 | +0.040476 |
| practical_low_suburban | 39.472619 | 75.861905 | 75.806548 | 75.880952 | +0.074405 |
| practical_high_urban | 41.439881 | 77.767262 | 77.655357 | 77.726190 | +0.070833 |
| practical_mid_urban | 35.132738 | 62.806548 | 62.776190 | 63.013095 | +0.236905 |
| practical_low_urban | 30.139881 | 51.244048 | 51.320238 | 51.642262 | +0.322024 |

所有百分比为 accuracy%，每个环境168000个固定物理样本；都市三场景均值：NoAs 35.570833%、原As 63.939286%、M2 63.917262%、M3 64.127183%。逐seed／逐row／场景数据见 [formal_metrics_summary.csv](E:/type10-7/automation_reports/CV-SincNet/20261019-phase1-FcrossRX-boundedRisk-e30-s2026092701-m1-r01/formal_metrics_summary.csv)，所有 RX/day/TX 切片见 [formal_metrics.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/formal_metrics.csv)。

| 环境 | M3 宏平均 F1（%） | M3 最差 RX（%） | ΔF1／M2（百分点） | Δ最差 RX／M2（百分点） |
|---|---:|---:|---:|---:|
| clean | 85.064575 | 78.629167 | +0.113619 | +0.295833 |
| practical_high | 80.941548 | 71.320833 | +0.011606 | +0.100000 |
| practical_mid | 78.247305 | 69.745833 | +0.044234 | +0.137500 |
| practical_low_suburban | 75.558718 | 67.379167 | +0.080114 | +0.175000 |
| practical_high_urban | 77.353425 | 68.470833 | +0.071064 | +0.154167 |
| practical_mid_urban | 62.580714 | 56.333333 | +0.230987 | +0.220833 |
| practical_low_urban | 51.143867 | 46.745833 | +0.285001 | +0.287500 |

M3 clean 最差 RX=1-1；其余六 practical 最差 RX=14-7。相对 M2，clean 修复398／新错219，净+179；mid urban 修复1067／新错669，净+398；low urban 修复1272／新错731，净+541。对原 As，low 修复2036／新错1367，净+669。完整逐环境净纠错与 margin÷30 见 [配对表](E:/type10-7/automation_reports/CV-SincNet/20261019-phase1-FcrossRX-boundedRisk-e30-s2026092701-m1-r01/formal_pair_summary.csv)。独立重计504张混淆矩阵，84项分区计数对齐；所有四个逻辑row的七预测均先固定后接 truth。对照预测与评分均复用，没有重新预测或评分。

M3 的有界风险未转化为源验证增量。相对 M2，mid urban 修复13、新错9，净+4／3600（+0.111111 pp）；low urban 修复13、新错19，净−6／3600（−0.166667 pp），合并净−2／7200（−0.027778 pp）。每场景为同一1800个源V样本的两次信道观测，不是两个模型seed。

下表为M3−M2；Δ间隔表示默认头分类间隔除以30后的均值差。不同维度存在重叠，不能跨维度累加。

| 源分层 | mid n | mid 净正确 | mid Δ间隔 | low n | low 净正确 | low Δ间隔 |
|---|---:|---:|---:|---:|---:|---:|
| 新增噪声比 <3 dB | 383 | +1 | +0.000842 | 1127 | +2 | +0.000890 |
| 3–8 dB | 718 | 0 | +0.001359 | 794 | −1 | +0.001290 |
| 8–15 dB | 293 | +4 | +0.001342 | 303 | −2 | +0.000439 |
| ≥15 dB | 2206 | −1 | −0.001933 | 1376 | −5 | −0.001704 |
| BLOCKED | 1334 | +5 | +0.001270 | 1979 | +1 | +0.001069 |
| locked | 2205 | −1 | −0.001935 | 1367 | −5 | −0.001721 |
| degraded | 715 | +4 | +0.001473 | 667 | −4 | +0.001010 |
| unlocked | 680 | +1 | +0.000941 | 1566 | +3 | +0.000954 |

3–15 dB间隔上升，low该区间却净−3；≥15 dB合计净−6，间隔下降。BLOCKED局部净增不能证明遮挡损伤已被解决；新增噪声比也不是总SNR。

固定原As20 student的1765个clean正确样本，M2→M3的clean保持为1763→1764／1765（99.886686%→99.943343%）；两draw的mid保持2603→2607／3530（73.739377%→73.852691%），low为2114→2109／3530（59.886686%→59.745042%）。该子集略有clean修复，不能代替完整V27000：M3 clean对M2仍−0.007407 pp，对NoAs−0.133333 pp。

M3的实际风险CE从E21的0.875668降至E30的0.832836；同logits nominal CE从0.767574降至0.724443，四组训练CE均下降，但源mid／low未净增。风险CE与M2目标不同，不能直接横比其绝对值。此处只是训练与验证的对应观察，不作物理因果归因。

来源：[源配对表](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/source_paired_effects.csv)、[源分层表](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/source_strata.csv)、[父正确集合](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/P43_As20_clean_cohort.csv)、[父来源](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/parent_cohort_summary.json)、[风险汇总](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/risk_summary.json)、[训练曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/training_curves.csv)、[已固定源判定](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/stored_source_decision.json)。


前三轮均未通过各自预登记的源判据，未扩 E50。P67 将 U 时间确认改为同一物理样本跨轮确认，As 行最终接受量由 8726 增至 456028／567000；但源 mid／low 均值对原 As 仅 +0.069444 pp，low −0.055556 pp，clean 对匹配 NoAs −0.118519 pp。接受覆盖增加不能替代伪标签精度或识别收益。[源判定][p67]、[训练计数][p67train]

P68 在既有 6L 预算内引入第二个独立完整信道视图。同父 INDEP−SAME 的源 mid／low 仅 +0.069444 pp、low −0.027778 pp；clean 对 NoAs −0.029630 pp，仍未达到 +0.5 pp 增量。两行共同使用 SAT18 AS_CE20 父，与原 P43 As20 的前期 KL 历史不同。[源判定][p68]

P70 共享 142080 个 nominal 观测，各用一次；质量分层对随机池的源 mid／low −0.027778 pp，clean 对 NoAs −0.125926 pp。保持 nominal 风险的分层组织未提供足够增量。[源判定][p70]

P71 唯一改变 As CE 内部风险：固定 γ=.25、v=[.15,.40,.25,.20]，π=.75p+.25v，总 CE 系数仍为 .25。复用 M2 的池与顺序，从原 P43 As20 独立训练；不改变 KL、网络、U 或前向预算，不称 nominal 无偏估计。目前训练中，尚无收益结论；历史均为单模型 seed，正式成绩不用于上述取舍。

[p67]: E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01/source_matrix_decision.json
[p67train]: E:/type10-7/local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01/retrieved_r01/analysis_r01/training_summary.json
[p68]: E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-fullAsPair6-e30-s2026092701-m1-r01/source_matrix_decision.json
[p70]: E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-qualityStrat-e30-s2026092701-m2-r01/source_matrix_decision.json


实现只改变 As CE 内部风险：gamma=.25，v=[.15,.40,.25,.20]，pi=.75p+.25v；p由既有142080个合法源训练nominal观测确定。3至15 dB内部风险份额23.992821%→34.244616%，但这不是“可恢复性真值”。总 As CE=.25，same-IQ KL=.25，ordinary/raw、ORD128 SupCon、native U规则、6L/768前向与模型均保持。KL不按pi加权；全体合法有标签As仍做CE。读取完全相同的M2观测包和顺序，每条观测恰用一次，0新增物理模拟；固定同logits反事实仅核对公式，不作为已训练M2的loss。

实际起点commit4206fc7e05b5817338f3d583d68f46d4aa68f995，代码/config提交e4983676ecdcb88eab465f7f197a43c457a04a15。父为原P43从零As20学生/自身EMA，不是P70已测试final；严格同契约加载后重置AdamW/lr2e-4/wd1e-4/cos10，追加2220更新，总6660更新／E30。NoAs来自自身合法NoAs20父，不能称所有row同父；M2/M3才共享原As20父。合法L6300／隐藏TX的U56700／V27000，RX1/3/4/6/8、day1/2/3，split392005，全源无留出训练RX。

实际导入 __file__ 来自已固定20261015_r02运行包的 experiments/adv3b02_xuc/code/leo_practical/channel.py、residual.py，以及同目录DataAugmentation.py与SSDG/train_ssdg.py，而非根code/副本。输入float32 B×2×256，25MHz，逐包RMS且center=false，再从raw产生As；residual/post_sync/noeq，EQ关闭G=1，reference_snr_db30、reflect，接收质量适配/相位跟踪/IQ补偿/AGC沿原引擎。训练fc2.462GHz、正式fc2.45GHz均保持历史约定；测试仍复用VALIDATED_ONCE固定received IQ，不改SNR、样本或信道。数据角色、checkpoint继承与实际模块证据保留在本地retrieved_r01/M3-s2026092701/source。

实际新增成本：E10／2220更新；native 3162962样本前向＋源12600＋正式1176002＋父student/ownEMA smoke4，总4351568；本地与CPU分析NN=0。As20父训练4797722 NN／1810.249秒已完成并复用；M3训练累计7960684 NN，含本轮源/正式/smoke为9149290 NN。M3尾段训练1136.030秒，源读出2.151秒，正式预测51.675秒，CPU分析38.540秒；提交启动至analysis完成墙钟1247.963秒。现成池加载0.176秒；历史池142080次生成／287.656秒单列，本轮0新增。不将墙钟差直接归因风险机制。四逻辑行去重共享父后总80 model-epochs，本轮只新10。预算耗尽不等于收敛。详见 [costs.json](E:/type10-7/automation_reports/CV-SincNet/20261019-phase1-FcrossRX-boundedRisk-e30-s2026092701-m1-r01/costs.json)。

实际修改代码/config：experiments/cvs_bounded_risk71/下18个Python文件，核心为design.py、risk.py、model.py、observer.py、dispatch.py；analysis/cvs_bounded_risk71_analysis_20261019.py与对应checks。固定π、loss与原6前向的接入、只读池、唯一owner、源判定→自动all7→独立score→CPU分析均可执行。必要测试为15协议负例、8风险检查、8CPU分析toy，20本地语法、24远端compile；真实原As20 student/自身EMA无query smoke4通过。独立P0/P1仅发现未展开路径，发布前修复并定向复核PASS；无未解决P0/P1。归档只做一次传输SHA比较，PID/CWD/argv/GPU/log增长与终态分别VERIFIED。

实际入口：N607普通非特权既有CVS-RFFI环境，CWD=PROJECT/releases/cvs_FcrossRX_bounded_risk_E30_20261019_r01，执行python -u experiments/cvs_bounded_risk71/runner.py。唯一owner codex/root/bounded-risk71；本次GPU0一行，最大每GPU2总进程含其他owner。输出PROJECT/runs/20261019-phase1-FcrossRX-boundedRisk-e30-s2026092701-m1-r01，原父/M2/其他健康任务未改动。详细本地验证命令及stdout存O71/protocol_checks_actual.json、analysis_checks_actual.json；传输与运行读回在O71，未将原始日志/权重/IQ提交Git。

四组PNG均已实际查看，PDF均存在并以%PDF-开头：

- [完整训练曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/full_training.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/full_training.pdf)
- [源state/lock/质量/新增噪声分层](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/source_state_quality_lock_noise.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/source_state_quality_lock_noise.pdf)
- [正式配对变化](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/formal_paired.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/formal_paired.pdf)
- [风险份额与真实CE曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/bounded_risk_readout.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/bounded_risk_readout.pdf)

本行无技术缺项。科学目标未完成：没有可重复的clean+星地双提升，low urban未达60%，没有正式逐包state标签，不能对正式BLOCKED或CFO/多径/噪声做因果归因。源端只见BLOCKED局部微小净纠错与间隔变化，尚未证明某类严重urban损伤得到稳定缓解。固定风险候选关闭，不凭正式成绩救活或扩种子；后续独立研究不得同时叠加本候选权重。
