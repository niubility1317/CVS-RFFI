# P78 最终结果：移除普通 AWGN／多径未带来有价值的新增星地收益

状态：ANALYZED。两条新增行均完成各自 P43 E20＋E10，总 E30；单模型种子 2026092701。固定默认分类头和七个正式环境全部完成独立 truth-last 评分，原两个对照复用原产物。实际代码提交 5d6514b00b96614a1bd8e2b276e286bf3bed37e6，起点 ca91644d9e13efdfb9fb7f343f9041743772af96。

## 结论与归因

保留原完整 practical 配方。新增机制相对原 As 的 low urban 仅 +0.059524 个百分点，mid urban −0.002381，urban 均值 +0.007540，clean −0.075000。没有证据表明该机制值得保留。默认头的新 As low urban 为 51.303571%，未达到 60% 目标。

按同预算、同种子、新 ordinary 配方下的新 NoAs 比较，新 As 的 clean +0.189881 个百分点，满足下降不超过 0.1 个百分点的阶段要求，urban 均值 +28.511508。但新 NoAs 自身相对原 NoAs 的 clean −0.171429，mid／low urban −0.268452／−0.319048；较大的 As−NoAs 差不能全部归因于新增机制。新 As 相对原 NoAs 的 clean 只有 +0.018452 个百分点，宏 F1 反而 −0.028495，且仅一个模型种子；尚无“星地增强稳定改善 clean”的可重复证据。新增机制的 urban 交互 +0.143056 个百分点主要包含无 As 对照退步，不能代替直接 As 收益。

源决定先于读取新正式成绩保存并由 owner 观察：full V27000 clean 新 As−新 NoAs 为 −0.077778 个百分点，clean 条件通过；固定 V1800×2 信道 draw 的 mid／low 均值新 As−原 As 为 +0.305556，低于预登记 +0.5 门槛，源资格 FAIL。故关闭该固定候选，不扩 E50、不扩模型种子、不在失败后拆 AWGN／多径或扫描强度。正式成绩未改变决定。两个 draw 不是两个模型种子。

## 哪类退化有变化

源域 low urban 的 BLOCKED、unlocked 与新增噪声比较低区间出现少量修复；mid urban 的变化更混合。以下比较始终为新 As−原 As，同一批物理样本和固定信道 draw：

|源场景|分层|样本数|修复|新增错误|净修复|准确率变化 pp|
|---|---|---:|---:|---:|---:|---:|
|practical_mid_urban|state=BLOCKED|1334|14|12|2|0.149925|
|practical_mid_urban|lock=unlocked|680|5|5|0|0.000000|
|practical_mid_urban|signal_to_added_noise_db=<3|383|4|1|3|0.783290|
|practical_mid_urban|signal_to_added_noise_db=3-8|718|5|10|-5|-0.696379|
|practical_low_urban|state=BLOCKED|1979|34|22|12|0.606367|
|practical_low_urban|lock=unlocked|1566|27|19|8|0.510856|
|practical_low_urban|signal_to_added_noise_db=<3|1127|20|11|9|0.798580|
|practical_low_urban|signal_to_added_noise_db=3-8|794|13|11|2|0.251889|

这些是源域分层关联，不能把同一包重复落入多个维度的计数相加。新增噪声比不是总 SNR，正式数据缺少逐包物理状态标签；本轮不能认定某一种物理损伤已在正式 urban 中得到可靠缓解。源面板有状态、锁定、CFO、核、着色、AGC、边界、白功率代理及实际功率等分层，完整 CSV 与图保留。真实 P43 E20 父 clean 正确集合缺少匹配缓存，未新增推理补表，也未混用 SAT18 集合；固定 NoAs30 clean 正确集合的保持率单独报告。

## 实现和实际执行

只在最后 E10 中让 L ordinary 的原生 AWGN／multipath 完整计算后返回输入，保留该调用的随机抽样与影子计算；相位、时移、增益、PA 等原有操作保留。完整 As 仍直接从 raw 生成，不存在先 ordinary 再 As 的串接修复。U 的原生强增强透传不受旁路影响，原 U／自身 EMA、RDrop／SupCon、六次 L 前向、损失权重和默认 cosine30 头正常联合训练。没有新头、分支、损失或参数。两条对应 P43 原 E20 student／自身 EMA 分别严格加载；LRaw 仅是固定运行代码依赖，不继承其训练权重，也不加载 P77／SAT18。

L6300、U56700 TX 隐藏、V27000，五个源 RX1/3/4/6/8 和 day1/2/3 全部保留，无源 RX 留出。每条尾段 L6300 与 U56700 物理样本全部覆盖；新 As 选中 142080 个完整 practical 包，6300 个 L 物理包均被覆盖，单包 12 至 34 次，源 RX 选中计数 28538／28501／28258／28560／28223。未使用目标数据训练或选模。

每行 2220 次 ordinary 调用、284160 包；MP 95 次／12160 包，AWGN 208 次／26624 包，同时命中 12 步，影子计算未省去。每行 U2220 次／567000 包，256 包调用 2210 次、末124包调用10次，透明传递与 patch 退出均核实。普通输出对 raw 的 MSE 均值 0.193619，说明没有把整个 ordinary 偷换为 raw。原语返回后的 RNG 状态保持，但后续输入与 sigma 可变化，不声称整个训练轨迹逐位匹配。

IQ2×256，25 MHz，RMS center=false，原 normalize_input 位置保留。实际 residual／post_sync／EQ=false（G=1）／reference_snr_db=30／reflect；接收质量适配、相位跟踪、IQ 补偿、session 硬件与 AGC 依赖保持。训练载频 2.462 GHz；正式固定 manifest 六景载频 2.45 GHz，25 MHz／ref30／residual／EQfalse，未改正式 IQ、SNR 或困难样本。源面板载频未经本轮独立核实，不填猜测值。实际模块 __file__ 指向固定 r02 的 experiments/adv3b02_xuc/code/leo_practical/{channel,residual}.py 以及同目录 DataAugmentation.py、SSDG/train_ssdg.py，非顶层 code 实现。

## 成本、验证与复现

新增训练 20 模型轮／4440 更新，两行各追加 E10／2220 更新、累计 E30／6660。新增 native 前向 6325924、源读出25200、正式2352004、共享真实 CP smoke1540，总计8704668次样本前向。本地 NN=0。两父 E20 各4797722 native 前向复用；两旧完整 E30 训练与读出共18298572前向不重复收费。两父E20＋两旧尾E10＋两新尾E10共80唯一模型轮／17760更新，四条完整E30曲线120行不代表120轮新增训练。

新 NoAs／As 训练进程墙钟分别899.554724／1128.245311秒，合计2027.800036进程秒；正式预测分别51.092420／51.604505秒。独立终态证据的启动至分析完成实际经过1287.972302秒（21.466205分钟）。源读出墙钟未单独归档，不记为0；源分析3.817753秒是读取现有聚合产物的分析耗时。相同前向量不等于相同墙钟，旁路保留完整影子计算，预算结束不等于收敛。分析自身新增NN／IQ／CP／更新均0。

已执行27项协议负例、真实原语OFF输出/RNG、强制MP＋AWGN独立重放、真实U256/124函数透传、finally与E20/E21/E30边界检查；14项分析fixture和新增最终报告链1项检查通过。一次独立P0/P1审查PASS。单归档本地／远端一次SHA比较VERIFIED、25文件远端编译PASS，launcher首步真实CP／默认头／自身EMA1540NN smoke PASS后直接训练。两行所属PID/CWD/命令/GPU及日志增长独立核实，最终所有所属进程已退出，技术缺项0。504正式混淆矩阵、2904状态矩阵及配对／交互／成本等聚合算术独立复核PASS；该复核不替代最初独立truth-last评分。

本地检查（CWD为worktree）：

~~~text
E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 experiments/cvs_ordinary_channel78/local_checks.py --dependency-root E:/type10-7/code/worktrees/cvs_iq_timing_20261016
~~~

远端训练（普通账号，CWD=PROJECT/releases/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01；PROJECT为既有项目根）：

~~~text
CVS-RFFI/python -u experiments/cvs_ordinary_channel78/runner.py
~~~

源／最终分析分别使用发布目录 analysis/cvs_ordinary_channel78_analysis_20261019.py 的 --stage source／--stage final，输出 PROJECT/runs/20261019-phase1-FcrossRX-ordinaryChannel-e30-s2026092701-m2-r01/analysis_r01；只有 final 阶段传既有固定 truth 路径。实际命令、环境、提交、行配置和不可覆盖根见 [预登记与执行报告](E:/type10-7/automation_reports/CV-SincNet/20261019-phase1-FcrossRX-ordinaryChannel-e30-s2026092701-m2-r01/report.md) 和 experiment.json；代码见 [增强实现](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_ordinary_channel78/augmentation.py)、[训练入口](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_ordinary_channel78/runner.py)、[固定配置](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_ordinary_channel78/design.py)。

## 交付与未完成事项

[全部逐行指标与配对变化](E:/type10-7/local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01/retrieved_r01/analysis_r01/formal_metrics.csv)、[源分层](E:/type10-7/local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01/retrieved_r01/analysis_r01/source_strata.csv)、[配对错误分层](E:/type10-7/local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01/retrieved_r01/analysis_r01/formal_paired_strata.csv)、[成本](E:/type10-7/local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01/retrieved_r01/analysis_r01/costs.json) 已落盘。以下四张 PNG 已实际查看，四张 PDF 文件头和交付存在性已核实：

- [完整训练曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01/retrieved_r01/analysis_r01/training_curves.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01/retrieved_r01/analysis_r01/training_curves.pdf)
- [源质量与状态分层](E:/type10-7/local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01/retrieved_r01/analysis_r01/source_quality_states.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01/retrieved_r01/analysis_r01/source_quality_states.pdf)
- [七环境配对表现](E:/type10-7/local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01/retrieved_r01/analysis_r01/formal_all7_paired.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01/retrieved_r01/analysis_r01/formal_all7_paired.pdf)
- [实际增强覆盖](E:/type10-7/local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01/retrieved_r01/analysis_r01/ordinary_channel_coverage.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01/retrieved_r01/analysis_r01/ordinary_channel_coverage.pdf)

本轮只有一个模型种子，无跨模型种子波动或均值证据；两个信道draw不补充模型种子。匹配真实E20父正确集合和正式逐包状态仍缺少，未编造。low urban≥60%、稳定clean正收益及跨seed复现尚未完成，目标仍ACTIVE。下一项方法需独立提出假设，不能用正式成绩复活本轮失败配方。

---

以下为同row自动分析的完整四格指标、宏F1、最差RX、错误修复、新增错误与成本表：

# P78：ordinary 通用信道与完整 practical 的尾段 2×2 检验

20261019-phase1-FcrossRX-ordinaryChannel-e30-s2026092701-m2-r01

两条新行分别从对应 P43 B00/B01 的 own E20 student 与 own EMA 追加 E10，累计 E30。只对 L ordinary 链中的原生 MP/AWGN 执行原计算后旁路输出；PA、幅相、时序等其余原操作按原生配置保留，DAC/class-signature 原本关闭并保持关闭。前 E20 完整 ordinary 历史保留。完整 practical 仍从 raw64 生成，原 U、全部 CE/RDrop/SupCon、默认 cosine30 分类头与六次 L 前向不变。该机制是移除可能重叠的普通信道扰动，不预先称为成功的新 Ag。

|环境|旧 NoAs %|旧 As %|OC NoAs %|OC As %|新 As−新 NoAs pp|新 As−旧 As pp|交互 pp|
|---|---:|---:|---:|---:|---:|---:|---:|
|clean|85.247619|85.341071|85.076190|85.266071|0.189881|-0.075000|0.096429|
|practical_high|42.685119|81.405357|42.920238|81.410119|38.489881|0.004762|-0.230357|
|practical_mid|40.995238|78.661310|41.115476|78.669048|37.553571|0.007738|-0.112500|
|practical_low_suburban|39.472619|75.861905|39.642262|75.924405|36.282143|0.062500|-0.107143|
|practical_high_urban|41.439881|77.767262|41.620833|77.732738|36.111905|-0.034524|-0.215476|
|practical_mid_urban|35.132738|62.806548|34.864286|62.804167|27.939881|-0.002381|0.266071|
|practical_low_urban|30.139881|51.244048|29.820833|51.303571|21.482738|0.059524|0.378571|
|urban_mean|35.570833|63.939286|35.435317|63.946825|28.511508|0.007540|0.143056|

新 As 相对同预算新 NoAs：clean +0.189881 个百分点，下降不超过 0.1 个百分点=True；low/mid urban +21.482738/+27.939881 个百分点，urban 三场景均值 +28.511508 个百分点。
交互=(OC As−原 As)−(OC NoAs−原 NoAs)，必须连同四行绝对性能解释；不能把 NoAs 退化算成 As 的直接改善。一个模型种子不足以证明可重复双提升。

|条件|环境|accuracy %|宏 F1 %|最差 RX %|
|---|---|---:|---:|---:|
|NOAS30|clean|85.247619|84.979636|78.450000|
|NOAS30|practical_high|42.685119|39.311416|34.304167|
|NOAS30|practical_mid|40.995238|37.954813|33.520833|
|NOAS30|practical_low_suburban|39.472619|36.758731|32.054167|
|NOAS30|practical_high_urban|41.439881|38.481030|34.000000|
|NOAS30|practical_mid_urban|35.132738|33.446603|29.495833|
|NOAS30|practical_low_urban|30.139881|29.184589|25.750000|
|AS30|clean|85.341071|85.044474|78.362500|
|AS30|practical_high|81.405357|81.084230|71.908333|
|AS30|practical_mid|78.661310|78.359706|70.212500|
|AS30|practical_low_suburban|75.861905|75.572741|67.741667|
|AS30|practical_high_urban|77.767262|77.423208|69.108333|
|AS30|practical_mid_urban|62.806548|62.450463|56.458333|
|AS30|practical_low_urban|51.244048|50.887035|46.379167|
|OC_NOAS30|clean|85.076190|84.797467|78.616667|
|OC_NOAS30|practical_high|42.920238|39.913055|33.325000|
|OC_NOAS30|practical_mid|41.115476|38.467907|32.641667|
|OC_NOAS30|practical_low_suburban|39.642262|37.358275|31.283333|
|OC_NOAS30|practical_high_urban|41.620833|39.068296|32.966667|
|OC_NOAS30|practical_mid_urban|34.864286|33.679461|28.295833|
|OC_NOAS30|practical_low_urban|29.820833|29.349756|25.145833|
|OC_AS30|clean|85.266071|84.951140|78.662500|
|OC_AS30|practical_high|81.410119|81.079930|72.258333|
|OC_AS30|practical_mid|78.669048|78.357753|70.412500|
|OC_AS30|practical_low_suburban|75.924405|75.632797|67.908333|
|OC_AS30|practical_high_urban|77.732738|77.385124|69.183333|
|OC_AS30|practical_mid_urban|62.804167|62.469624|56.587500|
|OC_AS30|practical_low_urban|51.303571|50.968971|46.537500|

|比较|环境|修复|新增错误|净修复|
|---|---|---:|---:|---:|
|OC_AS30−OC_NOAS30|clean|4099|3780|319|
|OC_AS30−OC_NOAS30|practical_high|68117|3454|64663|
|OC_AS30−OC_NOAS30|practical_mid|67131|4041|63090|
|OC_AS30−OC_NOAS30|practical_low_suburban|65492|4538|60954|
|OC_AS30−OC_NOAS30|practical_high_urban|64795|4127|60668|
|OC_AS30−OC_NOAS30|practical_mid_urban|53853|6914|46939|
|OC_AS30−OC_NOAS30|practical_low_urban|45161|9070|36091|
|OC_AS30−AS30|clean|585|711|-126|
|OC_AS30−AS30|practical_high|678|670|8|
|OC_AS30−AS30|practical_mid|811|798|13|
|OC_AS30−AS30|practical_low_suburban|956|851|105|
|OC_AS30−AS30|practical_high_urban|766|824|-58|
|OC_AS30−AS30|practical_mid_urban|1292|1296|-4|
|OC_AS30−AS30|practical_low_urban|1658|1558|100|
|OC_NOAS30−NOAS30|clean|856|1144|-288|
|OC_NOAS30−NOAS30|practical_high|3845|3450|395|
|OC_NOAS30−NOAS30|practical_mid|3731|3529|202|
|OC_NOAS30−NOAS30|practical_low_suburban|3850|3565|285|
|OC_NOAS30−NOAS30|practical_high_urban|3848|3544|304|
|OC_NOAS30−NOAS30|practical_mid_urban|3510|3961|-451|
|OC_NOAS30−NOAS30|practical_low_urban|3511|4047|-536|
|AS30−NOAS30|clean|3868|3711|157|
|AS30−NOAS30|practical_high|68469|3419|65050|
|AS30−NOAS30|practical_mid|67279|4000|63279|
|AS30−NOAS30|practical_low_suburban|65618|4484|61134|
|AS30−NOAS30|practical_high_urban|65141|4111|61030|
|AS30−NOAS30|practical_mid_urban|53261|6769|46492|
|AS30−NOAS30|practical_low_urban|44297|8842|35455|
|OC_AS30−NOAS30|clean|3921|3890|31|
|OC_AS30−NOAS30|practical_high|68460|3402|65058|
|OC_AS30−NOAS30|practical_mid|67281|3989|63292|
|OC_AS30−NOAS30|practical_low_suburban|65696|4457|61239|
|OC_AS30−NOAS30|practical_high_urban|65087|4115|60972|
|OC_AS30−NOAS30|practical_mid_urban|53326|6838|46488|
|OC_AS30−NOAS30|practical_low_urban|44431|8876|35555|

源决定在 formal 阶段之前保存：{"fullV_clean_vs_matched_NoAs_pp": -0.07777777777777661, "mid_low_mean_vs_original_As30_pp": 0.3055555555555589, "clean_vs_matched_NoAs_within_absolute_0p1pp": true, "mid_low_gain_at_least_0p5pp": false, "source_criteria_pass": false, "E50_source_eligible": false, "fixed_recipe_closed": true, "fullV_clean_vs_original_As30_pp": 0.04074074074074119, "fullV_clean_vs_original_NoAs30_pp": -0.06296296296296466, "fullV_clean_interaction_pp": 0.02592592592592924, "mid_low_mean_interaction_pp": -0.5694444444444446, "only_two_source_criteria": true, "scientific_negative_skips_formal": false, "target_scores_read": false, "automatic_extension": false}
仅两项源条件：新 As fullV clean−新 NoAs≥−0.1 个百分点，两个固定 draw 的 mid/low 等权新 As−原 As≥+0.5 个百分点。失败仍完成全部七环境，并关闭该固定配方的 E50/扩种子；正式分数不回流决定。无原语命中数、MSE、状态代理或 raw 预测不变的额外晋级条件。
训练 fc=2.462 GHz，既有固定正式 received views 的 fc=2.45 GHz；源 panel 频率本分析未独立核实，没有修改正式信道。源 V1800 状态/锁定/新增噪声比/波形与白功率代理、CFO、核、着色、AGC、边界分层及 margin÷30、固定 NoAs30 clean 正确集合保持率见 CSV。它们是源域关联描述，新增噪声比不是总 SNR；正式无逐包状态标签。源没有 high urban；fullV27000 只有原生完整 accuracy/RX，不虚构逐包 F1。
真实 P43 E20 父 clean 正确集合缺少匹配缓存，本轮未测；没有混用 SAT18 父缓存，也不新增推理。固定 NoAs30 集合与父集合不混称。
新两行各 2220 更新全部解析。MP/AWGN 影子调用、包数、RNG 前进计数、恢复和原 U256/U124 透明传递逐条核对；运行日志不声称对 U 双执行反事实。其他 primitive 下游输入和 sigma 可因旁路而改变，不能声称整个旧训练随机轨迹或波形完全相同。

|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|native wall s|
|---|---:|---:|---:|---:|---:|---:|
|NOAS30|6078780|918004|0|963900|7960684|2146.190|
|AS30|6078780|918004|0|963900|7960684|2860.909|
|OC_NOAS30|2271960|324002|0|567000|3162962|899.555|
|OC_AS30|2271960|324002|0|567000|3162962|1128.245|

新增 20 模型轮/4440 更新；新增训练与读出 8703128 NN，共享实际 smoke 1540 NN，新增总计 8704668 NN。两个父 E20、两个旧尾 E10 与两个新尾 E10 合计 80 唯一模型轮/17760 更新；四条完整 E30 曲线为 120 逻辑行，复用训练不重复收费。相同前向量不等于相同墙钟；影子原语计算未省去，预算结束不等于收敛。
分析不重新预测或评分历史 NoAs30/As30。所有 RX/TX/day 配对错误、独立混淆矩阵、完整曲线、源分层与成本写入同一输出目录。分析新增 NN/IQ/checkpoint 读取/更新均为 0。

实际生成图：
- training_curves.png
- training_curves.pdf
- source_quality_states.png
- source_quality_states.pdf
- formal_all7_paired.png
- formal_all7_paired.pdf
- ordinary_channel_coverage.png
- ordinary_channel_coverage.pdf
