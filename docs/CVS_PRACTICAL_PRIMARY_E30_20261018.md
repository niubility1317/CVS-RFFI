# 星地识别优先：E30 探索与 E50 匹配预算验证

2026-10-18 用户最新要求：提高固定六种 practical 的识别性能，重点 low／mid urban；阶段 clean 相对同预算、同种子、同数据角色的 NoAs 下降不超过 0.1 个绝对百分点。最终仍需取得可重复的星地与 clean 双提升。单种子或不足 0.1 pp 的上涨只能作为初步观察。无需必须构造 Ag＋As，也不设独立额外跨 RX 泛化门槛；RX／day 分层继续报告。历史判据与结果保留。

固定 F＋CROSS_RX、三路 SIG_TIME／统一 Sinc／joint160／默认 cosine 头随骨干联合训练、NoExplicitCFO 与内部补偿；原 L6300／U56700（TX 隐藏）／V27000，源 RX=[1,3,4,6,8]、day=[1,2,3]、split392005、现有 IQ256。全部训练 RX 保留。正式 target RX=[0,2,5,7,9,10,11]、day=[0,1,2,3]，原 received IQ／信道／SNR／所有包不变。source 检查聚焦真实标签破坏、数值异常和明显性能损伤。目标数据不参与设计、选模、晋级、选种或救跑。

## 已完成证据

原完整 practical 配方仍有最清楚的星地收益。三模型种子相对匹配 NoAs：low urban +17.2849±3.4169 pp，mid urban +21.9448±5.1968 pp，urban 三场均值 +22.3079±5.5014 pp；clean −0.3210±0.5189 pp。clean 逐种子为 +0.0935／−0.1536／−0.9030 pp；未满足三种子的阶段要求，也未取得稳定双提升。完整逐种子、六场、F1／RX／成本在 As-replication-r02 报告，不把不同研究增益相加。

最新完成 `20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01`：新 AS_CE／AS_DET 两行各 scratch20＋10，复用 P43 同种子 NoAs／原 As 两行，L 前向和原 U 匹配。AS_CE 仅去 satellite KL；AS_DET 再仅对实际 As 行绕过 feature dropout，完整生成原 mask，未修改部署／分类头或信道。

该候选四项预登记源域条件均 PASS，不能改写成 source 失败。正式单种子 AS_DET 相对 NoAs 的 clean −0.126190 pp、mid urban +26.505357 pp、low urban +20.466667 pp、urban 三场均值 +27.117063 pp；clean 超过 0.1 pp 代价。相对原完整 As，clean／mid／low／urban 均值分别 −0.219643／−1.168452／−0.637500／−1.251389 pp，全部七视图都低于原 As 与 AS_CE。此固定单种子没有新增机制收益；不据正式成绩微调参数或选择性救跑，不以 n=1 宣称机制全局被否定。源通过未转化成正式收益。完整证据见该 run 的 final_analysis.md。

## 本轮已检验的唯一假设与最小矩阵

用户允许对有合法源证据的好配方使用总 E50。保留原完整 As，先固定 seed01 的 NoAs／As 各 own FINAL30＋20，再以源域预算收益为依据，固定原定 seed02／03 的相同四行，不挑种子、不改变参数。轨迹 20＋10＋20／reset AdamW＋cos20，非连续 cosine50。原01预测与六个E30历史不重跑，新四行均自动 source→七视图预测→独立评分。该预算假设现已完成验证。

01 的源预算依据为 full V clean +0.118519 pp、mid／low +1.972222／+1.638889 pp；01单种子正式 clean +0.365476 pp 曾仅作为初步观察，不能外推为重复性。最新完整 run 为 `20261018-phase1-FcrossRX-As-budget-e50-s0203-m4-r01`；执行提交 `7bbb5536cb72c9b95c7c2045ba660e1e72e234e7`，分析提交 `5f3a0fc9c128944994f40d4f4858a5f8e7730b40`。

## E50 三种子完整结果

**E50 对 low／mid urban 的增益可在三个固定种子复现，但 clean 未达到阶段要求，尚无可重复双提升证据。** 相对各自 As30，As50 的 low／mid urban 平均增加 1.156746 ± 0.092682／0.954167 ± 0.156568 个百分点，clean 平均变化 -0.017659 ± 0.296162 个百分点。相对匹配 NoAs50，clean 为 -0.150198 ± 0.955701 个百分点，均值和全种子口径都未满足下降不超过 0.1 个绝对百分点。

01／02／03 的匹配 clean 变化分别为 +0.365476／+0.436905／−1.252976 pp，前两个通过、第三个失败。全部固定种子保留，没有剔除第三个种子、补选种子或选择性重跑。E50 只指总预算 50 轮，轨迹为 20＋10＋20、追加段重置 AdamW／cosine20，不是连续 cosine50。

本轮证实“更多预算可进一步改善完整 As 的 urban 识别”，否定“单靠此固定预算扩展已解决 clean 代价”的判断。NoAs 自身延长训练使 urban 均值下降 1.943320 pp，因此不能把 As50−NoAs50 的全部差距归因于多训 20 轮。

|视图|NoAs50 准确率 %|As50 准确率 %|As50−NoAs50 pp（配对均值±样本 SD）|As50−As30 pp|As50 宏 F1 %|As50 最差 RX %|
|---|---:|---:|---:|---:|---:|---:|
|clean|84.714087|84.563889|-0.150198 ± 0.955701|-0.017659 ± 0.296162|84.214124|77.687500|
|practical_high|47.103770|80.764286|33.660516 ± 8.238954|0.437500 ± 0.463044|80.492918|72.925000|
|practical_mid|45.710119|78.223016|32.512897 ± 7.553454|0.587698 ± 0.372703|77.979399|70.998611|
|practical_low_suburban|44.131746|75.784524|31.652778 ± 6.656782|0.699603 ± 0.207028|75.554855|68.918056|
|practical_high_urban|45.565675|77.272421|31.706746 ± 7.664944|0.575992 ± 0.447469|77.032625|70.173611|
|practical_mid_urban|38.417460|62.989484|24.572024 ± 5.066220|0.954167 ± 0.156568|62.754430|57.552778|
|practical_low_urban|32.764683|51.926587|19.161905 ± 3.110887|1.156746 ± 0.092682|51.537900|47.713889|

urban 三场均值：NoAs50 38.915939%，As50 64.062831%，匹配增益 25.146892 ± 5.278997 pp；As50−As30 0.895635 ± 0.219271 pp。表中最差 RX 为每个种子自身最差 RX 的均值，不是所有种子最小值；完整逐种子值见最终报告。标准差 ddof=1，非置信区间。

clean 配对逐种子：+0.365476／+0.436905／−1.252976 pp；两个通过、一个失败，均值与全部种子口径均不满足 −0.1 pp 阶段要求，尚无可重复双提升。原 E30 clean 负结果、失败 dropout 与全部种子保留。

## 源域与正式结论的边界

三种子 full V27000 的 As50−NoAs50 clean 均值为 -0.069136 ± 0.029937 pp，V1800 固定 panel 为 -0.259259 ± 0.084863 pp。full V 的三种子均在 −0.1 pp 内，但小 panel 均超过该损伤幅度；两种口径不能混用。As50−As30 的源 full V clean 为 0.106173 ± 0.035006 pp，源 mid／low 均值为 1.986111 ± 0.160174 pp。这些是已冻结源域观测，不保证跨 RX 正式 clean 同方向。源域通过不等于最终目标通过，也不能依据正式负结果回调源阈值。

原完整 As 仍作为有实际星地收益的匹配参照。当前证据不足以证明 clean 与星地收益已经兼得；不将“源 full V 基本不降”写成“跨 RX clean 基本不降”。suburban 与 clean 的实际差距逐种子保留，不能靠 clean 下降缩小差距来宣称保护成功。最初建议的 2 pp 探索目标不是本轮新增硬门槛。

## 实际执行与计数

新增四行全部固定 FINAL50：每行新增 20 轮／4440 更新，累计 50 轮／11100 更新；完整读取新增 17760 个 step、80 个 epoch。02／03 的匹配物理序列 mismatch 均为 0、普通增强 MSE mismatch 均为 0。原六个 E30 与 01 E50 的完整 audit 直接复用，没有重复计旧日志读出或训练。

每新增行学生 train／eval 分别为 4,543,920／594,002 次样本前向，own EMA 教师 train／eval 为 0／1,134,000；源读出 12,600、正式预测 1,176,002。新增四行训练及读出 29,842,096，加本 run 共享 smoke 3,074，合计 **29,845,170 次样本前向**。01 E50 原两行新增成本 14,922,586 另列，六行 E50 扩展合计 120 model-epochs／26,640 更新／44,767,756 次样本前向；六个 E30 父共 180 model-epochs／39,960 更新／54,895,716 训练段及读出前向另列，历史共享 smoke 不任意分摊。分析不执行模型、不读 IQ／checkpoint、不产生更新。

实际 native 训练段耗时（含其验证，非并行总耗时或加速比）：

|新增 row|秒|
|---|---:|
|NOAS50-s2026092702|1626.249|
|AS50-s2026092702|2195.350|
|NOAS50-s2026092703|1589.310|
|AS50-s2026092703|2170.395|

已核实原 dispatcher、四训练进程和分析进程正常退出，failure 为空；所有源读出、12 个逻辑条件×7 视图的固定预测和独立评分完整，独立核对 1512 个混淆矩阵、252 次分区合并。训练 release 提交 `7bbb5536cb72c9b95c7c2045ba660e1e72e234e7`，分析提交 `5f3a0fc9c128944994f40d4f4858a5f8e7730b40`。最终汇总单次归档传输 SHA 比较 VERIFIED：`a8850417ce00879ab91c06e7ef283ce2b1ad3db1a7a53a8066f3bb9dbac45b8c`；只取回公开汇总，不取回 IQ／checkpoint／原始逐包预测。无技术缺项。

当前无运行中的本侧 E50 训练或评分任务，技术缺项为空。另一对话复用此 E50 矩阵，进行源状态报表并考虑单侧间隔；已做过的 urban SSL、去 satellite KL／dropout 不换名重跑。

## 下一项研究边界

本轮唯一预算假设已得到 urban 增益支持，但“更多预算足以实现 clean 基本不降／重复双提升”未通过。现有源摘要不足以支持一个独立新机制，不强造新候选，也不继续超出 E50。下一候选仍聚焦完整 practical 的简单训练组织；在明确改变哪个源域可观察原因后，再固定一个主要变化、同预算同 seed NoAs／原 As 对照和放弃条件，正式分数不参与参数、候选、seed 或选择性重跑。当前最终研究目标未完成。

完整逐种子、六场、F1、最差 RX、错误修复和新错及预算交互见 [E50 最终报告](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-As-budget-e50-s0203-m4-r01/final_analysis.md)。

## 下一项已固定的源域最小比较

现已登记 `20261018-diagnostic-FcrossRX-ownEMA-source-e50-s2026092701-m4-r01`：仅固定01已有E50的NoAs／As，各比较student与ownEMA完整状态。代码核实原正式与source均为student；本次不新增EMA、训练、头或部署分支。历史SIGTIME／BASE_SSL clean-only E20的EMA曾多错3个，保留负结果；当前F／CROSS_RX／As50训练来源改变，不能用旧CE下降证明本次有效。

full V27000四条件逐ID读出，原V1800与两次固定三场practical观测保持，学生panel预测复用。固定源判定：AsEMA−NoAsEMA clean≥−.1绝对pp；AsEMA−AsStudent fullV净修复≥0；mid／low等权合并净修复>0。逐场／draw／RX／F1／错误配对全部报告，不设每cell涨分或预测保持门槛。差分交互只作归因，不把通用EMA收益写成As独有收益。负结果完整收尾后关闭固定读出，不扫描衰减／epoch／seed救跑。预期新增133204次前向，0训练更新；本轮source-only，暂不访问formal。单种子source通过也不证明最终重复双提升。

原ownEMA source r01已按技术停止规则保留：真实smoke4NN PASS、四fullV完成后，适配代码混用了opaque ID和native辅助元组；108004NN／0更新／0target，无完整科学结果。r02按既有opaque ID与TX／RX／day严格连接，定向正负例和唯一原问题定点复核PASS；科学矩阵、来源与阈值均不改。当前源实验为`20261018-diagnostic-FcrossRX-ownEMA-source-e50-s2026092701-m4-r02`。

既有源缓存实际含11个条件，r02误写轴长度1而停止，追加108004NN／0更新／0target。已实际0NN核对header/completion；r03固定原11轴结构与index0full，前移输入检查，定向测试及该问题定点复核PASS。两失败合计216008NN保留，不作科学负结果；当前修复run `20261018-diagnostic-FcrossRX-ownEMA-source-e50-s2026092701-m4-r03`，原科学矩阵不变。

## 既有ownEMA源比较：已完成负结果

`20261018-diagnostic-FcrossRX-ownEMA-source-e50-s2026092701-m4-r03`已完整完成四条件源读出，实际代码`649487724adda1b69243f9efd441f2c05160f781`。AsEMA−AsStudent：fullV clean修复5／新错7，净−2（−0.007407pp）；mid urban净−4（−0.111111pp），low urban净−7（−0.194444pp），两场均值−0.152778pp。AsEMA−NoAsEMA的fullV clean−0.066667pp在源阶段范围内，但EMA没有改善原As。固定候选关闭，不扩种／不访问formal／不调衰减或epoch救跑；单seed微小负值不外推EMA全局无效。

成功r03新增133204NN／6CP／0训练更新，两个技术失败另216008NN／12CP，路线总349212NN／18CP／0更新；fullV学生命中与原native完全一致，120CM／16分区合并核对PASS，全部源输出完整，无技术缺项。没有新formal结果；E50三seed正式clean−0.150198pp及未取得重复双提升的结论保持。详见[源比较最终报告](../automation_reports/CV-SincNet/20261018-diagnostic-FcrossRX-ownEMA-source-e50-s2026092701-m4-r03/final_analysis.md)。当前无本侧在训或源读出进程，后续继续以增强训练组织为研究范围，不从本次负结果捏造新的物理退化归因。

## 当前唯一训练假设：等量 As 的步骤分配

已登记 `20261018-phase1-FcrossRX-stepALT-e30-s2026092701-m1-r01`，实现中，尚未发布。固定原 As20＋10＝总 E30，复用原 As30／NoAs30；唯一变化为末段 mixed128 按累计奇偶 step 分别使用128／0个As，替代每步64。每步6L／总尺度1.25、末段142080As、原U规则、uniform三场与全部源RX保持。按物理ID的As次数不相同，报告实际曝光，不解释为纯梯度时机效应。另一对话的.2／.4／.4场景prior路线不重复。

源支持固定为fullV clean相对NoAs30≥−.1绝对pp，mid／low均值相对原As30>0；不通过则关闭固定方案，正式七视图仍自动收尾。通过只支持继续研究，不等于目标性能通过。用户最新授权允许表现有源证据的方案总E50，当前不追加；后续扩展需固定匹配预算对照，正式分数不作依据。

当前执行状态已更新：代码 `cc9a2dac5e2667dab9d730dd8b618e8e635fc5ca` 的唯一新增 E10 已在 GPU0 开始；真实As20烟测2NN＋128/0双步1536NN PASS，PID1233511／1233539、CWD／argv／GPU和日志增长独立VERIFIED。实际source_roles匹配、ownEMA严格恢复、信道与原As30一致。尚未训练／评分完成，不作新增性能结论。

## stepALT E30：完整收尾，固定配方关闭

`20261018-phase1-FcrossRX-stepALT-e30-s2026092701-m1-r01` 已完成一条新增E10（共2220更新、总E30／6660）及两个旧E30逻辑对照复用，全部clean＋六practical固定预测与独立评分。训练代码`cc9a2dac5e2667dab9d730dd8b618e8e635fc5ca`，零前向分析r02代码`2e08d8ddabd28ca4bad350485a7a98915c006046`。源判定在读取正式分数前已失败：fullV clean对NoAs−0.148148pp、mid／low对原As均值+0.111111pp，因此不E50／不扩种／不扫周期相位救跑。

正式n=1：candidate对NoAs clean−0.005952pp、mid+27.613690pp、low+21.127381pp、urban均值+28.331944pp；本seed clean阶段界限通过。新增机制对原完整As：clean−0.099405pp、mid−0.060119pp、low仅+0.023214pp、urban均值−0.036508pp，未获得新增收益，不能把完整As原有收益归给stepALT。low最差RX仍下降0.141667pp，suburban绝对值均略低于原As，不能以clean下降导致的小差距缩小宣称保护。无重复clean＋星地提升证据。

全部2220步／10轮及418行native日志已解析；6L／总尺度1.25，142080As，6300L物理ID全覆盖，次数11至36（不声称逐ID与原配方相同）；4440条L／U物理序列与2220个ordinary MSE记录对原尾段均零差异。新增4353102NN含smoke1538，父20原生4797722NN及旧对照不重计。独立378CM／63分区合并核对PASS，取回11汇总文件，无技术缺项。r01只读分析曾误映射新版totalL字段，保留0NN失败；r02基于真实首末记录修复，8项定点检查PASS，未重训／重复预测／评分。所有本侧进程已退出。

当前无已登记的新候选。原完整As继续作为有源域和正式星地收益的匹配参照，E50三seed的clean负结果保持；下一机制只根据合法源结果及未检验的关键假设确定，不用本次正式微差选seed／参数／方向。完整[本行报告](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-stepALT-e30-s2026092701-m1-r01/final_analysis.md)。研究目标仍未完成。

## 下一项唯一假设：现有As的监督份额

已登记`20261018-phase1-FcrossRX-Asweight-e30-s2026092701-m2-r01`，实现中，尚未启动。保持原As64／全部IQ和6L／总尺度1.25，仅将mixed槽ordinary／selected权重.25/.25改为.125/.375；.5primaryordinary、.25raw、原CROSS_RX和U保持。As行的selected是As(raw)，NoAs行selected实际是raw.clone（保留原代码路径），所以新增各自own20+10两行，与原两E30复用组成4逻辑条件。有效clean权重1.0→.875、As.25→.375是明确监督转移，不能冒称clean总份额不变。与peer普通PA旁路不同，不改物理信道或普通增强。

本轮源E50推进规则提前固定为：newAs fullV clean对原As≥−.1pp，源mid／low两draw等权对原As≥+.5pp；其余源结果及NoAs控制、交互完整报告，不加额外门槛。此增量规则不追溯改写旧stepALT关闭决定。任一失败关闭固定.375权重，不扫权重/seed救跑；科学负结果仍全部正式收尾。正式stage依旧newAs对匹配newNoAs clean≥−.1pp，并另报原NoAs。全部原数据／默认头／来源边界、预算和原测试IQ不变。

发布前按真实调用纠正文案：原P43 B00的NoAs.apply输入raw[mask]并返回raw.clone，不是配置文案所暗示的ordinary。新两行均保留原IQ／RNG；NoAs权重为ordinary.75/raw.5→ordinary.625/raw.625，As因子为selected raw→As(raw)。原代码／历史结果／健康任务不改。

Asweight当前已实际启动：训练／分析代码`3befd47c8041d81f9430eda53562d0821624ff04`；dispatcher1259300，B00 worker1259736 GPU0、B01 worker1259814 GPU2，actualPID／CWD／argv／GPU／日志增长VERIFIED。双父真实CP4NN＋两行6L1536NN共1540NN smokePASS后正常训练；source roles均EXACT_MATCH、target_contact=false、各自ownEMA严格恢复，物理配置与原row一致。两新尾段均完成1／10轮；0NN分析1260826处于已核实WAIT_SOURCE，truth未开，无新性能结果。


## As监督权重E30：完整评分及源支持E50

两新行已完成own20+10／总E30及全部七视图独立评分，0NN分析完成，无技术缺项。单seed weightedAs30相对原As30：clean−.488095pp、mid＋.605357pp、low＋.717857pp、urban均值＋.532738pp；相对匹配weightedNoAs30 clean−.326786pp，未满足−.1pp阶段界限。最差RX／F1／分层／错误修复和新增／成本完整保存，未证明重复双提升。sourcefullVclean对原As−.059259pp、两drawmidlow＋.777778pp符合预先固定的增量条件；source决定已在首次读新formal前保存，不能追溯改判或把source通过当正式通过。

据此已固定下一项`20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01`：只增加预算，matchedNoAs／As各ownweightedFINAL30 student＋ownEMA追加20轮，总50，resetAdamWcos20；.375权重／IQ256／全5源RX／L6300U隐藏V27000／默认头／原完整practical三场／6L1.25／U不变。ordinary仍原20参数epoch13，SupCon＋30，channel31至50／step6661至11100。只新训两行，其余原／加权30和原50六个冻结结果复用，形成weight×As×budget八条件。唯一审查P0/P1=0/0，68聚焦负例与AST PASS；真实烟测与发布待执行。

未来固定02／03扩种支持只看As50−ownAs30源fullVclean≥0pp且midlow均值≥＋.5pp。任一失败放弃固定配方扩种，仍完成本行all7正式；不扫权重／epoch／seed救跑。目标stage仍相对同预算匹配NoAs clean≥−.1pp。下一项唯一假设是更多预算能否缓解当前clean—星地矛盾；不从已见正式分数重新选择路线或承诺60%。[E30完成报告](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e30-s2026092701-m2-r01/final_analysis.md)；[E50预登记](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01/report.md)。

Asweight E50已实际启动：训练／分析代码`64176e7ff1aa5cd2477a504f8b442bb7966079dc`；dispatcher1276091，NoAs1276185／GPU0、As1276263／GPU1，真实双父＋两6L共1540NN smokePASS后训练，source roles／ownEMA／resetAdamWcos20／实际physics匹配。PID／CWD／argv／GPU／日志增长独立VERIFIED；0NN分析1277501已核实WAIT_SOURCE／truth未开，将自动闭合all7。新增段每条20轮，尚无E50成绩。


## 加权E50单seed完成：阶段通过，源支持固定02／03

`20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01`已完成两ownweighted30+20及全部七视图独立评分，代码`64176e7ff1aa5cd2477a504f8b442bb7966079dc`，0NN分析完整、1008CM／168分区合并核对，所有进程正常退出／无技术缺项。新段共40model-epochs／8880updates／14922588NN含1540smoke；每条累计50轮／11100updates。真实source fullVclean对自身weighted30＋.162963pp、两draw midlow＋2.361111pp通过固定资格，root先保存决定并登记02／03，随后才首次读取新formal指标，正式结果不作扩种反馈。

单seed weightedAs50：clean85.248810%、mid64.483333%、low53.242857%、urban均值65.508333%；对匹配weightedNoAs50 clean−.042857pp，本seed满足−.1pp阶段要求，仍非clean正收益／重复双提升。对自身weighted30 clean＋.395833、mid＋1.071429、low＋1.280952pp；对原As50 clean−.411905、mid＋.581548、low＋.855357pp，取舍仍在。clean／mid／low三因子交互＋.011905／−.269048／−.100000pp，不能把预算增益写成新权重独有协同。high／mid／lowsuburban为82.282143／79.757738／77.213690%，相对自身clean差2.966667／5.491071／8.035119pp，绝对值较自身30提高，仍未接近最初2pp探索建议。完整F1／最差RX／错误及成本见[完成报告](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01/final_analysis.md)。

现已登记`20261018-phase1-FcrossRX-Asweight-e50-s0203-m4-r01`，代码/分析准备完成但尚未发布。固定02／03各NoAs／As四轨迹均为原own20＋新weighted10＋新weighted20，总50，新增120model-epochs／26640updates，不重训seed01或旧20／30／50。nested weighted30/weighted50各自source／固定all7pred／独立score，中间数值不决定是否继续50；真实factoryseed同行02／03，ordinary始终ownoriginal20args epoch13，offset20/4440、30/6660。24logical＝3seed×original/weighted×NoAs/As×30/50；只新8stage读出，旧16条件复用。172聚焦负例／8native绑定／8预测路径／19py2AST与9分析检查PASS，唯一新适配P0/P1=0/0，真实3080NN烟测在launcher首步。当前无扩种成绩，不保证60%或重复双提升；保持同一假设，不变权重、数据、默认头、U或物理条件。[源决定](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01/source_decision.json)／[复现预登记](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e50-s0203-m4-r01/report.md)。

固定02／03已实际启动：代码`094a5e9e063b52c99b646c33c2b357a388ece5b8`；dispatcher1303723，四worker1303753／1303830／1304318／1304399分别GPU0至3。四源角色／ownEMA／actualFfactoryseed／physics匹配，3080NN真实烟测PASS后进入weighted30；PID/CWD/argv/CUDA/nvidia/log增长独立VERIFIED，全GPU compute进程各1。0NN分析1305922 WAIT_SOURCE／truth未开。后续weighted50各自从新30无条件继续，当前无扩种性能结果。


## 固定02／03：E30中间读出完成，E50继续

四条weighted30中间段已完成各10轮／2220更新与全部七视图独立评分；同PID已严格从自己的新30 student／EMA进入固定20轮50段。实际factoryseed02／03、empty resetAdamW／cos20、6L／As0或64／.125.375核实，无技术异常。中间数值不决定续训。

12个E30逻辑条件的现有独立scorer汇总：weightedAs−matchedweightedNoAs clean逐seed−.326786／−.505357／−1.227381pp，配对均值−.686508±.476843pp，全部seed及均值均不通过−.1pp阶段要求。weightedAs−originalAs clean−.388095±.089660pp，mid／lowurban＋.531151±.121889／＋.703770±.105766pp，urbanmean＋.490675±.130200pp；n=3表明该固定E30配方存在clean—urban取舍，未获得重复双提升。报告全部seed，不挑最好值。

本部分只读现有独立scorer，未读truth／IQ／checkpoint／rawpred，未增加预测／评分／训练。完整24条件重计数、逐包错误和E50结果仍待固定全部阶段完成。当前唯一假设仍是增加预算能否恢复clean并保留urban收益；本读出不改变已运行E50轨迹或发起新候选。[部分报告](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e50-s0203-m4-r01/partial_E30.md)。


## Asweight E50三种子：完整负结果与预算收益

固定02／03四轨迹已全部完成新10＋20及所有8新阶段source／clean＋六practical独立评分，所有本run进程正常退出。完整24条件×7视图独立重算3024CM／504分区，新增120model-epochs／26640updates／47251432NN含3080smoke；旧20/30/50不重复训练或计为新增。源先取回，随后正式取回，单次归档比较分别VERIFIED，无技术缺项。实际训练／分析代码094a5e9e063b52c99b646c33c2b357a388ece5b8。

weightedAs50对匹配weightedNoAs50 clean逐seed−.042857／−.177976／−1.797619pp，配对均值−.672817±.976447pp，仅01通过，均值及全seed阶段均FAIL。对自身weightedAs30，low／midurban＋1.199603±.076713／＋.995437±.083559pp、urbanmean＋.878571±.144700pp，clean−.171825±.492087pp；urban预算收益可重复，clean恢复不可重复。对原As50，low／midurban＋.746627±.095732／＋.572421±.057687pp，clean−.542262±.112959pp，不能把完整As原有收益全归给权重机制。

加权As50绝对均值clean84.021627%、midurban63.561905%、lowurban52.673214%、urbanmean64.536442%。suburban三场80.779960／78.424008／76.132341%，与自身clean差3.241667／5.597619／7.889286pp；不能用clean下降造成的差距缩小冒称保护成功。源fullV As50−ownAs30 clean＋.132099±.028288pp、midlow＋2.375000±.077330pp虽重复，不代表跨RXclean同向。正式目标、源判断和计算成本分别报告。

本固定.375配方作为双提升候选关闭，不再加预算、扫权重、挑seed或救跑；研究目标保持未完成。当前本侧无在训／预测／评分进程。下一项只读核对历史“保持clean绝对监督量”的真实代码、源证据和前向／总尺度控制，尚无新候选登记或训练；不依据本次formal设计参数。完整[最终报告](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e50-s0203-m4-r01/final_analysis.md)。


只读历史核对已完成首部分：旧additiveAs虽有8L／重复raw及总尺度控制，但使用heldRX1／L5040，跨seed源确认失败；cleanretain37为4L比例比较，不是当前6L的加性保留；HQ的BRIDGE_CE仍有joint anchor，并非真正无一致性控制。当前不据此重启加性路线。P41也曾把完整practical selected64加入现有SupCon（SAT192／DUP192／ORD128），不增前向，固定源晋级失败；不能将“让As进入SupCon”包装成完全新机制。后续只读区分其192 anchor／重复物理样本权重与当前128 anchor的真实调用，尚未选定或登记新候选。


## 下一项已登记：固定128个对比位置的视图替换

20261018-phase1-FcrossRX-mixedContrast-e30-s2026092701-m2-r01已登记PLANNED，尚未远端启动。唯一变量将原CROSS_RX SupCon的ORD128 maps替换为已生成MIX128，原L batch每位置一次、不追加SAT192的64positions，原sampler重复如存在保持。P41 SAT192已源失败，不宣称广义As进入SupCon全新；本次明确固定128positions／当前6L，与旧192／4L区别。不新加loss／层／branch／NN，原P43 .5ordinary+.5mixed+.25raw总L1.25／原U、all5RX/L6300/默认cos/IQ256保持。新NoAs／As01各own原20+10=总E30，原两E30复用。

Source资格固定newAs−originalAs fullVclean≥−.1pp且两drawmidlowmean≥+.5pp；无额外cell门槛。失败关闭固定候选、不E50／扫参数救跑，仍自动all7正式收尾。正式stage仍newAs−matchednewNoAs clean≥−.1pp；单seed不证明重复双提升。实现73聚焦负例和AST已PASS，analysis接口检查与唯一P0/P1／真实CP烟测／发布待完成；不把准备写成启动。完整矩阵和命令见[预登记](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-mixedContrast-e30-s2026092701-m2-r01/report.md)。


mixedContrast当前已实际启动：代码／训练／分析提交 6849fbf648895c055398e6db159844dba0de3187，一次普通preflight的源父／控制句柄与roles匹配，单archive比较／compile VERIFIED，1540NN首步CP＋真实模型梯度烟测PASS后训练。dispatcher1353621；B00 worker1353649／GPU0、B01 worker1353729／GPU1，PID／CWD／argv／nvidia／log增长核实。实际own20 EMA、empty resetAdamW cos10、6L／128原batch位置／128anchors／4mixed map terms／rawmaps0／extraNN0／As0或64均符合。原sampler重复如存在保持，不强制物理ID全unique。0NN分析1355032已核实WAIT_SOURCE／truthclosed。现在没有新性能结果或额外E50授权执行；既定所有row自动all7正式闭合。


## mixedContrast E30：完整负结果，固定视图替换关闭

20261018-phase1-FcrossRX-mixedContrast-e30-s2026092701-m2-r01已完成两条own原20+10轨迹，总E30／6660更新，全部source及clean＋六practical冻结预测与独立truth-last评分。代码／配置／分析提交6849fbf648895c055398e6db159844dba0de3187。四逻辑条件×七视图完成504CM／84分区核对，所有本run进程正常退出，无技术缺项；新增20model-epochs／4440updates／8704668样本前向（含1540smoke），原20及原E30控制成本分别保留、不重计。

root在新formal前先取回source并保存决定：newAs−originalAs fullVclean＋.033333pp通过clean护栏，但两drawmidlow均值仅＋.069444pp，未达预登记＋.5pp。固定MIXED128配方据此关闭，不扩E50／不扫mask、权重或种子救跑；该决定先于正式读取，正式结果不用于改判。科学负结果仍完成全部正式收尾。

正式n=1：MIXED_AS30 clean84.493452%、midurban62.749405%、lowurban51.187500%、urbanmean63.853968%。对匹配MIXED_NOAS30 clean−.665476pp、mid＋27.692262pp、low＋21.134524pp、urbanmean＋28.368651pp，clean−.1pp阶段界限FAIL。对原完整AS30 clean−.847619pp、mid−.057143pp、low−.056548pp、urbanmean−.085317pp，六practical绝对准确率均略低，没有新增机制收益；不能把完整As原有收益归给视图替换。原AS30在本seed对原NOAS30 clean＋.093452pp仅为不足.1pp的微小观察，不能证明重复双提升。

clean／mid／low交互−.758929／＋.018452／＋.030357pp，urbanmean交互＋.000198pp，不支持星地协同。新As三suburban81.250595／78.505357／75.793452%，与自身clean差3.242857／5.988095／8.700000pp；不能把clean降低造成的差距缩小算作保护。完整macroF1／最差RX／逐RX-day-TX／修复与新增错误、全部训练日志解析和成本见[最终报告](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-mixedContrast-e30-s2026092701-m2-r01/final_analysis.md)。

本次唯一假设已否定；当前本侧没有在训／预测／评分进程，也没有登记新的候选。E50授权保留给有源证据支持的配方，不为已关闭机制增加预算。已完成的加权E50三种子证明更多预算的urban收益可重复，但clean代价未满足阶段要求；整体“星地增强同时改善clean”的可重复目标仍未完成。后续只据合法源证据与未检验的关键假设确定新路线，不由本次formal微差推导或挑选方向。


## 下一项：As64批内三场景配额

20261018-phase1-FcrossRX-sceneBalance-e30-s2026092701-m1-r01已登记PLANNED，实现中，尚未发布。当前原As本来逐pid IID三scene，本项仅把每步64包scene counts改为21/21/22、cumstep轮换extra、private RNG打乱原64位置；不改物理依赖、损失、ORD128 SupCon／6L1.25／oldU／全源RX／IQ256／默认头。旧sceneCycle是每PID跨6次访问六场覆盖，heldRX1/L5040/8L且源失败，本项不是该周期救跑。仅新增own原As20+10一行，总E30；原NoAs30与As30冻结读出复用，operator对NoAs恒等且无globalRNG或NN改变。

源双条件固定BAL fullVclean−NOAS30≥−.1pp且two-drawmidlow BAL−AS30≥+.5pp；失败关闭配方不扩E50/扫参数/换seed，所有row仍自动all7评分。源通过只支持未来预算内E50资格，先保存source决定，正式不作反馈。唯一假设是减少每step场景混合权重采样方差可否改善practical；当前无性能证据。完整[最小报告](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-sceneBalance-e30-s2026092701-m1-r01/report.md)。


sceneBalance本地代码／分析已冻结：19py／47聚焦负例、全部2220配额／私有RNG／重复PID／实际下游mock／2remoteAST及4analysisAST／9toy PASS，唯一合并P0/P1=0/0。当前没有训练或性能结果；真实770NN烟测由launcher首步执行。nominal1/3不等于与原IID实际每包／逐ID／尾段次数匹配，不把固定分层效果直接称纯梯度方差证据。


sceneBalance已实际启动：code/config/analysis 25747acec3171d78ef4e841c90255242b96fd7aa；一次普通preflight/source句柄roles匹配，单archive比较/compile VERIFIED，oneCP2＋6L768共770NN首步smokePASS直接训练。dispatcher1381372、B01worker1381800／GPU4；PID/CWD/argv/CUDA/nvidia/log增长核实。当前strictownAs20EMA、resetAdamWcos10、originalORD128／6L1.25／原U／配额21/21/22符合，0NN分析1383360 WAIT_SOURCE／truthclosed。最新4/10新增epoch，无新性能结论；原全部数据域／默认头／测试IQ未变，健康其他任务未动。


## SceneBalance E30：阶段单seed通过，源规则失败关闭

20261018-phase1-FcrossRX-sceneBalance-e30-s2026092701-m1-r01已完成own原As20+10，总E30/6660更新，全部source及clean＋六practical固定预测、独立truth-last评分与三条件378CM/63分区核对。actualcode25747acec3171d78ef4e841c90255242b96fd7aa，所有所属进程退出、无技术缺项。新增10model-epochs/2220updates/4352334NN含770smoke，原20父和两E30控制读出不重计；完整2220步/10轮、原14native字段与物理曝光已解析，每场景尾段47360、每epoch4736，extraNN0，原6L1.25/ORD128/oldU/默认头/全源RX保持。

root先取回source并保存关闭决定：fullVclean BAL−NOAS30−.118519pp，two-drawmidlow BAL−AS30＋.041667pp，两个提前固定条件均FAIL，故不E50、不扫描配额/周期相位/seed救跑。该决定在首次读取新formal之前保存；科学失败仍完成all7。归因只针对固定分层分配，nominal1/3不等于原IID实际逐包/逐ID/尾段计数匹配，也不声称已隔离纯梯度方差。

正式n1：BALclean85.395833%、midurban62.796429%、lowurban51.233929%、urbanmean63.940873%；相对匹配原NoAs30 clean＋.148214pp、mid/low＋27.663690/＋21.094048pp，本seed满足clean−.1pp阶段护栏。对原As30，clean仅＋.054762pp、mid/low各−.010119pp、urbanmean仅＋.001587pp，没有实质新增星地收益；单seed和微小涨幅不证明重复双提升。三个suburban81.376190/78.624405/75.802976%，与自身clean差4.019643/6.771429/9.592857pp，比原As扩大。low<60%，整体目标仍未完成。

固定配方已关闭。当前本侧无已登记新候选或运行进程，E50授权保留给有合法源证据支持的配方；不由本次formalclean正差反向晋级、换父模型或选seed。原完整As继续作匹配参照。完整[最终报告](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-sceneBalance-e30-s2026092701-m1-r01/final_analysis.md)包括全七view、macroF1/最差RX、源两draw、错误修复/新增、成本与缺项，三组图95点核对通过。source inference wall未单记，不伪造；图和report不作为新的审批层。


## 本轮去重：N0 与延迟 As 均已有匹配训练

本轮核对实际发布源码、登记和源统计：完整 practical 内 0.5/1.5N0 随机化已有匹配 E30 训练并按原判据关闭；NoAs20→As10 也在全 5 RX/L6300 的 carrierGate AS_BATCH 与 shadowLHS AS_NATIVE 中完成，源 V1800 clean 对匹配 NoAs 为−.500000/−.444444pp。其4L/As128与当前6L/As64不同，不据此泛化为所有延迟As无效，也不把接口/剂量差异包装成新关键假设重开。N0源输入失败与后续独立训练关闭的依据分开记录。

原完整As及固定加权As的E50三seed已完成；相对ownAs30的low/mid urban分别＋1.156746/＋.954167pp与＋1.199603/＋.995437pp，但相对匹配NoAs50的clean为−.150198pp与−.672817pp，均未满足−.1pp阶段要求。更多预算的urban收益可重复，clean与星地双提升仍无重复证据。本轮0新增更新/NN/CP/IQ/远端执行/正式预测评分，无新候选登记；E50不超过总50轮，失败配方不加预算救跑。完整[去重与预算边界](../analysis/cvs_practical_route_dedup_20261018.md)。


## 下一项已登记：末段关闭 U

20261018-phase1-FcrossRX-tailNoU-e30-s2026092701-m2-r01已登记PLANNED，实现中未发布。仅新增NoAs/As各own原20+10，总E30；末段真实U/EMA/教师/学生U前向关闭，保留父20的原U历史、原6L1.25/ORD128/Asraw64/普通aug父epoch13与全部源RX/默认头。原NoAs30/As30冻结读出复用，共4逻辑条件。nativephase字段0/10保留，但实际U消费0；global RNG和前向成本变化如实报告，不说纯伪标签loss归因或全程NoSSL。

唯一假设：末段真实标签监督能否保留clean并改善practical。源资格固定新As fullVclean−新NoAs≥−.1pp且two-drawmidlow新As−原As≥+.5pp；失败关闭固定方案不扩E50/扫U/seed，all7正式仍自动收尾。当前F只有GN/LN，无BNrunningstats；连续scratch6L E30也已有源失败关闭，不重开两路线。详见[预登记](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-tailNoU-e30-s2026092701-m2-r01/report.md)。


tailNoU已实际启动：代码 17cc611299091815ea71f744a32dcac96818ca4b；dispatcher1417383，B00/B01worker1417411/1417493在GPU0/1。单次普通preflight/归档比较/compile VERIFIED，1540NN真实双父＋两6L首步smokePASS后训练。source roles/strictown20学生、父EMA验证但子不构建、actualNoU/NoEMA、原6L1.25/ORD128/defaulthead核实；PID/CWD/argv/CUDA/nvidia/log增长VERIFIED。0NN分析1419028 WAIT_SOURCE，truthclosed，尚无新结果或E50扩展。


## tailNoU E30：全部完成，源规则关闭

20261018-phase1-FcrossRX-tailNoU-e30-s2026092701-m2-r01已完成两条own原20+10，总E30/6660更新，全部source及clean＋六practical固定预测/独立truth-last评分；四逻辑条件504CM/84分区独立核对。代码/配置/分析17cc611299091815ea71f744a32dcac96818ca4b，所有所属PID结束，无技术缺项。新增20model-epochs/4440updates/6436668NN含1540smoke；两尾段真实U/teacher/EMA均0，childEMA=None，原6L1.25/ORD128/defaulthead/Asraw64保留，父20U7历史不改。去U改变RNG/计算，不能归为纯伪标签loss效应。

主线程先保存源决定：fullVclean新As−新NoAs−.133333pp、two-drawmidlow新As−原As＋.097222pp，两固定条件均FAIL，故关闭固定tailNoU，不扩E50/扫U/phase/seed。正式不反向改判。

正式n1：新As clean85.330357%、midurban63.033333%、lowurban51.437500%、urbanmean64.140675%；相对匹配新NoAs clean＋.216071pp、mid＋28.002381、low＋21.278571pp，本seed通过−.1pp阶段界限。相对原As，仅mid＋.226786、low＋.193452、urbanmean＋.201389、clean−.010714pp。新NoAs自身clean下降.133333pp，不能将整个相对差扩大归为As改善。三个suburban81.641667/78.769048/76.018452%，距自身clean3.688690/6.561310/9.311905pp；不接近最初2pp探索建议，也不能把clean下降贡献算作保护。

本轮支持“去U末段有n1小幅星地增量且clean对原As基本不变”的描述，不支持源晋级、稳定双提升或“U普遍有害”的结论。原完整As继续作匹配参照；当前无本侧训练/预测/评分进程，也无已登记下一候选。研究目标仍未完成，不由本次formal微差选择新机制。完整[最终报告](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-tailNoU-e30-s2026092701-m2-r01/final_analysis.md)含全7/F1/最差RX/错误修复新增/交互/成本及缺项。


## 下一项已登记：L ordinary 输入换 raw

20261018-phase1-FcrossRX-LRaw-e30-s2026092701-m2-r01已登记PLANNED，实现中，未启动。仅新增两条原P43 own20student+EMA+10，总E30，旧NoAs30/As30读出复用。唯一变量为L专属ordinary输出改precallrawclone；原oldaug仍执行一次作RNG/成本控制但输出弃用，Uweak/strong/EMA原规则保持。原6L1.25/Asraw64/ORD128（实际anchor输入raw）/defaulthead/全源RX和IQ256保持，前20ordinary历史不改。PA-only历史已做，本项旁路整链输出及其归一化尾部；原As本来raw，不称修复重复损伤。

源规则仅newAs fullVclean−newNoAs≥−.1pp且twodrawmidlownewAs−oldAs≥+.5pp；失败关闭固定方案，不E50/扫开关/seed，全部all7仍自动评分。局部call RNG可验证，整个训练轨迹不假定相同，原aug计算并未省掉。RXpopulation/calibration-refresh已有actualE30，不重开。详见[预登记](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-LRaw-e30-s2026092701-m2-r01/report.md)。


LRaw已实际启动：代码 d14450bf1b0d12cc281df1f4d0e79e8760e48282；双own20+10 worker1444640/1444720在GPU0/1，首步1540NN真实烟测PASS，原U/EMA/ordinary factory实际开关与源物理角色匹配。自动all7/独立truth-last及0NN分析1445589已接通；性能尚未产生，source先读规则不变。


## LRaw E30 完成与源域支持的 E50 延长

LRaw30已ANALYZED，代码 d14450bf1b0d12cc281df1f4d0e79e8760e48282，两条各2220新增更新、全部all7与4逻辑独立重计504CM/84partition完成，无技术缺项。source先读归档2188eb7864e49d75940239842b4ae463e1c9af7c3c059aedf9002d40e96307f2；newAs−newNoAs fullVclean−.029630pp、midlownewAs−oldAs＋1.138889pp，两固定规则PASS。先保存源决定并登记 20261018-phase1-FcrossRX-LRaw-e50-s2026092701-m2-r01 两条own30+20，再读取formal；新50不改参数/seed，不由formal救跑。

正式n1新As clean84.352381%、midurban62.907143%、lowurban51.597619%、urbanmean64.023810%；对匹配newNoAs clean＋1.633929pp（stage界限n1PASS）、mid＋29.510119、low＋23.147024pp。但newNoAs自身clean对oldNoAs下降2.529167pp，新As对oldAs clean−.988690pp，mid＋.100595、low＋.353571、urbanmean＋.084524pp；不能将stage相对差写成clean保护/重复双提升。六practical变化与F1/最差RX/逐RXdayTX/错误/交互/自clean差在完整报告中保留。

本次新增8704668NN含1540smoke，20modelE/4440newupdates；每新tail学生2271960NN、教师567000NN、nativeeval324002NN，原ordinary计算284160输出/row仍执行并丢弃。父20和旧30成本复用不重复收费，source预测wall未单独保存如实披露。E50已PLANNED实现中，当前无该50训练/评分结果。完整[报告](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-LRaw-e30-s2026092701-m2-r01/final_analysis.md)。


LRaw E50下一项固定矩阵已登记PLANNED：20261018-phase1-FcrossRX-LRaw-e50-s2026092701-m2-r01。sourcePASS决定和登记早于E30formal读取；两条strictownLRaw30student+EMA+20，总50/11100，原ordinarydraw-discard/raw返回、U/EMA/6L1.25/Asraw64不改。原NoAs50/As50为 20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01，实际releasecommit0f399c910324e21a8b1a60346d879be5c2964a9c。source50仅描述预算变化、无新晋级gate；完整8逻辑all7/13paired及成本分开报告。新增代码本地100focusednegative/19AST与analysis9synthetic/targeted20epochPASS；唯一合并P0/P1进行中，尚未启动50。详见[预登记](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-LRaw-e50-s2026092701-m2-r01/report.md)。


LRaw50已实际启动：代码 1cf3bf1c9232110b4199e01628b682b74a47d3a9；dispatcher1465885，双worker1465929/1466063在GPU0/1，总E50固定计划。1540NN首步smoke PASS、源角色/strictownLRaw30CP/EMA/20轮U实际开关及PID/CWD/argv/GPU/log增长VERIFIED；analysis1466559 WAIT_SOURCE。当前无50性能结果，计划不受已报告30formal改变。


## LRaw E50 已完成：预算未解决 clean 代价

20261018-phase1-FcrossRX-LRaw-e50-s2026092701-m2-r01已ANALYZED；训练代码 1cf3bf1c9232110b4199e01628b682b74a47d3a9。两新行各追加20/4440，总E50/11100；全部all7/独立truth-last和20轮全日志完成，8逻辑新2旧6重计1008CM/168partition，无技术缺项。source先读归档90b0de366386f865d59a1585447eeeb580ef8d35b18cc0ba6940abe088a5fd53，formal取回a8f62e3aea6d00d34330cbba9738199cc2aca1e9952819773f16be84bbaa21eb；E50资格早于E30formal读取已按source固定，本次source无新晋级gate。源As50−own30 clean＋.148148pp、midlow＋2.083333pp，没有在本次正式clean复现。

正式n1新As50 clean83.138095%、midurban63.342262%、lowurban52.470833%、urbanmean64.374802%。相对own30 clean−1.214286pp、mid＋.435119、low＋.873214、urbanmean＋.350992pp；相对原As50 clean−2.522619、mid−.559524、low仅＋.083333、urbanmean−.636111pp，低于.1pp的局部微增不当稳定改进。newNoAs50 clean82.573810%，newAs50−匹配newNoAs50＋.564286pp、net948包，点估计通过−.1pp阶段界限；对原NoAs50却−2.157143pp，不能将弱化NoAs对照写成原基线clean保护或重复双提升。

suburban三个accuracy80.645833/78.381548/76.017857%，距自身clean2.492262/4.756548/7.120238pp，差缩小有clean下降贡献；low未达60%。F1/最差RX/13配对错误/三阶交互/RXdayTX都在[完整报告](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-LRaw-e50-s2026092701-m2-r01/final_analysis.md)，E30补充[错误解释](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-LRaw-e30-s2026092701-m2-r01/error_analysis.md)已保留，无额外NN/评分。

E50实际新增14922588NN（smoke1540）、40modelE/8880更新；每行native6271922=学生train4543920＋eval594002＋教师eval1134000；source12600、formal1176002。两native墙钟1596.744385/2191.173243秒，source预测wall未单列记录。连同本段LRaw30共60modelE/13320更新/23627256新增NN（含两smoke3080），旧父与旧读出复用不重计。ordinary每行4440次/568320包仍计算并弃用、usedrawMSE0，不作省算力声明。

两个worker、dispatcher及0NNanalysis均已自然退出，未停止/重跑健康任务、未扩seed/超过E50。当前单seed预算矩阵收尾，最终可重复星地与clean双收益仍未证明；不把sourcePASS或匹配弱NoAs的阶段PASS当最终完成。


## 下一项：As 专属标签平滑 E30

20261018-phase1-FcrossRX-AsLS-e30-s2026092701-m1-r01已登记PLANNED，实现中，未启动。当前actual原CE本来LS=.01，不能称hard/onehot。本项唯一因素是mixed128内selected64 As两CE的q改eps=.05，普通64/raw/common/U全仍.01，128均值/RD两pass均值/外mixed.5、As.25份额、原KL/SupCon/6L1.25不变。全LAs全6类同公式监督，无预测/质量gate、不新抽As、不加forward/head/branch。父仅原P43 ownB01As20student+EMA+10，总30/6660；只一新B01，oldAs30直接方法控制、oldNoAs30阶段控制复用，NoAs无算子无需重训。

近邻CE_T2/CE_HALF已actual全5RX/6L E30且sourceFAIL，但两者仍LS=.01；新q熵与温度/整体权重梯度不同。本次不重开AsPairCE删KL、LRaw/NoU/weight/sceneBalance/stepALT/MIXcontrast等。另一对话最新在查同NN双独立完整As观测，本侧不重复。时间相关性审计显示scatter名义10.2us内约.0302度、重复收缩等已按各自源判据关闭，不重命名重开。

source固定newAs fullVclean−原NoAs30>=−.1pp且two-drawV1800 midlownewAs−原As30>=+.5pp；fail关固定.05/noepsscan/noE50，all7仍自动独立评分。通用label-smoothing文献仅作动机不替代CVS识别收益。最小[预登记](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-AsLS-e30-s2026092701-m1-r01/report.md)已固定；当前无新性能结果，目标保持未完成。

AsLS E30单行已于2026-10-18发布运行：code 306cd273a4af76cda62e02b45e19e6db9ae765e9，r02传输/编译VERIFIED、770NN首步smoke PASS；r01远端检查路径错误发生于0真实CP/NN/updates且原release保留，修复不改配方/run。源决定先于本轮正式分数读取，all7自动收尾，源失败不扩E50。

AsLS .01→.05 E30已完成唯一新行及全部七视图评分/完整日志：source双规则FAIL（fullV−NoAs−.107407pp，midlow−As−.138889pp），在新正式读取前关闭，无E50。正式newAs clean85.423810%/mid62.435714%/low50.901786%/urbanmean63.604762%；相对原As clean＋.082738pp、mid−.370833pp、low−.342262pp、urban−.334524pp且六practical全降。相对原NoAs clean＋.176190pp，阶段clean约束PASS，但n1微增不能称可重复双提升。新增4352334NN含770smoke，10modelE/2220updates；技术及评分缺项0（source预测wall未单列）。保留原As，本固定.05路线关闭、不扫参/种子/更换父救跑。
