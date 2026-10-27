# PAMF 单视图半监督三种子探索

2026-10-27 用户明确要求探索实验用三种子，避免随机波动。本条覆盖此前由Agent自行安排的单seed探索及依据单seed停止扩展的决定；不改写历史单seed数值或原负结果。后续所有带随机训练的候选固定model seed 2026092701/02/03，同一源划分/物理ID/预算，逐seed报告、配对差和三seed均值±样本SD，不择优报告seed。不重复训练已完成普通PL/监督对照，不操作其他owner。

当前协方差完整矩阵固定为CovSel-U、CovRandom-U、CovFull-U、CovSel-LU各三seed，共12个模型。seed01四份E50权重复用；其余两个seed各补四份，共8个新模型，各自复用对应seed合法SIG1 E20，新增10+20至E50/11100、累计13L37LU。原ordinaryPL .95/weight1、完整PAMF/单SIG_TIME/联合默认头、L/T1与全U相关项剂量保持。原seed01三个raw正式结果复用，只补尚缺的fullmask seed01目标预测及另八份新模型的预测，不重跑完成单元。

每个模型的特征通道基底不同。02/03必须在各自真实E20上用完整L6300/三H重新拟合自然支持集mask，不能把01的94/294项照搬；相同quantile/排序/至多10%/随机seed规则不变。各seed共同loss系数仍用固定9批训练L/U梯度同一公式决定，四方法共享该seed的系数，不看V/target挑权重。01掩码及实际alpha1复用。mask为空则明确报告方法在该seed未定义，不填入不合格项或替换seed。

三seed矩阵在新covariance seed02/03目标访问前固定。此次按用户要求取消原单seed源筛选后的种子扩展限制，所有已训练矩阵行固定final后均做F直接rawclean+六practical正式测试，不能只测试源分数好的seed；source评估只诊断。原source-only单seedrun含G判定保留；当前训练和新测试均无DURA权重/恢复IQ/协同损失。星地模拟仅复用冻结的信道数学代码，既有H缓存原样复用。

评价仍是同一方法达到原rawclean≥85.981746%、raw六景>72.096197%及严格同SIG1 PL/Sup与原逐景/TX/RX/day等要求，保留全部三seed而非拼接最好列。三seed未齐不宣布稳定收益；三seed也不自动等于统计显著。源/目标权限和truth-last保持。

低维接收响应诊断没有随机训练模型，model seed为null。其一次确定性L统计、固定18个TX/day留出fold不是模型seed，也不靠重复三次冒充稳定性。该方向后续任何带训练的F候选同样必须三seed。当前物理诊断只证明源响应公共模式存在，不证明身份保护或未见RX可校正。

当前动作：三份对应E20父源句柄已核实；02/03 mask统计已完整完成，分别112/282与91/362项，01的94/294复用。8模型训练及完整12模型raw补齐登记为`20261027-phase1-PAMF-Covariance-SIG1-threeSeed-m12-r01`，首步各自9批实际梯度校准和无query联合smoke，PASS后连续训练/测试。没有放宽原性能目标，也不根据已见seed01目标结果改变方法、mask规则、权重公式或预算。

发布后状态：上述 run 已实际运行，02/03 无 query 联合检查与固定校准均 PASS（共同系数分别 1.000000／0.9949095941）。4 个 seed02 模型已启动，seed03 四行排队；全部三种子矩阵保留，完成后再报告稳定性。

本run队列资源更新：原globalidle阻塞已完成行测试/seed03；独立控制run`20261027-control-PAMF-Covariance-threeSeed-queue-r01`改用own≤4、每卡own≤1/总CUDA+pending≤2、free≥10000MiB，保留原全部科学配置与产物，不动其他owner。该变更不新增训练矩阵或数据权限。

独立下一候选准备：[有界接收响应校正源计划](PAMF_BOUNDED_RESPONSE_SOURCE_PLAN_20261027.md)。仅有本地解析算子／合成检查，尚未发布真实源试验或训练；不依据当前单 seed/部分目标结果选参数，不改变协方差完整矩阵。任何新的 F 验证和训练继续覆盖三模型 seed。

2026-10-27最终更新：协方差四方法×三seed完整12模型/28job已ANALYZED，全部clean均低于匹配PL，六景差−0.0903至+0.1001pp，未达原目标。[完整结果](../../automation_reports/CV-SincNet/20261027-phase1-PAMF-Covariance-SIG1-threeSeed-m12-r01/results.md)。有界固定F源校正的三seed也已完整负结果；后续继续全U/RX训练期物理视图路线，尚无新训练，原目标保持active。

当前进展覆盖上一段“尚无新训练”：独立 `20261027-phase1-PAMF-RXMean-SIG1-threeSeed-m6-r02` 已实际开始，RXMean-U／RXMean-LU 各三seed，原始普通PL .95/weight1与完整T1保持，三份真实E20零更新校准全部通过。仅训练期同包RX均值差软一致性，推理原始IQ；无DURA。r01导入次序技术失败0optimizer已保留，r02复用完成全U均值统计，不重训PL/Sup。全部六行固定E50后自动rawclean＋六景／168000包／独立truth-last，不以部分seed或目标结果改配方。源校准与运行记录见[当前报告](../../automation_reports/CV-SincNet/20261027-phase1-PAMF-RXMean-SIG1-threeSeed-m6-r02/report.md)。

当前源梯度分析只读取既有36批校准记录；它不代表性能结论，也不触发权重或seed选择。结果工具要求完整三种子/每景/两个人群及同seed对照ID配对，完整训练日志核验另查每步raw/H交替、allU曝光及段间计数。源分数或梯度对齐不能替代原目标验收，原绝对目标与全部声明范围保持active。

RXMean 六模型现已全部 ANALYZED：U 的 clean/六景为 84.532341%/70.364649%，LU 为 84.525595%/70.358102%；相对普通 PL clean 分别 −0.161508/−0.168254pp，六景 +0.131415/+0.124868pp，未达全方面目标。详见当前报告。预先固定的 RXMeanMax 三 seed 已从等待转为真实训练；四方法×三 seed 冻结 CPU 源置信度诊断独立运行，0 新训练、0 正式控制重测。原目标与全部三 seed 要求保持 active。

已复用RXMean六份原sourceV raw/H概率完成三seed配对分析：raw高置信子集在low urban约38.8%从raw对变H错，但H高置信子集仍约99.77%正确；新增监督须区分参照证据强弱。这是源代理，0新F/训练/target/U真值，不改变当前Max配方。Max三seed完整E20→E30的6660步/30模型epoch已全部核验，正按原计划继续E50。当前未达标，原目标active。

新候选RXTrust-P三seed已独立固定并发布CPU等待队列（../../automation_reports/CV-SincNet/20261027-phase1-PAMF-RXTrustP-SIG1-threeSeed-m3-r01/report.md）：普通rawPL和全部L不变，H对raw接受者用raw/H argmax并集单标签或partialCE，拒绝者仍softKL；四RX风格Max选择和所有前向预算保持。源E50代理集合包含率约99.883%、RX最低99.565%，不是U/早期/target保证；11项测试、原Max5项和独立P0/P1 PASS。新GPU任务在当前Max全部完成后才启动，不读其指标决定是否运行；三seed同原E20+30至E50，PL/Sup/Max控制复用。实际P-smoke待首GPU步，目标active。

Max三seed已完整ANALYZED：rawclean84.544643%、六景70.262070%，相对匹配PL −0.149206/+0.028836pp；相对随机U +0.012302/−0.102579pp，未达原目标。完整19980step/90模型epoch及3528000正式判决通过核验。预先固定的RXTrust-P三seed已真实校准/训练，alpha分别0.372451/0.383520/0.405387，首整H epoch的singleton/pair/fallback全U计数已核实；P无性能结论，原全范围目标active。

P三seed E20→E30完整6660步/30模型epoch及每模型283500H三路计数已核验，原E50轨迹保持。36批同E20配对发现P改变H目标及raw/H剂量（H参考中位24.75%至26.96%，非逐批硬限幅）。独立PM三seed仅改为max_rawH_median(norm_aux/norm_primary)校准，alpha0.150513/0.145340/0.150372，复用同P源梯度，0新校准F/权重，整个P结束后才启动。代码/6tests/独立P0P1通过、CPU等待控制器已VERIFIED，真实PM smoke仍PENDING_FIRST_NATIVE_SEGMENT。原目标active，不读取P目标决定新矩阵。

P三seed完整ANALYZED：rawclean84.367857%、六景69.716799%，相对PL −0.325992/−0.516435pp且样本SD增大，失败。19980step/90模型epoch与3528000正式判决完整核验。预先固定PM已三seed接续，自己原E20的真实native checkpoint smoke均PASS、首整H epoch计数完整；校准0新F/权重，alpha0.150513/0.145340/0.150372保持，未参考P目标改变配方。PM尚无性能，原全范围目标active。

PM三seed完整E20→E30的6660步/30模型epoch及H/RX/尾批/原PL计数已核验，继续原E50。只读RX传递表几何诊断表明非自洽增益/相位能量0.033164%/0.281932%，投影变化很小且不能证明身份保护，故不为其新开神经候选；0新IQ/F/权重/UTX/V/target。PM健康且无新训练矩阵，原完整目标active。

已将协方差四方法、RX均值两方法、Max与P共八种完成方法的三seed正式计数合并为[完整同基座证据](PAMF_SIG1_COMPLETED_EVIDENCE_20261027.md)。630个方法/seed/人群/场景计数单元与重复引用的PL/Sup逐项一致；全部八方法clean均值低于PL，逐景/分层也未全胜。两项LU−U的配对比较均未显示同时覆盖L有平均收益，但不能推广为全部半监督机制。该汇总0新训练/F/IQ/权重/truth，不含仍在健康训练的PM，既有对照不重跑；原目标active。
