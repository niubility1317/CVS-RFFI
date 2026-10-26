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
