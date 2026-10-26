# 协方差四方法三模型种子完整补齐

状态：PLANNED。用户最新要求所有带随机训练探索用三个model seed，当前固定CovSel-U、CovRandom-U、CovFull-U、CovSel-LU各2026092701/02/03，共12模型。复用01四份E50与已完成三份raw正式结果；02/03各新训四份，共8份、240新增模型轮/53280步；每模型独立对应E20+10+20至E50/11100、13L37LU。不得从01或已见成绩更好的variant续训，不重训PL/Sup。

完整PAMF、单SIG_TIME、原联合默认头、原L/T1机制和普通PL.95/weight1不变。全部U参与相关一致性，不依赖伪TX；U真TX始终隐藏，已知RX用于组均衡。01mask94/294复用，02/03在自己的真实E20上全L拟合，分别112/282与91/362；规则/随机maskseed不变。每seed用相同固定9批训练L/U梯度公式确定一个四方法共同alpha，444步ramp。校准无optimizer/V/target更新，不把其学生状态继承进新训练。

只用clean与同包星地H训练，复用原H数组及冻结信道数学实现；不调用DURA网络/权重/恢复IQ。原已完成SIG1PL/Sup三seed目标结果直接配对。第一步真实对应E20/L768/U256/defaulthead/SIG1联合backward smoke后立即训练；11项泛化/旧行为针对性检查PASS，原候选P0/P1与seed基底绑定定点检查PASS，不额外重复全审查。

每行固定final后自动全V raw/三H诊断及正式cross_rx126000+cross_both42000=168000物理包的rawclean+六practical。完整12模型均测试，不按源分数剔除seed；source score仅诊断。01缺的fullmask第一次补测，其余01三行不重测；02/03新增8行。合计9新prediction/9独立CPU score，全12逻辑矩阵14112000raw决策。每row两任务七视图全部冻结后才连接truth，scorer无source/model/G句柄，其他健康训练/候选不受目标反馈影响。

唯一矩阵、实际命令、输入权限、代码/环境/CWD、GPU、输出、停止规则及产物见experiment.json。全局空闲卡、一卡一ownworker，CPU scorer最多1且计入其他owner；首个技术失败停止新启动，已有健康worker自然完成，旧root与完成产物不覆盖。不依赖负源分数提前停止未完成的注册seed。

最终报告逐seed、三seed均值±样本SD、与同seed PL/Sup配对差及逐景/TX/RX/day准确率、macroF1、混淆。三seed不是显著性检验或三个信道随机seed。原rawclean≥85.981746%、raw六景>72.096197%及其余原声明范围保留；模型或seed未齐不宣布稳定收益，不拼最好列，goal继续active。[当前计划](../../../docs/goals/PAMF_SIG1_THREE_SEED_FONLY_20261027.md)。
