# 固定 SIG_TIME 的半监督随机／PAIR 参照

用户最新明确现有SIG_TIME足够，重心为泛化与半监督训练机制。本轮依据报告31957f78第六节，将既有物理PAIR与固定SIG_TIME独立整合：两新scratch50的SIG_TIME RANDOM/SIG_PAIR PAIRS＋一历史旧gatecos，仅U的批次组织不同。它是后续半监督机制的一致起点，不是新算法或84%核心机制，不叠加当前签名排序或其他loss。

原PAIR在原网络的高覆盖与难度权重在SIG_TIME旧小门控集合的梯度结果不能混为一项实验。新比较固定单路SIG112/joint160与195041特征参数、普通默认cosine6×160分类头正常联合训练、旧batch_neighbor mask/ownEMA.999/cos日程、33label+17pseudo、λU=.16/λent=.01及acceptedmean CE。无专门训头、无新参数、无附加loss、forward或optimizer步骤。PAIR只使用许可rx/day/eq/sig/baseindex，不读取U标签、TX字段或IQ作排序；每epoch56700物理样本恰好一次，总963900 U曝光/3774Usteps；标签阶段不抓U。随机包装与原native顺序及全局Torch RNG一致。

L6300/U56700 hidden/V27000原契约、physicalIDs/split392005/RX1/3/4/6/8/day1/2/3/旧6TX、NoExplicitCFO内部补偿不变。V健康检查无梯度，无卫星/Phase2。训练初始化scratch，ownEMA来自本模型；历史baseline按原provenance/固定final50复用，preflight加载的已核实SIG checkpoint仅做noquerysmoke，不作为训练父权重。

实际epoch33/34/50每个222批保存t1PW3456/t2PW6912/实际SIGlocalprojection576的mainLCE/.16UCE/.16UCE+.01entropy梯度，autograd只读、不累计grad、不消耗forward/RNG、不改变loss返回。保存梯度复算，记录实际U接收数量、全部50epoch采样状态/17次delivery。不同L128/U256梯度不是同包，也不解释为累计AdamW贡献；覆盖率不是loss权重，也不能代替收益。

首seed2026092701三row：2新100model-epochs22200updates＋1历史50epochs11100updates；150完整源epoch/504000固定defaultdecisions/54组整体RX/day/TX指标、3996真实层批梯度。每row50终轮冻结后自动clean默认头预测168000，再独立truthlast评分。报告所有正负结果和修正/引入错误，不改epoch/门控/权重/种子重跑。后续固定02/03只在两新源finalLCE<=.5、精确预算/featureexport/W独立、PAIR−RANDOML720zL2>=1e-6与PAIR U接受>=.25全程条件成立后原样扩；先写源资格，后读新target数字，无目标排行。84%仍要求完整预定义3seed。

代码复现入口 python -u -m experiments.cvs_sigtime_pair50.dispatch；普通N607既有CVS-RFFI环境，release cvs_sigtime_pair50_20261014_r01，CWD/每row命令/输出/log/GPUowner/physicalcontract/seed/stoprule均在experiment.json。每GPU含其他owner最多3进程/free>=12000MiB；不修改健康任务。新不可变run/output，技术失败保留产物、不自动重启。

本地6negative、隐藏TX访问陷阱、56700一遍/tail124、原random与RNG一致、两actualnative同初态/正常head及t1/t2/SIG投影更新、原Uloss对象identity不变、导出一致/W独立和remotecompile PASS。一次聚焦P0/P1审查与真实checkpoint无querysmoke后发布。原始IQ/CP/预测/日志/梯度留远端，只上传实现/登记/必要聚合。当前未发布，暂无本轮目标分数。

代码/config提交 c98bfcd54c870a03195f220644ab4f7a46ba8d8b，全部模块编译及唯一直接P0/P1审查PASS；真实SIG checkpoint noquerysmoke为发布首步，PASS后直接继续。
