# 同一受损 As 的 CSIL 类别知识迁移

状态REGISTERED。当前用户要求继续完成目标开发及附件路线。P84同观测源诊断已完成：PAMF59.0833%、CSIL77.4444%、MoPC78.3056%（lowurban）；CSIL净修复661/3600，两个draw与6TX均净正，BLOCKED61.4452% vs32.7943%。CSIL原seed392005 scratchFINAL200、role_ids精确相同、默认原头、未target选模；加载实际cp的合同检查复用已核实结论。不同seed和E200/E30预算披露，专家优势不等于KD效果。

只新增KD一行从相同seed2026092701、相同初始化scratch20+own10，完整E30/6660更新。匹配控制直接复用P84r02的T1，健康T1/T6绝不重启/热改。V/target均不参与训练。学生原900722参数/206state、F+CROSS_RX/SIG_TIME_X3/joint160/defaultcosine scale30不变，所有原参数正常联合训练；不添加头、不训诊断头、不加部署模块。

同一原fullAs64 tensor同时喂冻结CSIL和已有两次学生Dropout前向，teacher原logit=5cos+5，q=softmax(zT−5)；studentKD=softmax(zS/6)，共同余弦尺度5。beta=1固定。每样本仅在teacher argmax==真实L标签时计入KD，仍按全部64均值；教师错时该项0，不按学生信心筛样本。原As native-scale CE01全部保留。混合128中占半数As，外层mixed权重.5，故总L增加.25×beta×mean(KL)。两组AsKL均0，ordinary/raw的RDrop KL、CROSS_RX、原hiddenU/EMA/原门控及学习率日程保持；不与T6叠加。

TEACHERS全局仅训练使用，未注册进学生/EMA/优化器；factory加载用fork_rng保护CPU/CUDA随机流，eval且requires_gradFalse。每段结束检查教师参数/BatchNorm状态完全不变、grad为空、实际teacherNN=64×updates，学生extraNN0。部署读默认学生头，无教师或温度操作。

新训练成本30模型轮/6660更新/5114880LNN，额外CSIL426240NN；原U学生/EMA各963900NN，验证/源面板/正式预测另计。匹配的是学生前向和训练轨迹；教师成本额外，不声称总计算预算相同。原CSIL E200历史成本披露，不能算作免费来源。源V只读面板/健康诊断；固定FINAL30。源预定KD−T1midlow≥.5pp、fullVclean≥−.1pp，失败仍全部七环境评分并关闭本固定beta/teacher，不扫描补救。正式target不回流选参、扩种子或重跑；已经暴露测试基准不宣称新盲测。

数据同P84：ManySig equalized1/256点RMS、Fs25MHz、源RX1/3/4/6/8/day1/2/3、L6300/U56700/V27000、split392005，U真值隐藏；fullpractical residual/post_sync/noEQ/ref30/reflect。训练fc2.462GHz与固定正式fc2.45GHz的历史差异保留。全部正式IQ不变。

唯一owner codex/root/sameAsKD85，N607普通账户CVS-RFFI/fullFP32/CPU2；CWD PROJECT/releases/cvs_FcrossRX_sameAsKD_E30_20261019_r01，命令CVS-RFFI/python -u experiments/cvs_same_as_kd85/runner.py；不可变输出PROJECT/runs/20261019-phase1-FcrossRX-sameAsKD-e30-s2026092701-m2-r01。各owner总进程≤3/GPU且free≥12000MB资源队列，仅启动KD，读取原T1完成状态。停止仅本run协议/来源/预算/非有限/碰撞/不能合法预测的技术失败，保留产物并核对poststate，不自动重启。

5项知识目标/隐藏标签负例与16文件AST通过；一次独立P0/P1审查进行中。launcher首步加载真实P43默认头和CSIL，并在两个segment的实际六前向hook上运行1666NN/0更新无query检查，PASS直接训练。只做单次发布归档比较与compile，不新增hash/签名/审批门槛。当前未读取T1/T6任何正式新分数。

本候选一次独立P0/P1检查完成：未发现阻塞问题；普通身份/剩余容量/新输出未占用/复用T1实际配置已VERIFIED，未改动健康任务。

实际发布已VERIFIED：代码commit 0976d1f1476f8a8e6124714435b607e19507c1ee，单归档比较/compile、真实学生与CSIL默认头首步检查1666NN PASS，直接进入训练；PID/CWD/argv/GPU/log已独立核实。教师额外检查前向128计入，学生1536与真实源CP2单列。
