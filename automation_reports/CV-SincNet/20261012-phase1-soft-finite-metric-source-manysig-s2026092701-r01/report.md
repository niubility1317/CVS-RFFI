# 软有限作用度量：源端预登记

目标仍是 clean 三种子84%。波形不变量未过源条件；真实有限间隔在五RX sourceV上62.4889%，ACTION_CE62.0519%，仅+0.4370个百分点、2/5RX正增益，未通过+0.5个百分点和4/5非负条件。没有降低条件、目标测试或选择性重跑。

本候选进一步按报告检验软降权是否比硬去除有用：复用全部五份固定CLEAN_REPLAY从零E20模型，绝不按分数选教师。每份骨干从未训练对应heldRX。严格核对实际config、执行commit、祖先为空、source角色、heldRX排除及checkpoint本体，冻结参数和buffers。父20epoch/4440更新成本保留；本批身份和头更新0。

## 固定数学与对照

从同一四RX L5040提取实际unit-normalized identity joint160，不使用mean/std虚拟读出。六类按同一公式建立原型mu。C_noise采用z-mu_y的共享类内外积平均，固定10%球形收缩和相对1e-6地板。C_finite采用同包四种实际数字操作的delta外积平均，区别于跨TX混合协方差；不打开U。

五个固定读出：原NATIVE_HEAD、普通COSINE_PROTO、MAHALANOBIS=C_noise^-1、SOFT_FINITE=(C_noise+C_finite)^-1、HARD_FINITE在noise白化坐标中移除覆盖80%作用能量的方向。倍率固定1，不跑权重网格；hard rank完全由fitL决定。全部类别同一公式、统一六类竞争，类标签置换不改变规则。

对应heldRX V5400的预测先固定再计分。fit没有heldRX／V原型、协方差或方向；V不更新，原角色不拆分。所有15 TX类对、day、混淆、准确率、macroF1和描述性间隔保留。各读出分数单位不同，不把NLL或间隔当已校准的横向证据。

## 科学条件与执行

SOFT_FINITE相对普通MAHALANOBIS源V平均至少+0.5个百分点、5RX中至少4非负；相对原NATIVE_HEAD均值最多下降0.5个百分点；最差RX不低于MAHALANOBIS最差值1个百分点以上。条件全过才进入正式原契约三seed固定身份比较和默认clean+六practical独立truth-last评分。不宣称有限数字操作解释未知物理RX，不读目标支持集或新增采集。

矩阵、commit、命令、环境、CWD、输出、GPU、owner与产物在experiment.json。一个新不可覆盖run，独立一次P0/P1检查，真实matching checkpoint无query smoke，archive一次比对和远端compile后继续；每GPU最多3个总CUDA进程。低性能保留，不重跑；系统技术故障仅停止可确认所属run，其他健康进程保持。
