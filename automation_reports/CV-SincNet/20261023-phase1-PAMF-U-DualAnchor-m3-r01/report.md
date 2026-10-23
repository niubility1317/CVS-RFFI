# PAMF 完整监督与 raw 身份双锚保护

状态：ANALYZED。三个seed均完成E50／11100、raw与固定DURA各七景完整预测及独立truth-last。raw clean84.254762%、六景71.805324%；固定DURA组合84.336310%／77.429729%，原目标仍未满足。102项检查、一次独立P0/P1审查及真实原E30源检查均PASS，全部13320条新增训练记录已核验。原源raw-gradient依据和目标访问前固定配方保留；完整分层、成本及匹配对照见target_results.md。

从原 RXPSM 同 seed E30／6660 学生及自身 EMA、原型和统计独立追加20轮／4440步，累计E50／11100、13L＋37LU。保留原 T1 六次 L 前向、全部有标签 scalar loss 与梯度、默认分类头联合训练；原 U 半普通噪声／半Practical、软CE、原型损失、EMA及置信规则不变。不叠加 CMR、MixStyle、UCRIC、PAIR 或恢复器训练。

在每个 t2／默认分类头／其余可训练参数组，取原完整 T1 有标签梯度 gL，取原已存在两个 raw L 前向的平均真实标签 CE 梯度 gR，取原已加权 U 梯度 gU。只将 gU 投影到同时满足 <v,gL>≥0 与 <v,gR>≥0 的闭凸锥，取与 gU 欧氏距离最小的 v。通过枚举无约束、单一边界、双边界和可行零向量求解；双边界对归一化 anchor 的 2×2 Gram 作 rtol1e-12 伪逆。若任一anchor范数平方≤1e-12，当前组U置零。可行容差为归一化内积1e-10×max(1,||gU||)，仅处理浮点边界；投影转回原dtype后若任一内积超出1e-6×max(1,anchor范数×v范数)负容差，保守置零。再按原 full-L 范数限制 ||v||≤||gL||，实际参数梯度为 gL＋v，loss标量仍报告原L＋U。

raw CE 直接使用原 T1 最后两次 raw L 真实标签前向已有计算图，不增加前向、分类头、教师、推理参数或额外损失项；只新增一次 raw CE autograd 求导。原clip／optimizer／一次EMA后置更新保持。该欧氏内积约束不保证原Adam动量／预条件后的实际损失下降，更不保证目标收益。必须验证精确投影几何、raw真实CE捕获、native原CE消除、默认头、状态／曝光和真实原E30无query梯度接入。

三seed固定2026092701／02／03，GPU5／6／7各一seed。等待同seed UGP 的 raw＋DURA全部14项GPU预测完成、CMR已派发后接续空出槽位；每卡最多两个本任务worker，含其他owner总计算进程不超过三个、启动free≥10000MiB。CPU scorer等CMR终态。不可覆盖root为PROJECT/runs/20261023-phase1-PAMF-U-DualAnchor-m3-r01。相同系统技术失败两次停止新派发并保留健康任务；smoke PASS立即续行，不按源或目标分数提前停止。

每seed固定最终学生，原teacher/student U2097900、physical1048950，新增双锚4440次、L-only原型累计1051392；记录各组原／后内积、激活集合、零fallback、范数限制和额外梯度成本。sourceV仅诊断，无选模／早停。冻结后自动raw PAMF与同一固定DURA v1.0各clean＋六Practical，每景168000，全14项决策固定后独立truth-last。与同readout的F20监督／F21普通PL三seed配对比较，逐景／RX／TX／day、SD、净纠错、曝光和成本完整报告。原raw clean至少+1pp、六景严格超过+1.5pp须同时超过两个基线，DURA单列参考。方法和父来源按源证据固定，无目标反馈调参、排序或选择性重跑，不称新独立盲测。


102 项检查覆盖双半空间解析投影、顺序投影反例、非正交／共线／反向边界、切向梯度保留、单边界／双边界／零梯度、分组／非有限拒绝、真实T1 raw pair CE观察者而非As／RDrop项、native原CE消除与上游缩放、隐藏U拒绝、ownEMA状态／无新增网络参数、原E30父来源／固定配方、三个seed GPU交接、独立CPU评分及raw＋同一DURA完整矩阵，以及原有SSL机制回归。一次独立P0/P1审查PASS，无发现，未读目标成绩或启动实验。真实原E30 smoke作为发布第一步，检查真实raw CE非零和各组raw梯度、双内积／范数、全部参数实际梯度与显式组合一致、默认头／教师／L768／U256／单次EMA；probe丢弃，正式三seed重新载原E30。


## 已核实实际双锚训练

实际执行commit `51e4ccfc45a77d7e39c387e1157f43faf05365e8`；一次发布包SHA本地／远端对照和远端编译PASS。独立核实CPU控制器、三个worker的PID／CWD／完整命令、实际GPU和原生step日志增长；均已重新加载原同seed E30正式续训。真实E30 probe raw CE=0.171661，t2选择双边界、默认头选择raw边界、其余参数选择full-L边界；全部实际参数梯度与显式组合一致，双内积／范数符合固定规则，L768／U256、默认头／无教师梯度与单次EMA正确。probe丢弃，未用其更新正式学生。

逐条检查全部已刷盘完整step前缀，原U／physical曝光、raw锚和引导调用数、各组双内积／范数／激活集合和fallback累计计数准确；gzip尚未闭合，不能称完整训练或收益。详见source_activation.json。GPU5／6／7每卡双锚＋原CMR各一条本任务worker，含其他owner总进程最多三个，未停止或热改任何健康任务。固定最终E50学生继续自动raw／同一DURA各七景、全部决策固定后独立truth-last，CPU评分在CMR之后。


## 三seed源训练完整，正式评测继续

VERIFIED_SOURCE_FINAL：三行各新增20轮/4440步，累计E50/11100；逐条读取全部13320条新增闭合step记录，源U曝光2097900、物理U曝光1048950、学生U视图2097900和自身EMA11100次均符合预定，父前6660步复用原完整核验记录。无U真值/目标训练或另训头。完整源证据及已发生父阶段/增量/累计时间见source_training_complete.json；程序经过时间来自共享GPU，不能等同独占GPU小时。

模型已按固定最终学生冻结，并已自动进入raw与固定DURA各七景预测；这是源训练完成，正式预测/独立评分尚未全部齐，不宣称整体实验完成或目标达成。部分目标成绩仍不读取，三seed全矩阵完成后统一分析。


## 完整目标测试结果

ANALYZED：各方法三个seed，共3模型、42项raw／固定DURA七景决策均完成独立truth-last评分，六个匹配监督／普通PL对照模型完整复用。逐景／seed／SD／Macro-F1／RX／TX／day、修复损伤及曝光／成本见target_results.md和附属完整表。原raw双门槛尚未满足，研发目标仍ACTIVE。

UDAP / raw clean：84.254761905% ± 0.376409389%（三个模型 seed 样本 SD）。
UDAP / DURA-LU clean：84.336309524% ± 0.412250405%（三个模型 seed 样本 SD）。
UDAP / raw six_mean：71.805324074% ± 1.053763100%（三个模型 seed 样本 SD）。
UDAP / DURA-LU six_mean：77.429728836% ± 0.295031232%（三个模型 seed 样本 SD）。

不依据本结果热改健康旧任务或替换父来源。
