# 完整源采样时钟＋practical：固定四因子E20

状态：RUNNING／启动独立读回VERIFIED。执行commit `a5ffee44d325b69cc360d4248b673ff124110d99`。3组新模型实际训练中，尚无新正式性能结果。

唯一假设：轻度、包内非恒定的采样速率观测变化，能在不新增部署结构／分类头／损失的情况下改善跨RX身份提取；与真实practical联合后保留其收益。它不是已失败的±.125点固定分数延时，也不是CFO或常相位旋转；没有依据声称它已经保身份或已经改善星地。

物理依据：仓库实际DataAugmentation的sample_dg_env_params包含5/25/80/150ppm采样速率范围；本候选一次性固定±80ppm，不拟合源RX或目标。GNU Radio原始信道模型也将采样速率差表示为接近1的步进插值器，而将CFO单列：[原始数据生成论文](https://pubs.gnuradio.org/index.php/grcon/article/download/11/10/23)、[官方动态信道说明](https://wiki.gnuradio.org/index.php/Dynamic_Channel_Model)。这些依据支持算子类型，不证明80ppm是本数据的实测接收机误差，更不证明RFFI收益。

Ag：对已生成共有ordinary的G槽执行u[n]=n+epsilon*(n−127.5)，epsilon均匀于±80ppm，每个物理包／epoch／step独立命名流，不使用TX标签。Lanczos半宽16，反射延拓，无循环接缝、无输出RMS、无新噪声／PA／CFO。最大几何偏移.0102点；不是改变包起点的恒定延时。数字变换不等同替换真实RX。As仍直接处理ordinary之前原规范化IQ，沿用旧完整residual/post_sync／noEQ／Fs25MHz／fc2.462GHz／SNRref30／reflect及质量、噪声、补偿依赖。只生成一次，不串接Ag与As；正式IQ和六场景保持原2.45GHz等既定配置。

底座F＋CROSS_RX，三路SIG_TIME、统一Sinc/CENTER_25K+CENTER_UNITS、joint160、默认cos30头正常联合训练、NoExplicitCFO与内部补偿。原ordinary、RDrop1、LS.01、浅中层SupCon仅共有第一对，原oldgate/ownEMA/cos及13Lonly+7LwithU规则保持。全部5源RX1/3/4/6/8、day1/2/3、L6300/U56700隐藏TX/V27000，split392005，不留训练RX，不切5580/720，V不训练或更新EMA。新模型全部scratch20，不以任何旧权重初始化。

每批L128先生成一次ordinary并做RDrop对。第二个mixed128视图按(batch_position+global_step)%2固定为G64/S64；不存在的因子回填同一ordinary。两个对各权重.5、各自按128物理包平均；共有ordinary监督.5稳定，总L尺度1。每个pair两次前向复用同一IQ，所有模型4次L前向，无新U／教师视图。标签、质量、置信度不控制分配，所有生成As都监督。

|行|G槽64|S槽64|实际新预算|
|---|---|---|---|
|B00|ordinary|ordinary|复用固定REPEAT；先以实际F合成loss/gradient/BN状态等价检查确认|
|B10|Ag(ordinary)|ordinary|scratchE20/4440|
|B01|ordinary|As(raw)|scratchE20/4440|
|B11|Ag(ordinary)|As(raw)|scratchE20/4440|
|FULL_AS_REF|旧ordinary128与practical128完整配对|复用固定CONCAT|0新训练，独立保留全剂量参照|

B01/B11每模型As284160包，为全As参照568320的一半；B10/B11每模型Ag284160包。不能把B11−B01的正差当作超越全As。B00和FULL_AS_REF来源为20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02，固定final20且未据目标选模；仅复用已冻结结果，不加载为训练父权重。匹配真实CP只作兼容性no-query smoke（2合成包），随后丢弃，3新模型从零。

固定model seed2026092701。3新模型×20E=60modelE、13320更新，每模型2273280 L前向；2个既有20E参照的历史40modelE单列，不冒充零累计成本。U、教师真实前向、Ag/As生成、评估和smoke另计。每row固定final20学生默认头后立即自动预测原clean＋六practical，每视图168000；同row全部7视图和logp固定后exec独立CUDA为空CPU scorer连接truth，复用已有CM/宏F1第二实现重计。固定所有row，不根据前一row分数停其他行。

比较B10−B00、B01−B00、B11−B00、B11−B10、B11−B01、B11−FULL_AS_REF；逐视图报告交互A11−A10−A01+A00、准确率／宏F1／最差RX、RX/day/TX分层、成对修复/新增错误、各suburban绝对值与自身clean差。单seed不能完成稳定联合目标。

预登记弃用条件：B10不能提供clean正增益或宏F1下降，或B11不能提供相对B00的clean正增益、lowurban>60%、三个suburban各差≤2pp、lowurban同时高于B01和FULL_AS_REF，则结束此固定时钟／分槽配方。记录所有负结果，不扫描ppm、系数、RX、seed、场景概率或追加E救援。源V只作原健康与机制证据，不据目标选候选或追逐源V微小涨分。指标不触发中途停止健康训练；系统技术停止只限协议／角色越界、非有限更新、输出冲突、无法执行或不能合法预测。保留产物、健康行和旧任务，无自动重试。

运行20261017-phase1-F-crossRX-clock-factorial-e20-s2026092701-m3-r01；代码experiments/cvs_clock_factorial20，实际Git commit发布时写入release_commit/resolved_config；本报告不额外要求设计或文件SHA。命令CVS-RFFI/bin/python -u experiments/cvs_clock_factorial20/runner.py，CWD PROJECT/releases/cvs_F_clock_factorial_e20_20261017_r01。N607普通账号、每worker CPU2、共享每GPU最多3总进程计入其他owner。唯一owner codex/root/F-clock-factorial-20261017；不可覆盖root PROJECT/runs/20261017-phase1-F-crossRX-clock-factorial-e20-s2026092701-m3-r01。实际code/config/source日志、最终CP、全部预测／scores、clock与As覆盖、全程曲线和交互分析保存在同run。期望产物与逐行字段以experiment.json为准。

本地与审查证据：48项配置越权负例通过；四行均为5个完整源RX、L6300/U56700，4次L前向；本地未读取IQ/CP或执行网络。唯一独立P0/P1审查为PASS，无阻断缺陷，不追加重复审查。预检查首次因PowerShell远程文本CRLF而FAILED；同一只读脚本改为LF的私有副本后VERIFIED，账号／项目／时间／8块3090可见、当时显存各1MiB。未修改公共脚本或旧任务。

实际发布：单归档传输比较／远端编译VERIFIED；实际F合成57项负例PASS、2576合成NN、128合成物理包、132采样时钟包、4次合成更新、0真实IQ/CP/target；B00与既有REPEAT的loss／全部gradient／BN状态等价。一次匹配真实CP no-query smoke PASS（2合成包、0查询／更新，丢弃CP），随后3行scratch。独立相隔62.85s核实dispatcher与3worker的PID/CWD/argv/GPU和各自step日志增长，GPU0/1/2各1个本run任务，无失败。启动证据见execution_evidence.json；不把检查通过称为Ag／As收益。
