# F＋跨 RX：完整源验证 TX／RX 风险诊断

本轮只做源域、零训练诊断。用户发现 PAMF 的源域弱 TX 在目标也偏弱，要求以 F＋跨 RX 同 TX 对比学习为基底改善差 TX。图中 PAMF 排名不直接复用为 F 的具体 TX 规则；先以原固定三模型的完整源 V 判断困难来自整类还是某些 RX／day。

逐行矩阵及实际源契约见 experiment.json。使用原 run 20261016-phase1-F-supcon-e50-clean-manysig-m9-r02 的三个 CROSS_RX fixedfinal50 student；原执行提交4df7b94d0fd8ce29056bd8236364e45afa92e07f。所有种子均保留，不按目标分数选种子。原数据 L6300/U56700（TX隐藏）/V27000、RX1/3/4/6/8、day1/2/3、split392005 与现成 VALIDATED_ONCE 角色沿用，加载前复用原 source_proof／validate_checkpoint；不继承其权重进行新训练，也不执行旧 EMA。

每模型默认头冻结，不拟合、替换或重训分类头。完整 V27000，每 TX4500、TX×RX900、TX×day1500。报告总体/每TX/逐RX逐day混淆矩阵、每TX CE、margin/30均值和10%分位数。3模型共81000预测，但只有27000唯一物理样本，不称8种子或独立81000样本。只用合法源 V 标签；不读取 target IQ/truth。所有梯度与参数更新为0，验证模型参数版本与持久buffer保持。

后续补偿的通用校准公式在读取本轮结果前固定：r_c=.5×总体sourceV错误率+.5×最差sourceRX错误率；w_c=1+.5×(r_c−meanr)/max|r−meanr|，风险相同则w=1。三个模型源混淆矩阵先聚合，再按相同公式产生全6类系数。均值1，范围[.5,1.5]，与具体类编号无关，类标签置换等变；不会将PAMF的14-7/20-19写成专属分支或预设权重。该系数是用户授权的源侧训练超参数校准，不反向传播V、不拟合原型/归一化/教师。实际训练机制与匹配对照在本结果后另行登记，本源诊断不是目标域提升结论。

当前入口：CVS-RFFI/python -u experiments/cvs_f_txrisk/runner.py；CWD PROJECT/releases/cvs_F_TXrisk_sourceV_20261019_r01；不可覆盖输出PROJECT/runs/20261019-diagnostic-FcrossRX-TXrisk-sourceV-m3-r01；唯一owner codex/root/F-source-TXrisk-20261019。仅GPU2/3、一个本任务worker，每卡不超过一个本任务worker；全owner最多3进程且启动前空闲≥12000MiB。已有外部方法的GPU2/3任务保持运行，不改GPU0/1或其他健康任务。

停止规则：错误数据角色/CP/默认头、任何模型更新、非有限值/计数错误或输出冲突立即保留产物停止本run；无目标测试、无按分数重跑。首步真实默认头checkpoint合成no-query smoke PASS后立即完整评估，不另设smoke凭证。实现/config Git提交后打包，仅一次归档SHA传输比较、一次远端编译和一次PID/CWD/argv/GPU/log增长核实。

本地6个协议负例、等风险基线精确、类标签置换等变与权重上下界检查PASS，7个Python文件AST通过；下一真实运行仅一次独立P0/P1审查。预期产物为各行smoke/sourceV统计、source_calibration、result及completion，最终统计与解释登记提交Git，原权重/原始日志/物理IQ不入Git。

REJECTED_EXTRA_GATE：brainstorming独立设计审批/固定spec/plan门槛被用户Exclusive Minimal Experiment Workflow覆盖；既有研发授权内直接落实，不新增审批。
