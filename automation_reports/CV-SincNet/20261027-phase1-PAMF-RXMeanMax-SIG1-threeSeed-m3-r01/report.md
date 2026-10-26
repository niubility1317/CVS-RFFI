# 全U最大预测变化 RX 风格：三种子预登记

状态：RUNNING。三个 seed 的真实 checkpoint/梯度检查和源校准均已通过，三模型正在实际 GPU 训练；尚无完整性能结果，原目标 active。下文的等待记录是发布时的历史状态。

固定完整PAMF／单SIG_TIME／原默认头，在普通PL .95、weight1、fullT1 L机制不变的基础上，只改变训练期U的RX风格选择。每个U取原始IQ或同包H基础（新增30epoch各15），当前F eval/no_grad的同base概率为q。枚举其余四个源RX的固定均值差作用，合为4B前向，逐包选择KL(q||p_style)最大的风格，再仅对选中视图带梯度重算。没有跨包重新指派或TX标签参与选择。

响应作用沿用已完成全U20统计和强度0.1／±0.5dB／±0.025rad，不扩大幅度、不减去单包响应、不把风格当真实接收机。该统计仍是混合响应，最大预测变化可能触及有用指纹，未获身份保护或性能保证。原始IQ推理保持，不加载均值bank，不增参数/头/视图或集成。

与历史单seed任意IQ方向VAT的差异：本候选的离散扰动集合由已知源RX平均响应定义，无输入梯度探测，student新增分支eval有梯度，H由同包F(H)自身软概率参照；不沿用历史G读出或旧X3对照。旧VAT负结果保留，不扫描其半径。本候选在当前随机RXMean目标结果尚未读取时制定。

比较对象固定为已完成SIG1-PL、SIG1-Sup，以及当前已登记RXMean-U三seed；当前run全部完成后复用其权重/预测/评分，不重新训练或预测对照。本候选三seed2026092701/02/03各自从原同seed合法scratch E20独立+10+20到E50/11100、13L37LU，绝不从随机风格优胜模型或F50继续训练。全部三行固定final并自动rawclean+六practical／cross_rx126000+cross_both42000后独立truth-last，不按源或目标成绩选择种子。

L额外64对仍为原随机风格KL且detach，保留与RXMean-U相同的标量归一化(nU*KL_U+64*stopgrad(KL_L))/(nU+64)及L额外前向。每seed12个源训练批6raw6H零更新梯度校准，原defaulthead/SIG1/fullL/ordinaryPL和U-only梯度检查通过后即继续。系数min(1,0.1*median(primary_norm)/median(aux_norm))、444步ramp固定；不是逐批10%硬上限，不是性能选择。

新增三模型共90模型epoch／19980更新，共享父成本单列。每模型新增候选eval包6804000，合计U教师包9355500、U学生包3402000、原L包5114880、额外L包852480，共18724860次训练前向包。相对随机RXMean-U多6804000包，不把一次4B合批称为没有额外计算，也不将包数比例当实测时间比例。

所有新GPU校准/训练均依赖当前随机风格完整run的complete.json，仅作资源依赖，不读其指标。控制器可先发布为WAITING_FOR_PREDECESSOR；前序全部结束后才按own≤4／每卡own≤1且总CUDA+pending≤2／free≥10000MiB、CPUscorer总≤1执行。这样两个本对话矩阵不会叠加超过当前容量，不停止/热改其他任务。前序缺失只等待；新技术故障停止新派发、保存健康子进程和独占输出；科学负结果不取消其他seed。

精确矩阵、代码commit、路径、命令、环境/CWD与artifact见experiment.json；一次archive传输SHA／remote compile与首步真实checkpoint smoke按最小流程。没有增加数据重验、签名、审批或报告完整度门槛。

原rawclean≥85.981746%、raw六景>72.096197%及全部声明范围严格超过匹配PL/Sup的目标保持。该固定F0候选未获任何正式性能，三seed也不自动等于显著或完整原目标达成。

执行 commit 05e320bb0c554d9b135cec2e43b0e667617a8375 已推送并独立比较上游OID；单archive传输校验VERIFIED、remote compile PASS。5项focused tests与一次独立P0/P1审查PASS。普通账号一次preflight核实新root不存在、原三份scratchE20合同匹配、R02全U均值bank完整；源数据未重验。

控制器PID/CWD/argv/UID已独立核实，两次观测log126→252bytes且pipeline时间增长，状态WAITING_FOR_PREDECESSOR、12job全QUEUED、GPUworker0，真实新最大选择checkpoint/梯度smoke仍PENDING。当前六行随机风格矩阵保持原进程/预算；前序完整结束后才执行本矩阵，并在首步真实smoke通过后自动继续。当前排队不当作神经训练成功。见launch_verification.json。

完整结果与训练核算入口已实现，尚未读取本候选目标结果。`experiments.pamf_u_history.rx_mean_max_results` 在三行及12job全部完成后，只读取冻结预测和已有评分；核实18组同seed ID配对，比较SIG1-PL、SIG1-Sup和随机RXMean-U三类已完成参考，共12份评分／候选3528000次raw决策。按cross_rx／cross_both混淆计数加权，报告逐seed、均值/样本SD、逐景/TX/RX/day及配对差。缺seed、错参考run或不完整pipeline均拒绝正式汇总，不用部分结果选方法。

`experiments.pamf_u_history.rx_mean_max_audit` 对全部完成行的每步／每epoch／完整stdout检查四候选前向成本、最大KL选择、各RX的全U曝光及段间状态。四种风格必须被计算，但最大选择允许只实际选中其中一部分，不强制伪造20个风格对均被选中。完整预算必须是候选eval6804000包、teacherU9355500包／studentU3402000包，source/U真值/target权限保持。当前只做等待状态的只读预检查：0行已冻结，3行pending，不把它当真实梯度smoke通过。

上述分析/核算新增5项测试及原随机风格统计4项、协方差统计3项、曝光核验3项均PASS；既有完整协方差结果在共用统计增加第三参考能力后逐字段不变。仅修改本地分析工具，远端神经执行commit仍为05e320bb0c554d9b135cec2e43b0e667617a8375，训练/选择配方未改。完整结束后分别执行max_audit、max_results collect、max_results analyze，使用已登记runtime；新指标不能反馈到当前已固定矩阵。

当前已从等待转为实际三 seed 训练。三份真实 E20 的 fullPAMF/SIG1/defaulthead/source-only 梯度 smoke 与 12 批校准全部 PASS，三 seed alpha=1；各模型参数保持且校准 0 optimizer。三条训练实际 CUDA PID/UID/CWD/argv/显卡映射与日志增长已核实，已观察每行 538 至 584 个非零连续 step，见 launch_verification.json。原 RXMean 随机风格矩阵已全部 ANALYZED；其负结果不会改动本版 seed、物理幅度、Max 选择或预算。本版尚无完整源或正式结果，不能宣布收益。
