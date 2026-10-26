# 全U最大预测变化 RX 风格：三种子预登记

状态：QUEUED_VERIFIED_NEURAL_NOT_STARTED，等待控制器已发布并核实，尚无新GPU校准或训练，效果未知，原目标active。

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
