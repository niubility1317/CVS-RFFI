# 教师伪标签目标反馈：三种子普通cos 30轮 clean

用户要求继续报告机制、每模型30轮、普通cos、不设源V性能门槛，只测clean。第一报告的后续机制是改变教师伪标签目标，而不只给现有自训练梯度加权。本轮使用先前未发布核心中已经固定的eps0.001、teacherweight0.1、studentKL0.25等，不据本轮target错例/分数调整；未启动的旧sourcepilot改为合法全源三seed直接clean方案。已完成的报告36模型负结果全部保留，不重新排序或选择性重跑。

固定TEACHER_SSL与FEEDBACK_TEACHER各三primary组2026092701/02/03，6对联合从零轨迹、12模型。每student和teacher独立随机初始化、各30轮/6660optimizerupdates，每pair13320，总79920更新/360模型轮。student实际init为primary，teacher实际init为primary+10000；教师角色三偏移seed仅作诊断，不能代替student primary三seed84%目标。两角色均普通30×cos(joint160,W6x160)，各177025实际optimizer参数；无RX头/路由/margin，Phase2可丢弃类别方向另拟合。

源实际RX索引1/3/4/6/8、day索引1/2/3、六TX，L/U/V7/63/30，训练池标签占10%。L6300、U56700隐藏TX标签，源V27000仅合法加载，不前向或performance资格筛选。L每batch五RX26/26/26/25/25、day/TX平衡，轮流反馈RX25/26；U仅其余四RX各32。student/teacher CE都使用相同128合法L，teacher dropout独立RNG不改变student轨迹随机数。无旧checkpoint、oracle80、预训练teacher、EMA或星地增强。

机制采用MPL启发的一阶有限差分：用trainL反馈RX的student CE梯度gM及冻结当前AdamW二阶状态对角P形成归一化方向d=P*gM；对student66个实际optimizer骨干参数（排除普通cos权重）暂时theta±0.001d，deterministic前向当前sourceU，得到各类logp的中心方向差分。精确恢复全部被暂改参数，且在恢复前不存在student主训练图；不改变optimizer，不访问query。奖励按六类中心化、batchRMSfloor1e-4、clip2并detach。教师最小化mean(qteacher*r)的正号对应一阶student反馈风险下降代理，aux梯度仅teacher66骨干，不作用commoncosW；teacher普通L CE照常更新W。此不是原论文完整MPL、也不是精确虚拟AdamW。数学玩具符号、实际梯度、精确restore及privateRNG检查PASS。

前5轮两模型纯CE；之后5轮线性ramp至teacheraux0.1、studentKL0.25。teacher own clean eval概率作为student软目标，conf0.95及相邻完整两pass稳定门控，student Ustrong噪声0.015。对照TEACHER_SSL保留相同teacher/student数量、虚拟前向、梯度诊断，只将teacheraux加权为0。每模型30轮cos2e-4至1e-6、AdamW WD1e-4、clip10、batch128、222steps/epoch，均固定末轮，不按V/target选epoch。

每pair固定STUDENT/TEACHER两个末轮后，全部clean各168000独立全六类决策，两角色预测都完成才exec独立评分。总2016000预测、216条same-row overall/RX/TX/day指标，完整实际modelseed及paired_model_seed、MacroF1、三seed均值/SD和对照差保留。teacher诊断即使三偏移seed均值≥84%也不冒充primary学生目标。没有practical/LEO/query拟合。已有benchmark暴露，不声称新独立盲测。

一个owner、普通N607账号、不可覆盖run/release/output，共享capacitylock每GPU总实验进程≤3/free≥12000MB。实际Git代码/config提交、一次独立P0/P1、聚焦负测和真实合法checkpoint无querysmoke、一个archive SHA比对/compile/N607preflight、postlaunch PID/CWD/argv/GPU/loggrowth。只按runowned系统技术/权限/数据/输出碰撞/非法预测失败停止并保留产物，不按分数停止，不热改健康旧任务。commit、精确命令、环境/CWD、矩阵、路径/GPU和产物见experiment.json与release/readback，完成后完整日志/真实预测独立重算、登记与Git交付。
