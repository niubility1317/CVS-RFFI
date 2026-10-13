# 浅中层 TX 条件矩对齐：三种子 30 轮普通 cos clean

报告目标是保留TX差异、抑制接收条件代理变化。源码中的旧条件矩机制此前未启动；本轮保留其.1权重、warm5/ramp10、时间均值/总体std、detached标准化floor1e-4和实际浅/中层位置，不按target分数调整。用户取消sourceV性能门槛、只clean，本轮直接全源三seed；U source半监督与已经固定的UNIFORM_SSL目标及采样匹配。两独立变体UNCONDITIONAL/TX_CONDITIONAL，各primary2026092701/02/03，从头各30轮、6660更新，总39960/180模型轮。

同普通30×cos(joint160,W6x160)和原native/residual-fusion身份结构，177025optimizer参数；不增加RX头、专家路由、margin或辅助神经分类头。对time_down[B,48,128]及t2.pool[B,96,32]当前合法L前向提取每packet channel时间mean与populationstd，按当前Lbatch detached均值/标准差归一。TX_CONDITIONAL在每TX内比较各RX质心的平方偏差，再等RX/等TX/等通道/等层平均；min组2样本。UNCONDITIONAL不按TX分组，其余相同。普通CE保留当前六类身份区分，不能证明未来新TX所有信息完整。实际autograd验证alignment对commoncos W梯度None。矩统计仅训练loss、无持久state，不用于query归一/投影/适配。

实际source RX索引1/3/4/6/8、day索引1/2/3、全部6TX，L/U/V7/63/30，训练池标签10%。L6300、U56700无TX标签，V27000仅合法加载、不前向或性能筛选。L五RX26/26/26/25/25、TX/day平衡；U只轮流排除一个RX后的四RX各32，与现有UNIFORM_SSL完全相同source采样。当前student own clean eval软目标、conf0.95+相邻完整两pass、noise0.015强视图、warm5/ramp5至KL0.25；无外部teacher、EMA或卫星训练。捕获hooks仅L主CE前向，退出后才弱/强U前向，避免多次捕获和误对齐U隐藏类别。原30cos2e-4至1e-6、AdamW WD1e-4、clip10、batch128、222步/轮。

复用此前完成UNIFORM_SSL三个seed的固定合法clean预测作为同loss/采样控制，不加载其权重训练，不重跑目标。关闭alignment的真实loss/backbone梯度与UNIFORM_SSL一致测试PASS；后者额外meta梯度诊断在新模型不执行，所以不声称同计算成本。每row末30固定后自动clean168000预测，再exec独立truthlast；总1008000决策和108条same-row overall/RX/TX/day指标，完整三seed均值/SD、MacroF1与配对差保留。源V不构成测试门槛，不追加practical/LEO。

现有benchmark已经暴露；本轮配置/数据角色/三个seed/预算在目标访问前固定，所有负结果保留，不按target错例或分数调参/挑seed/选epoch/重跑。84%依实际三primaryseedclean均值判断，当前尚未达成。仅声明地面代理纯域泛化，不称真实星地部署或Phase2已审计；commoncos九类refit接口已smoke。

单owner、普通N607账号、不可覆盖run/release/output、共享capacitylock每GPU总实验进程≤3/free≥12000MB。一个实际Git代码/config提交、一次独立P0/P1、聚焦负测/真实合法checkpoint无querysmoke、单archiveSHA/compile/N607preflight、postlaunch PID/CWD/argv/GPU/loggrowth，只有runowned系统技术/数据/权限/输出或非法预测失败停止，保留产物、不干预健康旧任务。commit、命令、环境/CWD、路径/GPU、矩阵及产物见experiment.json/release/readback，完成后完整日志/原始预测独立重算、登记和Git交付。
