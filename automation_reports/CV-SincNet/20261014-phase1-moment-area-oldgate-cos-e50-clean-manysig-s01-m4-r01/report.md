# 同投影统计形状与有序面积：最小首种子验证

按用户附件报告三·路线一严格实现，共同路径 μ6、variance6 保留，最后15槽位为 offdiag covariance 或 skew area，使用同一固定 full-rank27→28 lift；四窗口、单6D局部投影、112→160、LN/增益0.05/0.2相同。两条件的全部初始可训练参数相同，各增加18976参数；默认骨干初态同SIG_TIME anchor。描述子内在信息/秩不宣称相同。

矩阵：seed2026092701 × OLD_GATE_COS50只读历史对照、SIG_TIME新同路径锚点、COV_SHAPE、AREA_SHAPE。三新模型各独立scratch50/11100，新增150model-epochs/33300更新；历史50单列复用。无父checkpoint或额外头继承，不从已暴露目标高分权重续训。当前单seed不能满足三seed84%目标。

旧batch_neighbor gate+EMA0.999+cos、33label+17pseudo、AdamW lr2e−4→1e−6/wd1e−4、lambda_U0.16/lambda_ent0.01/smoothing0.01固定。L6300/U56700/V27000、split392005、原RX/day/TX/physicalIDs不变；U真值隐藏，V只健康eval不梯度/选模。NoExplicitCFO与内部补偿保持，无星地训练新增。默认cosine6×160scale30随骨干联合训练，禁止任何单独TX/RX/day/channel头拟合，native诊断fits按既有scope关闭。新特征作用于L/U/自身EMA teacher和实际导出的joint160，不引入类别索引或RX路由。

唯一核心假设：保留水平和方差后，用有序面积替换瞬时协方差是否提供增量。t2完整输出hook保持，Dropout前取点、连续4→2→1层级、Sinc、PAIR、SiLU和U难度权重全部后置独立实验。反转/置乱敏感只证明算子使用顺序；性能只能由真实固定默认头结果证明。此方向来自用户报告与源代码，未用目标分组错误或新首seed分数调结构、参数或停止规则。

本地真实native三模型联合梯度/新增参数数值更新/同初态/同参数量/同fixedlift/W独立/精确导出/teacher deepcopy/非法head/seed拒绝及float64代数检查PASS。源比较控制名已修复唯一P1并通过定向回归；一个独立新增执行审查及仅该问题确认PASS。无源/目标IQ读取。实现提交3be0974b17e4e958475b25ab52ffe0a1fd1b85f3，具体落实记录 analysis/moment_area_traceability_20261014.md。

入口 python -u -m experiments.cvs_moment_area50.dispatch；普通N607既有CVS-RFFI fullFP32/2threads；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_moment_area50_20261014_r01。不可变run/output /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-moment-area-oldgate-cos-e50-clean-manysig-s01-m4-r01；共享capacity锁计其他owner每GPU≤3、free≥12000MiB，按实时状态分配。旧SIG统计组合六个健康worker不变。

发布流程：Git代码/config、聚焦协议负例、唯一独立P0/P1、此最小预登记；一个archive传输和local/remoteSHA比较、remotecompile、一轮N607 preflight及旧seed01真实checkpoint zeroIQ/no-query smoke，PASS即接实际run。烟测不产生授权token；复用VALIDATED_ONCE handles，不重验数据或增加hash/seal链。

Stop：本run自身的角色/来源/非有限数/默认头/预算/export/物理ID/prediction/truthlast技术失败停止新增派发，保留产物和健康已有进程，不自动重试或按目标重启。所有新模型固定final50/11100。原firstsource资格：全部LCE≤0.5、新模块实际更新、真实export/W独立、AREA vs COV固定L720 zL2≥1e−6；PASS才未来原样扩预定义02/03，不能按目标分数决定扩种子。

每行完成冻结后自动clean168000逐包defaulthead预测；全部声明prediction固定后独立truth-last评分；四行共672000默认decisions/72组CM、accuracy、MacroF1。记录同seed配对差、最差RX/day、修正/引入错误、完整50epoch曲线/U接受率、实际模型时间及源特征变化。只有完整3预定义seed候选均值≥84%才达主目标。不得把source/V、单seed或另训读出替代目标。

预期产物：release preflight；run launch/queue、各新行initial/final/frozen/resolved/actual_args/initialization/source_contract/50epoch/11100step/source_feature_witness；各行default prediction/complete与scores/metrics/completion；source_features、source_qualification、analysis、completion，失败时failure。IQ、weights、rawlogs、predictions留既定产物；实现、配置、报告、曲线及必要聚合Git上传并核对remoteOID。本轮仅暴露ManySig clean旧六TX，未验证卫星部署、Phase2或未见TX迁移。
