# SIG_TIME＋RDrop特征一致性：E20/E50三种子直接测试

状态 RUNNING。E20 六个新增模型与三个匹配原版 RDrop 控制已完成完整三种子 clean 测试及独立核验：原版均值 83.942857%，POINT 82.787500%（−1.155357 个百分点，1/3 正差），DYNAMIC 80.797817%（−3.145040 个百分点，0/3 正差）。本次两项特征约束在 E20 均未改善平均性能；完整逐种子、样本标准差、分组指标与修正/新增错误见 [E20 测试汇总](E20_summary.md)。E50 六行继续按固定 50 轮运行，尚无完整结果，不据 E20 成绩改参、停止或选择性重跑。

按用户“直接跑测试查看性能结果，而不是看源域留出的接收机”的最新要求，本批使用完整源域从零训练，取消新增留出接收机筛选及源域分数放行条件。每行固定终轮后立即预测 clean 测试集，再由独立 CPU 进程连接 truth 评分。测试分数不改变剩余矩阵、参数、终点或种子。

| 条件 | E20 | E50 | 动作 |
|---|---|---|---|
| 原 SIG_TIME＋RDrop | 三个固定种子 | 三个固定种子 | 复用 6 行匹配的已评分产物 |
| POINT | 三个固定种子 | 三个固定种子 | 6 次新增从零训练 |
| DYNAMIC | 三个固定种子 | 三个固定种子 | 6 次新增从零训练 |

种子固定2026092701/02/03。完整源 RX1/3/4/6/8、day1/2/3、六TX，L6300/U56700隐藏真值/V27000；E20=13L+7U，E50=33L+17U，每轮222更新。默认6×160 cosine头随骨干联合训练，RAW_HZ、普通增强、LS.01、旧门控、自身EMA、AdamW lr.0002/wd.0001、cosine、NoExplicitCFO保持。源折权重训练集合不同，禁止作为初始化/教师；本批只复用其机制代码。

POINT 在原双 CE 均值＋完整对称 KL 上增加未中心化特征一致性 P=.5mean(stop(T)·||normalize(z1)−normalize(z2)||²/160)，T 为默认头对应的 detached Fisher trace。DYNAMIC 使用相同 P，系数是特征边界上停止梯度的 ||gC||/||gP||；C 为两视图共享可微批均值中心化的同型约束，仅供梯度参照。直接复用 gain/center 已发布算子，不新增截断、时序或权重扫描。两者都复用实际增强后的同一L128，保留完整 KL，不重抽增强、增加参数或另训头；梯度经真实联合160特征进入t1/t2/SIG_TIME，辅助损失不直接更新默认W，原CE/KL继续联合更新W。

历史结果边界：POINT 在两个 RX1 源实验改善错误数与CE，但 RX3 错误数多1；DYNAMIC 的 RX3探索少错24，而 RX1确认比FULL多错102且finalLCE.5081，原科学门槛失败。这些负结果保留。本批依据最新用户要求直接检查完整训练后的实际测试性能，不把上述源分数当放行条件，也不据目标结果事后修改公式。特征一致性和梯度范数平衡均有前人研究，未声明全球首创。

12个新模型共420model-epoch、93240优化更新、11934720额外L身份前向、8164800 U观察及2016000 clean决定；复用6个原RDROP控制此前210epoch/1008000决定，总比较18行/630epoch/3024000决定/324同row指标。源探索成本另列。历史6行实际配置、数据角色、空继承、默认头完整预测和truth-last评分元信息已核验；E50 seed02/03来自原技术恢复run，其余原run。当前输入核验没有读truth、分数值或真实IQ，不加载历史模型，不重训控制。原报告和所有负结果保留。

固定终点student/defaulthead，不按V选模，不平均checkpoint、不另训或替换头。只测clean，每行168000既定物理ID、18个overall/RX/day/TX指标；源V可记录健康诊断但没有科学分数停机条件。无新增卫星、Phase2、新TX或独立盲测声明。完整三种子后报告均值、样本SD、同预算对原RDROP配对差、修正/新增错误、MacroF1和最差RX/day；分别检查E50_POINT/DYNAMIC均值≥84%、相对RDROP均值正且至少2/3seed正。E20仅作预定预算比较，禁止据分数挑预算或选择性重跑。

本地CPU/CUDA各151协议负例、3资格否决、198相同初始张量、1024合成身份包及实际t1/t2/SIG/defaultW梯度检查PASS；零真实IQ/CP加载/优化/头拟合/目标访问。唯一新增接线P0/P1审查PASS。每worker首步own初始CP保存严格重载及无query smoke，PASS立即训练；不设额外签名/数据重验/文档审批。

环境ordinary N607 CVS-RFFI，CWD PROJECT/releases/cvs_SIGTIME_rdrop_feature_mechanisms_main20_50_20261016_r01；命令`CVS-RFFI/bin/python -u experiments/cvs_sigtime_rdrop_features_main/runner.py experiments/cvs_sigtime_rdrop_features_main/configs/matrix.json`；root PROJECT/runs/20261016-phase1-SIGTIME-rdrop-feature-mechanisms-e20e50-clean-manysig-m12-r01；唯一owner `codex/root/SIGTIME-RDrop-features-direct-test-20261016`。actual Git身份由release_commit.txt记录，代码/config/登记提交后发布一个archive、一次本地远端SHA比较和remote compile。共享资源锁计入所有owner及preCUDA，每GPU最多3实验进程，CPU2/worker、loader0、独立scorer CUDA为空。发布后独立核验PID/CWD/argv/GPU/日志增长。Only protocol/data/checkpoint/defaulthead/nonfinite/incomplete execution/worker/scorer technical failures stop pending and preserve outputs;healthy and unrelated processes continue. No source scientific score gate;all12 predetermined rows run regardless of source/target scores. No automatic retry,head refit or target-feedback tuning/reordering/selective rerun.

逐行矩阵、原始依赖commit、物理角色和预期产物在experiment.json。初末CP、真实数据、训练日志及逐包预测留既定输出存储，不入Git。发布前尚无本批新增模型的测试结果，不把历史84.340079%当成本批成绩。

<!-- FEATURES_MAIN_LAUNCH_VERIFIED -->
执行提交`c0393fc8c5de7700933ba4f5f16d6cc0a73c108f`已推送且远端OID匹配。单archive传输比较与remote compile VERIFIED；12行PID/startticks/CWD/argv/实际GPU、首轮日志增长、own初始CP无query smoke、完整源角色、原配方和特征约束实际调用已核实。所有owner当前GPU进程数≤3。模型仍在训练，尚无本批新测试成绩。见launch_verification.json。

<!-- ROW_REPORTS E20_DYNAMIC-s2026092701,E20_DYNAMIC-s2026092702 -->
同row分析新增：[E20_DYNAMIC-s2026092701](row_reports/E20_DYNAMIC-s2026092701.md)、[E20_DYNAMIC-s2026092702](row_reports/E20_DYNAMIC-s2026092702.md)。这些完整单row结果不构成整批三seed结论，其他健康训练保持冻结方案。

<!-- ROW_REPORTS E20_DYNAMIC-s2026092703,E20_POINT-s2026092701,E20_POINT-s2026092702,E20_POINT-s2026092703 -->
同row分析新增：[E20_DYNAMIC-s2026092703](row_reports/E20_DYNAMIC-s2026092703.md)、[E20_POINT-s2026092701](row_reports/E20_POINT-s2026092701.md)、[E20_POINT-s2026092702](row_reports/E20_POINT-s2026092702.md)、[E20_POINT-s2026092703](row_reports/E20_POINT-s2026092703.md)。这些完整单row结果不构成整批三seed结论，其他健康训练保持冻结方案。

<!-- E20_FEATURE_COMPLETE_BUDGET -->
E20六新增行及三匹配历史控制形成完整三seed比较，见[E20测试汇总](E20_summary.md)。其他训练保持原矩阵，不因该分数改变。

<!-- E20_FEATURE_SOURCE_DIAGNOSTICS -->
完整E20训练信号、系数和U接受量见[训练机制统计](source_diagnostics_E20.md)。只描述已完成轨迹，不改变当前E50方案。
