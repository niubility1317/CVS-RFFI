# 报告落实记录：统计形状与有序面积（2026-10-14）

依据用户附件 e2e8f861-e763-457f-821f-b9041c097e79/已粘贴的文本.txt，现按报告路线一先实施单因素描述子比较。已有 MixStyle 三种子结果已完整回收，更新为80.7246%，相对同路径控制无稳定收益；SIG_TIME统计组合已完成三种子闭环，组合82.4972%，相对三路SIG_TIME −0.6883个百分点，0/3正差；全部负结果已上传。此轮依据报告及实际算子结构，不依据当前首seed目标分数调参数或选父权重。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| R01 | 三·路线一 | 同6D投影、四窗，μ/方差/协方差 vs μ/方差/有序面积，27D同固定lift→28 | experiments/cvs_moment_area50/features.py | verified | local_check.json：公式、同lift、同初始参数PASS | 只替换15个交叉关系槽位 |
| R02 | 三·路线一 | 同输出112→160/norm/gain/参数量及初始化，加入真实joint160 | model.py/features.py | verified | 三真实native：联合grad、新参数数值更新、精确导出、W独立、teacher deepcopy PASS | 保持t2完整输出，暂不改Dropout取点 |
| R03 | 五·约束 | 默认cosine联合训练，禁止额外头/探针拟合 | source.py/predict.py | verified | 首seed3实际source closures均零拟合；feature-only导出/W独立PASS | 旧门控EMAcos/scratch50/33+17 |
| R04 | 五·顺序1 | MixStyle全部种子回收且不按负分改计划 | 已提交三种子MixStyle报告 | verified | f1d1e860e；1512000预测/162指标独立复算 | 历史RX/day探针违规如实披露 |
| R05 | 二·结构分析 | 区分面积、统计形状和常数时间坐标；不把顺序敏感视为泛化证据 | checks.py/report | verified | float64：cov置乱不变、area反转/内部顺序及时间项恒等式PASS | 性能只由真实冻结测试证明 |
| R06 | 三·路线三 | 独立比较t2与t2.pool取点 | experiments/cvs_source_dropout；后续独立训练矩阵 | implemented | 固定L720/8mask源诊断和63统计重算VERIFIED，训练对照未发布 | 未改当前descriptor实验；无性能结论 |
| R07 | 三·路线二 | 连续4→2→1层级含三个边界增量和统一全局时间 | 后续独立矩阵 | deferred | 当前不加入层级 | 避免同时改变窗口/容量 |
| R08 | 三·路线四 | 可信门控内有界stopgrad难度权重及真实训练梯度 | 后续训练侧备选 | deferred | 当前保持原U损失 | 报告建议不抢占结构路线 |
| R09 | 五及目标 | 首seed同路径SIG_TIME与两个描述子、旧基线；源PASS才扩02/03；自动默认头truthlast、Git | design/source/dispatch/score/report | verified | 完整三seed12row／99900steps／2016000默认预测／216组指标复算VERIFIED，Git交付 | 不使用目标分数控制扩seed |

当前最高风险未完成项：新描述子能否在真实source训练中更新并保持合法默认预测路径；局部算子通过不等于识别提升。实施按路线一的公式严格对齐，其他路线按原报告顺序有意后置；无需新增报告审批、签名或数据重验。


已执行：verified Python -m experiments.cvs_moment_area50.checks、package_check.py、check_control_context.py，均 PASS。本地不读源或目标 IQ。实际训练/预测/评分尚未发布，不将这些局部检查当作性能结果。审查发现的 FEATURE_CONTROL 参数比较硬编码已修复，COV→AREA 定向回归 PASS。


实际发布 VERIFIED：run 20261014-phase1-moment-area-oldgate-cos-e50-clean-manysig-s01-m4-r01，执行提交 512c10a267b5e654cfd2c983ab1debdeebc7f626；真实checkpoint无查询smoke、归档传输比较、remotecompile和三worker进程/CWD/命令/GPU/日志增长均PASS。R03/R09仍是implemented：完整训练收尾零拟合及预测/独立评分须等真实产物，不用RUNNING冒充verified。


首seed真实执行/独立复算已完成：200epoch记录、33300新增steps、672000固定默认头预测、72分组指标 VERIFIED；R03升级verified。R09源PASS→原样扩02/03已实际执行，完整三seed仍待六模型结束，维持implemented。现为5项verified、1项implemented、3项deferred、0项rejected/blocked。严格按路线一公式实现，无报告近似替换。最高剩余风险是三seed有序面积能否提供稳定性能收益，首seed实际为负；不依据此目标分数改参数或取消健康实验。

实际验证入口：python -m experiments.cvs_moment_area50.checks（既有一次局部PASS）；python -m experiments.cvs_moment_area50.audit（已固定产物复算，无新forward）；python -m experiments.cvs_moment_area_seeds.publish --output <delivery> --inspect（实际PID/CWD/argv/GPU/日志增长VERIFIED）。


R06已独立实施源诊断：20261014-phase1-source-dropout-descriptor-manysig-s01-m1-r01，执行b9144fee45b26342442e8a39bbe475bd19c03bd5，源feature area扰动比31.9940%，8mask不是模型seed；真实训练取点对照仍待落实，因此标implemented而非性能verified。现5verified／2implemented／2deferred，0rejected/blocked。


R06独立真实训练已发布：20261014-phase1-read-position-oldgate-cos-e50-clean-manysig-s01-m3-r01，实现087b3f61abb4444f05e7c22f4af6f631865188df，执行b4d55dba50c8601017604a2a0eb0d037f8d835e8。只移descriptor取点，主干Dropout.1/公式/参数保持；本地真实两模型gradient/数值更新/teacher/export/W独立/初态参数和eval函数PASS，实际resolved取点/零fit配置正确。完整训练结果仍pending，R06维持implemented；5verified/2implemented/2deferred、0blocked/rejected。既有R1六模型继续，不热改旧release。


R09完整3seed闭环 verified：9新/3历史，12row／450新增model-epochs／99900steps／2016000默认预测／216指标；所有9新模块数值更新/导出/W独立/零拟合真实证据完成。AREA82.0833%相对COV −0.5355pp、SIG −0.7097pp，均1/3正差；84未达。R09是执行及交付要求已验证，不代表性能目标满足。最新为6verified／1implemented（R06实际取点训练仍pending）／2deferred，0blocked/rejected；最高剩余风险是取点优化完整3seed性能，不能以源Dropout回放证明收益。路线一公式严格对齐，无近似替换。验证命令：python -m experiments.cvs_moment_area_seeds.audit（只读完整固定产物），剩余取点run尚未发布，未宣称远端执行；仅已做4实际模型context/源资格负例/两种子source汇总检查，不冒充远端执行。
