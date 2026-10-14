# 报告落实记录：统计形状与有序面积（2026-10-14）

依据用户附件 e2e8f861-e763-457f-821f-b9041c097e79/已粘贴的文本.txt，现按报告路线一先实施单因素描述子比较。已有 MixStyle 三种子结果已完整回收，更新为80.7246%，相对同路径控制无稳定收益；SIG_TIME统计组合剩余六个健康模型保持不可变计划。此轮依据报告及实际算子结构，不依据当前首seed目标分数调参数或选父权重。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| R01 | 三·路线一 | 同6D投影、四窗，μ/方差/协方差 vs μ/方差/有序面积，27D同固定lift→28 | experiments/cvs_moment_area50/features.py | verified | local_check.json：公式、同lift、同初始参数PASS | 只替换15个交叉关系槽位 |
| R02 | 三·路线一 | 同输出112→160/norm/gain/参数量及初始化，加入真实joint160 | model.py/features.py | verified | 三真实native：联合grad、新参数数值更新、精确导出、W独立、teacher deepcopy PASS | 保持t2完整输出，暂不改Dropout取点 |
| R03 | 五·约束 | 默认cosine联合训练，禁止额外头/探针拟合 | source.py/predict.py | implemented | 复用已验证默认头入口；待实际运行 | 旧门控EMAcos/scratch50/33+17 |
| R04 | 五·顺序1 | MixStyle全部种子回收且不按负分改计划 | 已提交三种子MixStyle报告 | verified | f1d1e860e；1512000预测/162指标独立复算 | 历史RX/day探针违规如实披露 |
| R05 | 二·结构分析 | 区分面积、统计形状和常数时间坐标；不把顺序敏感视为泛化证据 | checks.py/report | verified | float64：cov置乱不变、area反转/内部顺序及时间项恒等式PASS | 性能只由真实冻结测试证明 |
| R06 | 三·路线三 | 独立比较t2与t2.pool取点 | 后续独立矩阵 | deferred | 当前不改变取点 | 依报告顺序，避免和描述子混杂 |
| R07 | 三·路线二 | 连续4→2→1层级含三个边界增量和统一全局时间 | 后续独立矩阵 | deferred | 当前不加入层级 | 避免同时改变窗口/容量 |
| R08 | 三·路线四 | 可信门控内有界stopgrad难度权重及真实训练梯度 | 后续训练侧备选 | deferred | 当前保持原U损失 | 报告建议不抢占结构路线 |
| R09 | 五及目标 | 首seed同路径SIG_TIME与两个描述子、旧基线；源PASS才扩02/03；自动默认头truthlast、Git | design/source/dispatch/score/report | implemented | 待登记、唯一直接P0/P1、preflight及真实运行 | 不使用目标分数控制扩seed |

当前最高风险未完成项：新描述子能否在真实source训练中更新并保持合法默认预测路径；局部算子通过不等于识别提升。实施按路线一的公式严格对齐，其他路线按原报告顺序有意后置；无需新增报告审批、签名或数据重验。


已执行：verified Python -m experiments.cvs_moment_area50.checks、package_check.py、check_control_context.py，均 PASS。本地不读源或目标 IQ。实际训练/预测/评分尚未发布，不将这些局部检查当作性能结果。审查发现的 FEATURE_CONTROL 参数比较硬编码已修复，COV→AREA 定向回归 PASS。
