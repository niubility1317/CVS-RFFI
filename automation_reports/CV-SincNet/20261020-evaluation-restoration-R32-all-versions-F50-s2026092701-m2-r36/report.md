# R36：R32 两版恢复器均做正式测试

状态：LOCAL_VERIFIED，等待 R32 完整结束。固定两版均测试，源域涨跌不删行；R36 尚未发布，不宣称实际最终 checkpoint smoke 或正式测试已完成。方法 commit `318658985067b511f5e98089a23db1e34990929b` 已推送并独立核实；24 项聚焦检查、CLI help、一次独立 P0/P1 及接口问题定点复核通过。


用户于 2026-10-20 明确要求：后续网络新版的收益不能只看源域，每版都要正式测试。本次从尚在健康训练的 R32 两版开始执行。固定矩阵为 `pair_aug_control`、`observable_response`，不按源域收益、原 R32 资格判定或目标分数删行。原 R32 训练、源诊断及其条件测试队列保持原样；尚未发布的 R33 条件对照补测被 R36 替代，保留原登记。

两版均使用原 R32 固定 E20/4940 完整 checkpoint，实际 L6300/U56700、累计 L126000/U1134000 物理曝光及三倍波形前向。恢复器训练使用模拟成对自监督，0 TX 标签、0 F 权重或输出、0 target；不是无 clean 参照的盲训练。R02/R07 原合法源域上游和原 F50 的既有来源核对复用；严格原 schema/完整状态加载，不另训识别头，不训练或适配任何参数。新增训练成本为 0。

为保护健康任务并防止重复测试，R36 等待 R32 根状态 ANALYZED 且所属 controller、训练和测试 worker 全部退出。两版固定正式矩阵先写入 `formal_matrix.json`，使用明确的 `FORMAL_MATRIX_FROZEN` 状态；不伪造源域 `qualified`。启动首步用两条实际 L_s IQ 对两版完整最终 checkpoint 和原 F50 默认头做无 query smoke，通过后立即继续。原 R32 如已完成某版完全相同的七视图正式预测及独立评分，则按该版 checkpoint、物理面板、F50、七视图和原 truth-last 完整性复用，不依据准确率决定复用。存在部分或不兼容产物时保留并报告技术问题，不重复发起推理。

每版测试原 R05 固定 168000 个物理样本及原 received IQ：clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。恢复器只接收当前 `[B,2,256]` IQ，输出一个 IQ；原冻结 F50 seed2026092701 自动计算 FFT、PA、时域及统计输入。全部七视图的 IQ、logits、features、variance、物理 ID 验证完成后，独立 CPU scorer 才能读取 target truth。各版评分不回流其他训练、选模、候选排序、阈值或重跑。

报告同一 F50、同一 seed 的逐场景 accuracy、Macro-F1、TX/RX/day、净纠错、clean 损伤、IQ/FFT 误差及已固定的身份配对诊断。low urban 目标 70%；其余五个受损场景距原 clean 84.67619047619047% 不超过 5 个百分点，即至少 79.67619047619047%。恢复后的 clean 另报损伤。此为已暴露历史基准上的固定矩阵审计，不宣称新的独立盲测。即使识别率达标，单一 F 和合成局部响应仍不足以证明每个新增细节真实属于输入 TX。

N607 普通账户、固定 release commit、不可覆盖 run/output root；GPU 池 0/1/5/7，每卡最多一条本任务预测 worker，最多两条，原共享 `receiver_residual_capacity4.lock`、全 owner 每卡少于两进程且空闲显存至少 12 GB 才启动。保留其他 owner 和所有历史产物。输入、来源、输出冲突、非有限数或技术预测失败时保留证据，停止安排待启动行、等待健康已启动行结束；不自动重试，不结束其他进程。

登记包含逐行矩阵、方法 commit、精确命令、CWD/环境、路径、GPU、stop rule 和预期产物。发布前一次预检、一个 archive 的一次两端 SHA 比较和远端 compile；启动后独立核实 PID/CWD/argv/GPU/日志及真实产物。完整结果返回前状态为未完成正式测试，不凭源域结果宣称收益。


精确命令、环境、路径和逐行矩阵见[登记](experiment.json)。


## 最新状态：REPLACED

REPLACED：R36 尚未发布，原 R32 训练后源评分接口技术故障，无法满足 R36 原 parent ANALYZED 条件。新 R37 显式恢复已保存源评分并执行同一两版正式矩阵；0重训，源域涨跌不删行。

后续[两版正式测试及评分恢复 R37](../20261020-evaluation-restoration-R32-recovery-all-versions-F50-s2026092701-m2-r37/report.md)。此处没有本轮正式收益结论。
