# R41：以源域物理约束继续压缩和性能联合优化

状态：RUNNING / VERIFIED。32项聚焦检查、CLI帮助及一次P0/P1通过。方法commit `be54678266c95e28bbc18f5a71387c8becf463fa`；真实checkpoint首步smoke已通过，两卡均完成E1，固定E20训练及源物理/七场景正式结果尚待完成。


状态：RUNNING / VERIFIED，N607 两档训练均已完成首轮。基于 R40 的固定源域证据，检验给紧凑网络增加粗恢复、复合信道作用和噪声约束，能否兼顾中间物理准确性、恢复细节与最终识别。结果未知，不宣称已有新收益。

两档结构及参数与 R38 相同：`dense_three_quarter` 1709713 参数、`dense_half` 794529 参数。完整 coarse/action/prior 联合优化，共享三步 prior 的长卷积权重，保留各步 bias/GN/pointwise。每版都从原 source-selected R17 `unroll_unshared` E20/4940 的合法数据契约独立初始化；不从 R38 学生或其他 owner 的 F 蒸馏学生续训，不依据已暴露的正式分数选择宽度、父模型、损失、预算或重跑。R38 两档已完成产物作为同预算控制，不重新训练或覆盖。

来源：原 R02 coarse、R07 action、R17 prior 的完整 L6300/U56700、原 source RX1/3/4/6/8、day1/2/3 与 7/63/30 物理 ID。R17 原 source-only 选择在 target 访问前冻结。继承检查复用原记录及实际 metadata；本轮仍 0 TX 标签、F 权重/输出、V 或 target 恢复器优化。F50 只在 no-query smoke 和训练冻结后的原默认头评测中使用，不另训任何分类头。

数据为原 R07 全部 20 个 received epoch（15% clean、85% 六 practical），同时读取与当前 received 严格匹配的已缓存合成 FIR、时变 main/image 系数和绝对实轴噪声方差。L_s 与 U_s 的 63000 条都作为 TX 无标签 IQ 参与。准确术语是“TX 无标签的模拟配对自监督，加已知合成信道参数监督”；有合法 clean 配对与合成算子，因此不称不使用 clean/参数的盲无监督学习。

每样本损失事前固定为：

`L = final_clean_NMSE + λ(epoch) mean3stage_R17_IQ_NMSE + 0.1 coarse_clean_NMSE + 0.1 composite_action_probe_NMSE + 0.01 [log(predicted_native_variance / simulated_native_variance)]²`。

前两项完全保留 R38，λ 从 E1 的 10 线性降至 E20 的 1。新增项按源域 R40 的偏差与原 R07 的 `action + 0.1 logvariance_MSE` 物理配方，以 0.1 系数加入该物理块及 coarse clean 项；不进行 target 调权。IQ NMSE 使用 clean 每实坐标平均能量下限 1e-4；信道项使用两个共同复高斯探针，目标输出功率下限 1e-4。探针由独立 GPU generator、seed2026102041 生成，两档具有相同数据顺序及探针序列。比较整体作用，不用具有增益分解歧义的 kernel 参数 MSE。噪声使用绝对每实坐标方差，不除 prior `SCALE²`；clean 缓存的 1e-8 是数值下限。

训练预算每版 E20/4940、batch256/micro64，最后 24 条按真实 batch 数归一；AdamW2e-4、warmup247、cosine 至2e-5、WD1e-4、clip1、FP32/noTF32，modelseed2026092701、dataorder2026102001。每版 L 曝光126000、U 曝光1134000，teacher/student/context 及合成参数各1260000次，训练探针2520000个。原 R02/R07/R17 各20轮加本轮各组件20轮，共120组件轮次、80顺序训练遍历；F50 原E50另计。新增物理项增加计算，不承诺比 R38 训练更快。

首步 actual checkpoint no-query smoke 使用原缓存两条 L_s 当前 received、对应 clean/合成算子，以及真实原 R17/F50；检查所有组件有限非零梯度、教师/F 不变和完整新 schema checkpoint 精确输出一致，通过后直接继续。N607 普通账号、GPU0/1各最多一个本任务 worker，最多两个；遵守各卡总容量，不干预其他 owner。CPU评分两线程并在本 run 的 score.lock 中串行。独立不可覆盖 run/release/output root，Git pin 后单归档传输与远端编译。

每 row 固定 E20 后自动执行：源域七视图 IQ/F 预测、源域七视图 context-only 估计、匹配 batch1 成本、独立 CPU 源评分及物理诊断；随后无性能门槛地执行七正式视图预测，全部 prediction 固定后独立 CPU truth-last 评分、每场景原 R17 净正确数最多下降1680条（1个百分点，含等号）与原 lowurban70%/其他五场景距 raw clean≤5pp 的分别验收。源域没有收益也不能取消正式测试。每版独立开始测试，不等另一版训练结束，不依正式成绩修改另一行。

源物理诊断复用已验证 R40 NumPy 分析，对全部七估计固定后才读 clean/模拟 oracle，四共同探针 seed2026102040；每版63个 overall/RX/day汇总，报告均值/中位数/P90，与原R40可同口径比较，不拟合参数或用目标反馈。最终报告保留 accuracy/Macro-F1、TX/RX/day、repair/damage、原R17配对丢失/修复、IQ/FFT误差、全部失败与参数/权重大小/MAC/实测延迟和显存。推理仅当前 IQ2x256，输出一份 IQ，由原冻结 PAMF 生成 FFT/PA/时域/统计输入。

停止规则：错契约/输入、非有限、输出碰撞、无法执行或无法合法闭合 prediction 的技术错误保留 affected root，不自动重试，不停止健康其他行；低性能不触发技术停止。精确矩阵、命令、commit、环境/CWD、路径和预期产物在本 run 的 `experiment.json` 中登记。该历史基准已暴露，不声明新的独立盲测；物理项和 F 识别率均不能单独证明逐细节 TX 真实性。


精确命令与矩阵见[登记](experiment.json)。


## 训练已启动

VERIFIED：实际原R17/两学生/冻结F50 no-query smoke PASS；教师/F不变，各组件梯度有效。GPU0/1各一worker的PID/CWD/argv/CUDA及首epoch/日志增长已核实；两档物理约束全链联合训练正在运行，尚无正式性能结论。每版固定E20后自动七场景正式测试，源分数不筛行。
