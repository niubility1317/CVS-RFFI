# 浅中层跨 RX 监督对比：固定三种子 clean

报告强调同 TX 跨接收条件变化，以及条件均值相同不等于分布相同。本轮独立检验样本级监督对比约束，属于报告思想的训练扩展，不冒充斜投影、条件分布独立或真实硬件因果分离。原条件矩实验保持原配方运行，本轮不从其 target 分数选择权重、层位或 seed。

|ID|来源|要求|目标文件|状态|验证|边界|
|---|---|---|---|---|---|---|
|R1|2.1/2.2/6|合法 L 的 TX 条件跨 RX 样本级约束及所有 RX 同类对照|contrastive.py|verified|CPU/CUDA真实梯度、两个层位非零梯度、分组及标签置换 PASS|训练扩展，非严格报告公式|
|R2|用户覆盖|普通 cos、160 维接口、从零 30 轮三主种子、U 隐藏、无 V 性能门槛|design/source/model.py|verified|6真实模型source契约/普通cos/实际初始化及39960更新完整核对|不增加标签或训练轮次|
|R3|用户覆盖|每行自动 clean 固定预测再独立 truth-last|dispatch/predict/score.py|verified|1008000原始clean预测/108指标独立重算一致|84% 尚未达成|

两个 arm：ALL_TX 的正例为其他同 TX 包；CROSS_RX_TX 的正例为同 TX、不同源 RX 包。后者把同 TX、同 RX 包从分母中排除，避免作为负例。两者负例都是其他 TX 包。对每个 anchor 的全部正例平均负 log-softmax，减去 log 正例数量仅消除不可约常数，不改变梯度。温度固定 0.2。L 当前 batch 在五 RX 和六 TX 中平衡，每个 anchor 有至少一个正例及负例；每 anchor/每层等权。

只捕获 L 主 CE 前向的 time_down[B,48,128] 和 t2.pool[B,96,32]。每包 channel 时间 mean 与 population std 拼接，按当前 L batch 的 detached feature mean/std（floor1e-4）标准化，再每包 L2 归一（eps1e-4）。约束角度防止缩小特征规避损失；统计不保存为推理状态。普通 CE 保护当前六 TX 区分，不能证明完整未来 TX 信息。对比项不直接连接 cos W。

沿当前未改的辅助预算固定权重 0.1、warm5/ramp10；不搜索 temperature 或权重。源 L6300/U56700/V27000 原 7/63/30、训练池标签10%，全部五源 RX 与 day1/2/3。六个新模型分别 seed2026092701/02/03、6660 更新、30 轮，总39960 更新；普通 cos scale30 和177025 optimizer参数。source U、采样、噪声、门控、KL及dropout 与已固定 UNIFORM_SSL 相同；没有 EMA/外部 teacher/卫星增强，源 V 不前向或筛选。匹配控制复用原 UNIFORM_SSL 三种子固定预测，不继承其权重、追加训练或重测。

每行固定 FINAL30 后自动168000 clean预测，再独立进程 truth-last评分；总1008000预测/108条指标。全部负结果保留，不选seed/epoch，不声称新盲测。源训练原始数据与历史模型保持原位置，query不得拟合或更新模型。

一个 owner/不可覆盖root/普通 N607 账号。最多3个总实验进程每GPU，余量至少12000MB，现有健康任务不热改或停止。仅协议、数据、输出碰撞、非有限或无法产生合法预测等系统技术故障停本run。唯一独立P0/P1、一个代码/config提交、单archiveSHA/compile、一次真实checkpoint无query首步、postlaunch PID/CWD/argv/GPU/log增长。精确命令、环境、路径、GPU和矩阵见experiment.json。

唯一独立只读 P0/P1 审查 PASS：损失与 L-only 捕获、实际发布/依赖、6 行配置、FINAL30及独立评分入口一致，未发现阻断缺陷。CPU/CUDA 聚焦检查 PASS，两个实际层位损失梯度非零；全部 cos W 对比项梯度 None，9 类 head refit通过。原文不可约常数的符号描述已按代码改为减 log 正例数，无行为修改、无需重复审查。

正式 ANALYZED/VERIFIED：ALL_TX80.3891%、CROSS_RX_TX80.4974%、旧UNIFORM_SSL80.0313%；对SSL三seed分别均正向，提高0.3577和0.4661个百分点。两方案彼此仅CROSS_RX_TX在1/3seed提高，不宣称专属跨RX作用已成立。完整39960更新、180epoch、1008000预测和108指标核对一致。未达84%，保留所有负结果及完整曲线，见final_analysis.md。
