# 源域冻结的轻量 CVS：clean 确认测试

状态：LOCAL_VERIFIED，尚未启动新候选预测。

依据当前用户活动目标，评估源域联合六候选已自动冻结的 residual_fusion；固定四个model seed、E200权重、原ManySig物理划分、纯CE无数据/星地增强、域骨干关闭。实际仅新增4个候选clean预测；原四基准16个已完整固定预测只读复用，不重跑、不覆盖，不更改其config、resolved或provenance。

新方法保持原时域/频域/PA特征提取，只替换身份融合分类头为 LN(base)+tanh(gain)*LN(PA)，再用cosine scale30分类。164225个参数，相比原有效CE参数减少48.30%；MAC只减少1.56%，轻量性还须报告实际时间、显存与常驻状态。该结构对数学/通信/物理/RFFI的假设和限制见 docs/CVS_RESIDUAL_IDENTITY_RESEARCH_20261001.md，不把残差头作用单独归因于某种物理因素。

源8行全部E200，source-only 选择及联合六候选重算均与自动冻结相同。它选中 residual_fusion，且源训练结束发生在基准clean启动之前，未使用基准或目标分数。完整源证据与冻结记录见本报告evidence。当前清洁目标历史上已曝光，属于固定研究确认，不声称新盲测；结果不能回流当前候选调参、重排或选择性重训。

合法权重来源为同一L6300/U56700未用/V27000、sourceRX1/3/4/6/8 day1/2/3 split392005契约下从零训练。新4权重加载前检查真实scratch祖先、physical角色、输入/类映射、E200/10000、resolved与payload全量一致；prediction只读clean.npy与opaque IDs，不读取satellite或truth，每包面对全部6类。复用只允许既有固定基准run四网络16个complete marker、baseline-only冻结范围及原输出路径。全部20预测身份/类映射一致且完整固定后，独立scorer才连接truth。

200轮×50更新、batch128/no drop、AdamW2e-4/wd1e-4/cosine1e-6、FP32/no clipping，源每轮遍历完整6300L；无伪标签、EMA、teacher或附加损失。测试batch256，7个目标RX共168000包，逐包argmax，不使用全局配额或类别数作弊。Phase1无support适应、新类注册或unknown，D92三阶段/K表/H不适用。

15项本地协议检查通过，包括旧contract真实schema、固定16及联合20矩阵、缺预测/ID不符保持truth关闭、旧基准只读复用与错误旧run拒绝、clean-only、污染来源拒绝、外部GPU占用和冷进程。登记launch-ready验证通过，发布前只做本次复用改动的单次独立P0/P1审查。唯一launch owner，只在实际空闲且12GB可用的GPU运行一个评估进程，不干预健康训练，输出独占，故障不覆盖或自动重试。

评分将报告候选对native及三基准的四seed配对差值、整体/RX/TX、Macro-F1、参数/ConvLinearMAC、GPU训练/推理时间、预测峰值显存与模型状态。未测量项写N/A；不将源域小幅提升当作clean提升。
