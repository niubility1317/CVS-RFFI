# 原50轮F＋跨RX的冻结源域恢复诊断

状态：LOCAL_VERIFIED，尚无本行正式目标测试结果。用户共同优化目标保持 active。

本行固定使用原 `20261016-phase1-F-supcon-e50-clean-manysig-m9-r02/CROSS_RX-s2026092701`，执行提交 `4df7b94d0fd8ce29056bd8236364e45afa92e07f`。原三种子 clean 均值 85.113889% 仅为基底背景，本行一个种子源域诊断不能与其直接相减。

现有 R04/R05 的 cleanF E200 与 AsF E30 是不同读出，不能代表用户指定的 F50。新增独立读出保留原三路 SIG_TIME、居中 Sinc 的实际叶坐标及 checkpoint 内 6×160 默认头，全部冻结；不另训、替换分类头。

固定三组：原输入、R02 `tf_unet_mixed` 粗恢复、R03 `residual_deterministic` 修正。仅复用 1,800 条原源 V、clean＋六 practical 固定观测，不重新合成信道、不训练、不选择或重排恢复器，不访问目标 IQ/真值。已有负结果保留。每视图共同使用相同物理 ID，七视图不能当作七个独立 seed。

原 F 为 scratch50＝33L＋17U、11,100 更新，普通增强、LS0.01、旧门控、ownEMA、cos 与原默认头联合训练。初始化无祖先/外部权重，实际原始角色 ID 与当前源缓存完全匹配。R02 E20、R03 累计恢复训练 E40 的历史成本分别报告；本行新增优化更新与头拟合为 0。

已完成 15 项聚焦检查；两条真实 L_s 无 query smoke 严格加载原 checkpoint，并与原 CoordinateSinc、CompositePool 和 CompositeReference 实现比较，特征与 logits 最大绝对差均为 0。检查覆盖来源、错误配方、目标接触、换头、完整七视图先于标签读取、负类别拒绝及批次拆分。原始证据在私有 artifact，不提交权重、IQ 或机器状态。

全部七视图预测完成后，独立 CPU scorer 才读取 V 标签，报告 accuracy、Macro-F1、修复/损坏/净修复及逐 TX/RX/day。只作源诊断，结果不选模型或扩种子；目标提升仍须之后合法冻结的正式测试证明。具体矩阵、命令、输出、环境与停止规则唯一维护在 [experiment.json](experiment.json)。

协作分工：原“设计星地 IQ 无监督去信道网络”对话保有物理展开实现及其 run 的唯一启动权；本对话保有本模块和本源诊断的启动权。CPU 两线程、GPU worker 为 0，不影响其他健康实验。

REJECTED_EXTRA_GATE：额外设计审批、完整运行参数 JSON 恒等（包含 NaN 和序列化类型变化）不属于本次阻断条件。直接核查实际科学配方、checkpoint 来源与原默认头；未增加成员哈希、报告审批或重复数据验证。
