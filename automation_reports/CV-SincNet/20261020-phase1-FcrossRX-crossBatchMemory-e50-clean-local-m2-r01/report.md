# F 跨 RX：短期跨批次特征记忆

状态：ANALYZED_SOURCE_NOT_PROMOTED_SINGLE_SEED_SIGNAL。两行的固定训练、clean 预测、独立评分和完整日志核对均完成；未扩种子。

目标仍是提高源域固定弱 TX 14-7、20-19 在目标 clean 上的性能，同时保留其余 TX。整体准确率超过 84% 本身不代表弱类目标达成。复用原 F＋跨 RX 同 TX 对比学习及本机同种子 BASE，历史 85.1139% 仅作方法锚点。

本轮比较 ANCHOR 和 MEM4。ANCHOR 将当前批次的对比参照特征停止梯度，只对锚点回传；其损失值与原 SupCon 相同，梯度不同。MEM4 在同样规则下加入最近三个真实 L 批次的参照特征，按物理 ID 保留最新项并剔除当前重复项。每个层位、每次 RDrop 分别维护历史；批次当前的 detached 均值和标准差同时用于当前与历史 raw 描述子。

该机制检验跨批次配对的作用，未预设弱类缺少有效正样本。此前完整诊断中的弱类锚点均有效；本方法增加配对范围，同时引入特征陈旧性，两者不能混为纯难负样本收益。所有类别按原锚点平均，不添加弱类权重或额外弱类损失。

原 SIG_TIME_X3、统一 Sinc 物理更新和 clip5、RDrop1、LS0.01、普通增强、旧门控＋ownEMA＋cos、默认分类头联合训练均保持。从零 50 轮＝33L＋17U，共 11100 次更新。无星地信道增强，只测 clean；U 真值隐藏，记忆不由 U/V/EMA/目标域更新。记忆不导出，不增添推理分类器、网络前向、参数或 IQ 样本曝光。

先并行完成种子 2026092701 的两行固定预算，再按 experiment.json 中的源域默认头规则冻结剩余两种子的扩展。源规则继承既定要求：两个弱 TX recall 均严格改善，其他 TX 不低于 BASE 0.25 个百分点，整体不下降，最差源 RX 不低于 BASE 0.25 个百分点，特征无零向量。源域微小改善不构成泛化证明。全部首种子行自动 clean 预测与独立 truth-last 评分，保留负结果，不凭目标结果改变扩展或配方。

完整矩阵、数据与模型来源、命令、环境/CWD、输出根、GPU0 全局最多 3 worker、RAM 至少 4.5 GiB、CPU scorer 最多 1、停止规则与产物见 experiment.json。初始两行并行，保持其他健康任务。技术失败只停止新调度，保护已存在产物，不自动重启。

数学及协议负测试、CPU/CUDA 实际网络检查均通过。真实 checkpoint 无 query 检查一次通过，加载模型丢弃，正式模型从零训练。新增辅助梯度对默认头全部 6 个参数均为 None，主联合头梯度与原生逐值相同；Sinc/浅中层梯度非零，物理更新、EMA eval、导出一致性均通过。一次独立 direct P0/P1 审查通过。

[XBM 原论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html)提供跨批次嵌入记忆的思路。其图像检索慢漂移观察不能替代本任务的实际训练和目标 clean 结果。

启动后实际控制器、两行 worker 的 CWD/命令/GPU 与日志增长均 VERIFIED。两行已完成 E1、各 222 次原生更新，辅助机制尚在固定五轮预热中，未发现技术错误；尚无新增目标域结果。详见 launch_verification.json。

首次机制启用轮 E6 已核实：两行各完成 1332 次原生更新和 222 次对比更新，辅助 loss 有限、权重 0.01；两次 RDrop 复用同一次实际增强 IQ，额外网络前向/星地样本/U 真值/目标样本均为 0，无非有限 loss/梯度跳步。逐步历史键数将在最终完整日志中核实；尚无目标 clean 结果。见 first_active_epoch.json。

两行已完成首次满权重轮 E15，各 3330 次原生更新、2220 次对比更新；权重达到固定 0.1，损失有限，无非有限跳步，额外前向/星地样本/U 真值/目标样本均为 0。该记录是运行健康证据，不是目标性能。详见 full_weight_epoch.json。

两行均已完成首个 U 轮次 E34：各曝光 U 56700，ANCHOR 接受 877（1.5467%），MEM4 接受 876（1.5450%）。U 真值可见数为 0，旧 batch_neighbor 与 rx_day_quantile 保持；原 RDrop 同 IQ、权重和默认头联合方式不变，星地样本/目标样本/额外前向/非有限跳步均为 0。该接受率不是伪标签准确率，不能替代目标性能。详见 first_U_epoch.json。

最终：ANCHOR 85.1083%（配对 -0.0310 个百分点），MEM4 85.3268%（+0.1875）。MEM4 的弱 TX 均值 +0.4286，但只有首种子，20-19 仅多识别 6 个样本；对 ANCHOR 的该类增量为负。两行预定源门槛均失败，不追补种子，不改 SYM4 或原基底。完整逐类、RX/day、Macro-F1、成本、负结果与证据见 [results.md](results.md)、result.json、results_summary.json、completion_verification.json。目标 ACTIVE_UNMET。
