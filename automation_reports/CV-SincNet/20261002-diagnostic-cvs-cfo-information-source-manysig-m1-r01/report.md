# CVS 相对频偏坐标的源域身份信息诊断

状态 LOCAL_VERIFIED，尚未发布。上一轮完整源训练中，全路径同步满足受限仿射相位性质，却没有源域性能收益；固定源规则保留原残差网络。当前问题是同步删除的 received 相对频率坐标携带什么信息，不从任何历史目标成绩制定新特征或调参。

本诊断只迭代原 L_s6300 和 V27000，U_s56700 不使用；RX13468、day123、6TX、split392005、equalized1/center256/RMS/25Msps 与完整物理角色保持一致。CPU-only，无 checkpoint、神经网络训练、优化步、增强或域骨干，不构建 target。所有原产物保留，单一launch owner codex/root/cfo-information-20261002。

固定0:80、80:160、160:240内三个20点周期相关。对每包估计arg(C)，以cos/sin表达圆周频偏坐标，避免±625kHz边界被线性均值误读；周期1.25MHz。C退化零校正并记录。4组特征固定为80:160频偏圆周坐标、相干度、二者联合、三窗口坐标与相干度。不是恢复TX晶振参数：相对量同时包含TX/RX/链路/处理影响。

固定ridge=1e-3的诊断探针，pooled及5种留一源RX，共4×6=24拟合。每个探针的均值、尺度和6类权重仅在L_s（全部6300或排除对应RX后的5040）拟合；V不拟合、不回传、不选择特征或ridge。pooled评分全27000 V；leave-one-source-RX-out评分同一V的对应RX5400，其他源RX不冒充target。6类共用同一公式，预测逐样本argmax，无真实类配额或全局重排。

保留全部24结果、混淆矩阵、F1、每源RX及180个TX×RX×day×role单元的三窗口统计。圆周坐标的TX/RX/day及TX×RX方差分解只描述平衡源样本关联，不是硬件因果分解。该closed-form探针不是正式CVS候选，不进入原源选模，也不使用其V分数宣称最终识别优势。

默认测试收尾N/A：本次只做源域机制诊断，不发布可选正式CVS分类器，不访问query。后续正式神经网络仍使用scratch、普通CE、无增强、相同划分、性能优先源冻结后默认clean独立测试；本诊断不能替代性能目标完成。所有负结果保留。

9项聚焦检查PASS：单位/相位与增益/1.25MHz别名、退化输入、探针标签置换与V只读、因子关联反例、checkpoint/target/增强拒绝。独立P0/P1审查PASS；发布脚本缩进P1修复后，实际REMOTE/INSPECT定点编译和回归均PASS。首次发布保留不可覆盖run/release/log；仅当前技术错误让所属任务失败，不停止其他进程、不因负结果重跑。

[原源实验](../20261002-phase1-cvs-synchronized-identity-manysig-m8-r01/report.md) · [只读远端资源核实](evidence/remote_preflight.json)。
