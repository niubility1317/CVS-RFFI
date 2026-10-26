# 单视图 PAMF 同包选择性相关一致性

状态：PLANNED_SOURCE_ONLY。四个新模型独立复用同一合法单视图普通 PL E20，追加 10＋20 轮到 E50/11100；累计 13L＋37LU。完整 PAMF、单视图 SIG_TIME、原 L/T1 跨 RX、默认头联合训练和普通 PL .95/weight1 保持。矩阵、实际命令、环境/CWD、GPU、输出、停止规则及预算见 experiment.json。

CovSel-U、CovRandom-U、CovFull-U 检验选择规则，CovSel-LU 检验有标签 L 是否应共享新增优化。前两层自然支持集固定 94／294 项，随机控制同尺寸，full 为 1128／4560 项。只从已完成训练 L 的统计推导，不重复拟合，不用 U 的 TX、V 或 target 选掩码。低单变量 TX 关联不等于没有身份信息。

复用原 L 的 raw／As 前向和原 PL 的 U raw 前向，只为 U 增加一个 As 前向。全部隐藏 TX 的 U 参与新一致性，不依赖 .95 门控；已知 RX 用于组间等权。CovSel-U／LU 标量目标及 U 剂量相同，仅分离新损失的 L 梯度。原 L 监督与跨 RX 不关闭。

第一步在真实合法 E20 上运行默认头／SIG1／L 梯度范围 smoke，并用固定 9 批训练 L128/U256 校准一次共同系数：min(1,0.1×median(primary norm)/max_method median(covariance norm))，不读 V，不优化模型。PASS 后立即继续四行，配置在训练前一次冻结。444 步线性 ramp，其余科学配方不变。11 项聚焦测试通过，唯一独立 P0/P1 审查 PASS。

严格匹配的 SIG1 普通 PL／监督三 seed 对照已完成，直接复用，不重复训练。仅对已有 seed01 普通 PL 最终模型做一次冻结 source V 八视图读出，写到本 run 的 baseline_source，不改旧 root。此 pilot 不访问 target，原对照的目标预测不重跑。

每行完成固定 E50 后自动运行全 V27000 的 raw、Graw 和三 source practical H/GH，共八视图；全部模型与 DURA 保持冻结。预定必要条件为相对已有 PL：raw clean 不少于 −27 条、raw 三景总正确数不减少、G 三景总正确数不减少。各行独立判断，不排名选最好列；通过才进入另行固定的正式目标矩阵。源域通过不等于 clean／六景／DURA 或原三 seed 全目标达成。负结果全部保留。

仅使用全局空闲卡，每卡一条本 run worker；首个技术失败停止新启动，已有健康 worker 自然完成。唯一 run/output root 不覆盖；不操作另一对话的任务。详细方法及原固定 10% 规则失败的边界见 [研究方案](../../../docs/goals/PAMF_SIG1_SELECTIVE_COVARIANCE_20261026.md)。
