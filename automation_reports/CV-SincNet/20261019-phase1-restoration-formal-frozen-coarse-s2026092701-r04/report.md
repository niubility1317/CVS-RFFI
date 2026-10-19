# 冻结复时频恢复器的完整正式观测确认

状态：RUNNING / VERIFIED（完整正式观测预测）。方法代码 `a173cc5002262f243abeb5a904433fcac42ee1cf`。5项新增与23项回归检查、compileall和入口help通过；既有R02候选审查结论复用，不新增重复runner review。实际GPU4约24GB空闲、原输入/父权重路径可用、新root不存在。尚无本轮效果分数。

候选固定为R02源波形A与源身份B共同选中且源资格成立的复时频U-Net。网络、原cleanF/AsF及默认头全部冻结，训练、适配、分类头拟合均0；历史R E20、cleanF E200和AsF scratchE20+ownE10分别计成本。只按已登记源规则进入正式确认，不按目标分数改型、改阈值或重跑。

矩阵、commit、命令、路径、GPU与停止规则唯一维护于[experiment.json](experiment.json)，适配设计见[追踪](../../../docs/PAMF_FORMAL_RESTORE_R04_20261019.md)。复用原168000条固定输入和opaque IDs，clean+六practical全部视图，VALIDATED_ONCE句柄不重验。receiver_seed2027和原namespace与1200配对面板不同；同一原生引擎，不称独立物理族或在轨验证。

先真实源2样本复合checkpoint/no-query检查，然后只读适配原IQ链接和index.ids。全部7视图prediction完整固定后，独立CPU scorer才连接原truth。rx/day占位仅为旧入口格式，正式分层由truth-side原字段替换，不输入模型。原准确率、macroF1、净纠错和波形指标保持；辅助paired feature margin仅对固定512参考记录分块比较，显式标注，不能与完整最近其他TX检索胜率混同，不用于推理或选择。

R03健康三行保持b01b534代码/源矩阵，本轮使用独立GPU4。R04结果在R03源决策冻结前不作为研发反馈回收；不追逐两run的单项最高分。技术异常保留产物、不自动重试，不以低性能停止。单模型seed仍不能替代多seed与独立物理族最终证明。

一次归档传输比较一致、远端 compile/import 通过。独立 PID/CWD/完整 argv/GPU 读回匹配 GPU4 单一 predictor；真实两源样本复合 checkpoint 和原默认头 smoke PASS，TX/目标读取为 0。日志由 0 增至 102 字节，clean 168000 条预测已完成；七视图尚未全部固定，truth-side 评分未开始。R03 源决策已独立冻结，仍不回流目标调参。
