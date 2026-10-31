# 新接收机固定 PCD 系统的 DURA 配对诊断

Run：`20261031-phase3-fresh-dura-s2026092701-r01`。唯一 owner：`codex-phase3-fresh-dura-20261031`。本轮不训练新候选，不因辅助 raw 分数改变 PCD 研究基线或 DURA 默认；完整 Phase3 目标保持 ACTIVE。

研究问题是：在新增开发 RX3-2 上，同一份固定观测经 DURA 后，原 F0 身份纠错、正确接收、错误接收与未知拒识分别如何变化。考虑的路线包括继续目标伪标签训练、继续纯源统计增强和先补输入路径配对；前两条已有联合指标负结果，本轮采用冻结配对诊断以区分后续需要解决的环节。它不会证明任意恢复器或 raw 训练系统的一般优劣。

复用 `20261031-phase3-fresh-development-rx3-2-s2026092701-r01` 的全部三类 DURA 预测、E0 与 TARGET_PCD 的冻结权重、两套源密度及四组六类阈值（24 个 FP64/192 B）。从旧六读出选列 0/1/4/5、三个开放表征选 0/2。固定四方法 E0 pooled/predicted、PCD pooled/predicted，唯一主为 `PCD_predicted_ratio`；新拟合与神经更新均为 0。E0 纯源，PCD 明确保留该 RX 的无标签星地 U576 适配祖先；当前新 U 值、U IDs、Us、L/OE 读取均为 0。

矩阵是四方法×六路径：source_raw/source_DURA、clean_raw/clean_DURA、star_raw/star_DURA。三条 DURA 是主测试，精确复用旧产物，不重新前向；raw 仅为辅助。只新增源 V27000/NV3150、未来 clean/star 各10238的原固定 received 冻结预测。源 V 已用于阈值校准，不能视为独立已知测试。所有路径按自己的原 F0 预测身份路由冻结密度和阈值，不能用 DURA 身份替代 raw 身份。主保护基准始终为原 F0 同 DURA 闭集。

数据保持 source contract392005 与已有 VALIDATED_ONCE handles：源五 RX 为1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23；已注册六 TX 为14-10/14-7/20-15/20-19/6-15/8-20。NV 的19-15/19-16/7-1各1050，仅评分端读取。目标 RX3-2/index34 的未来03-15/23共10238条，已知4508、未知5730；四未知5-5/10-7/2-19/7-11仅评分端可用，后者未在早日 U 出现。03-23缺少两个旧类，所需完整名单的A_K/C_K/H/闭集保持N/A。源/Q、早日U/未来Q互斥，clean/star是同物理记录的配对输入，不增加独立样本。

恢复器保持 DURA v1.0 R69、FP32/TF32off、reuse/graph开启/native关闭；G10仍为600km/25MHz/2.45GHz、residual/post_sync/noEQ、refSNR30dB、reflect、input RMS一次/output RMS关闭，既有canonical received不重抽。source/channel/scene、future与receiver种子继承reference，模型seed2026092701；其余本轮无新随机采样。已有源和开发Q曝光、单seed/单开发RX、global physical history UNKNOWN保留；正式RX24-18不读。

先做一次实际F0/E0/PCD/G合成无query smoke，PASS立即继续。全部六路径及状态固定后，独立CPU scorer才读取NV/clean/star三truth各一次；不读Utruth。报告4×6逐TX宏A_K/C_K/R_U/H、原闭集、AUROC/OSCR、TX/RX/day/scene/day×scene、完整覆盖与逐类Wilson区间。DURA−raw记录数分别报告闭集纠错、正确接收增失、错误接收、未知拒识与宏差，不混淆宏平均和净条数。

主DURA绝对目标继续为clean R_U≥95%、star R_U≥90%、C_K≥90%、原F0同G闭集−A_K≤2个百分点，各预定主景完整名单分别核对。源known12与额外42诊断分报，不取消既定测试。辅助raw不用于调阈值、换主或判定新方法晋级；本诊断固定`NO_PROMOTION`、`diagnostic_completed=True`、`adaptation_success=None`、`new_target_DA=False`，A_N/H_OW仍N/A。

新增真实F0行数30150＋2×10238＝50626，E0/PCD合计101252；真实G0，因为DURA结果全部精确复用。合成smoke和既有模型/源统计/缓存/适应成本另列，不能以阶段墙钟宣称B1时延。部署PCD系统仍F0 232993＋open334721＋G266951＝834665，无source/U行bank或训练原型。

环境/CWD/实际commit/命令/路径/GPU写入本run登记。默认N607，GPU0至7实际free≥4096MiB，每卡本任务新增最多2worker，其他owner保持。不可覆盖run/output/release root；一次单包SHA本地远端比较、remote compile、启动后PID/CWD/argv/GPU/log核验。停止规则限协议越界、错误数据/祖先/输出碰撞、无法执行或合法prediction无法闭合；技术失败保留原root，不按低分停止、不盲目重发。预期产物为configs/eval、smoke、models与τ、source/targets六路径prediction、complete、CPU score、pipeline/logs及最终报告登记。
