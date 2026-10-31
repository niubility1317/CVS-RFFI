# Phase3 冻结 Sinc 的开放骨干原型适配（2026-10-31）

本轮检验目标原型适配是否受固定信号编码器限制。当前选定的 `4af` 条件密度比保持冻结主基线；上一轮投影原型 run 的 NO_PROMOTION 保留。本轮从原 source-only r03 独立初始化两臂，不继承失败学生、不用历史 query 训练。

Run：`20261031-phase3-body-prototype-s2026092701-r01`。候选为 SOURCE 和 TARGET_PCD，各固定 444 次更新。主读出唯一为 `PCD_predicted_ratio`，与 E0_predicted_ratio 和本轮 SOURCE_predicted_ratio 配对；pooled 三列为完整冻结对照，不按结果换主。

## 方法与可证伪差异

上一轮只训练 102688 参数投影。本轮冻结独立开放分支 Sinc 的两个 24×1 坐标以及所有 BN/其他 buffer，允许其余信号网络 231985 参数与投影 102688 参数训练，共 334673 个参数。Sinc 无梯度、无优化器状态/步数；采用普通 AdamW lr=1e-4、wd=1e-4、eps=1e-8，全可训练参数 norm clip=5。原 F0/defaulthead、DURA、原 r03 及原 checkpoint 全部不变；学生信号骨干变化与 buffers/Sinc 保持分别核实，不声称整个编码器冻结。

源每步 192 L＋64 OE，SOURCE/TARGET 源索引完全匹配（原 seed 2026103017/18）。源目标仍为原 crossset_metric＋OE 邻域 KL＋10×父源 z 点对点误差，L/OE 两组等权；父缓存仅作停止梯度锚点和 OE 教师。学生每步直接读取同一 DURA P1 IQ 并实际信号前向，不能以父缓存投影冒充骨干适配。两臂均独立从原 r03 初始化，SOURCE 完成且冻结后才打开 U。

TARGET 另用与上一轮完全相同的 384 条 U 和固定两份 phase/gain 派生视图（seed 2026103114），不重抽星地信道。父弱 U z 的 16 容量 spherical kmeans++（seed 2026103113、25 Lloyd）初始化临时单位原型；16 是算法容量，不是真实类别数。父弱最近其他物理记录确定近邻。每步当前学生弱全 U 的停止梯度分配由 log Sinkhorn eps=.05、20 轮得到；随机 64 U（seed 2026103112）的两强视图各自预测自身分配和固定近邻分配，cosine temperature=.1。总损失为 source＋.1×(global＋local)，原型每步单位化，训练后不部署。两强视图、父弱缓存、当前弱分配与学生强分支分别计成本；不宣称完整 SwAV/UniOT、未知类发现或匿名注册。

跨视图分配一致性的参考为 [SwAV](https://arxiv.org/abs/2006.09882)，目标全局判别和局部一致性的参考为 [UniOT](https://papers.neurips.cc/paper_files/paper/2022/hash/bda6843dbbca0b09b8769122e0928fad-Abstract-Conference.html)。本轮没有实现 UniOT 的公共类部分对齐，不继承论文指标或类别发现结论。

SOURCE 与 TARGET 各 444 步后才冻结、拟合两份源 known90/OE32 GMM/normalizer；四组六类阈值仅由源 V 按已固定 known12 规则计算。原 E0 两组六阈值精确复用，总六列×三条件，36 个阈值。所有阈值冻结后完成全部预测，再独立 CPU truth-last 评分。禁止目标 TX/K/比例/配额、V/NV/Q 神经训练、目标早停或 Q 选阈值。

## 数据、来源与测试

原 F0 为 Tri-SIG-TIME SSL 三视图 E50/11100/default6，原源合同 392005；r03 为 `dc01a402582f93e452dbaa3ef59a9fbb44151818` 的 source-only 开放表征（原 2220 更新），DURA 固定 R69 E20/4940、reuse/graph/FP32/TF32off。匹配来源结论复用原 4af baseline；新模型加载仍检查实际祖先，不能用其他适配权重回退。

Phase1 源 L6300/U_s56700，开放训练仅星地 P1 L6300/OE6134（OE 仅成员监督和公共 RX/day，不读 TX），源 5 RX×3 day；V27000 为额外源标签校准，NV3150 仅评分。SOURCE 不读取 U。TARGET 只读取合法 G10 星地 discovery U384，RX24-1/2021-03-08；未来 Q 为同 RX/2021-03-15 的 864 条物理记录，六已知类各96＝576、三未知类各96＝288，clean/star 配对均经同 DURA。U 与源及 Q 物理互斥，clean 仅用于冻结测试。

G10 继承既有 MIX01/600km/25MHz/2.45GHz/practical residual/post_sync/noEQ/ref SNR30dB/reflect 和固定 receiver2027/scene2026103001/channel2026104001，不重新生成 received。复用 VALIDATED_ONCE 的 U capsule `p3dev-star-g10-6e1292b62ab09286ebf1a0df`/split `83c5cc1fe2f2a1512dbc1287` 以及原 clean/star Q 句柄。目标 U 已接触设备只称未注册未知，不声称严格训练未见。历史 NV/Q 已曝光、单 RX/单模型种子、source day 重合、high 缺未知19-15 导致宏 R_U/H=N/A 均保留；正式 RX24-18 不访问。

## 预定科学判定与执行

沿用最新人类目标 v3：clean 宏 R_U≥95%、star 宏 R_U≥90%、C_K≥90%，A_K 较同 DURA 原 F0 闭集下降≤2个百分点；主报宏 A_K/R_U/H。源 known12 须通过，42 组完整诊断，必需场景覆盖。适应额外要求主 star A_K 同时严格高于 E0 和 SOURCE，R_U 不低于两者。任一失败不晋级、不扩种子；source 未过也完成事先登记的全部三条件预测，不能选择性省略 Q。完整 Phase3 无标签发现、未来匿名注册仍未完成，A_N/H_OW=N/A、目标 ACTIVE。机器旧95%全星地目标卡不覆盖最新人类指标，亦不声称已满足任何一套完整目标。

唯一 launch owner 为 root。默认 N607，按实际空余显存≥4096MiB选择一张卡，train/predict 单 worker 顺序运行，每卡本任务新增≤2，不停改其他 owner。独立 CPU scorer CUDAempty。新不可覆盖 run/release/output；预登记精确 command、commit、环境/CWD、路径、GPU、产物及停止规则。launcher 首步真实三 checkpoint 无 query smoke，PASS 立即继续；一独立 P0/P1 review。只对数据/祖先越界、nonfinite、冻结状态破坏、碰撞或不能闭合合法预测等技术故障保留证据停止，科学低分不触发停机，不原地覆盖重跑。

训练完整 888 步保存损失、实际采样/曝光、clip 前 norm、首两步分模块梯度；逐臂核实 Sinc/buffers 不变且 body/projection 实际变化，报告新增实际 IQ 前向与父缓存成本。完整预测后报告 18 点、两配对、TX/RX/day/scene/coverage、成本和 NO_PROMOTION/晋级判断；提交代码与公开结果并 push/OID 核验。额外独立设计审批/重复审查属于 `REJECTED_EXTRA_GATE`，本轮依已有自主研究授权推进。
