# Phase3 源端跨接收机簇几何排障（2026-10-31）

Run `20261031-phase3-source-cluster-geometry-s2026092701-r01`。前一全簇注册行已完成，混合簇和旧身份损伤使其 NO_PROMOTION。本行只在原 source L6300 上检查表示是否保留设备簇及跨日稳定性，不再访问 target，也不改变所选 PCD 基线。

## 固定模型与输入

固定原 F0/default6、source-only r03、fresh f4b5 TARGET_PCD444、DURA v1.0 R69。三模型全部冻结，同一 source G10 P1 L6300 各前向一次，获得 F0 raw_fused160、r03 z160、PCD z160；原 F0 logits 作为同 P1 闭集基准。PCD 有既有 U576 神经祖先，不能称为纯源模型；本行输入仅 source 不消除该祖先。

四种固定坐标为 F0 单位化 raw160、r03 单位化 z160、PCD 单位化 z160、PCD 原固定白化坐标160；不扫描归一化、HDBSCAN 参数或候选预算。白化使用现有 PCD density 的 origin/whitener，无新 normalizer/GMM 或旧开放判决 τ 的拟合；下述源支持半径与簇半径属于本行新统计拟合。先执行原 F0/r03/PCD/G 的实际 checkpoint 合成无查询烟测，PASS 即继续。

source L 为六 TX、RX1/3/4/6/8、day1/2/3，每 TX×RX×day70，共6300。源 RX 名为1-19、18-2、19-2、2-19、3-19，日期2021-03-08/15/23，全部原 G10 received 已经过冻结 DURA；不重生成星地信道。V27000、NV3150、OE6134、原 U_s56700、本次 target U/Q/private truth、正式 RX24-18 的逐行值均不读取。仅允许原 checkpoint、父 complete/config、模型聚合与公开 source L 元数据，不扩大角色。

## 20 个固定统计任务

每种表示依次以一个源 RX 作为原型留出接收机：其他四 RX 三日共5040条提供有标签 source 支持，计算六类均值及其类内欧氏距离95%总体分位 r_s（floor1e−6）。留出 RX 的 day1 共420条作为无标签发现输入，day2/3 共840条作为后日推理输入；支持、发现和推理 ID 互斥。五个 RX×四种表示共20个任务，表示只算一次后复用。

发现输入按 opaque ID 稳定排序，使用原固定 HDBSCAN min_cluster_size16/min_samples8、euclidean/brute、eom、n_jobs1/alpha1/epsilon0/allow_single_clusterFalse/copyTrue/max_cluster_sizeNone。保留所有非噪声簇，簇数由算法产生。每簇中心与95%分位半径仅来自该簇成员，r_j=min(r_s,簇半径)，同 floor。后日记录逐样本选最小距离/r_j，≤1 输出簇索引，否则拒绝；不使用早日/后日 TX 标签、真实簇数、后日统计或标签来拟合这些状态。另报其他四 RX 六类中心的最近类预测，作为跨 RX 几何诊断，不另训身份头。

全部20个发现状态和后日预测固定后，才评价源端标签。以早日 source L 的合法 TX 标签做严格多数单元、一对一 Hungarian 对齐；每簇只有某个真实 TX 占全部成员严格多数才允许匹配。后日标签不得重新匹配。六源 TX 完整等权分母，漏簇/拒绝/未匹配簇接受均计错；未匹配接受不是拒识。保存全部预测、对齐、各类/RX/day、噪声、簇大小/纯度、ARI 和平均类内/类间几何。

本行 source-screen 仅在五个 RX 均满足后日宏正确接收≥80%、宏接收≥90%、相对原 F0 同 P1 闭集损失≤2个百分点、早日六类全部一对一匹配时，标记该表示可进入下一次预登记的小规模研究；不自动访问 target 或更换 PCD。四种固定表示全部完成后分别报告，不以一个条件的最佳值拼接主方法。

## 声明与执行

六 TX、五 RX 及这些 L 记录都曾进入模型训练；“留出”只指本次原型支持，不是参数训练留出接收机。它是训练角色上的统计排障，不是未见 RX 泛化、严格未知拒识、匿名新类注册或独立验证。A_N、R_U、H_OW 均 N/A；即使 source-screen 通过也不宣称完整目标达标。

本行新真实 F0 前向6300、signal前向12600（r03/PCD各6300），G0；合成烟测另列。新神经/optimizer/GMM/阈值更新0，20组源支持统计与20次HDBSCAN为实际新拟合；所有状态仅诊断，不部署。唯一 owner root，N607 单顺序 GPU worker，每卡新增≤2且启动实际 free≥4096MiB，保留其他任务；新不可覆盖输出、实际代码 commit、聚焦协议负测及一次直接 P0/P1 后发布。

预登记技术停止条件为角色/来源不符、冻结状态改变、nonfinite、输出碰撞或无法闭合诊断产物；低科学分数仍完成全部20任务。不新建数据验证链、不补读目标或私有真值、不依据后日目标测试改参数。完成后同 row 报告、登记和 Git 推送读回，完整 Phase3 目标保持 ACTIVE。
