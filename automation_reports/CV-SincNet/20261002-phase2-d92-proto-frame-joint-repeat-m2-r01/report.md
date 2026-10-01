# ProtoFrame 完整复用基准预登记

状态 READY，未发布、未启动、未读取本候选的部分 support/query 指标。方法已按数学推导固定：Phase1 source-only final200 不变，5 个原型切向校正参数，B/C 各至多 1 次完整 GGN 球内方向与 12 次真实 RMSCE 回读。LocalRidge/自由截距/全 old×new barrier gate 完整求解，C 继承同物理任务的实际 B；new0 精确复用 B。地面原型仅固定特征几何，不作 teacher 或新增训练样本。

完整 2 model × rx3/rx1，共 4 rows/2400 parents；旧类 6，K1/5/10/20，new0/2/5/10/20，全部 4 receivers × 3 practical residual/post_sync/noeq 场景 × 5 support seeds。训练使用本 split 的完整合法 support K，区别 support 诊断的折内 train K。A 原生六类头、B 旧类适应、C 全注册类竞争及 R0_B/R0_C 在相同物理 query 上固定，然后全矩阵独立 truth-last 评分。禁止 source observations/逐记录 source features、query 拟合、truth/role/真实类别数量/配额/全局重排和评分回流。

query producer22、控制面13、support analyzer25、最新scorer42个合成用例通过；实际producer→预检→拟合→NPZ→独立parser接口通过。49个runtime/51个发布文件的隔离导入及实际spec元数据验证VERIFIED。完整关闭前不评分、不据性能停止、不自动重试、不干预健康 Group 或 ProtoFrame support。原 data 的 VALIDATED_ONCE 沿用，不新建数据验证或其他 gate。

结果必须包括完整 K×新增类数表、配对 A/B/C旧/C新、适应 gain、注册后 drop、absGap/H、row/receiver/scenario/seed 分层；理想目标是方向，不强制每轮达到。报告真实参数、factor/RHS、失败工作、训练/推理耗时、峰值 RSS 与保留状态、归档和两文件 ground bytes；wire/能耗/星载设备未测记 N/A。复用评估不称全新独立验证。

两个直接P1已闭合：逐个实际NPZ读回，以及实际Cstage/preparation继承同split返回的B状态；最新42用例包含9个truth读取前失败负例。保留此前4失败、修复后33通过与发现时审查文档，未把历史33通过视为当前完整验证。代码仍未发布，实际query性能N/A。
