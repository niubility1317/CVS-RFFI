# R42 与原版 LocalRidge 的主对比

用户明确指定冻结 R42 重跑此前主对比。固定完整 Tri-SIG-TIME SSL 四个原模型种子、DURA v1.0、3 个物理池、7 RX、MIX01–08，主工作点 K5/N20，共672任务单元。每单元B/C、严格S0/S1、query raw/G-only/同记录融合，共12行、总8064行。已有首seed/P01/MIX01/1-1的12行按capsule/split配对精确复用，其余不重抽物理观测。

R42算法保持：同处理路径CorrCCA/fullNN五强度，当前SUP物理OOF600候选选择，带宽/gate/KRR各折仅TRAIN；F/G/defaulthead/buffers冻结，query逐记录无拟合。没有R43组间偏移。seed1复用R41纯源NN，其他三seed按匹配F和原固定R41源配方重新拟合各自编码器，每orbit600步，5个源RX/TX留出+最终拟合共7200源NN更新/seed，部署权重仅1200步；源L6300、ep91/92配对12600，target/query/U真值训练访问0。K1源规则记录但主点K5不使用其几何选择。

原版LocalRidge的6720条A/B/C评分记录来自已完成原TriSSL主对比，B/C同S/Q配对，A0用于适应增量；不重训/重跑旧方法。主指标逐任务C_old/C_new宏H及配对差、B_old/A→B/B→C、宏F1/逐类下限；672单元等权，另列每seed、RX、pool、MIX及seed标准差。融合对原G-only属于路径改变，不能冒充同输入增益。

此前先导两个all16格未达+5仍保留。本次按用户明确指令扩展主工作点，不按目标分数筛选任务或回调参数；已曝光开发评估，不宣称独立盲选。主工作点完成也不等于最终全部K/N目标完成。

新queue为每单元独立root，GPU0–6每卡最多两个新增本run预测worker，源阶段每卡一个、全源冻结后才目标；启动前每卡实际空余至少6000MiB，其他owner不改。任一源/预测技术错误停止新任务入队，已有健康本run任务排空保留，不盲重试；科学低分不停止。全部8064行固定后才独立CPU逐单元评分。一次新runner P0/P1审查、针对性边界检查、真实checkpoint首步无query smoke、Git代码/config commit、一次归档SHA/远端编译、实际PID/CWD/argv/GPU/log增长；无额外seal/hash/审批。

精确命令、环境、数据句柄、逐行矩阵、stop rule和预期产物以experiment.json为准。当前状态PREPARED，尚未发布。
