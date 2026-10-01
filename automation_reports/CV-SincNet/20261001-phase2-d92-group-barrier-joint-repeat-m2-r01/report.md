# GroupBarrierJoint完整三阶段重复基准预登记

当前状态：LOCAL_VERIFIED／已预登记，尚未远端启动。没有本候选的 query 准确率结论。

run_id：`20261001-phase2-d92-group-barrier-joint-repeat-m2-r01`；[实际配置与来源](experiment.json)；[事件](events.jsonl)。完整固定矩阵为两模型 × rx3/rx1 四行，rx3每模型900、rx1每模型300，合计2400父任务。旧类6个，K表示每类合法support数，新增类数0/2/5/10/20；四receiver、三practical residual场景、五个固定support seed。此前评分库存只作透明重复基准，新增独立验证按用户要求暂缓。

## 联合学习原理

B仅使用本split旧类support，通过解析LocalRidge对小adapter的监督目标反传，保持已固定CE-only/白化球算法。C真实继承同split的B与U_B；冻结旧类条件分类函数，联合学习新类专用解析Ridge、Bernoulli新旧类gate和小adapter。gate包含全部旧点/新类对的严格正log barrier，ζ=N×1e-4/(m_old×q_new)来自固定目标近似预算，不是目标分数驱动的参数搜索。详细推导和边界见[数学推导](../../../docs/D92_GROUP_FACTORIZED_JOINT_DERIVATION_20261001.md)与[实现决策](../../../docs/D92_GROUP_BARRIER_JOINT_IMPLEMENTATION_DECISIONS_20261001.md)。

C保持B旧类内部winner；这不保证旧类query不被gate分到新类，也不保证≤1个百分点遗忘。有限barrier不等于原hard QP的精确最优。旧模型的零/负train margin不擅自裁零；K1/零rank/零bandwidth保留合法不更新语义，new0精确复用实际B。每条query只面对全部已注册类，不使用query角色、truth、真实类别数量、配额或全局重排。

## 三阶段配对口径与完整表

A是固定地面原生头适应前的旧类query准确率；B是仅旧support适应后、尚未注册新类；C继承实际B后注册新类，新旧类统一竞争。同row/split的三阶段用同一物理旧类query，跨新增数量还核对旧support/query一致。适应提升=B−A，注册后下降=B−C旧，绝对差=|C旧−C新|，H先逐parent计算再平均。两个model seed先各算parent均值再等权平均并报告描述性样本标准差，不称置信区间。K1 query正常评分；new0新类/H/gap为N/A。

|K|旧类数|新增类数|A旧(%)|B旧(%)|C旧(%)|C新(%)|B−A(pp)|B−C旧(pp)|绝对差(pp)|H(%)|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|6|0|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|1|6|2|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|1|6|5|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|1|6|10|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|1|6|20|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|5|6|0|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|5|6|2|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|5|6|5|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|5|6|10|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|5|6|20|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|10|6|0|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|10|6|2|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|10|6|5|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|10|6|10|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|10|6|20|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|20|6|0|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|20|6|2|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|20|6|5|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|20|6|10|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|20|6|20|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|

理想目标是适应提升≥10个百分点、注册后旧类下降≤1个百分点、绝对差≤3个百分点；作为逐步改善目标，不是每轮启动硬门槛。未固定完整预测或未评分的项目保持N/A，不能用support-held结果补query结果，不能混用其他方法的B。

## 来源、验证与成本

Phase1保持合规source-only scratch final200；固定practical residual/post_sync/noeq/25MHz。输入为现有VALIDATED_ONCE胶囊、同模型原始received分支cache和冻结地面A头packet。不存在source样本、source逐记录特征、历史adapted state或query拟合；不会reload checkpoint或执行encoder。完整继承/六类seed与实际路径见experiment.json，support训练状态不继承到此query基准。

[聚焦合成验证](../../../docs/D92_GROUP_BARRIER_JOINT_QUERY_VALIDATION_20261002.json)包含query32、scorer39、support analyzer11个不同case（多次有界调用，首次失败保留）。[入口审查](../../../docs/D92_GROUP_BARRIER_JOINT_BENCHMARK_ENTRY_REVIEW_20261001.md)与[评分器审查](../../../docs/D92_GROUP_BARRIER_JOINT_BENCHMARK_SCORER_REVIEW_20261001.md)的P0/P1已关闭；精确38路径的隔离import禁止np.load/torch.load且未载入encoder。合成验证不表示真实准确率改善。

CPU两lane，每lane BLAS两线程，CUDA为空。最多5888可训练坐标不等于实际星载省算力。须报告实际训练/三阶段推理秒数、峰值RSS、常驻state、payload bytes和硬件；160MiB guard仅针对gate H+Cholesky，不能冒充总内存。outer query计时与internal score计时范围不同，不伪造相等。未知GPU/星载能耗/真实无线传输成本为N/A，新增source payload与ground统计为0。

## 执行、评分与交接

当前runtime/PID/实际成本N/A，preparation parent9518d46d2不冒充将来的执行版本。先提交、push并独立核对远端OID，再唯一发布新release/run；记录实际argv/CWD/PID/start ticks及resolved config。root是唯一launch owner，不重试、不干预已有健康实验。完整四行A/B/C预测全部固定后才由独立scorer连接truth；技术失败保留，不用成功子集宣称全矩阵。分数不回流选参、晋级或重跑。Goal仍ACTIVE。
