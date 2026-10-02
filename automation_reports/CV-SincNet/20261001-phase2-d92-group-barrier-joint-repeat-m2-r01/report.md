# GroupBarrierJoint完整三阶段重复基准预登记

当前状态：ANALYZED。完整4row/2,400任务已独立评分，旧类6、新增0/2/5/10/20与A/B/C全部报告。含新类任务A旧60.42%、B旧66.73%、C旧59.13%、C新42.97%、H48.83%；没有达到三项理想目标。详见[完整绝对结果和实测成本](query_result_20261002.md)。原r01评分失败与全部原始记录保留，以下历史段落按原时点理解。

run_id：`20261001-phase2-d92-group-barrier-joint-repeat-m2-r01`；[实际配置与来源](experiment.json)；[事件](events.jsonl)。完整固定矩阵为两模型 × rx3/rx1 四行，rx3每模型900、rx1每模型300，合计2400父任务。旧类6个，K表示每类合法support数，新增类数0/2/5/10/20；四receiver、三practical residual场景、五个固定support seed。此前评分库存只作透明重复基准，新增独立验证按用户要求暂缓。

## 联合学习原理

B仅使用本split旧类support，通过解析LocalRidge对小adapter的监督目标反传，保持已固定CE-only/白化球算法。C真实继承同split的B与U_B；冻结旧类条件分类函数，联合学习新类专用解析Ridge、Bernoulli新旧类gate和小adapter。gate包含全部旧点/新类对的严格正log barrier，ζ=N×1e-4/(m_old×q_new)来自固定目标近似预算，不是目标分数驱动的参数搜索。详细推导和边界见[数学推导](../../../docs/D92_GROUP_FACTORIZED_JOINT_DERIVATION_20261001.md)与[实现决策](../../../docs/D92_GROUP_BARRIER_JOINT_IMPLEMENTATION_DECISIONS_20261001.md)。

C保持B旧类内部winner；这不保证旧类query不被gate分到新类，也不保证≤1个百分点遗忘。有限barrier不等于原hard QP的精确最优。旧模型的零/负train margin不擅自裁零；K1/零rank/零bandwidth保留合法不更新语义，new0精确复用实际B。每条query只面对全部已注册类，不使用query角色、truth、真实类别数量、配额或全局重排。

## 三阶段配对口径与完整表

A是固定地面原生头适应前的旧类query准确率；B是仅旧support适应后、尚未注册新类；C继承实际B后注册新类，新旧类统一竞争。同row/split的三阶段用同一物理旧类query，跨新增数量还核对旧support/query一致。适应提升=B−A，注册后下降=B−C旧，绝对差=|C旧−C新|，H先逐parent计算再平均。两个model seed先各算parent均值再等权平均并报告描述性样本标准差，不称置信区间。K1 query正常评分；new0新类/H/gap为N/A。

|K|旧类数|新增类数|A旧(%)|B旧(%)|C旧(%)|C新(%)|B−A(pp)|B−C旧(pp)|绝对差(pp)|H(%)|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|6|0|60.42|50.56|50.56|N/A|-9.86|0.00|N/A|N/A|
|1|6|2|60.42|50.56|47.30|27.99|-9.86|3.26|23.51|32.08|
|1|6|5|60.42|50.56|42.97|29.46|-9.86|7.59|16.87|33.34|
|1|6|10|60.42|50.56|40.81|27.00|-9.86|9.75|15.70|31.42|
|1|6|20|60.42|50.56|39.91|22.48|-9.86|10.65|18.15|27.94|
|5|6|0|60.42|68.41|68.41|N/A|8.00|0.00|N/A|N/A|
|5|6|2|60.42|68.41|63.44|43.96|8.00|4.97|20.19|50.91|
|5|6|5|60.42|68.41|60.23|45.34|8.00|8.19|15.06|51.10|
|5|6|10|60.42|68.41|58.55|41.05|8.00|9.86|17.50|47.82|
|5|6|20|60.42|68.41|58.28|36.75|8.00|10.13|21.54|44.59|
|10|6|0|60.42|72.32|72.32|N/A|11.91|0.00|N/A|N/A|
|10|6|2|60.42|72.32|67.91|51.44|11.91|4.42|16.73|57.79|
|10|6|5|60.42|72.32|64.70|51.82|11.91|7.62|12.93|57.12|
|10|6|10|60.42|72.32|63.38|47.66|11.91|8.94|15.72|53.99|
|10|6|20|60.42|72.32|63.54|44.33|11.91|8.78|19.21|51.69|
|20|6|0|60.42|75.62|75.62|N/A|15.21|0.00|N/A|N/A|
|20|6|2|60.42|75.62|71.34|57.40|15.21|4.29|14.10|63.07|
|20|6|5|60.42|75.62|68.38|57.12|15.21|7.24|11.44|61.94|
|20|6|10|60.42|75.62|67.52|53.41|15.21|8.11|14.11|59.27|
|20|6|20|60.42|75.62|67.76|50.31|15.21|7.86|17.46|57.21|

理想目标是适应提升≥10个百分点、注册后旧类下降≤1个百分点、绝对差≤3个百分点；作为逐步改善目标，不是每轮启动硬门槛。未固定完整预测或未评分的项目保持N/A，不能用support-held结果补query结果，不能混用其他方法的B。

## 来源、验证与成本

Phase1保持合规source-only scratch final200；固定practical residual/post_sync/noeq/25MHz。输入为现有VALIDATED_ONCE胶囊、同模型原始received分支cache和冻结地面A头packet。不存在source样本、source逐记录特征、历史adapted state或query拟合；不会reload checkpoint或执行encoder。完整继承/六类seed与实际路径见experiment.json，support训练状态不继承到此query基准。

[聚焦合成验证](../../../docs/D92_GROUP_BARRIER_JOINT_QUERY_VALIDATION_20261002.json)包含query32、scorer39、support analyzer11个不同case（多次有界调用，首次失败保留）。[入口审查](../../../docs/D92_GROUP_BARRIER_JOINT_BENCHMARK_ENTRY_REVIEW_20261001.md)与[评分器审查](../../../docs/D92_GROUP_BARRIER_JOINT_BENCHMARK_SCORER_REVIEW_20261001.md)的P0/P1已关闭；精确38路径的隔离import禁止np.load/torch.load且未载入encoder。合成验证不表示真实准确率改善。

CPU两lane，每lane BLAS两线程，CUDA为空。最多5888可训练坐标不等于实际星载省算力。须报告实际训练/三阶段推理秒数、峰值RSS、常驻state、payload bytes和硬件；160MiB guard仅针对gate H+Cholesky，不能冒充总内存。outer query计时与internal score计时范围不同，不伪造相等。未知GPU/星载能耗/真实无线传输成本为N/A，新增source payload与ground统计为0。

## 执行、评分与交接

唯一runtime `73aa1b31fc68aeddfb3d0ef7882acafb5be5a155`；supervisor PID1049548/start11917841，rx3模型01/02 PID1049715/1049721实时PREDICTING，rx1两行已完成preflight等待原排程（2026-10-01T16:21:03Z）。actual argv/CWD/CPU2环境由[evidence/query_runtime_1790871724715695800.json](evidence/query_runtime_1790871724715695800.json)独立核实；preparation parent仍9518d46d2，后续metadata提交不改变执行版本。全部预测与完整实际成本尚未完成。先提交、push并独立核对远端OID，再唯一发布新release/run；记录实际argv/CWD/PID/start ticks及resolved config。root是唯一launch owner，不重试、不干预已有健康实验。完整四行A/B/C预测全部固定后才由独立scorer连接truth；技术失败保留，不用成功子集宣称全矩阵。分数不回流选参、晋级或重跑。Goal仍ACTIVE。

发布证据：[evidence/publication_20261002.json](evidence/publication_20261002.json)；源码push独立OID证据：[evidence/source_delivery_20261002.json](evidence/source_delivery_20261002.json)。log实际为每row `prediction.log`，runner在`state.json`记录row PID/argv；原登记的support-style row_startup/log名称只作本地元数据纠正，没有更改或重复远端执行。

2026-10-01T20:17:05Z仅元数据独立读回VERIFIED：{"rx3-cvs-daot-rc4-s2026092701": "PREDICTING", "rx3-cvs-daot-rc4-s2026092702": "PREDICTING", "rx1-cvs-daot-rc4-s2026092701": "PREFLIGHT_COMPLETE", "rx1-cvs-daot-rc4-s2026092702": "PREFLIGHT_COMPLETE"}。未读取部分预测、标签或训练指标；现有immutable runtime不变。

状态更新：PREDICTIONS_COMPLETE。2026-10-01T23:22:23Z原不可变runtime73aa的4条row、2400episode已全部完成并固定，尚未读取truth或调用scorer。独立原scorer的一次r01执行已预登记，输出为全新独立summary.json；不重训练、不改原预测/源码、不根据评分改参数或重跑。发布与完整评分结果仍待独立读回，不能以退出码推定成功。原大量完整terminal metadata保留本地原路径，Git登记紧凑字段及来源，不复制固定预测或truth。

<!-- GROUP_COMPLETE_QUERY_SCORE_PREREGISTERED_20261002 -->

原评分r01已独立读回FAILED，returncode1，8.49秒，子进程峰值RSS1116221440B；scorer在_source_identity入口因feature_contract失败，未连接truth或生成summary。真实四row producer.branch_keys只有z_id/t_emb/f_emb/pa_local，FFT在fft_dim96独立声明；原scorer字面常量误含fft。原预测、方法、矩阵及source检查保留。独立scorer-contract-r02仅修一处字面列表，strict整个dict相等不放宽；新输出-score-contract-r02，预登记未发布。新11+37=48合成回归通过。

契约修复评分r02已唯一发布并启动，sourcef291；00:20UTC独立PID/argv/cwd/environment读回VERIFIED_RUNNING，wrapper1333307/child1333329，子进程仅原固定预测独立评分。此时summary仍无，尚不声称评分完成或方法改善。

<!-- GROUP_COMPLETE_QUERY_RESULT_REPORTED_20261002 -->
