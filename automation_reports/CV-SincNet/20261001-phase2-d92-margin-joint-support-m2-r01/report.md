## 当前：完整 support 数学分析与三阶段补充完成，训练诊断提取运行中

四行、160 parent、1800条路径、3240阶段的独立分析已COMPLETE_MARGIN_JOINT_PROBE_VERIFIED，原分析PID787831退出，local79064已结束。训练runtime e50de0ad4，分析source ccb7f3469。

实际地面A已另行完成并与原固定B/C配对，见[完整三阶段与K×新增类数](../../../docs/D92_MARGIN_GROUND_A_PAIRED_SUPPORT_RESULT_20261001.md)及[结果解读](../../../docs/D92_MARGIN_JOINT_SUPPORT_RESULT_OVERVIEW_20261001.md)。本结果只描述support持出诊断，不是query或新增独立验证，未全面改善，不晋级或反馈改参数。

完整训练-only collector唯一local78869，remotePID875239/start10109892，当前extract-stdout；snapshot已捕获250238670字节。证据[evidence/collector_runtime_1790853818675083800.json](evidence/collector_runtime_1790853818675083800.json)。不重启、重复提取或将诊断进程RSS当作星载方法内存。

以下为保留的预登记和历史状态；运行判断以上述当前证据为准。

# D92-MarginJointLocalRidge support联合实验预登记

状态：SUPPORT_TRAINING_COMPLETE_ANALYSIS_RUNNING。四行原训练已完成且原进程退出，独立数学分析PID787831/start9221520正在执行；原runtime e50de0ad4，分析源码ccb7f3469。root唯一launch owner。

四行共160个parent；旧类6个，K=1/5/10/20，新增0/2/5/10/20类，两个固定模型seed与两个cohort。矩阵与input身份沿用显式预登记元数据；不依赖成绩选择。practical residual/post_sync/noeq/25MHz。

B使用解析LocalRidge与合法旧类support微调。C继承当次同物理fold实际B，在全部注册列上求解旧类间隔约束QP，同时用完整KKT梯度监督adapter。结构已冻结，依据[数学推导](../../../docs/D92_MARGIN_JOINT_STRUCTURAL_DERIVATION_20261001.md)和[论文对应](../../../docs/D92_JOINT_SFT_PAPER_MAPPING_20261001.md)，没有参数网格。

QP技术上限max_transitions=4096、每head自有factor buffer=167772160B（160MiB）。[资源推导](../../../docs/D92_MARGIN_JOINT_RESOURCE_BOUND_20261001.md)明确最坏容量和guard排除项；不保证4096次内收敛，不视为RSS/星载内存。实际工作与时间独立测量，未测=N/A。

Phase1固定，完整source-only scratch final200来源见experiment.json。仅当前合法support；源样本、逐样本源特征、query和历史成绩反馈禁止。已VALIDATED_ONCE的received/capsule不重建不重验。支持监督微调不是query拟合。

A、B、C三阶段按同一物理旧类held support配对：本run先固定B/C预测，后独立Ground A补充。A及B−A在补充完成前为N/A。K1无独立held，不能制造准确率。完整K×新增类数及receiver/model分层、H、适应提升、注册下降和新旧差距必保留。

理想目标B−A≥10个百分点、B−C旧类≤1个百分点、新旧绝对差≤3个百分点是逐步改善目标，不是本轮硬门槛。support结果不能称独立query验证；新独立数据确认按用户暂缓。本轮没有晋级结论。

每row独占输出；两CPU lane，每lane BLAS2，GPU不可见。技术失败只失败所属lane，保留产物；健康lane继续。没有低成绩停止、自动重试或历史适应状态复用。

日志保留实际参数、类RMSCE和损失/梯度/Armijo/QP状态、耗时、source validation=N/A原因、完整NPZ、紧凑JSONL/CSV。独立分析核验全部QP约束/KKT/自由截距VJP与实际SUM/MAX计费；配置容量不是实际执行量。

|row|model seed|cohort|实际evaluator命令与配置|
|---|---:|---|---|
|rx3-cvs-daot-rc4-s2026092701|2026092701|rx3|见experiment.json逐行command/config_ref/log_path|
|rx3-cvs-daot-rc4-s2026092702|2026092702|rx3|见experiment.json逐行command/config_ref/log_path|
|rx1-cvs-daot-rc4-s2026092701|2026092701|rx1|见experiment.json逐行command/config_ref/log_path|
|rx1-cvs-daot-rc4-s2026092702|2026092702|rx1|见experiment.json逐行command/config_ref/log_path|

[预登记](experiment.json) · [冻结与容量证据](evidence/method_freeze_and_resource_choice_20261001.json)

远端只读元数据preflight：VERIFIED，四个缓存来源均绑定；新run/release/archive均不存在。未读取特征值、query或源样本。Ground A扩展150项通过，12项为两个旧方法fixture中的Margin专用断言跳过，Margin本身已执行；[证据](evidence/ground_a_entry_validation_20261001.json)。上述为启动前preflight证据；其后已唯一发布，启动状态独立VERIFIED。

启动时尚无完成marker；当前训练已完成，但实际性能仍待独立审计。不能将旧方法A/B/C拼给Margin。四行原训练现已结束，正在执行一次独立数学摘要。随后生成完整训练诊断，再用同物理held记录做Ground A配对。保留所有失败与健康lane，不重复publisher、preflight或request生成。

历史训练观察（2026-10-01T08:20:51Z）：前两行MARGIN_JOINT_PROBE_COMPLETE，每行810个FINAL/648个STEP；原两个evaluator已退出。后两行PID771304/start9043046、PID771324/start9043867在同一supervisor756554/start8891521下健康执行；整体complete尚不存在，无失败产物。完整性能尚未审计，不报告部分行准确率。证据[evidence/training_progress_1790842913353830900_metadata.json](evidence/training_progress_1790842913353830900_metadata.json)。

此前只读观察的两个rx3训练进程VmHWM为1066416KiB/1058448KiB（约1.02GiB/1.01GiB）；仅当时进程高水位，非全run最终峰值或星载测量。因子buffer160MiB不涵盖进程RSS。完整资源将在结束后按实际scope报告。

新的[AI标量导出工具](../../../docs/D92_MARGIN_TRAINING_AI_SCALARS_20261001.md)已通过26项合成检查，支持完整3流前缀标量展开与JSONL/CSV读回，拒绝巨型audit/ref/text伪标量；源验证null并说明Phase2禁止。只在完整训练诊断生成后使用，不改当前日志或模型。真实转换尚未执行。

训练完成的独立证据：[evidence/runtime_readback_1790844546417236400.json](evidence/runtime_readback_1790844546417236400.json)。原声明160个parent、1800条物理路径、3240个stage全部完成，四行均MARGIN_JOINT_PROBE_COMPLETE。数学分析已唯一发布到新release d92_margin_joint_analysis_20261001_r01，实际commit ccb7f3469cc1a28ea8bbd7eb7adce937b41a430a，当前启动证据[evidence/analysis_runtime_1790844653598279800.json](evidence/analysis_runtime_1790844653598279800.json)；不是分析完成或性能结论。

最新完整训练事件观察[evidence/training_progress_1790843809457814700.json](evidence/training_progress_1790843809457814700.json)记录当时rx1两进程VmHWM为1273280KiB/1218616KiB（约1.21GiB/1.16GiB），只为观察时进程高水位，不是最终全run/星载峰值。完整成本仍待诊断。

完整query重复评测源码由query-blind子agent分工实现，仅读源码及合成输入，不读取实际成绩、不改冻结方法。入口完成前不启动query评测；该开发不修改当前数学分析。
