# D92-MarginJointLocalRidge support联合实验预登记

状态：TRAINING_ON_SUPPORT，启动VERIFIED。root唯一launch owner。实际不可变runtime e50de0ad4e7d0570dce62ba88c3791fdf1c14951；supervisor PID756554/start8891521。两CPU evaluator PID756564/756565已启动，另两行PENDING。独立证据见[evidence/runtime_readback_1790841357922706700.json](evidence/runtime_readback_1790841357922706700.json)，frozen startup已绑定原spec。

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

当前没有完成marker或实际性能。不能将旧方法A/B/C拼给Margin。待原四行结束后，执行一次独立数学摘要和完整训练诊断，再用同物理held记录做Ground A配对。保留所有失败与健康lane，不重复publisher、preflight或request生成。
