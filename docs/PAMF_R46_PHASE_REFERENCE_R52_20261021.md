# R52：接收相位补偿与坐标传递的源域参考对照

VERIFIED/ANALYZED：128源样本、1536主前向及2Ls真实checkpoint smoke；108单元的五种误差独立复算通过，R51三个旧指标共108数组逐项一致。两received非零C4 final均值：等变0.82160833、不变1.49335690、固定clean1.50956389、坐标传递clean0.84315633。0F/TX/V/target/truth/训练，原350521父逐张量未变，所属CPU进程已退出；原完整性能和TX真实性目标ACTIVE。

|源视图|输入旋转角度|等变 NMSE|不变 NMSE|固定 clean NMSE|坐标传递 clean NMSE|原输出 clean NMSE|
|---|---:|---:|---:|---:|---:|---:|
|clean|0|0.00000000|0.00000000|0.00004318|0.00004318|0.00004318|
|clean|90|0.04124547|1.93308880|1.93245922|0.04104720|0.00004318|
|clean|180|0.03508788|3.81672193|3.81508007|0.03511086|0.00004318|
|clean|270|0.01655705|1.97325203|1.97251449|0.01662308|0.00004318|
|received01|0|0.00000000|0.00000000|0.02099969|0.02099969|0.02099969|
|received01|90|0.82435531|0.70292225|0.73071856|0.84532782|0.02099969|
|received01|180|1.25366267|2.55765574|2.56952720|1.27871434|0.02099969|
|received01|270|0.54749858|1.08209461|1.08815349|0.56901462|0.02099969|
|received20|0|0.00000000|0.00000000|0.01984630|0.01984630|0.01984630|
|received20|90|0.79332548|0.71151055|0.72848409|0.81663617|0.01984630|
|received20|180|1.02641374|2.77710889|2.79369595|1.04712840|0.01984630|
|received20|270|0.48439409|1.12884950|1.14680439|0.50211660|0.01984630|

固定 clean 定义只旋转 received，比较 G(Ry) 与原 clean x；坐标传递定义把输出逆转再与 x 比较。等变误差比较 R⁻¹G(Ry) 与 G(y)，不变误差比较 G(Ry) 与 G(y)。两个定义对应不同目标，不能互换。所有 NMSE 按同一原 clean 的 I/Q 平均能量归一化，下限 1e-4。

原输出与 clean 的小误差，仅说明原 R07 输入参考下的重建；非零旋转后的误差不能单独定位为信道估计、先验网络或物理不可辨识性的哪一种原因。C4 90°/180°/270° 是较大相位压力，未被证明等同于当前六个 practical 的残余相位分布。未据本诊断加入等变损失、选择相位门限或改写恢复器。

同一 64 L_s＋64 U_s、抽样 seed2026102051、clean/R07 received01/20、三stage与四C4角度；完整108单元和mean/median/P95/max见[汇总](../automation_reports/CV-SincNet/20261021-diagnostic-restoration-R46-phase-reference-s2026092701-r52/summary.json)与[逐组CSV](../automation_reports/CV-SincNet/20261021-diagnostic-restoration-R46-phase-reference-s2026092701-r52/row_metrics.csv)。独立 NumPy 算术核对五指标、原采样和108个旧指标数组，原模型状态不变。

本源域诊断不证明跨RX测试收益或每个细节属于输入TX。原八个压缩版的七场景正式测试均保留；没有新增训练网络版本。本次 CPU 耗时包含加载与探测，不是 GPU 推理时延。训练使用 TX 无标签模拟配对 clean/received 的权限没有变化，不能称无 clean 盲训练。

部署状态 VERIFIED：普通账户、一次预检、单归档两端SHA比较、远端compile、真实2Ls smoke、存活实际PID/CWD/argv/空CUDA/CPU2、日志增长和最终退出均已核实。配置、代码commit、路径、停止规则和预算见[登记](../automation_reports/CV-SincNet/20261021-diagnostic-restoration-R46-phase-reference-s2026092701-r52/experiment.json)。

性能优先继续有效：原 low urban ≥70%、其他五受损场景距 raw-clean≤5pp 和更强TX细节保真目标尚未完成；本次不更换用户指定350521直接父，不用正式分数回流调参。下一步回到现有压缩结构的合法源训练目标优化，新训练行自动进行七场景正式测试。


## 固定方法与历史发布记录

# R52：相位补偿与坐标传递的源域参考对照

状态：LOCAL_VERIFIED，尚未发布。21项聚焦检查与CLIhelp通过，唯一限P0/P1审查PASS。

目的：R51测得非零C4输入旋转的反向输出与原输出存在明显差异，但接收链逆问题本身可能需要补偿载波相位。R52同时保留坐标传递定义和固定clean定义，不把“应该等变”预设为任务正确性。只测原R46 prune_refine350521，父source-final E20/4940、方法bbb8daf817de050b958d217b70d4c33f4aefe341；不选择新父/通道/损失，不训练或改变推理网络。

固定同R51原6300Ls/56700Us池、seed2026102051、64Ls+64Us、clean/R07received01/20及精确C4旋转。三stage和overall/Ls/Us共108聚合单元、1536主前向样本，加实际2Ls noquery smoke。新增invariance_NMSE=||G(Ry)-G(y)||²/Eclean、fixed_clean_NMSE=||G(Ry)-x||²/Eclean；保留equivariance_NMSE=||R^-1G(Ry)-G(y)||²/Eclean及两个旧pairedclean指标。Eclean为每例I/Q平均能量，下限1e-4。报告全部mean/median/P95/max和两个received视图三个非零角度final的固定均值。旧0.001仅保留R51描述字段，不是新训练/选择门槛。

21项聚焦检查包括两个可解析反例：身份等变算子能有0等变误差却有非零固定参考误差；相位补偿算子恰好相反；混合响应可同时不满足两者。两种误差测量的是不同性质，不能把其中一项直接当TX保真或正式性能。C4为较大坐标/相位压力，不能冒充六种practical残余相位分布。

复用已验证original7/63/30角色和物理ID、source READY/EXACT_MATCH、R07缓存与R46/R44/R38/R17/R07/R04源祖先；运行时同一parent/source句柄核对。TX标签/F/V/target/truth读取与训练更新均为0。原G eval/inference_mode且所有参数冻结，最终逐张量核对不变。无新恢复器版本/分类头/target预测，已完成八版正式七场景测试保持。本诊断不替代正式测试，不是无clean盲训练，不声称逐细节真实性或新独立盲测。

CPU2/0GPU、空CUDA、FP32/TF32关闭；R50独立复算与R51源检验均已结束，不叠加本任务CPU工作。新不可覆盖run/release/output，唯一owner codex/root/R46-phase-reference；普通账户，一次preflight/单归档两端SHA/compile。首步实际2Ls全链smoke PASS即继续；启动器立即读取实际子PID/CWD/argv/空CUDA，之后读取增长日志或完成artifact，适用于短任务。技术失败保留影响输出，不自动重试/热改/回退，不触碰其他owner。

独立逐组算术及旧R51三指标核对均在完成后执行，数据/权重/原始逐例指标/日志不入Git；公开聚合结果与报告。原70%、其他五景距rawclean≤5pp及更强输入TX细节目标保持ACTIVE。现有额外设计审批等不在用户八项最小流程内，记REJECTED_EXTRA_GATE，不等待。精确commit、命令、路径、seed、预算与停止规则见experiment.json。
