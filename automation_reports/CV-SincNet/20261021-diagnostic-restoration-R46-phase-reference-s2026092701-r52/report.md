# R52：相位补偿与坐标传递的源域参考对照

状态：LOCAL_VERIFIED，尚未发布。21项聚焦检查与CLIhelp通过，唯一限P0/P1审查PASS。

目的：R51测得非零C4输入旋转的反向输出与原输出存在明显差异，但接收链逆问题本身可能需要补偿载波相位。R52同时保留坐标传递定义和固定clean定义，不把“应该等变”预设为任务正确性。只测原R46 prune_refine350521，父source-final E20/4940、方法bbb8daf817de050b958d217b70d4c33f4aefe341；不选择新父/通道/损失，不训练或改变推理网络。

固定同R51原6300Ls/56700Us池、seed2026102051、64Ls+64Us、clean/R07received01/20及精确C4旋转。三stage和overall/Ls/Us共108聚合单元、1536主前向样本，加实际2Ls noquery smoke。新增invariance_NMSE=||G(Ry)-G(y)||²/Eclean、fixed_clean_NMSE=||G(Ry)-x||²/Eclean；保留equivariance_NMSE=||R^-1G(Ry)-G(y)||²/Eclean及两个旧pairedclean指标。Eclean为每例I/Q平均能量，下限1e-4。报告全部mean/median/P95/max和两个received视图三个非零角度final的固定均值。旧0.001仅保留R51描述字段，不是新训练/选择门槛。

21项聚焦检查包括两个可解析反例：身份等变算子能有0等变误差却有非零固定参考误差；相位补偿算子恰好相反；混合响应可同时不满足两者。两种误差测量的是不同性质，不能把其中一项直接当TX保真或正式性能。C4为较大坐标/相位压力，不能冒充六种practical残余相位分布。

复用已验证original7/63/30角色和物理ID、source READY/EXACT_MATCH、R07缓存与R46/R44/R38/R17/R07/R04源祖先；运行时同一parent/source句柄核对。TX标签/F/V/target/truth读取与训练更新均为0。原G eval/inference_mode且所有参数冻结，最终逐张量核对不变。无新恢复器版本/分类头/target预测，已完成八版正式七场景测试保持。本诊断不替代正式测试，不是无clean盲训练，不声称逐细节真实性或新独立盲测。

CPU2/0GPU、空CUDA、FP32/TF32关闭；R50独立复算与R51源检验均已结束，不叠加本任务CPU工作。新不可覆盖run/release/output，唯一owner codex/root/R46-phase-reference；普通账户，一次preflight/单归档两端SHA/compile。首步实际2Ls全链smoke PASS即继续；启动器立即读取实际子PID/CWD/argv/空CUDA，之后读取增长日志或完成artifact，适用于短任务。技术失败保留影响输出，不自动重试/热改/回退，不触碰其他owner。

独立逐组算术及旧R51三指标核对均在完成后执行，数据/权重/原始逐例指标/日志不入Git；公开聚合结果与报告。原70%、其他五景距rawclean≤5pp及更强输入TX细节目标保持ACTIVE。现有额外设计审批等不在用户八项最小流程内，记REJECTED_EXTRA_GATE，不等待。精确commit、命令、路径、seed、预算与停止规则见experiment.json。
