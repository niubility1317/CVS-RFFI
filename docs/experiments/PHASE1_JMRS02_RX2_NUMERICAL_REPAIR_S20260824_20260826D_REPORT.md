# PHASE1_JMRS02_RX2_NUMERICAL_REPAIR_S20260824_20260826D预登记与追踪报告

本文件镜像本地正式报告。D聚焦修复RX1零点范数导致的非有限梯度，矩阵为`B0,RX0,RX2`；实际代码commit为`b62a17bffd0071754703871ce3ae42ef7bcdf5de`。RX0是可训练的同容量全局校正对照，RX2额外读取IQ接收机条件；RX2必须同时优于B0和RX0才允许晋级。

实际发布使用Git归档`PHASE1_JMRS02_RX2_NUMERICAL_REPAIR_b62a17bf.zip`，SHA-256=`6923571670d23834a555a29de52b0309f544880a4286a89a8f31fa294749928b`，本地/N607一致。远端编译与launcher语法检查通过。

真实checkpoint无query反向smoke已通过：RX0/RX2 prediction agreement均为1.0，非有限梯度均为0，estimator gradient norm分别为0.0389837和0.134840。formal唯一主进程PID=`3614644`，在GPU1运行；C继续在GPU0运行且未受干预。

完整报告位于本项目实验报告承载面对应run ID。当前状态：`RUNNING / REAL_CHECKPOINT_BACKWARD_SMOKE_PASS / NO_PERFORMANCE_RESULT_YET`。

## 最终结果

D完成151200条prediction/truth闭合、14份20-epoch完整history和3个独立评分JSON；280个正式epoch的非有限/清洗梯度总数为0。B0、RX0、RX2三LEO final均值分别为89.9233%、89.9259%和89.9048%。RX2相对B0为-0.01852pp、相对RX0为-0.02116pp，故`passes_rx2=false`。

值得保留但不能晋级的信号是：未门控RX2 candidate三LEO均值90.0767%，相对B0为+0.15344pp，raw rescue/harm=168/110；现有gate只选中15个救回和22个伤害，把正增益翻转为负增益。数值修复已验证，当前瓶颈转为truth-blind gate选择性。

C的receiver候选和D1P均失败，D的final gate也失败，因此不启动`BEST_RECEIVER+D1P`或其他joint。D仍是RX0–RX6 source receiver LORO，不是RX7–RX11 target-DG或ADV3B02同协议实验。

最终状态：`ANALYZED / NUMERICAL_REPAIR_VERIFIED / SCIENTIFIC_GATE_FAILED / NOT_TARGET_DG`。
