# PHASE1_JMRS02_RX2_NUMERICAL_REPAIR_S20260824_20260826D预登记与追踪报告

本文件镜像本地正式报告。D聚焦修复RX1零点范数导致的非有限梯度，矩阵为`B0,RX0,RX2`；实际代码commit为`b62a17bffd0071754703871ce3ae42ef7bcdf5de`。RX0是可训练的同容量全局校正对照，RX2额外读取IQ接收机条件；RX2必须同时优于B0和RX0才允许晋级。

完整报告位于本项目实验报告承载面对应run ID。当前状态：`LOCAL_VERIFIED / NOT_YET_LANDED / NO_PERFORMANCE_RESULT`。
