# PAMF严格监督结构时延：修复来源替代

状态：本地复现与定点修复完成，待源侧检查和实际发布。

失败发生在模型构建期：GPU参考bank与CPU固定phase相乘。修复仅把同一FP64相位移至bank设备，随机值、种子、原L-only CE配方和预算保持。失败行及其下游用新root替代，旧产物保留；未启动同结构消费者只换执行版本，全部健康PID保持。专用时延仍为原450 case-round，旧统一500不重测。

[恢复范围](../../../docs/PAMF_WRONG_REFERENCE_DEVICE_RECOVERY_20261023.md)；命令、种子、来源、资源、停止规则和预期产物见experiment.json。新增性能仍pending。
