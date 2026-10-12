# 浅中层 Fishr 与 MixStyle 的 AdamW 源诊断

完整 18 个固定 final20 模型、216 批、1,728 次临时 AdamW 调用已核对。持久身份更新为 0，无目标数据访问。下表为 FULL 相对 IDENTITY 的全模型位移，每行汇总固定三种子各 12 批。分层、逐种子、四目标的全部结果见 summary.csv。

|模型|优化器模式|平均梯度余弦|平均位移余弦|平均位移差/身份位移|平均相对参数位移|
|---|---|---:|---:|---:|---:|
|calibrated-CAP_JOINT|saved_final|0.991083|0.997124|0.0688413|6.6167e-12|
|calibrated-CAP_JOINT|reset_peak|0.991083|0.866733|0.509948|6.3761e-09|
|calibrated-DELAYCAP_JOINT|saved_final|0.997546|0.999144|0.0382285|6.78926e-12|
|calibrated-DELAYCAP_JOINT|reset_peak|0.997546|0.927195|0.377884|6.37501e-09|
|calibrated-DELAY_JOINT|saved_final|0.996315|0.999107|0.0405728|6.65537e-12|
|calibrated-DELAY_JOINT|reset_peak|0.996315|0.922832|0.386493|6.36631e-09|
|identity-CAP_JOINT|saved_final|0.990949|0.997659|0.0624112|6.63872e-12|
|identity-CAP_JOINT|reset_peak|0.990949|0.873379|0.496804|6.38358e-09|
|identity-DELAYCAP_JOINT|saved_final|0.996892|0.999248|0.0368283|7.01387e-12|
|identity-DELAYCAP_JOINT|reset_peak|0.996892|0.922526|0.390205|6.35943e-09|
|identity-DELAY_JOINT|saved_final|0.99614|0.999092|0.0385412|6.6494e-12|
|identity-DELAY_JOINT|reset_peak|0.99614|0.920567|0.392368|6.34666e-09|

保存状态与重置状态的学习率也不同，跨模式差不能仅归因于动量。模式内部的 FULL 与 IDENTITY 才隔离 Fishr 项的影响。浅层组为 time_fuse，中层组为 t2；其余参数另报，不能把组名等同于完整网络深度分区。
独立 L 参照与当前主 L 不重叠，但原训练已见过，并可能与辅助 A/B 重叠。梯度和位移的内积仅为一阶方向导数，没有执行更新后前向，不能称为实测误差下降或未知 RX 泛化。
保存状态用于 final20 权重上的局部反事实；诊断位置是完全启用的目标，并非恢复历史训练步骤。负结果和零范数均保留。未生成新候选，不据此读取或挑选目标测试。
正式诊断共 2,592 次模型前向、247,104 次 L 样本读取、55,296 次 U 样本读取。发布前真实源预检额外 30 次前向、1,180 次 L、256 次 U、8 次临时 AdamW；成本与正式诊断分列。
