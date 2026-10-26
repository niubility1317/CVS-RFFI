# NoSIG 原始／恢复域标签专家源证伪

状态：**RUNNING，两个专家已实际训练并独立核实**；尚无本轮标签质量或目标性能。不是三个学生种子的结果，原联合目标未达成。

执行commit `107d092fde1292a66c5c6aef8c547712c524f813`。一次archive传输比对VERIFIED、远端compile PASS；控制器和两个worker的PID/CWD/argv/GPU驻留及训练记录增长已独立核实。RawLabeler在GPU0，GLabeler在GPU4。真实NoSIG E30／原默认头／native Sinc／固定G的原始与恢复L128 backward smoke PASS，头和Sinc梯度有限非零、零optimizer、零U训练／V／target读取。

完整原L6300×12恢复IQ缓存已完成（75,600 G包、154,828,800 float字节），G参数及全部注册buffer不变。两个专家同父E30、同顺序同预算：各新增10轮／2220步L128，新增适配只用RX3/4/6/8的5040条，RX1新优化曝光0。骨干与原默认头及native Sinc联合CE，CENTER_UNITS／物理clip5；RawLabeler只输入raw，GLabeler轮换12份同包G。父和G已见过RX1，不能称真正未见接收机验证。

每行final自动源V预测，原始专家27,000包、恢复专家12×27,000包。全部预测固定后再计标签质量：原E30 raw confidence<.95困难池中，两专家各confidence≥.95且同类。固定条件为认证≥200、实测precision≥99.5%、相对原E30净纠正≥100；RX/day/TX与单专家质量一并报告。未通过便保留负结果；不调阈值，不生成U标签，不自动训练学生。

RGPL固定双图已失败并归档，本轮不读取其伪标签。7项聚焦检查和唯一独立P0/P1审查完成；审查发现的probe冻结状态检查问题已修复并通过一次定向复查。它们只证明执行条件，不能替代科学结果。

[源计划](../../../docs/goals/PAMF_RESTORATION_LABELER_PLAN_20261026.md)、[预登记](preregistration.md)、[实际启动证据](launch_evidence.json)、[图方法负结果](../20261026-diagnostic-PAMF-NoSIG-RGPL-sourceGraph-m1-r01/report.md)。
