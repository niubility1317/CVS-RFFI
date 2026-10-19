# 原F跨RX50：隐藏标签源U置信度诊断

仅分析已预登记并完成的BASE/BOTH各三个seed final50，每个student与自身EMA快照各前向完整源U56700个IQ，共12快照／680400次前向。无模型、optimizer、EMA或分类头拟合更新，无星地增强，无目标或任何truth访问。当前RX_MARGIN健康训练保持原计划。

只统计预测TX分布、置信度、熵和0.92/0.95/0.97置信度尾数，按源RX/day分组，并比较student/ownEMA预测一致性。U_s的dataset.y必须为None、返回y全部−1、tx_label_visible全部False；真实checkpoint零输入smoke通过后立即继续U诊断，参数和buffer检查不变。既有source角色、原始训练实现commit31467086c、默认头和数据缓存复用，不重新验证数据或训练模型。

现有训练日志不能还原逐TX历史伪标签接受数。本诊断不重放普通增强、批内邻居和强一致门控，冻结快照的阈值尾数不等于训练接受率；没有U真值，也不能计算精度、召回或真实类别覆盖。阈值只作事先固定的描述性统计，不用于修改当前训练。具体矩阵、环境、路径、命令、资源和停止规则见experiment.json。

一个GPU0诊断worker顺序运行，全owner最多3个GPU worker且空闲RAM至少4.5GiB。任何角色/隐藏标签/来源/数值/模型状态/输出异常即失败并保留产物，不停止其他健康进程。仅聚合数值进入Git。
