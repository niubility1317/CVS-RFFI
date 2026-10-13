# 50 轮收尾计数故障

VERIFIED：12 行的完整日志均达到 50 epoch、11100 optimizer updates，合计 133200 次更新和 11508480 个合法 U 包。原代码版本 75135508034ece4120d8b27ae2f2dbc93f5078c9 的收尾字段仍混入 30 轮的 epoch/6660/5550 旧值，纯 CE checkpoint/header 与预测检查不匹配，U 模型在固定预算 assertion 处退出。

没有任何目标 prediction 或 score，状态为 STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT。三个 CE checkpoint 保留原样，但其旧计数 header 不用于新 run；九个 U 模型未保存 final checkpoint。全部日志、源角色和 partial 产物保留。

独立核实停止时所有相关 PID 已自然退出，没有实际发送 SIGTERM，没有影响其他 owner。新 r02 保持四方案、三 seed、每模型从零 50 轮、普通 cos、相同数据/半径/日程与 clean 测试，原 r01 checkpoint 不继承。原失败尝试成本与新 run 成本分别报告，不能把总成本冒充一次 50 轮。

修复将训练端 checkpoint/done/freeze 与预测端检查统一从当前 config 推导。跨阶段回归在修复前复现拒绝合法 50 轮产物，修复后通过，并拒绝旧 30 轮元数据。目标数据未参与故障诊断或修复。
