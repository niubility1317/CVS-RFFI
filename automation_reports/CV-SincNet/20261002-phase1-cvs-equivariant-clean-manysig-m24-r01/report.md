# 全路径复相位等变记忆 CVS：选中候选 clean 确认

状态 LOCAL_VERIFIED，尚未访问本轮新 query。固定源规则（4 新+4 原残差控制）选择 `equivariant_memory`；四个选中 checkpoint 加二十份已有冻结预测只读复用，共 24 行、原 168000 物理 query/6 TX/7 RX，完整预测固定后独立 truth-last 评分。

模型、源执行、clean 三个不同范围独立 P0/P1 审查 PASS。完整源日志已核对；源原角色/预算/scratch 和 checkpoint payload 在预测前再次确认。真实新矩阵仅四个选中 seed，默认完成测试和同 row 报告。匹配 VALIDATED_ONCE 数据复用；性能优先、参数成本次要。目标成绩不回流调参或选择性重跑。全网具有公共常相位约束，仍不能宣称任意信道/RX不变或TX参数辨识。

[源完整报告](../20261002-phase1-cvs-equivariant-identity-manysig-m4-r01/report.md) · [冻结源选择](evidence/performance_selection.json) · [验证](evidence/local_validation.json)。


GPU冻结合成诊断的最大全网相位logit误差为0.0268415213，超过预先报告容差0.001；最大单位嵌入距离0.00186185061。实数精确运算下的等变推导与实际有限精度分别报告，不声称本次GPU数值容差通过。该诊断不参与源排名或新门槛；按既定源规则继续clean，待用固定source权重/公共合成输入定位数值来源，不改待测模型，不以测试反馈修补或重跑。
