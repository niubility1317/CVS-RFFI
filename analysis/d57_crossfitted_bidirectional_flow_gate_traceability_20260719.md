# D57交叉拟合双向混淆流门追踪

|需求|实现|验证|状态|
|---|---|---|---|
|D46/D56底座|继承D46分数与D56support混淆流|D57/D56/D46共31项回归|LOCAL_PASS|
|held排除|折`r`只用其余K−1折建流|exact complement重算与审计回放|LOCAL_PASS|
|双向类门|positive不降且false-positive不增|安全坐标与拒绝坐标反例|LOCAL_PASS|
|联合安全|accepted坐标联合伤害时原子D46 fallback|交互反例精确回退|LOCAL_PASS|
|类对称|无ID/role/scene/receiver/顺序|类标签置换等变测试|LOCAL_PASS|
|K1/K2|精确D46 fallback|参数化测试|LOCAL_PASS|
|资源边界|复用D56拟合，只增加标量门控|资源公式闭包|LOCAL_PASS|
|协议与性能|query0、105行、完整同排报告|receipt/summary/report|PENDING_RUN|

D57为support-only、强制nonpromotable开发探针；未通过开发门前无formal/125权限。
