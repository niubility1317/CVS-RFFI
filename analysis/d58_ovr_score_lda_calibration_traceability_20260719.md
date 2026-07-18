# D58逐类one-vs-rest分数LDA校准追踪

|需求|设计|验证|状态|
|---|---|---|---|
|D46底座|保留B20、full/block类级权重与RMS|D58/D56/D46共33项回归|LOCAL_PASS|
|连续双向证据|每类正分数与负类吸收分数的均值/方差|手算、平移/尺度和重算闭包|LOCAL_PASS|
|仿射校准|闭式一维LDA正斜率＋midpoint，同时改W/b|exact affine重算|LOCAL_PASS|
|类对称|无ID/role/scene/receiver；标签置换等变|类/rank置换|LOCAL_PASS|
|失败闭锁|非正分离、零方差、非有限值整fit回退D46|反例测试|LOCAL_PASS|
|K1/K2|精确D46 fallback|参数化测试|LOCAL_PASS|
|资源边界|复用D56拟合，只增加矩与affine标量运算|K1/2/8公式闭包|LOCAL_PASS|
|协议与性能|query0、105行、完整同排报告|receipt/summary/report|PENDING_RUN|

D58为support-only、强制nonpromotable开发探针；开发门全部通过前无formal/125权限。
