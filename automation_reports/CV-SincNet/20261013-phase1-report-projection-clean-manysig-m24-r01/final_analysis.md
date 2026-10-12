# 报告判别投影：三种子clean结果

|方法|三seed均值|SD|≥84%|
|---|---:|---:|---|
|NATIVE|79.4562%|1.8033%|False|
|COS_MEAN|75.7496%|1.6404%|False|
|CENTERED_COS|75.6827%|1.6089%|False|
|GEIV|75.2329%|3.0261%|False|
|ORTHOGONAL|73.0067%|1.7978%|False|
|OBLIQUE|74.9480%|1.5946%|False|
|RIDGE|75.9490%|3.0594%|False|
|SOFT_RIDGE|76.0157%|3.1848%|False|

每row168000 clean，全部source参数与同seed8模型固定后预测，再独立truthlast评分；没有sourceV性能筛选/practical测试/query拟合。三个已严格核实的纯CE30父模型复用，共24冻结派生状态，新增梯度更新0；校准只L6300，源角色不变。所有输出为常规cos，投影保留160输出接口；硬删方向实际秩及15TX均值保持、Gaussian有效Fisher与原始source统计边界见source_fit_diagnostics，不声称新类完整身份证据或已通过Phase2源状态审计。
基准已暴露，方法/seed/矩阵/参数在测试前固定；全部实际结果保留，不按target挑seed、调参数或重跑。

独立核对VERIFIED：4032000个原始physicalID预测及432条完整same-row混淆指标重算；三个NATIVE各168000决策与原CE30同父clean完全一致。源L6300重新前向并拟合，非GEIV矩阵与真实GPU状态相对误差≤0.002；GEIV退化特征空间的基向量不逐元素断言相同，保存实际特征值/谱规则/机制数学检查。父模型无更新、源V不前向、U不打开。源统计原始cache未导出到预测状态，Phase2资格仍未审计。

登记文字更正：旧源先导模板残留source_days0/1/2、source_only/no target、全部U未打开；本轮实际依据最新用户clean授权及冻结rowconfigs执行，源contract/resolved显示day1/2/3、正式两SSL使用隐藏U、全部clean测试。仅纠正登记元数据，原不可变release/run/spec及所有训练/预测/评分产物保留；不改变划分、roleIDs、方法、预算或已冻结矩阵。
