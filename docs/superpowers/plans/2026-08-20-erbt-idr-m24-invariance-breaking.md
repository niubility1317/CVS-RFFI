# ERBT-IDR M2.4非等价机制完整125实验实施计划

## 目标

修复`M24-D1-REFIT`与去RF32的D92 E0在特征投影和LDA拟合上的代数等价问题，构建不读取query或truth的support-only新头，并在同一完整125输入身份上与D92 E0直接比较。

## 方法矩阵

- G0：`M24-D0-HISTORICAL-F1`，去RF32的D92 E0主基线。
- G1：冻结平衡IF256原型头，不拟合target全协方差，K1不回退历史F1。
- G2：G1加类别中心张成空间正交的rank-1接收机干扰硬投影；K1因无法估计类内方向而退回G1。
- G3：G2加类别对称的support离散度不确定性惩罚。
- G4：G3加K≥5的确定性双原型与按类归一化log-mean-exp；K1/K2保持单原型。

每个方法均运行5个receiver×5个seed×5个K/new条件，共125组；总计625个方法行和1875个场景单元。

## 实施顺序

1. 先写机制测试，确认在生产模块缺失时失败。
2. 实现平衡IF、非可逆投影、不确定性和双原型状态。
3. 接入truth不可见row executor、完整125预测runner、truth-last scorer和分析器。
4. 运行聚焦负测、真实checkpoint无query smoke及一次P0/P1正确性审查。
5. 固定Git提交、发布单一release归档到N607、校验一次归档SHA并远端编译。
6. 启动不可覆盖run ID；prediction完整后再连接truth评分。
7. 生成总体、K/new、receiver、seed、scene、old/new、class、margin、中心角距、help/harm、`F_within/F_std`及资源分析，写回正式报告并发布。

## 科学停止规则

- 协议越权、错误矩阵、输出覆盖、错误checkout、确定性重复异常、prediction不闭合或scorer连接错误时停止并保留证据。
- 低性能不停止实验。
- 如果新头在完整prediction后与G0完全无分歧，仍完成truth-last评分并判为机制无效，不把该诊断解释为新的发布审核门。

