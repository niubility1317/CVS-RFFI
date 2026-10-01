# SupportMetric 联合方法实现验证（2026-10-02）

**新候选源码已实现，75项合成用例通过；没有新的真实数据性能结果。** 方法名为 `D92-ProtoFrameSupportMetric-GGN1-LocalRidge`，状态schema为 `d92_support_metric_joint_local_ridge_v1`。旧ProtoFrame/GroupBarrier运行版本未改动，也未创建新配置或启动实验。

新候选保留LocalRidge解析头与完整组合导数，在冻结原型可达空间中定义物理坐标，以实际Gram加合法support预测Fisher构造GGN度量球。它从完整受限方向eta=1开始回读真实RMSCE并固定折半，最多一步接受。它不加载旧adapter，C只继承本次真实B；rank0/K1完整拟合最终head，new0直接复用B。

## 直接验证

|模块|最新合成用例|覆盖行为|
|---|---|---|
|精确字典几何|23通过|精确零与冗余、最小subnormal及投影后更小非零残差、整数平方根包络、输入/返回不可变、资源失败与真实账|
|support度量步|34通过|解析一维内点/边界、非正交换坐标、Fisher类均衡/中心化、实际非单位Gram、SPD与非法输入、后续factor/triangular失败与快照|
|联合分类器|18通过|U坐标的完整kernel/Ridge/阈值/gate/RMS差分、实际B→C与old-inner隔离、两新类非零JVP和单新类零导数、r0/r1/K1/new0、纯几何显式复用及成本|

首次step为30通过/1失败，原因是secular按长度差终止、最终按平方范数互补回读的尺度不一致。修复将停止条件直接与最终原坐标互补量对齐，没有放宽最终容差或改变半径/目标。独立审查另修后续attempt失败混入旧数组的问题，现逐attempt保存当前RHS、因子和已完成forward。

首次joint为16通过/1失败：只有一个新类时，条件softmax恒1、阈值Jacobian正确为零，测试却要求非零。production未改；主差分改两个不同新类，新增一个新类的精确零回归。所有首次失败原日志保留；Git保存可逐字节解压的gzip，不改原stdout空白。

测试由root串行执行，使用官方CmdExeActivator stack激活的ssr-gpu Python3.10、NumPy2.2.6、SciPy1.15.3、Torch2.1cpu；未启动Conda CLI/hooks。它们只使用合成字典与合成合法support，没有真实support/query、checkpoint或源样本读取。完整证据与各环境见[验证JSON](evidence/d92_support_metric_20261002/implementation_validation.json)。

[限定独立审查](D92_SUPPORT_METRIC_STEP_CORE_REVIEW_20261002.md)结论为无未解决P0/P1；审查者未自审自己的basis，root另读basis源码并执行数值验证。无新问题时不重复未改行为的审查或测试。

## 已证明与未证明

字典包络覆盖已存binary64 Q的精确有理几何，不恢复舍入前原型。step只提供FLOAT64_SUBPROBLEM_DIAGNOSTIC_NOT_COMPLETE_HEAD_CERTIFICATE；完整Ridge/gate JVP区间误差界仍未建立。有限差分、正定性和KKT残差不是全输入正确性或query泛化证明。

新坐标属于U[r]而非旧Q五坐标，禁止历史适应state继承。整数证书、实际三角RHS、Fisher和全部head/trial的工作照实计费；预建冻结basis可以显式复用，构造费用和绑定费用分开。进程RSS、星载训练/推理、功耗、native wire、部署序列化及实际增量传输未测，均为N/A；不得把r≤5解释为已降低整机计算开销。

下一步在源码交付后接入独立support入口、逐路径状态归档与完整分析；先预登记一条固定候选，保留原日志和失败，不产生参数网格。固定Phase1/practical residual、源数据禁止、VALIDATED_ONCE复用及全矩阵prediction固定后truth-last的边界不变。已有query实验继续原排程；结果不回流调参或选择重跑。

[数学定义](D92_PROTO_FRAME_GGN_COORDINATE_SCALE_NOTE_20261002.md)、[数值边界](D92_PROTO_FRAME_SUPPORT_METRIC_NUMERICS_NOTE_20261002.md)、[集成计划](D92_SUPPORT_METRIC_JOINT_IMPLEMENTATION_PLAN_20261002.md)分别保留理论、严格认证未完成项和当前实现状态。
