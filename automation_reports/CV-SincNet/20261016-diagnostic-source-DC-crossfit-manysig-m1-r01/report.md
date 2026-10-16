# 源 RX 加性 DC 诊断：旧增强已覆盖，固定方向关闭

**VERIFIED／ANALYZED，科学判定 FAIL。** 当前 F＋跨 RX 同 TX 对比学习基线不变。L5040 的源 RX 相关均值差在同时留 TX／day 的 18 折中有 16 折可预测，聚合风险降低 29.912510%；但 RX 均值对差的逐分量最大值 p95 为 0.004160463488，低于旧 ordinary 实际 DC 参数上界 0.005012006679，仅为其 83.01%。四项固定条件中的“超出旧幅度覆盖”失败，不进入新增 DC 合成训练。

这说明该统计包含可重复的 RX 关联，并未证明它是物理接收机硬件的因果效应，也未证明增大、减除或交换 DC 能保持 TX 身份。TX index3／day2 和 TX index4／day2 的留出风险变差保留，不选择性删除。旧参数上界只代表 DC 算子本身，不代表所有旧复合增强输出的全局上界。按预登记关闭本固定 RX-DC 合成方向，不换窗口、坐标、幅度、TX/day/RX 子集或阈值救援。

|固定条件|实际值|结果|
|---|---:|---|
|零效应总风险非零|True|PASS|
|聚合预测风险降低至少 10%|29.912510%|PASS|
|18 折至少 12 折正收益|16/18|PASS|
|RX 对差 p95 大于旧 DC 参数上界|0.004160463488 < 0.005012006679|FAIL|

只读取当前 C0 留 RX1 合同中 fit RX3/4/6/8 的原 L5040；6 个真 TX×4 RX×3 day，每 cell 70 包。原 L6300 只是既有缓存载体，先按合法 RX 元数据选定 5040 个物理 ID，再读取这些 IQ。held RX1 的 IQ、U/V 值及 U TX 真值不进入统计；原源缓存可能同时反序列化其他源角色容器，明确保留此事实。复用 VALIDATED_ONCE 数据，无目标 loader、模型 checkpoint 或身份／RX 头拟合。

描述量固定为完整 256 点 mean(I,Q)/sqrt(mean(I²+Q²))；先取 TX/RX/day cell 均值，再减同 TX/day 的四 RX 均值。每折只用其他 5 TX×其他 2 day 估计 RX 对比，预测未参与估计的 TX/day；零 RX 效应作平方欧氏风险参照。没有 IQ 变换、窗口／坐标扫描或新的输入归一化。数据实际 fs25 MHz／IQ256／equalized1／native RMS center=false 不变。先前原始长度只读核查确认全部 L6300 原数组恰为 256×2，因此真实长窗 crop 也没有额外观测可用，不能用反射／补零代替。

实现与验证：15 个协议负例、5040 合成描述量／18 折稳定效应及交互／零效应／旧幅度覆盖反例通过；同时留 TX/day 的预测隔离通过。隔离发布导入、实际单行 config 及 C0 真实增强 metadata 的旧上界还原通过。一次限定 P0/P1 审查 PASS。G0 不使用模型／checkpoint，no-query checkpoint smoke 不适用，没有加载无关权重来制造 smoke。

执行：实际代码提交 `f3c33bbee0d175f71444b7bd8c2f86a94618dd4b`。一次普通账户 preflight、一个 release archive 本远 SHA 比较及 remote compile 通过。独立读回 submit、execution、completion、resolved config 和 355 字节终止日志；PID／CWD／argv／CUDA 隐藏相符，工作进程自然完成，无 failure 或重试。任务只运行 1.421083 s，首次独立读取时已结束，因此不声称观察过 live PID 或运行中的日志增长。终止产物足以确认本次实际完成。

成本：CPU 2 线程、5040 个合法 L IQ 描述量、0 新模型、0 网络前向、0 checkpoint、0 优化更新、0 头训练、0 星地生成、0 目标访问；E20 模型训练预算消耗 0。离线图表和风险复算只读取保存的 JSON，0 IQ／网络／checkpoint。全部 18 折表和量级图已生成并查看；派生描述量与 RX profile 只留远端私有 NPZ，不提交 Git。

精确 row／命令／环境／CWD／路径／owner／停止规则见 [experiment.json](experiment.json)，预登记见 [preregistration.md](preregistration.md)。本轮不是识别训练或正式评分，没有 clean／urban 准确率，也没有跨模型 seed 证据。此前 paired-As 的主要源收益在 LOS／PARTIAL／locked，BLOCKED／unlocked 改善有限，RX6／RX8 clean 保护失败；其负结果不被本统计覆盖。独立有效 Ag、联合四格交互、正式七视图、low urban >60% 及跨 seed 共同突破仍未完成，目标保持 active。

- [全部 18 折误差表](E:/type10-7/local_artifacts/cvs_source_DC_crossfit_20261016/final/all_18_folds.csv)
- [风险与旧幅度覆盖图](E:/type10-7/local_artifacts/cvs_source_DC_crossfit_20261016/final/source_DC_risk_and_coverage.png)
- [独立复算摘要](E:/type10-7/local_artifacts/cvs_source_DC_crossfit_20261016/final/analysis_summary.json)
