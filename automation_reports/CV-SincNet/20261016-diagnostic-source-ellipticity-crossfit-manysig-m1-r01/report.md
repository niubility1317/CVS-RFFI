# RX 椭圆性：通过源统计，尚未证明增强有效

**VERIFIED／ANALYZED，PASS_SOURCE_STATISTIC_ONLY。** 当前 F＋跨 RX 同 TX 对比学习、从头 E20、joint160／默认 cosine 头联合训练及正式测试条件不变。完整 256 点 AC 椭圆性的 RX 均值差在同时留 TX/day 的 18 折中有 12 折可预测，聚合风险降低 27.486642%；RX 均值对差 p95 为 0.032791289171，高于预登记 0.01。全部四项固定统计条件通过，仅支持另登记一次有界广义线性接收 IQ 的 E20 源识别验证。

**六个失败折不能忽略**：TX index1 和 index3 的三个 day 全部风险增加，其中 TX index3/day1 为 -612.5%、day3 为 -487.4%。相对风险可以低于 -100%，因为零效应风险很小；绝对零／预测风险全部保留。正折数恰好达到 12/18，并非所有 TX 通用或强物理因果证据。不能删去这两个 TX、选择 day 或复数方向救援，也不能把统计 PASS 写成标签保持或 clean／urban 准确率提升。

|固定资格|实际值|结果|
|---|---:|---|
|零效应总风险非零|True|PASS|
|聚合风险降低至少10%|27.486642%|PASS|
|至少12/18折风险降低|12/18|PASS|
|RX对差p95至少0.01|0.032791289171|PASS|

只读取当前 C0 留 RX1 合同的 fit RX3/4/6/8 原 L5040，6 个真 TX×4 RX×3 day，每 cell70。原 L6300 为既有缓存载体；先按源 L 物理 ID 与 RX 元数据过滤，再使用这些 IQ。held RX1 IQ、U/V 值和 U TX 真值未使用；缓存可能同时反序列化其他源角色容器，但这些值不进入统计。复用 VALIDATED_ONCE 源数据，无新数据验证或任何 target loader。fs25MHz／IQ256／equalized1／native RMS center=false 不变。

描述量只用 `abs(mean(x_ac**2))/mean(abs(x_ac)**2)`，完整256点、AC去均值仅用于统计；没有训练／部署前处理变化。只用椭圆性大小，没有复数方向、频率、窗口或 RX 子集扫描。标量相位、增益及加性 DC 不变性是代数事实；旧 PA、多径、噪声等复合增强仍可改变此量，不能断言新旧输出支持不重合。TX／调制／通道也能影响椭圆性，不把关联解释为 RX 硬件因果或反演未知 TX 效应。

18 折同时留一个 TX 和一个 day，只用其他5 TX×其他2 day 估计 RX 对比。离线工具独立从全18折保存风险重算聚合值和正折数，核对角色、实际配置及终止产物；不读取新 IQ、checkpoint 或网络。派生逐包描述量与 RX profile 仅留远端私有 NPZ，不提交 Git。

实际代码 `5d1061781635f5227dfb7b9b1bee1868084d497d`。16 协议负例、5040 合成描述量、广义线性公式、标量相位／增益／DC不变性、稳定效应／TX交互／零效应／极小效应及留 TX/day 预测隔离均PASS。隔离发布模块路径与真实注册config一致，未导入torch；唯一限定P0/P1只读审查PASS。G0无模型／checkpoint，no-query checkpoint smoke不适用，没有加载无关权重。

一次普通账户CPU发布 preflight、一个release archive本远SHA比较和remotecompile通过。独立读回submit、execution、completion、resolved config及294字节终止日志，PID／CWD／argv／CUDA隐藏及代码身份一致，无failure或重试。CPU任务已在首次独立读取前完成，不声称观察过live PID或运行中的日志增长；实际终止产物VERIFIED。

成本：CPU2线程，5040合法L统计，墙钟1.737601 s；0模型、0网络前向、0checkpoint、0优化更新、0头训练、0星地生成、0目标。E20训练预算消耗0。图表与复算0额外IQ／网络／checkpoint；一幅实际分层图已查看。当前没有新clean／urban预测、分类头收益或跨seed证据，整体联合目标保持active。

精确row／命令／环境／CWD／路径／owner／停止规则见[experiment.json](experiment.json)，预登记见[preregistration.md](preregistration.md)。本次不自动启动训练或测试。后续最小身份检验必须另登记：保持共同旧ordinary和第一对原监督，只在第二对有界增加 `x+b*conj(x)` 型接收观测变化；参数按RX/day的虚拟session稳定，不由TX标签定义，不按包估计或校正TX硬件。保留同前向C0、原U／ownEMA／SupCon／默认头；其是否有效由实际E20识别决定。该训练目前尚未实现、登记或启动。

- [全部18折绝对／相对风险](E:/type10-7/local_artifacts/cvs_source_ellipticity_20261016/final/all_18_folds.csv)
- [源风险与幅度图](E:/type10-7/local_artifacts/cvs_source_ellipticity_20261016/final/source_ellipticity_risk_and_scale.png)
- [独立复算摘要](E:/type10-7/local_artifacts/cvs_source_ellipticity_20261016/final/analysis_summary.json)
