# residual_fusion 两组固定权重的六环境测试

状态：PLANNED。用户要求全部六个 residual 环境的详细测试，本轮固定纯 clean 训练和 mid／low urban 拼接增强训练的两组 residual_fusion，各 4 个种子；只读取各自 E200 最终权重，不重新训练。每个环境完整覆盖同一批 168000 个物理 query，clean 加六环境共 7 个视图、8 行模型，合计 9408000 次预测。

六环境为 practical_high（郊区45°至80°）、practical_mid（郊区20°至45°）、practical_low_suburban（郊区10°至30°）、practical_high_urban（城区45°至80°）、practical_mid_urban（城区20°至45°）、practical_low_urban（城区10°至30°）。残余路由 residual／post_sync／noeq，25 MHz，equalized1／中心256点／单位RMS。低、中仰角区间本来存在重叠，按代码原定义保留。这是以地面收到的 IQ 叠加工程代理信道，不是实测在轨结果。

既有 clean capsule 的物理ID、源目标隔离及 VALIDATED_ONCE 结论复用。独立 builder 只读 clean IQ 和 opaque ID，使用固定 evaluation_seed=392005、receiver_seed=2027、namespace=phase1_full_six_fixed_20261002_v1、无类别信息的 virtual_target_session0 生成六个新观测；每个物理ID各六个观测，两组模型共享全部观测。新 received 视图由 builder 一次性核验覆盖、有限性、ID、路由和仰角记录。旧六环境包 IQ 顺序与当前包不一致，未混用。

每行核对父实验从零初始化、无 checkpoint／teacher／EMA／祖先继承、L／U／V 实际物理角色精确匹配、原始角色契约所有字段及运行时表示、E200／10000步、实际 resolved 与 payload 完整一致；严格加载 plain 或 identity 包装形式的原权重。真实 checkpoint synthetic 无 query smoke 通过后才执行该模型 query 推理。全部 8 行预测固定且通过 ID、形状、类别范围及完整性核验，独立 scorer 才连接 truth。每样本独立面对全部6类，无 query 拟合、配额、全局重排或测试反馈。

原纯 clean 来源为 20261001-phase1-cvs-residual-identity-manysig-m8-r01；增强来源为 20261002-phase1-cvs-selected-concat-manysig-m4-r01。两者固定同一结构164225参数和相同物理 L／U／V=6300／56700／27000，源 RX1、3、4、6、8及day1、2、3，U不训练。增强父实验仅 practical_mid／practical_low_urban，E80起 clean CE＋0.68 satellite CE 拼接。域骨干不启用。原纯 clean 训练没有完整显式 backend 记录，本次两组推理统一完整 FP32；差值是描述性配对比较，不单独证明增强的因果贡献。架构已依据既有公开 benchmark 指定，不宣称首次盲测或目标无接触的结构选择。

此前增强实验 high／mid／low urban 来自三种互不重叠样本子集，本轮六个视图各完整168000条、使用新的固定 realization，不能直接混合旧子集结果。结果将报告4seed均值±样本标准差、逐seed、逐RX、逐TX、每环境混淆矩阵和同seed增强减纯clean差值。总体及RX Macro-F1按6类计算；逐TX只解释准确率，单类行的6类Macro-F1不作性能结论。

每行输出独占；失败保留产物，无自动重试和性能停机。只新建授权测试，不影响其他健康任务。原8权重保留。未实测或缺少的day／资源指标记N/A。

[逐行预登记](experiment.json) · [一次只读发布前核实](evidence/preflight.json)。实际启动、预测、评分、报告完成状态将由独立产物读回补充。
