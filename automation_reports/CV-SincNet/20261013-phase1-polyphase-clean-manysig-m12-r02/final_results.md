# 采样结构四组：完整目标域 clean 结果

VERIFIED：12条从头30轮轨迹、360epochs、79920updates、2016000原始clean预测、216条same-row指标全部核验。每row先预测完整固定，再由独立新解释器评分。目标域84%尚未达到。

|结构|三seed准确率均值|seed间SD|Macro-F1均值|
|---|---:|---:|---:|
|ORDINARY_GLOBAL_GN|79.3752%|1.6937%|0.790090|
|ORDINARY_POINT_GN|79.8659%|0.6854%|0.795451|
|LOWPASS_POINT_GN|79.7534%|0.0015%|0.793658|
|CARRY_POINT_GN|77.7887%|1.6872%|0.772549|

|比较|三seed均值差（百分点）|正向seed|
|---|---:|---:|
|ORDINARY_POINT_GN − ORDINARY_GLOBAL_GN|+0.4907|2/3|
|LOWPASS_POINT_GN − ORDINARY_POINT_GN|-0.1125|1/3|
|CARRY_POINT_GN − ORDINARY_POINT_GN|-2.0772|0/3|
|CARRY_POINT_GN − LOWPASS_POINT_GN|-1.9647|0/3|
|CARRY_POINT_GN − ORDINARY_GLOBAL_GN|-1.5865|0/3|

CARRY在三个seed上均低于原结构以及逐点GN对照；保留采样相位的局部性质没有带来这套配方的目标识别收益。逐点GN均值较本轮原结构高0.4907个百分点，但仅2/3seed正向，尚不足84%。该结果不回流调参、重新排序或选择性重跑。原30轮矩阵及失败release全部保留。

低通三个seed接近同分的现象已核对：预测数组各自独立，seed间判决一致率约85.8%至89.6%，并非复用同一预测。所有配置、原始预测和源日志仍保留；不因这个小SD声称总体稳定泛化。

本轮原结构均值79.3752%为本轮重新从零训练结果；历史CE30均值79.4562%单列，不以旧成绩替换新row，也不声称逐权重重现。源V前向现已关闭，角色及主预算相同。

## 完整训练与计算成本

|结构|identity可训练参数|每模型训练时间均值（秒）|168000 clean推理时间均值（秒）|末轮CE均值|
|---|---:|---:|---:|---:|
|ORDINARY_GLOBAL_GN|177025|210.486|4.631|0.035725|
|ORDINARY_POINT_GN|177025|228.435|4.796|0.033206|
|LOWPASS_POINT_GN|177025|226.767|4.966|0.030414|
|CARRY_POINT_GN|177025|261.163|5.295|0.037940|

时间为共享GPU实测墙钟，非专机benchmark。carry最后t3时间位置64而其余32，参数相同仍增加时间/激活成本；同epoch不是同compute。其卷积在更细网格上执行，也改变有效感受野，因此这四组并不能完全隔离感受野与采样位置的贡献。没有把局部carry误差0解释为全网平移不变或保证未知RX性能。

完整日志逐条读取：每row6660连续步骤、30epoch，所有scalarfinite；step均值与epoch汇总一致；cosineLR .0002→.000001正确；L曝光852480、U曝光/teacherforward0，V不forward，EMA和satellite关闭。全stdout未见Traceback/CUDA/OOM技术异常。每row18个混淆/accuracy/MacroF1从原始prediction独立重算，RX/TX/day各分层混淆均求和回overall。完整记录不是tail抽样。

源模型只在Lclean拟合，固定末轮冻结，无祖先权重；当前基准已暴露，不能称新的独立盲测。仅clean地面代理，不作为LEO/星上部署或Phase2注册结果。

## 后续预算授权

用户在本聊天进一步确认：后续训练epoch数可由Agent自主选择，原则上每模型不超过50。新方案在target访问前固定轮数，并使用同预算独立对照；本轮30epochs历史记录保持。84%目标继续活动，后续候选只能由合法源机制证据和研究方案提出，不按本轮target成绩调强度或选择性重跑。
