# 采样结构四组独立消融：30 轮三主种子 clean

源诊断 r03 的 18 个单元已完整完成，只有采样结构在三主种子上均满足原固定适用性条件；这不代表目标识别性能已经提高。本矩阵对原池化、逐点归一化、低通降采样和两相位 carry 分别从头训练 30 轮，普通 cos160 与合法弱标注数据角色保留。唯一主方案为 CARRY_POINT_GN，84% 判定只来自其完整三 seed clean 均值；所有对照也完整报告。

|结构|identity.t2.pool|identity.t3|t3 时间长度|
|---|---|---|---:|
|ORDINARY_GLOBAL_GN|原 MaxPool|原全时间 GN|32|
|ORDINARY_POINT_GN|MaxPool|逐点通道 GN|32|
|LOWPASS_POINT_GN|[.25,.5,.25] 低通后 stride2|逐点通道 GN|32|
|CARRY_POINT_GN|保存偶/奇采样位置 S2|共享 DW/PW＋carry＋逐点通道 GN|64|

所有结构输出 joint160 / cos6，身份 trainable177025；保留两相位并不保证前序整网完全平移不变。通过与普通逐点 GN 和低通比较，区分归一化、滤波与保留采样位置的贡献。carry 的 t3 激活和部分计算增加，不能以参数相同声称计算相同。

固定三 model seeds2026092701/02/03，现有同物理角色 source L6300/U56700/V27000。身份训练只 Lclean，U 真值隐藏且本矩阵不打开 U，V 不前向；老师、EMA、伪标签和卫星训练分支实际关闭。每模型6660 updates、852480 L exposure，cosine LR .0002→.000001、AdamW WD.0001、clip10、label_smoothing0；最终30轮冻结，无源或目标性能选模。训练不加载任何 checkpoint。发布首步只用已核实同契约CE30真模型做无 query smoke，不继承其权重。

每 row 自动训练→冻结/来源检查→168000 clean predictions→独立新进程 truth-last→18 same-row metrics；12rows总2016000 predictions。逐RX/day/TX、混淆、Macro-F1、三seed均值/SD和五个固定配对差完整报告。来源capsule及rolephysicalIDs沿用既有验证句柄；目标IQ/真值不进入训练或选择。

精确配置/seed/命令/环境/CWD/input/output/log/GPU/owner和stoprule见 experiment.json。一个run一个root owner；GPU计入其他owner最多3进程、free>=12000MB。仅预登记系统故障处理，无低分停止或选择性重跑。已暴露测试基准，不宣称新的独立盲测，不作卫星部署性能声明。

本地四结构实际native路径的shape/cos160/CE可达梯度/全部optimizer参数/strictstate roundtrip，以及sourceU配置负例、训练不读checkpoint、prediction不全时truth保持关闭、完整12rows汇总及主方案goal不可被对照高分替代检查均PASS。一次独立候选P0/P1审查待完成；真实checkpoint首步和实际训练/测试仍待发布。

一次独立 P0/P1 候选代码审查 PASS：实际 12 个配置与登记一致，原结构保留、三变体只改两个目标层，纯 CE／fixed30／逐 row 独立 truth-last／全部 owner GPU 容量闭环无阻断问题。

## r01 实际发布失败

FAILED / VERIFIED：commit a486d1a15 的归档校验与remotecompile通过，preflight导入缺少cvs_repeated_rx_margin，未创建run/submit，0训练、0checkpoint/query访问。原release保留。替代 20261013-phase1-polyphase-clean-manysig-m12-r02 只补发布依赖，预算矩阵不变。
