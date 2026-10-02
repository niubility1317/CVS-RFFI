# SupportMetric完整query入口验证

2026-10-02，VERIFIED_LOCAL_FROZEN_IMPLEMENTATION_READY_NOT_LAUNCHED。当前方法公式与完整矩阵在实际support及query成绩读取之前固定；本次只完成执行与独立评分入口，没有根据成绩调参。

root在激活的ssr-gpu环境串行验证147个distinct用例：producer34、scorer70、control43。初次scorer66通过/4失败的根因是测试AST不能解析geometry kernel常量，以及独立scorer混淆core的stage与archive的state；源码分别严格匹配，物理身份没有放宽，修复后完整70通过。control初次42通过/1失败，真实publisher超时分支没有保存原始stdout/stderr；补齐字节文件与路径元数据后只重测受影响用例，1通过。没有重复未变更的42项检查。所有失败与通过的raw输出按字节gzip保留在automation_reports/CV-SincNet/20261002-phase2-d92-support-metric-joint-repeat-m2-r01/evidence/query_source_validation_20261002。

独立审查发现REMOTE缺getpass导入会在Popen后无法写launch.json，owner已在REMOTE自身导入并在Popen前核普通用户；两个fresh namespace离线回归通过。独立审查文档标记NO_UNRESOLVED_P0_P1；root同步了登记中probe→benchmark引用及实际raw feature cache producer文字。review、test和runtime状态各自记录，不以源码审查替代测试。

真producer合成档案验证actual B→C继承、new0对象复用、每row一次basis构造、准备/训练/预测四scope的实际SUM/MAX费用，以及五个固定流A/R0_B/R0_C/B/C。独立scorer不导入拟合模块；全部4/2400预测、来源、basis及数值档案闭合后独立重读，最后才连接truth。全部注册类别逐样本竞争，不读取query角色、真实类数量、配额或全局排序。

使用Git承载工作树的57文件精确source bundle隔离导入成功，data_access/native_encoder_loaded均false；部分旧公共依赖只位于承载工作树，实际发布源以该树为准，新SupportMetric源码与spec均WS/WT逐字节镜像。此验证没有加载实际数据或模型，没有SSH、拟合或评分。Torch2.1/NumPy2 ABI warning原样保留；合成A推断采用受支持list路径，测试结果不证明其他tensor.numpy边界。

当前run 20261002-phase2-d92-support-metric-joint-repeat-m2-r01 READY、actual_runtime_commit null、publication NOT_LAUNCHED、results null。发布必须从本次push并独立OID核实后的HEAD，unique r01、root sole owner、CPU2/BLAS2。全4/2400终态后才原immutable release独立truth-last scorer；完整报告必须old6/K1/5/10/20×新增0/2/5/10/20，同物理A/B/C、新旧准确率、gain/drop/absGap/H及实测成本/N/A。实现或数学推导不构成性能提升或星载省算力证据。
