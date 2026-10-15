# SIG_TIME＋RDrop：轻度响应增强的冻结源域验证

状态：ANALYZED / 真实源验证与独立全量决定核对 VERIFIED。当前只验证设计已固定的A_g是否保留源域默认头判别信息，不训练、不把输入干预当作性能提升。

固定原E20_RDROP seed2026092701，full L6300/U56700/V27000来源，默认头、RAW_HZ与NoExplicitCFO。使用原source_proof核对source-only scratch/final20/4440/实际配置与物理角色，严格加载后先做真实checkpoint两包无query smoke，PASS立即继续。只用完整源V27000，不读U缓存或目标IQ/truth；V不拆新角色，也不更新参数、EMA、BN或任何模型buffer。

预登记矩阵为RAW、G_E1_REFLECT、G_E20_REFLECT、G_E1_EDGE、G_E20_EDGE。候选|a|≤0.06与虚拟RX/day/eq/epoch session已在上次设计固定，edge只作边界控制。真实前向预算135002，0优化器/EMA/另训头更新，CPU2、CUDA隐藏。父模型20轮/4440更新与原实测时间另报。

科学证伪：两项reflect候选相对RAW准确率下降均不得超过1个百分点；未通过即保留负结果，不缩幅救场或根据目标成绩重跑。本规则只证明源输入安全性，不能证明跨RX泛化有效、训练增益或联合目标完成。

实际代码与配置由本次Git提交固定，远端resolved_config记录release_commit。命令、环境/CWD、输入输出、owner和逐行配置以同目录experiment.json及release内probe_registration.json为准。注册run/output root不可覆盖，唯一owner为codex/root/mixed-IQ-source-20261015。远端按普通账户发布，一个归档只比较一次SHA并编译一次。任何技术失败保留输出并退出本run；不干预健康任务。

本地已执行4项聚焦检查：源V物理角色/非法标签/外部RX拒绝、阈值边界和不读target、纠错/新增错误/固定集合、0更新预算。沿用已通过的8项算子与归一化检查；不重复测试未改行为。一次限定P0/P1审查已PASS，未发现直接P0/P1；见review_evidence.json。真实checkpoint smoke在launcher第一步执行，不建立额外许可。

预期产物：真实checkpoint smoke、resolved_config、五条件全量源决定、RX/day/TX分层、RMS/边界/session统计、源判定、实际前向/墙钟、state不变证据，以及preflight/submit/log。尚无新结果或图表。


## 完整源结果

|条件|正确/27000|准确率 %|相对RAW pp|修复|新增错误|默认头间隔/30|
|---|---:|---:|---:|---:|---:|---:|
|RAW|26425/27000|97.870370|+0.000000|0|0|0.216999769|
|G_E1_REFLECT|26409/27000|97.811111|-0.059259|25|41|0.215260550|
|G_E20_REFLECT|26392/27000|97.748148|-0.122222|36|69|0.212779462|
|G_E1_EDGE|26411/27000|97.818519|-0.051852|26|40|0.215271831|
|G_E20_EDGE|26391/27000|97.744444|-0.125926|35|69|0.212773472|

两个reflect候选下降0.059259/0.122222个百分点，均通过预定1个百分点安全规则。它们并未提高冻结模型准确率：epoch1修复25却新增41个错误，epoch20修复36却新增69个错误，间隔也略降。只能据此允许进入训练检验，不能宣称A_g已经增强跨RX身份提取。

RX3是原默认头较弱的源RX，RAW5093/5400，两个reflect为5080/5400和5072/5400，下降0.240741/0.388889个百分点；全量RX/day/TX分层保留，不只报告有利RX。虚拟session共有30组（两个epoch×5RX×3day，eq1），同RX/day/eq/epoch不随TX或batch变化。V的RX全部属于原训练源RX，所以该验证不能证明未见RX泛化。

RMS相对功率误差最大4.768372e-7；恢复RMS后仍有真实波形变化，reflect MSE为0.000196742/0.000301632。边界控制在撤销各自RMS标量后，有效区[1,255)最大差4.768372e-7，未触及参考窗的人工边界；全网仍可能受端点及RMS标量影响。edge较reflect正确数仅+2/−1，不据此改选边界或重跑。

实际135002个默认头前向（含真实CP smoke2）、74.915770秒、0优化器/EMA/头拟合更新，参数和buffer逐项相等，Torch随机状态不变，无U缓存或目标访问。父E20训练成本为20轮/4440更新、923.560061秒，诊断不增加训练预算，也不与未来scratchE20当作权重继承。

独立本地完整解析135000条源决定，逐条件27000个相同物理ID、混淆矩阵、修复/新增及分类间隔均与远端完整结果一致。该核对不新增IQ/CP读取、预测或训练。两张源分层/响应图均真实生成并已逐图检查：

- [默认头与RX分层](E:/type10-7/local_artifacts/cvs_mixedIQ_source_20261015/final/figures/source_default_head.png)
- [实际响应与功率保持](E:/type10-7/local_artifacts/cvs_mixedIQ_source_20261015/final/figures/source_response_and_power.png)
- [全部源统计](E:/type10-7/local_artifacts/cvs_mixedIQ_source_20261015/final/results.json)
- [完整决定核对](E:/type10-7/local_artifacts/cvs_mixedIQ_source_20261015/final/independent_scalar_audit.json)

实际代码提交为7025e13daf9f743438ba2ca7e9871c37201a9fcf，单归档比较与远端编译VERIFIED/PASS。曾独立确认运行中的PID/CWD/argv，末次进程自然结束、完整产物与日志增长466字节均核实；没有重启源诊断或干预其他进程。

首次本地取回把Windows Path字符串用于Linux路径，误报未完成；修正为POSIX后直接复用原完成产物。canonical实验元数据的描述性远端output路径同步纠正，原immutable release与source_registration保留历史内容，实际Linux运行路径从始至终正确。该问题不涉及IQ、模型或成绩变化。

下一项是假设本身的训练检验：原SIG_TIME+RDrop参照及B00/B10/B01/B11、每模型scratch总E20、原默认头/旧U规则、逐行自动clean+六practical预测与独立truth-last。尚未启动新混合训练，没有clean或urban正式增益及跨seed复现证据。源安全PASS不替代联合目标。
