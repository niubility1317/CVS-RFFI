# SIG_TIME＋RDrop：轻度响应增强的冻结源域验证

状态：LOCAL_VERIFIED，尚未发布。当前只验证设计已固定的A_g是否保留源域默认头判别信息，不训练、不把输入干预当作性能提升。

固定原E20_RDROP seed2026092701，full L6300/U56700/V27000来源，默认头、RAW_HZ与NoExplicitCFO。使用原source_proof核对source-only scratch/final20/4440/实际配置与物理角色，严格加载后先做真实checkpoint两包无query smoke，PASS立即继续。只用完整源V27000，不读U缓存或目标IQ/truth；V不拆新角色，也不更新参数、EMA、BN或任何模型buffer。

预登记矩阵为RAW、G_E1_REFLECT、G_E20_REFLECT、G_E1_EDGE、G_E20_EDGE。候选|a|≤0.06与虚拟RX/day/eq/epoch session已在上次设计固定，edge只作边界控制。真实前向预算135002，0优化器/EMA/另训头更新，CPU2、CUDA隐藏。父模型20轮/4440更新与原实测时间另报。

科学证伪：两项reflect候选相对RAW准确率下降均不得超过1个百分点；未通过即保留负结果，不缩幅救场或根据目标成绩重跑。本规则只证明源输入安全性，不能证明跨RX泛化有效、训练增益或联合目标完成。

实际代码与配置由本次Git提交固定，远端resolved_config记录release_commit。命令、环境/CWD、输入输出、owner和逐行配置以同目录experiment.json及release内probe_registration.json为准。注册run/output root不可覆盖，唯一owner为codex/root/mixed-IQ-source-20261015。远端按普通账户发布，一个归档只比较一次SHA并编译一次。任何技术失败保留输出并退出本run；不干预健康任务。

本地已执行4项聚焦检查：源V物理角色/非法标签/外部RX拒绝、阈值边界和不读target、纠错/新增错误/固定集合、0更新预算。沿用已通过的8项算子与归一化检查；不重复测试未改行为。一次限定P0/P1审查已PASS，未发现直接P0/P1；见review_evidence.json。真实checkpoint smoke在launcher第一步执行，不建立额外许可。

预期产物：真实checkpoint smoke、resolved_config、五条件全量源决定、RX/day/TX分层、RMS/边界/session统计、源判定、实际前向/墙钟、state不变证据，以及preflight/submit/log。尚无新结果或图表。
