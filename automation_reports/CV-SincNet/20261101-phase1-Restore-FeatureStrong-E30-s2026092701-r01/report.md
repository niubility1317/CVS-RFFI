# 连续源特征回归权重1：从零单阶段30轮

- run_id：`20261101-phase1-Restore-FeatureStrong-E30-s2026092701-r01`
- group_id：`restoration-feature-strong-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

与固定FeatureCond仅连续latent回归权重.1→1，结构/单位功率/输出F指导/源顺序/优化器/30-7410匹配。源路径诊断提供动机，原Main七景冻结后独立验收。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 单因素与源端依据

与固定FeatureCond/c8659b15…相比，只把预测连续latent回归权重由.1提高到1：IQ NMSE+.05STFT+.1(输出F0 cos+源varMSE)+1.0(预测latent cos+源varMSE)。原全局参数clip1、输出F指导、共享encoder/四FiLM/单位功率输出、4670178参数、源顺序、初始化、优化器和预算全部保持。选择权重1用于检验与归一化IQ目标同量级的连续特征约束，不扫描权重或按Main分数选择。

原源端冻结路径诊断1b8b3625…使用Ls128/Us384固定512包×20draw，低档城区Fclean一致率正常88.72%、关闭条件约29%、预测latent86.56%、sourceclean特征参照97.49%。条件确已使用，参照提示预测特征仍可能限制该模型；它额外知道clean、处于不同输入分布，不能当作可学习收益或理论界。raw平均功率约75.83，故不直接移除既有投影。该机制依据只来自训练源角色，不消费Main错误做优化。

新G完全从零单阶段30轮/7410；原R07 Ls6300/Us56700，20draw后复用前10份，modelseed2026092701/dataorder2026110102/B256tail24/247steps/AdamW.001-warm247-cos2e-5/WD1e-4保持。原F0 clean63000特征缓存和f57源尺度共用于输出与latent指导，在线1890000行F0特征，不增加教师行。无旧G/coarse/DURA/神经IQ教师祖先，连续latent不是六类分类头，不读TX/logits/CE/类原型；原F0/defaulthead冻结。这是F-guided方法。teacher只loss，部署只当前IQ；V/OE/novel/目标不训练、早停或选模，OE仅冻结诊断。

固定final30后自动原Main cross_rx126000及cross_both42000各七景，所有42单元预测固定后独立CPU truth-last/原共享锁。原raw/DURA同IQ-ID预测精确复用，新G全量推理。逐景至少140345/168000正确=83.538690%，距raw clean86.538690%≤3pp且保持clean；源指标不取消测试，默认DURA不变。旧FeatureCond仅冻结对照，无权重继承。

3项针对性检查及一次P0/P1通过，真实Ls2 smoke/E0严格重载为launcher第一步。静态推理结构完全相同，复用260546560实数Conv/Linear/attention MAC（含QKV/out/QK/AV）、0FFT/CG；排除F0、norm、activation、softmax、FiLM逐点、插值、投影和内存，不是实测时延。N607原CVS-RFFI/Torch2.1+cu121/FP32-TF32off，GPU4/6 UUID准入free>=12GB、原compute<=2，只新增1条本任务NN/单卡总<=3，其他owner不改。固定所有30恢复文件仅技术用途，不据测试或源质量选checkpoint。
