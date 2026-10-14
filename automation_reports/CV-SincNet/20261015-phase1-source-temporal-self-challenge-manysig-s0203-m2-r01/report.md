# 固定 SIG_TIME：时序自挑战剩余两个模型源复核

状态：PLANNED / 等待原父模型自然final50；实现、本地检查及唯一适配器P0/P1已通过，尚未创建远端输出或启动本矩阵。代码`1f5c259e9945913d7bb33fb88b1e270d861275dc`。本轮不是训练，0轮/0优化器步/0头拟合，不读取新目标准确率。

沿用首模型源PASS决定`737b65502bc5d9f204a605108b80f430bfe59541`，仅原样检查预定义seed2026092702/03。首模型不重跑；actual t2梯度定位、1/3时间与批分位、同包同零数随机对照、label_smoothing.01、900源cohort和maskseed全部保持。既有900个物理包在每模型各观测一次，新增2模型1800包×模型观测但仅900唯一物理包；与首模型合计2700包×模型观测，仍不能写成2700独立物理样本或多个seed的目标性能。

固定父来源是`20261015-phase1-receiver-dro-oldgate-cos-e50-clean-manysig-s0203-m8-r01`的SIG_TIME-s2026092702/03各自final50，实际执行`a283291279cb01b939325cf5ccba1d0a671b28c4`；这是预定义对应seed控制，不按目标成绩换父。两个父目前处于原健康50轮矩阵，必须等SOURCE_TRAINED/final50/source_frozen自然形成，发布前逐一source_proof核实契约/物理ID/scratchlineage/自身seed/33+17/11100/defaulthead。无中途CP、其他高分模型或其他祖先回退。不改变、停止或重启任何原健康训练。

只用合法L6300中同一前5个L180 cohort，每TX/RX/day2包，原data/cohort seed2026210712及随机maskseed2026307402保持；U56700TX始终隐藏且无U IQ/真值读取，V27000由sharedVALIDATED_ONCE builder返回后丢弃，无V前向/梯度。split392005、源RX1/3/4/6/8、day1/2/3和物理ID/输入策略不改，无query、星地增强或显式CFO坐标。

每模型900唯一L、3600源前向包次、5定位张量、20层记录/60参数向量；两模型合计7200源前向、10定位张量、40层/120向量；与首模型累计10800源前向、15定位张量、60层/180向量。实际对象仍是t1PW3456、t2PW6912、SIG local projection576、原默认W960。正常CE图仅求导不更新，原始网络、默认ResidualPhysicalHead和195041个部署特征参数保持，临时hook移除、.grad不累计、mask RNG独立。前置初始化synthetic检查/CP smoke单列，不充当源样本。

首源core measure_batch/summarize/qualification直接复用，未修改数值算法。仅将recount改为显式design/row上下文，防止不同seed报告错配；每row保存独立原始PT和analysis，再由第二进程重算mask、损失、向量、cohortID及固定决定，确认两模型使用相同900物理包。sourcePASS仍只说明有区别的挑战已实现，不表示正收益；首模型t1/t2梯度与原CE反向的限制保留。新两模型完整结果形成前，不设计或发布正式RSC训练、SSL扩展或目标测试。

普通N607既有CVS-RFFI、FP32、CPU2线程、CUDA hidden，不分配GPU；每GPU所有owner最多3总实验进程的现行规则和原DRO任务保持。CWD `<PROJECT>/releases/cvs_source_self_challenge_seeds_20261015_r01`；入口 `python -u -m experiments.cvs_source_self_challenge_seeds.audit`；唯一owner `codex/root/source-self-challenge-seeds-20261015`；不可复用输出 `<PROJECT>/runs/20261015-phase1-source-temporal-self-challenge-manysig-s0203-m2-r01`，日志 `<release>/dispatcher.stdout.log`；本地 `local_artifacts/source_self_challenge_seeds_delivery_r01`。

输入ready后仅一次普通预检、单归档local/remote SHA比较、远端compile、真实seed02默认CP2个synthetic IQ无query smoke，通过即诊断两个模型。仅run-owned角色/query/父来源/未定义mask/非有限数值/状态或梯度累计变动/预算或关键产物缺失/独立复算不一致/输出冲突停止，保留现场不自动重启；科学FAIL保留，不扫mask或换层/父。必须有完整分析与completion和独立recount才称本source矩阵完成。

本地13个新配置/来源资格负例及两个真实native model seed上下文检查PASS；入口、生成远端/inspect脚本compilePASS。唯一新适配器直接P0/P1 PASS，旧数学审查复用，不做重复全审查。首源资格及后续source权限不变，没有额外hash/seal/审批或数据重验。原始IQ/CP/PT/logits/masks/日志留运行根，Git仅实现、预登记和必要聚合结果。
