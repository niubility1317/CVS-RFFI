# 固定 SIG_TIME 的 E50 urban 增强首轮

状态：RUNNING，已独立核实启动与日志增长。run_id：`20261015-phase1-sigtime-urban-ssl-e50-manysig-s2026092701-m4-r01`。

本轮按用户“拉长到 E50”执行，从头训练四行、一个模型 seed 2026092701，不继承 E30 权重，不自动扩种子。代码与配置以实际发布 Git commit 标识；dispatcher 从 release_commit.txt 记录该值。此前 E30 的全部权重、结果和负结果保留。

| 行 | 与同预算基线的差别 |
|---|---|
| B0 | SIG_TIME＋旧门控＋cos，clean 与 low_suburban/mid_urban/low_urban 普通拼接 CE |
| C3 | 分别归一化的 1:1 配对 CE＋有界状态课程＋单侧默认头间隔保持 |
| SSL_CONF | C3＋同一样本 clean EMA 置信度 ≥0.95 的 urban 软标签监督，系数 0.16 |
| SSL_WEAK | SSL_CONF 另要求同一样本独立弱视图预测一致且置信度 ≥0.95 |

全部固定 50 轮＝33 轮有标签阶段＋17 轮旧门控半监督阶段，222 步/轮，共 11100 更新/模型。每行父训练成本 0，增量与累计均为 E50；总计 44400 更新。默认头、内部融合、参考响应、SIG_TIME 结构和原联合训练方式不变。原旧门控只继续控制原半监督损失；新增 urban SSL 不使用学生 urban 置信度，也不再叠加 batch-neighbor 拒绝。V 仅在新增 SSL 首个半监督步进行一次冻结的自有 EMA 健康校验：precision≥0.95、coverage≥0.10；未通过时记录负结果并跳过额外 SSL，继续原训练。所有合法有标签 satellite 仍参与 CE。

新增监督资格依据上一轮独立源 V 校准：置信度 0.95 的 precision 为 99.82% 至 99.87%，coverage 为 90.97% 至 91.95%。不根据目标测试调整阈值。主候选预先固定 SSL_CONF。科学探索目标：low_urban≥60%；相对 B0，urban 均值 +3 个百分点、low_urban +5 个百分点、clean 下降≤0.5 个百分点。目标不是结果保证，也不触发自动扩种子、重跑或延长预算。

源 L/U/V 为 6300/56700/27000，原物理角色与 split 392005 不变，U TX 标签隐藏。25 MHz、IQ256、中心裁剪，原录音归一化后进入信道；实际 practical 模块为 experiments/adv3b02_xuc/code。residual/post_sync、EQ 关闭（G=1）、reference_snr_db=30、质量适配/相位跟踪/IQ 补偿开启，reflect 边界，无额外输出 RMS。训练三个场景不变，几何、传播、噪声、质量和补偿残差保留依赖；session 不由 TX 定义。正式测试复用原 VALIDATED_ONCE received IQ，绝不重生成或调弱。

首步执行一次真实、已核实来源的冻结 E50 源模型 no-query smoke，该权重仅供 smoke。已有同一参考模型和未改物理引擎的源诊断直接复用，记录旧路径与原成本，不计为新诊断或训练初始化。

环境：普通 N607 账户，Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_sigtime_urban_ssl_e50_20261015_r01`。远端命令：`python -u -m experiments.cvs_urban_sigtime50.dispatch`。本地发布：`E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_urban_sigtime50.publish --output E:/type10-7/local_artifacts/cvs_urban_sigtime_e50_20261015/delivery_r01`。唯一 owner 为 `codex/root/urban-sigtime-e50-20261015`。

GPU 由实时容量选择，每 GPU 含其他 owner 最多 3 个实验进程，至少 12000 MB 空闲。唯一输出根 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-sigtime-urban-ssl-e50-manysig-s2026092701-m4-r01`。发布只比较一次归档传输 SHA，进行一次远端编译和预检。启动后独立核实 PID/CWD/argv/GPU/日志增长。协议、数据角色、预算、非有限数、输出冲突或无法运行的技术故障停止未启动行，已启动健康行继续；不按性能停止，不自动重试，不修改其他任务。

每行冻结 final50 后立即自动预测 clean＋六种 practical；该行七视图 prediction 全部固定后由独立 exec scorer 连接 truth，再生成同 row 报告。所有测试分数不回流参数、候选、选模或其他健康训练。四行全部完成后汇总逐行七场景、RX/day/TX、Macro-F1、源状态/质量/门控/间隔、覆盖、50 轮曲线、更新数、前向量和墙钟。预期路径见 experiment.json；原始 checkpoint、预测、日志不入 Git。

验证：E50 10 项检查 PASS（含错误预算/头/测试 SNR 拒绝、教师冻结与 U 标签拒绝、真实结构参数逐项相同、默认头/骨干联合梯度、发布入口）；旧 E30 9 项兼容检查 PASS。独立 P0/P1 检查已完成；发布器导入和远端 E50 路径的 P1 已修复并定点核实，未发现剩余 P0/P1。来源为 reviewer `/root/urban_p0p1_review`；不增加重复审查。

本轮只有一个模型 seed，不能回答 E50 收益是否跨 seed 重现。无实际结果前不声称达到 60% 或性能改善。后续状态和结果在本报告续写。

发布 commit：`7a366c506a736c34e5bba335821f4867f2d3706b`，已 push 并核对远端 OID。归档传输、远端编译、普通账户预检和 no-query smoke 均通过。四行实际 resolved 为 scratch E50/33+17/11100 更新，parent 与 teacher checkpoint 均为空。B0/C3/SSL_CONF/SSL_WEAK 分别运行在 GPU 3/4/5/6；PID 为 2826320, 2826397, 2826552, 2826633。独立证据：`E:/type10-7/local_artifacts/cvs_urban_sigtime_e50_20261015/monitor/latest.json`。本次仅确认运行，尚无 E50 目标成绩。

逐行正式结果已返回：[B0-s2026092701](rows/B0-s2026092701.md)。该行完成不代表整个 E50 矩阵完成；其他行继续原计划。

逐行正式结果已返回：[C3-s2026092701](rows/C3-s2026092701.md)。该行完成不代表整个 E50 矩阵完成；其他行继续原计划。

逐行正式结果已返回：[SSL_CONF-s2026092701](rows/SSL_CONF-s2026092701.md)。该行完成不代表整个 E50 矩阵完成；其他行继续原计划。

逐行正式结果已返回：[SSL_WEAK-s2026092701](rows/SSL_WEAK-s2026092701.md)。该行完成不代表整个 E50 矩阵完成；其他行继续原计划。


## E50 完整结果与负结果（2026-10-15）

状态：ANALYZED。四行均完成source-only final50冻结、七视图全部预测、独立exec truth-last评分；另完成267120次源分析前向/0更新，10分析段均通过模型状态不变检查。44400真实更新、每行50轮完整step日志11100条全部分析，未发现非有限跳步或技术错误。
固定正式IQ每视图168000条，共4704000条预测、504条分层指标记录；每行全部七视图prediction固定后才评分。仅一个model seed2026092701。

| 方法 | urban(%) | low urban(%) | mid urban(%) | clean(%) | suburban均值(%) | urban Macro-F1(%) | 最差urban RX(%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 65.0927 | 53.9696 | 64.2851 | 81.2929 | 78.0657 | 64.7060 | 59.1569 |
| C3 | 65.4464 | 54.9351 | 64.8137 | 80.4161 | 77.4732 | 65.1370 | 59.6972 |
| SSL_CONF | 65.5373 | 55.0988 | 64.9167 | 80.1304 | 77.4923 | 65.1920 | 59.7194 |
| SSL_WEAK | 65.5758 | 55.1839 | 64.9500 | 80.1268 | 77.4933 | 65.2128 | 59.7528 |

预登记主候选SSL_CONF的low为55.0988%，相对B0增加1.1292pp，urban增加0.4446pp，clean降低1.1625pp；未达low60%、urban+3pp/low+5pp及clean下降≤0.5pp。SSL_WEAK对B0的low增加1.2143pp、urban增加0.4831pp、clean降低1.1661pp，也不满足目标。
C3对B0：low+0.9655pp、urban+0.3538pp、clean−0.8768pp；SSL_CONF对C3只增加low0.1637pp/urban0.0909pp，同时clean−0.2857pp。弱视图资格比clean资格额外增加low0.0851pp，不能据此宣布跨seed最优。E50尚无跨seed复现证据；预算结束不等于收敛，保留全部负结果。

源机制：固定原SIG_TIME参考模型clean正确718/720集合，low保持率B0 63.2312%、C3 65.3203%、SSL_CONF65.5989%、SSL_WEAK65.4596%。训练后源low的BLOCKED从B0 41.0072%升至C3 44.1247%（+3.1175pp），degraded从63.0137%至66.4384%（+3.4247pp），unlocked从35.9375%至38.7500%（+2.8125pp）；LOS保持97.4747%。这些是曾训练的合法源L面板，不替代目标成绩或未见域证明。
同一诊断命名流中，B0/C3去噪的固定状态效应仍为+10.28/+10.42pp；重算状态总效应+9.86/+11.39pp。去多径固定状态+2.08/+3.61pp，去carrier仅+0.56/+0.42pp，去注入RX IQ约0；因此能观察到受遮挡/失锁样本部分分类错误减少，不能宣称噪声、多径或Doppler已被特异解决。更换为edge边界导致B0/C3 −9.31/−10.83pp，表明短IQ边界仍敏感；正式测试始终沿原reflect未修改。实际波形功率代理对low准确率影响约±0.14pp，但状态切换属于引擎诊断，不能因精度差小就称没有切换。新增噪声SNR不是总SNR。
总效应移除损伤后重算接收机状态；固定效应复用原状态和补偿参数，均保留各自结果，不混为单因果解释。完整实测源分层和控制结果见local_artifacts/cvs_urban_sigtime_e50_20261015/source_mechanism_results.json及source_mechanism_aggregate.json。旧机制分支节点是原投影/贡献诊断接口，不冒充新附件要求的实际T/F/P/R/S训练接口。

| 方法 | 父/增量/累计epoch | 更新数 | 主前向样本量 | 训练墙钟(s) |
|---|---|---:|---:|---:|
| B0 | 0/50/50 | 11100 | 4769400 | 4452.785 |
| C3 | 0/50/50 | 11100 | 4769400 | 6457.786 |
| SSL_CONF | 0/50/50 | 11100 | 5733300 | 9553.112 |
| SSL_WEAK | 0/50/50 | 11100 | 6697200 | 9492.168 |

主前向含L clean/satellite和原U教师/强视图，新SSL额外urban/weak均计入；另列每组源面板144000、SSL首步源教师健康检查27256/54256、原生V健康前向、smoke、每组1176000正式预测，原生V总前向细账仍待单独核验，不把主前向误称全部计算。四组后分析267120前向、0更新、602.628秒；其中720为固定父模型clean集合建立，模型未用于续训。并发资源负载不同，墙钟不作孤立方法速度结论。

完整50轮曲线/技术审计：local_artifacts/cvs_urban_sigtime_e50_20261015/final/training_audit.json。图已实际生成并目检；原先source_controlled_attribution.png是启动前父模型图，训练后应使用以下新图：

![训练曲线](E:/type10-7/local_artifacts/cvs_urban_sigtime_e50_20261015/final/figures/training_curves.png)

![训练后源状态与质量](E:/type10-7/local_artifacts/cvs_urban_sigtime_e50_20261015/final/figures/trained_state_quality_strata.png)

![训练后固定与总效应](E:/type10-7/local_artifacts/cvs_urban_sigtime_e50_20261015/final/figures/trained_controlled_attribution.png)

正式训练代码commit仍为7a366c506a736c34e5bba335821f4867f2d3706b；本次独立源后分析release为cvs_sigtime_urban_ssl_e50_source_20261015_r01，commit89bf9577d43c0fa3763ddb2f2c67a96d27d7b89e。正常退出与完成产物独立回读VERIFIED，不覆盖或重跑旧训练/预测。后续八组位置小试预登记由用户附件驱动，保持源选择隔离；旧目标分数不用于其系数、课程或分支排序。
