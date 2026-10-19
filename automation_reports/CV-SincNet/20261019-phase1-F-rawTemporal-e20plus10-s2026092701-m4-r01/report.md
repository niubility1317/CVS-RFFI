# 合法源证据后的 raw 时间补充路径比较

状态：RUNNING / VERIFIED，真实父模型6NN无query smoke通过，四行PID/CWD/argv/GPU及日志增长已核实；尚未完成正式评分。

## 为什么现在提出此比较

附件七.4允许在浅中层修正仍不足时比较低容量raw IQ补充通路。固定W与联合W两轮已完成：联合W的TEMP实际训练low urban面板55.4054%，FULL66.0473%；源urban TEMP59.9583%、FULL69.6250%。不是训练期受损识别已充分学好后才出现跨RX失败。

新无参数源诊断20261019-phase1-F-prepool-source-s2026092701-r02，首次AvgPool消去半差d的固定描述子在B0及FULL的两个low draw均有正的跨半TX变化。FULL对应散布比率0.137402/0.139922，池化均值描述子为0.061571/0.062717。全部300分层独立重计，0额外NN/0拟合/0目标访问。此证据只支持局部TX相关时间变化被该池化消去，不证明它是其他分支没有的必要硬件身份信息。后续比较正是对它是否有识别用途的因果检验，不把散布换算为准确率。

## 唯一新机制与严格匹配输入控制

保留原F、所有物理分支、三路SIG_TIME、实际joint160及原6×160 cosine头。原默认W与新增特征路径同一优化器联合更新；原F其他权重冻结，无单独头阶段，无新分类器。

共享补充编码器直接读取同一个2×256输入x：Conv1d(2→8,k5,stride2,padding2,bias=False)→SiLU→Conv1d(8→12,k1,bias=False)→DWConv1d(12,k5,padding2,bias=False)→SiLU，得到12×128的h。B1(h)为72×128，加入原t1 native pw后、GN前；B2(AvgPool2(h))为96×64，加入原t2相同位置。B1/B2均为无偏置1×1卷积并置零，其他新增层正常初始化。没有额外零门控。初始函数必须精确等于原B0；原t1/t2对象与SIG_TIME hooks保留。

两个输入条件共用完全相同的架构、初始状态、参数、优化、NN和观测：RAW补充编码器读取x；PAIR仅在补充编码器输入前做AvgPool2(x)再repeat2。原F本身始终读取完整x，因此PAIR不会删掉原骨干的输入或弱化NoAs原功能。RAW与PAIR比较补充路径接触细时间差异的价值；PAIR变换不等同于已测得的非线性time_fuse内m/d分解，不能把两者混为一个数学操作。

每个补充编码器2252参数：80+96+60+864+1152；加原W960，合计3212可训练参数，部署总参数902974。补充卷积MAC/包214528，忽略激活和非参数池化/复制；既有TEMP2384参数、卷积MAC216576。RAW/PAIR参数完全一致，与TEMP相差132参数及约0.95%补充卷积MAC；完整模型耗时另实测，不用该局部MAC声称整体加速。

## 矩阵和预算

新4行RAW/PAIR×NoAs/As，seed2026092701，新增路径初始化seed20261019901，共同原B0 FINAL20；每行追加E10/2220更新，累计E30。不得继承失败TEMP、FULL或按正式成绩选择的E30状态。既有FULL/TEMP的联合W NoAs/As共4行作为同预算控制直接复用，不重训、不重测；原B0 FINAL20只作不同成本的父参照。

完全复用R02实际As bank284160、L索引、actual-first L1800及newL/V面板；无新物理模拟。保持用户确认的L-only CE0(raw)+CE0(second)+mean_all128[parentCorrect×KL(parent_raw||student_raw)]，tau1/lambda1；NoAs第二槽仍为相同raw。全模型eval但学生autograd、无U/EMA/Dropout、同reset AdamW2e-4/min1e-6/cos10/wd1e-4/globalclip5。每行568320学生NN和284160父参考NN，初始/最终源读出各50400NN。原W与补充路径联合，冻结原Sinc，无新的CENTER_UNITS叶更新。

所有新行固定FINAL30，逐行clean+六practical各168000，源矩阵决定固定且该行所有prediction完成后独立truth-last。保留所有负结果。测试仍单包，不使用clean/HQ/信道真值或额外接收样本。

## 固定源解释与放弃规则

RAW_As须同时达到：源V mid/low两draw均值相对同轮PAIR_As至少+.5pp、相对既有FULL_joint_As至少+.5pp；fullVclean相对PAIR_As和FULL_joint_As分别不低于−.2pp；RAW_As相对RAW_NoAs fullVclean不低于−.2pp。同时报告实际训练As、同L新信道和源V，不能仅凭梯度或训练loss推进。RAW−PAIR在NoAs和As的差分用于区分一般新增特征收益与As专属增量。

若不通过，关闭这一固定2252参数/k5/width12/E10 RAW/PAIR比较，不改rank/kernel/lambda/seed救回。不存在自动E50；若通过，最多再追加20轮，须先固定所有继承/优化器/采样器/计数衔接并保留同预算PAIR/FULL/NoAs。正式结果不参与上述决定或下一参数选择。

## 实现追踪

| ID | 来源 | 要求 | 目标 | 状态 | 验证 | 说明 |
|---|---|---|---|---|---|---|
| R01 | 附件七.4/源probe | 新增raw信息接入的具体依据 | 本计划/源报告 | verified | 完整源产物与独立散布重计 | 非唯一信息证明 |
| R02 | 附件七/九 | 固定低容量编码器/RAW-PAIR匹配 | layers/model | verified | 参数/MAC/实际输入 | 不扫尺寸 |
| R03 | 目标三/七 | 实际t1/t2/joint160及原默认W | model/train | verified | 零函数/特征与W更新 | 原block hooks保留 |
| R04 | 附件七.2 | 最后B零、前层正常、无双零门控 | layers/checks | verified | 初始等价/两步梯度 | 不冻结学生autograd |
| R05 | 目标三 | 合法B0来源、源角色与原始输入 | model/data | verified | strict own_model/既有bank | 无U/V拟合/target |
| R06 | 用户L-only/目标六 | 三项目标与同NN/同As/NoAs | train/config | implemented | 全量更新/计数/实际IQ | 固定总尺度 |
| R07 | 目标二/七 | 4新行E30、4既有匹配控制复用 | design/report | implemented | 原成本/增量/MAC/耗时 | 不冒称全新盲测 |
| R08 | 目标二.5 | RAW-PAIR/FULL增量及clean源门槛 | report/dispatch | implemented | source-only固定判断 | 非As-NoAs已知收益 |
| R09 | 目标七/最小流程 | 全七景/原默认头/独立truth-last | evaluate/dispatch | implemented | 逐row预测/独立重计 | 单包/不读目标选模 |
| R10 | 目标七 | error redistribution/最差RX/F1/seed | report/analysis | implemented | 全分层与配对 | model-seed n1 |
| R11 | 最小流程 | 一次review/commit/首步CP/单归档/后态 | checks/publish | verified | 实际证据 | 健康任务保护 |
| R12 | 目标八 | 所有失败保留/不重启旧root | 所有执行路径 | implemented | immutable run/output | 无自动扫参/E50 |

唯一owner codex/root/F-raw-temporal；输出PROJECT/runs/20261019-phase1-F-rawTemporal-e20plus10-s2026092701-m4-r01，每GPU所有owner最多2进程/free≥12000MB。停止规则：仅所属技术/协议/数值/来源/接口/输出/无法生成合法prediction的失败；保留健康行，不自动重试。预期逐row source/initial/final/steps、源决定、7景prediction/独立score、analysis/complete或failure。实际commit、命令、环境、路径及逐row配置在experiment.json登记。

本地完整模型检查PASS：34合成NN、7负测，0真实CP/IQ/target；原输入与RAW/PAIR补充输入、2252参数/214528MAC、零函数、真实joint160和原W同一步更新、编码器内层/B1/B2梯度、原F固定状态、深拷贝及实际checkpoint表达式/loader回放均通过。源判定0NN检查PASS。见local_checks.json。

唯一一次独立P0/P1审查PASS，15文件/2个发布内嵌脚本/4行配置及参数MAC、0NN源flow核对通过；无新增模型/CP/IQ/target/remote，见review.json。代码/config commit 906fd841ffcc0a91d2cb88112c641c42d2d2802b 已本地提交；push/OID状态 UNKNOWN_NETWORK_UNREACHABLE，见git_delivery.json。命令 CVS-RFFI/python -u experiments/cvs_raw_temporal30/runner.py，CWD=PROJECT/releases/cvs_F_rawTemporal_E30_20261019_r01，普通账号既有Torch2.1/FP32/TF32off/CPU2；首步真实B0 checkpoint6NN无query smoke通过即继续复用bank并启动四行。逐行路径、资源、停止条件和产物见experiment.json。REJECTED_EXTRA_GATE：不把暂不可达的远端Git OID追加为实验发布门槛；交付仍继续推送/独立核实。

启动证据见launch_evidence.json：单归档传输与远端compile VERIFIED；四行复用既有实际As bank，新增物理模拟0。没有据运行状态声称性能目标已完成。

连接恢复后普通客户端发布和Git push成功；当前本地HEAD与远端OID独立一致，git_delivery.json已更新VERIFIED。模型/数据/实验配置未变，没有重复启动。

源端完整回放及固定决定见source_closure.md；本分析未读取正式分数。
