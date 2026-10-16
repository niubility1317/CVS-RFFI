# F＋跨 RX 同 TX：独立有界公共相位视图 E20 源配对

状态：PLANNED；尚未启动或获得训练识别结果。逐行矩阵、实际路径与 seed 的唯一登记见 [experiment.json](experiment.json)。执行代码由实际 Git 提交及 release_commit.txt/resolved_config 标识。

## 假设与最小矩阵

上一零更新源检验 `20261016-diagnostic-F-crossRX-commonphase-sourceL-s2026092701-m1-r01` 已完整核对：当前 F＋CROSS_RX 的训练 fit L5040 原始准确率99.841270%，固定+0.37rad为75.337302%，−0.37rad为63.015873%，两个方向在4/4源RX下降超过1pp。该结果只证实脆弱性，不能证明相位增强有效。此前 CNN 相位训练失败与旧E200敏感性结果保留，不作为当前F训练收益。

唯一新假设：在不改变身份来源的前提下，用独立轻度公共相位视图减少接收观测相位依赖，同时保护 raw 监督，可提高留出 RX clean 默认头识别。

| row | raw RDrop 对 | 共有旧普通增强 RDrop 对 | 第三 RDrop 对 |
|---|---:|---:|---|
| B00-RX1-s2026092701 | 50% | 25% | 25% raw |
| B10-RX1-s2026092701 | 50% | 25% | 25% phase-only(raw) |

每物理 L 包六次前向，每对为平均LS0.01 CE＋alpha1对称RDrop KL；总监督权重1。第三视图仅将原始IQ乘以常数单位复数，每包/step独立uniform[−0.37,+0.37]rad，私有CPU generator seed20261016201+step，不推进原生训练随机流；同对复用一次生成IQ。全部128个合法phase L包参与监督，无confidence/TX/RX过滤。无新增噪声、多径、PA、CFO、移位、量化、剪裁或RMS归一化，不叠A_s，也不声称数字旋转等于更换真实接收机。

F三路SIG_TIME、统一完整Sinc、NoExplicitCFO及必要逐包补偿、真实joint160、默认cosine头正常联合训练保持。CROSS_RX只取共有旧普通增强对的time_down、t2.pool；同TX异物理RX为正，同TX同RX排除分母，其余TX为负。温度0.2、权重0.1×clip((epoch−5)/10,0,1)不变，不随CE份额缩小，不作用raw/新phase/U/teacher/eval。新机制没有部署参数、分支、辅助分类头或特征一致性损失。

共有旧普通增强沿用已核实的实际调用（含PA-normal/CFO，DAC0、class-signature关闭），不能称B00完全无增强或把旧PA当作已验证的身份保持增强。已有普通增强审计见 `analysis/cvs_joint_ordinary_IQ_audit_20261016.md`；同对IQ不重抽。新的phase直接作用raw，不与普通AWGN/multipath叠加。

## 数据、初始化、预算和合法读出

固定原6TX、day1/2/3、split392005。训练RX3/4/6/8，L5040/U45360；U TX始终隐藏，原13L＋7U/oldgate/ownEMA/原weak-strong路径不变，无新增U视图或教师前向。已见源V21600仅健康；held RX1原L1260仅final20读出。该契约不是完整L6300或L5580/Ldev720。精确物理ID来自已有合法carrier与固定RX1 fold，复用一次验证结论，无新数据重验链。

两行各从零E20、4440实际优化更新；同seed初始化规则和原生sampler/dropout/ordinary流，仅candidate/output路径不同。无父权重、E50/E200、旧F-only或看过held包的checkpoint进入训练；EMA来自自身scratch student。dispatcher真实checkpoint smoke仅加载自有匹配四RX F+CROSS_RX B00/final20，严格载入后只跑2个合成包，不初始化新训练，PASS自动继续。来源初始化/完成/冻结和精确角色已在原run核实，本次保持。各worker另有自身scratch checkpoint smoke，默认头随骨干更新，head-fit=0。

每行final20冻结后立即自动读出held L1260的clean＋六practical，共8820决策；不等待另一行。两行相同命名物理信道，full residual/post_sync/noeq、接收质量与补偿依赖保持；不改测试SNR/信道/receivedIQ，不删困难包。该读出是源留出诊断，0target，无目标真值评分。所有源视图完整后独立重算CM/F1、修复/新增错误、TX/day及suburban绝对准确率和相对自身clean差距。正常最终全源/多seed/正式target四因子尚未开展。

固定科学条件：clean≥+1pp、clean macro-F1≥0pp、seenV≥−0.5pp；两行均完整有限4440更新、3409920 L前向、3330非零SupCon调用、最终LCE≤0.5、0As训练/0目标接触；B10完整568320 phase观测。任一不满足则关闭固定公共相位训练配方，不调幅度、改seed/RX或选择性重跑。practical结果全部报告，但不作为本次Ag候选晋级阈值。PASS只支持单独登记独立源确认，不能证明跨目标RX Ag收益或联合目标。

## 成本、执行和停止规则

总新增40模型轮/8880更新，无续训或继承父训练。每行L NN包前向3409920（原单对RDrop1136640，新增2273280，两行匹配）；U学生、ownEMA、V、smoke和held8820按实际分别记。B10相位IQ568320，B00为0；A_s训练均0，held-source practical各7560，不计作训练。上一冻结源诊断15122前向/0更新单列；本地合成验证累计4更新/6160网络包前向（含两次检查断言失败），0真实IQ，不计入E20真实更新。

owner `codex/root/F-CROSS_RX-commonphase-E20-20261016`。普通N607账号，CPU dispatcher两线程；共享容量锁，所有owner合计每GPU最多3实验进程、启动时至少12000MiB空闲。实际GPU由launch记录，不干预健康任务。CWD `PROJECT/releases/cvs_F_crossRX_commonphase_sourceRX1_e20_20261016_r01`，命令 `CVS-RFFI/bin/python -u experiments/cvs_commonphase20/runner.py`；唯一不可覆盖输出root见experiment.json。复用不可变backend代码commit `0ae066adb10a001e834c8c679c5b7b81c20b3ac8`，不复用训练状态。

一次N607只读preflight已PASS。发布一次归档SHA比对、一次远端编译；smoke为launcher第一步。启动后独立核实PID/CWD/argv/GPU/log增长。只在角色/初始化/输出/预算/nonfinite/不可合法预测等预登记系统技术失败时停止新pending，保留所有产物，健康worker继续，无自动重试；低性能不停止。观察超时先只读核实原run，不重复提交。

保存实际args/config/模块/来源、完整20epoch/4440step/unitstep、全部phase角度/功率误差、物理L/U训练序列、各view默认头预测、成本、独立分析和图。checkpoint/IQ/物理ID/逐包预测/完整日志私有；代码、配置、登记、汇总报告入Git。训练完成或源V健康不代替完整实验结果。

本地11个负例及真实F合成检查PASS：同scratch初始化、六前向/同IQ对、私有phase RNG、默认头联合更新、导出160维一致；新B00的loss和head梯度与旧匹配B00完全相同。两次检查脚本问题分别为浮点精确比较和原生未启用诊断NaN比较，已修检查，训练实现未改。一次独立限定P0/P1审查PASS，无发现。实际发布/训练/读出状态待独立证据追加。
