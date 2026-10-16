# F＋跨 RX：单侧边缘可见性源身份检验

状态：ANALYZED／VERIFIED，固定源身份保护条件FAIL，关闭本算子。当前只检验一个有界IQ候选的源fit身份保持，不训练模型、不拟合额外头、不访问target。唯一逐row矩阵和完整模型/数据配置见[experiment.json](experiment.json)。

候选及原始文献、负面证据与适用限制见[研究说明](../../../analysis/cvs_edge_visibility_source_20261017.md)。本次采用一个端部16点的实数窗：外8点归零、内8点半余弦上升到1，右端镜像。没有中段遮挡、时移、重采样、额外噪声/传播、PA/量化/剪裁或全局功率恢复；归零不是幅值阈值剪裁。窗口不读TX/RX/day或置信度，参考[80:160]及支持外输入精确不变，不要求所有内部特征相同。它是部分观测候选，不声称替换RX硬件或本来就标签保持。

原基线F＋跨RX同TX、三SIG_TIME、统一Sinc、joint160、NoExplicitCFO/defaultcos30保持。两份既有own scratchE20 final20 B00只作冻结源诊断，原训练完整来源、final-only/4440更新、实际L5040/U45360/V21600与RX1held角色核对后，在各自原不可变后端独立CPU进程加载。两个模型原训练都遵守隐藏U/EMA规则；本次无U前向不表示父训练未用U。未来增强训练仍从零E20，不把诊断CP当初始化。

统计和默认头前向只用fitL5040，RX3/4/6/8、day1/2/3、6TX、72格各70包。既有载体反序列化L6300、未用源V和历史特征；RX1原L1260先排除，U仅角色ID、无IQ或TX标签。全部包参与，不删低质量或难包。不是完整L6300或L_train5580/L_dev720训练契约，不触发新数据验证。

每模型raw/left/right各5040，预计30240源NN＋一次匹配真实CP的2包合成no-query smoke=30242NN，20160边缘视图生成。0新epoch/更新/教师/头拟合。两个参照历史成本单独40 modelE/8880更新/9373356NN，不重复训练。

原条件在源IQ访问前固定：两模型、每个端部均要求accuracy与macro-F1相对raw≥−.5pp，每RX/TX/day accuracy≥−1pp，raw正确集合保留率≥.99；相对IQ RMS变化median≥.01、max≤.35，reference/支持外精确不变且总功率无增加（浮点相对容差1e-6）。输入变化/接线项通过不能替代身份判据。科学FAIL也完成预定双模型全部三视图，再关闭固定16/8zero8taper；不缩长度、换边、平滑形状、阈值、RX或seed救跑。

PASS仅支持另行登记两组匹配E20 source-heldRX反证：共同raw.75＋ordinary.25/SupCon完整保留，额外.25 raw或edge同IQ RDrop对，8L前向，原U/默认头联合训练；clean至少+1pp、macroF1不降、各practical/seenV下降≤.5pp。该训练尚未实现/登记/启动。默认头fitguard本身没有DG或As证据，不因通过自动访问target。

唯一owner codex/root/F-edge-visibility-source-20261017。普通N607现有CVS-RFFI Python，CPU/OMP/MKL/OpenBLAS2，CUDA_VISIBLE_DEVICES为空，GPU none，两worker顺序；不干预健康旧任务。CWD PROJECT/releases/cvs_F_edge_visibility_fitL_20261017_r01；命令 CVS-RFFI/bin/python -u -m experiments.cvs_edge_visibility17.dispatch。各row精确runner命令在experiment.json。私有runtime保留实际Python/账户信息，不入Git。

发布入口 F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_edge_visibility17.publish，CWD当前managed codex/cvs-iq-timing-e20-20261016工作树。代码身份使用本次Git commit，在release_commit.txt/resolved_config记录；不新增配置/报告hash。独占输出 PROJECT/runs/20261017-diagnostic-F-crossRX-edgevisibility-fitL-manysig-m2-r01；日志为该root的各seed.stdout.log和发布根source_edge_visibility.stdout.log。一个N607预检、一份archive及一次传输SHA比较、一个remote compile、launch后PID/CWD/argv/CUDA/log或自然终态独立读回。smoke为首个worker的第一步，PASS立即继续。

技术异常、错输入/来源/角色、输出碰撞、非有限或越界保存failure/partial后停止，不自动重试或修改旧作业。本地19个非法输入/配置与7科学判断反例通过、120合成IQ窗/0真实IQ/NN/更新；参考/支持外、相位/正增益等变、原输入/RNG保持已核实。旧本地Torch有NumPy桥警告，本检查未调用该桥、没有装包。一次独立P0/P1审查PASS，未发现直接阻断缺陷。

产物包括root request/active/launch/result/completion、每seed实际配置/来源/原默认头完整logits与hard判决/全部72格、私有物理ID和逐包窗统计、完整日志、独立全30240判决与分层/保留率/CE/成本复算、图和同row报告/登记。IQ、权重、原始判决、样本ID、机器状态不提交Git，只保存聚合元数据。无target预测或truth-last target评分。

完整联合目标仍ACTIVE：尚缺有效Ag、稳定As、同完整契约四格/交互项、跨seed与正式clean＋六practical；lowurban>60%及suburban距自身clean≤2pp探索目标没有完成。另一对话的practical正负噪声路线不属于本候选。


## 完整实际结果

执行commit 06be5bd7d97c25f5ba3a3fec3b1f8f984687bd62 已独立核对远端OID。一次归档传输/remote compile及调度进程PID/CWD/argv/CUDA已观察；两worker全部预定视图完成，实际进程自然结束，无技术failure或重启。第一模型先完成一次匹配真实CP合成2包no-query smoke，PASS立即进入源数据。独立取回时两worker已结束，保留其launch/active/CWD/resolved、完整日志和终态，不声称取得未观察的worker实时/proc采样或日志增长，不为补采样重跑。

| 模型seed | raw fit accuracy % | 左端 % | 右端 % |
|---|---:|---:|---:|
| 2026092701 | 99.841270 | 99.424603 | 99.464286 |
| 2026092702 | 99.841270 | 98.452381 | 99.186508 |

均为原模型训练契约中的fitL5040，不是heldRX1、V或target，也不是新训练收益。两个冻结模型复用同一5040物理包，不能将30240视图判决视为独立样本。

| seed／端部 | accuracy差pp | macroF1差pp | raw正确保留率% | 修复／新错 | 最差RX差pp | 最差TX差pp | 最差day差pp |
|---|---:|---:|---:|---|---:|---:|---:|
| 01/left | -0.416667 | -0.416614 | 99.562798 | 1/22 | -1.031746 | -0.952381 | -0.654762 |
| 01/right | -0.376984 | -0.377873 | 99.602544 | 1/20 | -0.555556 | -1.428571 | -0.654762 |
| 02/left | -1.388889 | -1.387743 | 98.608903 | 0/70 | -4.047619 | -4.047619 | -1.607143 |
| 02/right | -0.654762 | -0.653826 | 99.304452 | 2/35 | -1.825397 | -2.976190 | -1.011905 |

原判据分别要求accuracy/F1≥−.5pp、raw正确保留≥99%、每RX/TX/day≥−1pp。seed01左端RX3−1.031746pp，右端TX3−1.428571pp；seed02左端RX3−4.047619pp、TX5−4.047619pp，右端RX6−1.825397pp、TX0−2.976190pp。所有分层均保留，未将“仅略低于门槛”四舍五入为PASS。

| 固定条件 | 01左 | 01右 | 02左 | 02右 |
|---|---|---|---|---|
| overall_accuracy | PASS | PASS | FAIL | FAIL |
| overall_macroF1 | PASS | PASS | FAIL | FAIL |
| raw_correct_retention | PASS | PASS | FAIL | PASS |
| nonzero_IQ_change | PASS | PASS | PASS | PASS |
| bounded_IQ_change | FAIL | PASS | FAIL | PASS |
| no_power_increase | PASS | PASS | PASS | PASS |
| reference_unchanged | PASS | PASS | PASS | PASS |
| outside_support_unchanged | PASS | PASS | PASS | PASS |
| every_RX | FAIL | PASS | FAIL | FAIL |
| every_TX | PASS | FAIL | FAIL | FAIL |
| every_day | PASS | PASS | FAIL | FAIL |

左右端的实际输入统计在两模型间逐包完全一致。

| 端部 | IQ相对RMS改变量min／median／max | 超过.35的包数 | 功率相对变化median／min |
|---|---|---:|---|
| left | 0.037596 / 0.185034 / 0.496313 | 11 | -4.038972% / -29.589364% |
| right | 0.053432 / 0.180669 / 0.269386 | 0 | -4.002717% / -8.921564% |

左端11包超过预定轻度上限，最大相对IQ变化.496313；不删这11包、不按包自适应缩强度或恢复功率来凑通过。右端虽满足轻度上限，也不能覆盖身份条件失败。所有参考[80:160]和支持外输入精确保持、没有增加总功率，但模型内部完整观察仍改变，因此“参考保护”并不保证默认头保真。本轮不据此断言信息理论上的TX身份已经被完全删除或所有mask训练均无效，只关闭这一固定候选。

## 独立复算与成本

独立reader未导入本候选的score/qualify/window或调用模型：核对全部30240默认头硬判决和logit argmax，标量logsumexp复算CE、accuracy、六类macroF1/混淆矩阵、全部RX/TX/day与72单元×三视图×两模型、修复/新错、raw正确集合及margin/30分布；全部源metadata物理ID唯一数5040且两模型完全一致。每视图40批（39×128＋48）完整，全部逐包窗统计和原条件重建一致。没有传回IQ或独立重做原IQ变换；该边界明确保留。

实际30242NN＝30240源默认头前向＋一次真实CP合成2包smoke；20160次边缘视图生成；2checkpoint读取；0新增模型epoch/优化器/教师/头拟合/target。两worker的参数和buffer逐项与进入时比较保持，.grad为空，train/联合SupCon/As调用为0；报告该运行时断言，不冒称另一次远端权重字节比较。两个原父模型历史成本40modelE/8880更新/9373356NN单独披露，未续训。

worker主体耗时 8.685004／9.659704 秒，顺序主体合计 18.344708 秒；总启动墙钟未单独测量，不把该和当作全部调度成本。本地120合成IQ窗、0NN/更新；独立分析/绘图0源IQ/NN/更新。本地旧Torch NumPy桥警告没有触发失败或数值桥调用，没有装包。真实run没有技术异常、热改、重启或干预其他任务。

完整聚合和条件见[source_analysis.json](source_analysis.json)，执行范围见[execution_summary.json](execution_summary.json)。全部原始预测/logits、逐包窗统计、源元数据/ID及完整日志留在 local_artifacts/F_edge_visibility_source_20261017/final，不提交Git或取回IQ/权重。图已生成并实际查看：[双模型完整源检验图](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_edge_visibility_source_20261017/final/figures/edge_visibility_source_guard.png)，同目录PDF保留。

## 判定与研究状态

FAIL_FIXED_EDGE_VISIBILITY_IDENTITY_GUARD_CLOSED。两模型和两侧均未完整通过固定源身份条件；不缩16点、改单侧、换平滑形状、阈值、RX或seed救跑，不启动此算子E20，不改原目标IQ或practical难度。该负结果进一步否定“只保护参考区间就足够”的操作假设，没有证明跨RX泛化、As适应或联合收益。

当前唯一候选已由最小真实源检验否决，尚未指定/登记/启动后继IQ候选。下一步需另找独立依据，不复活周期残差、旧IQ插值、MC2或其他固定失败配方。基线F＋跨RX同TX、默认头联合训练及单模型E20保持。联合任务继续ACTIVE：有效Ag、As跨seed稳定、同完整契约B00/B10/B01/B11与交互项、正式clean＋六practical、lowurban>60%和suburban接近自身clean仍未完成。
