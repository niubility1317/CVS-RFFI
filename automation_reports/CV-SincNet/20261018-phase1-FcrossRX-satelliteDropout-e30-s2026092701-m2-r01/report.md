# 星地优先 E30：satellite 随机子网络扰动

状态：ANALYZED；四行单种子完整匹配，source判据PASS，AS_DET正式clean阶段约束FAIL，未验证新增机制收益。Run 20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01；不宣称已有新成绩。

用户最新目标覆盖旧 Ag＋As 必须联合及独立跨 RX 泛化门槛。优先提高固定 practical，重点 mid/low urban，报告全部六种。阶段 clean 相对同预算同seed无As对照不低于 −0.1 个绝对百分点；最终要求通过As训练获得可重复clean正收益。阶段保持不等于最终完成。模型、默认头、NoExplicitCFO及内部补偿、全L6300/U56700隐藏/V27000、全部RX1/3/4/6/8与day1/2/3、split392005、IQ256、正式receivedIQ和信道固定。每模型E30，无新增结构或预算。

## 证据与唯一假设

目前保留完整clean监督份额的真实practical As最有价值。既有三seed配对low +17.2849±3.4169pp、mid +21.9448±5.1968pp、urban均值 +22.3079±5.5014pp；clean −0.3210±0.5189pp（逐seed +.0935/−.1536/−.9030），不满足新的0.1pp阶段要求，也没有稳定双提升。引用既有As-replication-r02完整报告，SD为模型种子样本SD，不把两sourcechanneldraw当模型seed。历史正式数值仅作现状披露；下一机制来自双重扰动问题，参数和晋级不从target选取。

唯一新假设：在完整As输入已受传播/噪声/补偿影响时，特征dropout继续丢弃随机子网络，可能损害弱身份证据。只在satellite包移除这种模型扰动，而ordinary/raw/U保持原dropout，可能以更小clean代价保留或改善As适应。该假设不要求增强前后原预测、置信度、间隔、分支或门控一致。

旧AsPairCE源RX1/E20已失败，只去除As KL不作为新候选复活。这里AS_CE仅为必要归因控制。旧dropoutCoupling仅改跨视图mask相关性，边际丢弃概率保持；本次AS_DET直接绕过实际As行的dropout输出，检验的是不同关键假设。另一对话研究连续E30轨迹；最新只读状态为NoAs已完成FINAL30及七视图评分、As在E21因旧E20包装器拒绝而技术终止（实际E20/4440更新，无可续训权重），其从零单As补跑尚未获新增预算答复，当前没有完整配对结果。本轮仍own20+10，不重复其轨迹路线，不修改其他任务。原覆盖耦合待检验解释让位于本次新方向，未启动新覆盖实验。

## 矩阵与实际机制

复用P43同seed01的B00无As、B01原As-RDrop固定E30及已有预测评分，保持原产物，不重训重测。新增两行AS_CE与AS_DET，各scratch E20(13L+7U)＋自身student/EMA续10轮(10U)，累计6660更新；L贯穿30轮。每段resetAdamW/lr.0002/WD.0001/cos20或cos10，同原P43。不得拿P43训练权重初始化新行；固定工厂、数据角色、普通增强、U、前向与初始化规则匹配。

所有逻辑行L每step6×128前向，.5ordinaryRD＋.5mixedRD＋.25rawRD＋原CROSS_RX SupCon，系数1.25。mixed中原step奇偶交替的64行是完整As（无As控制相应为raw），并非固定前64/后64。每次RDrop对复用同一已生成IQ。原As场景low_suburban/mid_urban/low_urban，残差residual/post_sync、EQoff G=1、referenceSNR30、fs25MHz/fc2.462GHz、reflect及原质量/相位/IQ/AGC处理保持；raw输入不预叠普通损伤，全部64合法有标签As参与CE，0置信度过滤。

两新行只在mixed项的satellite半份去掉KL，保留ordinary半份KL和全部原CE(.01平滑)。AS_CE保留所有原dropout；AS_DET在两次mixed前向的实际satellite行使用nn.Dropout前输入，其他行用原随机输出。所有原mask仍完整生成，所以RNG消耗与前向数相同。common ordinary、额外raw、U和eval均调用原行为；不修改p、不永久eval、不冻结骨干、不另训头、不增加部署参数。若真实F仍有未控制随机操作使两次As读出不同，smoke作为实现缺陷失败；这不是每个增强包必须保留clean预测的科学门槛。

## 最小验证、判据和放弃

本地聚焦固定scope负例；本机无Torch，不伪称通过Torch测试。远端合成toy验证native RNG、ordinary/raw输出与梯度、eval和异常恢复；实际launcher第一模型步骤为一份合法匹配CP的2包无query smoke，随后两个真实F合成L完整hook/backward探针，0优化器更新。默认头/Sinc梯度、206state/900722params、原U隔离、6L/实际64As、目标函数尺度和satellite-only作用检查。只做一次独立P0/P1审查，不加审批、设计SHA或成员seal。

固定FINAL30。每新行训练后自动source readout、clean+六practical全部预测固定、独立CPU truth-last评分，科学失败亦完整评分。source资格只决定后续保留/扩展：AS_DET完整V27000 clean相对NOAS≥−.1pp；固定V1800×两既有draw的mid/low均值相对NOAS≥+5pp；相对原AS_RD以及归因AS_CE，分别须 urban均值≥+.5pp且clean≥−.1pp，或clean≥+.1pp且urban≥−.1pp。不附加逐包/全RX过程保持门槛，完整RX/day/TX分层照报。任一失败关闭固定satellite dropout绕过方案，不扫描概率、层、槽、窗口、seed或预算救跑。此source规则不是正式阶段目标的证明；正式clean亦须独立衡量，single seed不能宣称可重复双提升。

## 执行、成本与产物

唯一owner codex/root/satellite-dropout18。CWD PROJECT/releases/cvs_F_satellite_dropout_E30_20261018_r01；命令 CVS-RFFI/bin/python -u experiments/cvs_satellite_dropout18/runner.py。既有N607普通账号/P43钉住runtime/FP32、每workerCPU2；共享GPU每卡所有owner最多3进程且free≥12GB。输出PROJECT/runs/20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01不可覆盖。一个archive、一次本地远端SHA比较、compile，启动后核实PID/CWD/argv/GPU/log增长。权限/父源/非有限值/预算或输出冲突为技术失败，保留原产物、不自动重试；不动其他健康任务。

预计每新行6660更新、L5114880NN、完整As426240包、旧U/ownEMA不增；按原计数训练/验证7960684NN＋冻结source12600NN＋正式预测1176002NN=9149286NN。两新增模型60model-epoch/13320更新/18298572NN，shared真实CP smoke2＋真实F合成1536NN另计，预计新合计18300110NN；source数值/梯度toy无CVS前向。两复用历史控制60model-epoch/13320更新/18298572NN单列，不能当免费父模型。额外teacher/headfit/VJP/部署参数均0；额外损失算术、真实信道生成和wall分别记录，不声称同NN等于同wall。

私有产物包括每段checkpoint/完整原生日志、reference_steps/shape_steps/satellite_subnetwork_steps、L/U出现序列、模块/增强实参、source27000和固定1800panel、逐row7view预测/独立评分。自动汇总为SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS；最终报告还需全日志成本核对、准确率/F1/最差RX/urban均值、成对修复/新错及source与正式阶段判定。输出不进Git，入库只保留配置和汇总。实际execution Git commit在release_commit/submit及报告记录。阶段0.1pp、最终正收益与可重复证据分别判定。

发布前P1修复：新增行baseline_reuse误标为True会触发既有backend禁止重训reference；已修为两新行两段均False，历史NoAs/As-RD只在logical_rows复用。同步登记配置并加入该负例；本地11负例PASS，Torch仍待远端。

实际代码/config commit：e781e8a4f0a1011dfc6d8ee7984f04378d91cdd3；一次P0/P1及该P1定点复核PASS。11本地权限负例/compile/2远端AST通过；本地Torch不可用如实保留，Torch作用域和真实F smoke由远端执行。

真实发布与启动：execution commit `735e171fc064094f227fe6807d75836a0b3b0e3a`；archive传输/compile VERIFIED。远端13负例与Torch RNG/ordinary/raw/梯度/eval/恢复PASS；真实CP smoke2NN、两真实F合成1536NN、0更新PASS。两新worker PID/CWD/argv/GPU环境及nvidia-smi独立观察一致；原完整6300/56700/27000角色、scratch/13L7U实际日志已核实。训练最新进度见execution_verification.json。训练/冻结/正式预测/评分尚未完成，不能判定0.1pp阶段目标或重复clean收益。

## 独立分析补充预登记（原训练继续，不修改配方）

source全部冻结后，独立CPU程序只读完整两段原生日志/metadata/已保存source logits，0新IQ/CP/NN/更新，核对全部4行×6660steps、物理L/U序列、原U隐藏/ownEMA offset、6L成本和satellite dropout/KL代数，再独立固定source资格。Source27000clean取原生冻结指标，不伪称有全27000逐包重算；V1800×两draw原预测完整argmax/F1/RX/day/TX/状态/质量分层重算。全部4行七视图预测/评分完成后才读正式truth，复算全部CM和六组成对修复/新增错误、固定NOAS clean正确集合保持率、宏F1、最差RX、urban均值；0.1pp用整数net×1000≥−N，single seed无论正负不称可重复双提升。源/正式state不混用，新增噪声比不称总SNR。

单一CPU分析owner，CUDA隐藏/2threads；新不可覆盖release PROJECT/releases/cvs_F_satellite_dropout_analysis_20261018_r01；output PROJECT/runs/20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01/analysis_auto_r01（源/正式图分目录保留），日志output-launch。命令 CVS-RFFI/bin/python -u analysis/cvs_satellite_dropout_pipeline_20261018.py --project PROJECT --output ANALYSIS_ROOT --truth EXISTING_TRUTH --code-commit ANALYSIS_COMMIT。等待只依据本run真实PID/CWD/cmdline和完结metadata，观察失败不当作进程终止，完整产物不要求PID仍活；不操作训练进程、不自动重跑失败。完整日志解析、指标/等待13tests、11个独立日志用例、合成图12files以及发布3AST/9file模拟已通过，后者全部网络动作MOCKED；一次全日志suite工具观察超时结论UNKNOWN，不纳入PASS。真实source/final统计与图尚未产生，当前不是ANALYZED。

独立分析实际发布：代码 commit `579f91dd4946e90b74fdb8bafa52ae88a48c5fb7`；单 archive 本地/远端 SHA 一致，remote compile/import 与 13 个聚焦测试 PASS，0 IQ/CP/NN/更新。分析 PID 1108786 的 CWD/argv/CUDA 隐藏已由独立进程观察核实，状态 WAIT_SOURCE，仅等候本 run 三个真实训练句柄，不操作训练。最新两行均 E13/30、2886 累计更新，GPU 实际在用，source 和正式尚未完成；真实统计与图仍未生成。分析启动/等待成功不等于实验结果成功。

E20→30 实际衔接：两新行均已完成 E21/30、累计 4662 更新，沿各自本轮 scratch20 parent 进入 source10；原 U 规则保持，U TX truth/U As/target access 均为 0，worker PID/CWD/argv/GPU 和日志增长继续核实。progress 的 L_NN/L_As 是当前 segment 计数，完整累计成本由最终两段日志独立核对。只读分析仍 WAIT_SOURCE、truth unopened、NN0。公开汇总 collector 已提交为 `50e9860874185583c9d3ac069706ecf1a10ddcc1` 并核实 remote；10 项合成/mock 检查 PASS，真实回收尚未执行。不得将这些运行/交付检查当作 clean 或 practical 收益证据。

只读分析技术恢复预登记：两新行均 FINAL30/6660 更新，所有 source/prediction/scoring completion 已存在，原 worker 正常退出，无训练失败。analysis_auto_r01 因原生 CSV 动态字段并集解析失败停止，NN0、truth 未打开，失败原件保留。已依据真实 producer 修复三项元数据格式对应，6 个定向回归与真实四行八段共 26640 步/120 model-epochs 完整源日志重算 VERIFIED，L/U/ordinary/As scene-state 配对全一致，每行 native7960684NN，新/复用各15921368NN单列；不修改任何训练/模型/原日志/预测/评分。代码 commit `115dc1c9e5d7d392e820d6230bbf66aaae08f8bc`。新不可覆盖 release cvs_F_satellite_dropout_analysis_20261018_r02、output analysis_auto_r02，命令沿前述 CPU pipeline，仅换新 output/code commit；仍 source 分析先于 formal truth，0 新模型NN，无需重训重测。r02 尚未启动，性能结论仍待完整独立汇总。

r02 启动即因旧入口只接受 r01 目录而退出，PID1135066 已终止，stdout0、output 未创建，未读取结果。该可避免遗漏已修正并加入5项实际publisher-config→pipeline入口集成/不可覆盖回归，AST及mock发布PASS；等待/source-first/truth规则不变。新code `95bd405f041e7017776fce83a7c20291424dc20a`，唯一恢复目录 release cvs_F_satellite_dropout_analysis_20261018_r03、output analysis_auto_r03、launch其-launch，仍CPU2/零CVS前向。旧r01/r02均保留，现报告后续以r03为准。

完整结论见 [final_analysis.md](final_analysis.md)：r03独立完整重算/原scorer一致、29公开汇总文件回收及两实际图检查VERIFIED。AS_DET相对NoAs clean−.126190pp、mid+26.505357pp、low+20.466667pp；相对原As clean−.219643pp、mid−1.168452pp、low−.637500pp；本seed没有新增机制收益。source四条件PASS不替代正式识别收益。只有一个模型种子，阶段clean不达标、无可重复clean正收益，目标未完成。原模型/预测/评分与所有失败分析尝试均保留，分析不增加模型NN。
