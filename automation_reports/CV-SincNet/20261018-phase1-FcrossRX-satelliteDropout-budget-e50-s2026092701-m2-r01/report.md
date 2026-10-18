# P62：源域通过 satelliteDropout 的匹配 E50 预算验证

状态：**ANALYZED／VERIFIED，missing=[]**。两条总 E50 续训、clean＋六 practical 预测、独立 truth-last 评分和全量分析已完成。预登记源判据 FAIL，关闭本配方扩展；不追加种子或 E70。完整逐行结果、成本、曲线与负结果见 [最终分析](final_analysis.md)。下文预登记和运行阶段记录保留其当时口径。

## 简短判断与唯一假设

最值得保留的是完整 practical、稳定 clean 监督份额，以及已取得源判据 PASS 的 AS_DET。AS_DET 只停止实际 satellite 行上的现有 Dropout 输出扰动，仍抽取完整原随机掩码；AS_CE 为必要归因控制，只去 satellite RDrop KL、保留 Dropout。ordinary/raw/U 不变。它们没有新部署模块，原默认头仍联合训练。

选择依据仅既定合法源证据：旧 AS_DET−NoAs clean=−0.007407pp，AS_DET−原 As 的 pooled mid/low=+1.013889pp，AS_DET−AS_CE=+0.763889pp，四个既定源条件全 PASS。源证据见原 satelliteDropout 的 analysis_verification.json.source_decision；本次设计未消费 formal_stage。禁止根据新正式成绩换种子、调参或选择性重跑。

唯一待验证假设：原 AS_DET 在自身 E30 后继续训练到总 E50，能否保留对同预算 AS_CE／原 As 的增量，并在同预算 NoAs50 的 clean 下降≤0.1绝对百分点条件下改善固定 urban。更多预算不是必然收敛；E50 后不自动扩 E70。

## 最小匹配矩阵与来源

| 方法 | E30 | E50 |
|---|---|---|
| NoAs | 复用原 P43 B00 | 复用已完成原 budget50 B00 |
| 原 As-RDrop | 复用原 P43 B01 | 复用已完成原 budget50 B01 |
| As-CE：只关 satellite KL | 复用原 SAT18 AS_CE | **自身 FINAL30＋新20** |
| As-DET：再旁路 satellite Dropout | 复用原 SAT18 AS_DET | **自身 FINAL30＋新20** |

新矩阵只有 seed2026092701 两行，各累计50轮/11100更新；新增40模型轮/8880更新。两个旧SAT18父各为自身从零20＋自身10；原NoAs/As50继承原P43各自30。六个逻辑参考复用原预测与评分，不因当前命名重新运行。报告八行实际预算与继承关系，共享父只算一次。

主要比较 E50 DET−CE、DET−原 As、DET−NoAs；预算效应分别为各自50−30；交互=(DET50−CE50)−(DET30−CE30)。不同父的训练差异不全部归因于新增20轮或Dropout。跨seed不足，不能称稳定双提升。

## 真实执行配置

保持 F/SIG_TIME_X3/统一Sinc/joint160/defaultcos30、现有门控/参考响应与联合更新；无结构、额外头或新教师。L6300/U56700（TX真值隐藏）/V27000，sourceRX1/3/4/6/8、day1/2/3、6TX、split392005，没有源RX留出/Ldev。按当前 source_contract exact role_ids 和两个own20→30祖先核对，复用未变VALIDATED_ONCE；不加载其他预算或目标选模权重。

每行严格加载各自 FINAL30 student＋ownEMA，父局部epoch10/update2220，祖先scratch20(13L＋7U)。追加段本地epoch1–20/update1–4440，累计E31–50/update6661–11100；重置AdamW/cos20/cache，ownEMA不startup average。ordinary冻结参数必须来自各自scratch20 args在E13的日程，不能用FINAL30的epochs10。SupCon继续累计epoch+30，信道随机流继续累计epoch/step，不重播原E21–30。

原6L每步ordinary128×2/mixed128×2/raw128×2，768LNN；总监督尺度1.25，原SupCon0.1/ORD128、旧U保持。As64/step直接由raw进入完整practical，三个训练场景lowSub/midUrban/lowUrban各1/3，不先叠普通PA/通用AWGN。所有合法As继续真实标签CE，不要求学生高置信。AS_DET只在这64行旁路已有Dropout输出，仍执行原随机掩码抽样；ordinary/raw/U dropout与RDrop保持。NoExplicitCFO/PA/TX硬件边界不变。

实际pinned runtime来自SAT18/P43；launcher将保存__file__/resolved config/actual channel。fs25MHz/IQ256/原逐包complexRMS center=false位置；训练fc2.462GHz/正式固定2.45GHz；practical residual/post_sync/EQfalse即G1/refSNR30/接收质量适配＋相位追踪＋IQ补偿/reflect，均继承原实际值。不得改正式IQ、SNR、边界或删困难样本。

## 判定与自动评价

固定源判定在任何新formal结果读取前写定：DET50 native fullVclean−复用NoAs50≥−0.1pp；两draw pooled mid/low DET50−CE50≥+0.5pp；low≥CE50。任一FAIL就关闭本固定预算复验，不扫强度/系数/种子/E70；全部固定两行仍完成正式评价。除明显协议/数值/标签破坏，不加每样本身份保护或额外泛化晋级门槛。

每行final20（累计50）后自动source读出、clean＋六固定practical预测、该行全部7prediction固定后独立truth-last评分；sourceFAIL不抑制测试。跨行源决定只读source，再collectformal；最终CPU分析检查八逻辑行全部7view后才读truth进行独立CM重计/错误配对。源/目标口径与modelseed/channeldraw分开。

报告 accuracy、macroF1、最差RX、全部RX/day/TX，重点low/mid/urban三均值、clean与三个suburban差距；同样本修复/新增错误、固定NoAs30clean正确集合保持率、间隔/30；全部源state/lock/质量/新增噪声比分层和小/空组、15sessions及TX×RX×scene/state覆盖；不把新增噪声SNR当总SNR或相关性当唯一物理因果。

## 成本、命令与停止规则

每新行20epoch/4440更新/LNN3409920/As284160；预期原生studenttrain4543920/eval594002、teachertrain0/eval1134000，共6271922NN；源12600、正式1176002另计。真实两父无query smoke4NN/2CP/0IQ/0更新；本地focused checks及两模式真实F合成最多1536NN按实际记录。预测、父训练、历史共享smoke、CPU后处理wall各自报告，不把估计写成实测。

CWD=PROJECT/releases/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01；命令`CVS-RFFI/python -u experiments/cvs_satellite_budget62/runner.py`；run/output=PROJECT/runs/20261018-phase1-FcrossRX-satelliteDropout-budget-e50-s2026092701-m2-r01，owner=codex/root/satellite-budget62。CPU自动分析`analysis/cvs_satellite_budget62_analysis_20261018.py --project PROJECT --out BASE/analysis_r01`，NN0；分析失败保留，不重训/重预测/评分。

每GPU≤2总PID包含其他owner/free≥12000MB，只用普通账号。保持另一聊天Asweight30等健康任务，不停止/热改/重启。仅wrongrole/provenance/runtime/nonfinite/output冲突或预登记系统技术失败停止所属run；失败不自动重训，保留全部产物。唯一run owner，先对原run/root/进程读回，不能因超时重复提交。

本地Git代码/config→一次聚焦检查/唯一P0P1→commit/push独立OID→一次N607预检/单归档SHA/remote compile→真实CP smoke首步PASS直接训练→一次PID/CWD/argv/GPU/log增长核实。额外design approval、报告SHA、成员hash或重复审查均不作门槛；generic brainstorming额外审批记REJECTED_EXTRA_GATE，按用户授权与最小协议继续。

预期产物与精确逐row矩阵见experiment.json；实际commit、检查、GPU/时间在证据返回后记录，未验证前不称启动/完成。

本地执行证据：33项协议负例、4容量toy、4流程检查PASS；两模式实际F合成1536NN，CE/DET各768，satellite双前向maxdiff分别7.8440475/0，26次原Dropout调用均保留，DET旁路3391488个元素；ordinary logits和global RNG匹配，默认头/Sinc梯度非零finite。无真实IQ/CP/target/更新；异步合成未单独计时，不补造时长。两个本地namespace错误及一个toy配置拒绝均0NN，已修本地路由/fixture，不是真实run失败。18py compile及observer计数保存单文件compile通过。分析21项纯toy1.508秒PASS，另schema定向回归PASS，NN0。唯一P0/P1审查PASS：source schema、cap2、early/terminal source异常闭合三项修复已定向核实，无额外全审/NN。N607一次预检核实两own20+30 metadata/roles与固定参考句柄，普通账号、八GPU空闲、磁盘充足，0IQ/CP/NN；尚未执行真实父CP smoke或训练。

实际发布：代码/config/分析提交e5c25a54461439e6c33c76ce92ec50def7242524已push，独立GitHub compare identical；单次release archive SHA本远端一致，远端20py编译PASS。复合真实自身FINAL30 checkpoint smoke4NN/2CP/0IQ/0更新PASS后直接训练，未重复远端合成。dispatcher1276889；AS_CE worker1276916/GPU2，AS_DET worker1276994/GPU3，PID/CWD/argv/CUDA及nvidia进程占用独立核实；日志64379→421510、31753→421346字节，均增长。两行初始化source_roles EXACT_MATCH/target_contact false，own FINAL30 EMA严格继承，AdamW空state/resetcos20/lr.0002/WD.0001；actual_channel configurations与各自原SAT18完全一致。审计保存六场景参数不等于训练采六场景，实际训练仍从原TRAIN_SCENES三场景均匀采样。

自动源判定／clean+六practical预测／独立truth-last评分／八行CPU分析已由同一dispatcher衔接，尚未完成，不报告新准确率。已保存原live TX×RX×scene/state/lock计数供完成后分析，新两行各284160检查总量、无逐格门槛。P61草稿与其他健康任务未运行或修改。

持续监控：两条原worker日志正常增长，最新各完成追加10轮（累计E40），原dispatcher/worker存活，failure为空；源读出/正式预测/评分尚未开始。本轮未改配方、未重启，未启动额外训练或诊断前向；原续训的前向持续发生，按完成后的实际成本计。已补充[E30源域背景](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-satelliteDropout-budget-e50-s2026092701-m2-r01/E30_source_context.md>)，明确mid正收益与low负分层，未读取正式结果或用于改变已启动的E50。


收尾：DET50正式clean85.358333%、mid urban62.995238%、low urban52.027381%、urban均值63.774603%；相对自身DET30 low+1.420833pp，但相对同预算CE50 urban−.903175pp、RD50−1.236310pp。相对NoAs50 clean+.063095pp，单seed阶段约束通过，尚无跨seed稳定双提升。原源判定未重算；新两行各20轮/4440更新全部完成，新增远端14921052NN，本地1536NN另列，CPU分析65.657922秒/0NN，全部五图已实际查看。原进程自然退出。详细数据与缺项见final_analysis.md。
