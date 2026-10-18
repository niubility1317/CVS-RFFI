# P69：保留 AdamW 状态的匹配 E30

状态：ANALYZED。两新行E30及四逻辑条件七环境／独立truth-last／完整CPU分析已完成，missing=[]。源两项FAIL，不延长E50；整体目标ACTIVE。详细结果见 [final_analysis.md](final_analysis.md)。以下保留运行前设计与执行记录。

最值得保留的是完整 practical 有标签 CE 和稳定 clean 监督份额。P68 以及原时间确认没有通过源晋级；RXpopulation／校准刷新、时机、场景安排和比例方向已做，不重复改名启动。另一对话正研究 common_channel，排除。全部已源合格的原 As／Asweight／LRaw／SAT18 路线已做 E50，没有待补的合格 E50。

本轮唯一假设：同一合法父、相同新段 cosine10 和 IQ／损失／U 规则下，保留 AdamW 的一阶、二阶矩及每参数步数，是否比清空状态更好地继续星地学习并保留 clean。已有 continuous30 同时改动全程 LR，viewMoment 两边均重置状态，不能回答此问题；旧纯 CE 一步诊断不是实际 practical 训练证据。保留 moment 但新开 cos10 属于带状态微调，不冒称恢复完整原优化轨迹。

## 固定匹配矩阵

| 条件 | 来源 | 预算 |
|---|---|---|
| NOAS_RESET／NOAS30 | P43 B00 原已完成 FINAL30 | own20＋RESET10，复用预测评分 |
| AS_RESET／AS30 | P43 B01 原已完成 FINAL30 | own20＋RESET10，复用预测评分 |
| NOAS_KEEP | 本轮 B00，严格加载 P43 B00 stage20 student／ownEMA／optimizer | 新增 KEEP10，总 E30 |
| AS_KEEP | 本轮 B01，严格加载 P43 B01 stage20 student／ownEMA／optimizer | 新增 KEEP10，总 E30 |

仅改变优化器初始状态。加载前核对原保存 state／参数组／顺序、shape、finite 和 step，加载后逐项与合法父相等；恢复新建参数组 LR 等超参，防止父 E20 末 LR 覆盖新 cos10。CENTER_UNITS 新段 hook、clip 和日志计数从0开始，模型参数、ownEMA 和原 U 更新流程保持；不另训头。

固定 F＋CROSS_RX（SIG_TIME_X3／统一 Sinc／CENTER_UNITS／joint160／默认 cosine30），900722 参数／206 state keys，正常联合训练。全 L6300／hidden U56700／V27000、RX1/3/4/6/8、day1/2/3、六 TX、split392005，无 heldRX／Ldev；U TX 始终隐藏。P43 父源契约、scratch20／4440 更新、final_only、无目标选模来源沿用并在 smoke 实际加载核实。

原 6L／每步768 L 前向、总 L 尺度1.25、ordinary clean 份额、fullAs64、同IQ RDrop、ordinary128 SupCon、batch_neighbor 旧门控、ownEMA／原 raw U 教师与 strong U 学生保持；没有新 IQ 扰动、损失、头或教师。As-off 对应位置仍原 identity control。新增机制收益分别以 AS_KEEP−AS_RESET、NOAS_KEEP−NOAS_RESET 读出，并报告交互 (AS_KEEP−NOAS_KEEP)−(AS_RESET−NOAS_RESET)。As 整配方效应以 AS_KEEP−NOAS_KEEP 读出，不能混作优化器的收益。

训练 full residual／post_sync、EQ=false即G=1、质量适配／相位跟踪／IQ补偿／AGC开启、ref30、reflect、fs25MHz／IQ2×256、trainfc2.462GHz不变。实际 pinned leo_practical 仍为 evidence20261015_r02 的 experiments/adv3b02_xuc/code，保存 __file__。RMS 位置和内部 CENTER_UNITS 原样；15个RX/day/eq硬件session按原分布稳定，不由TX定义。全部有标签As参与监督，无学生置信度门；正式 IQ／fc2.45GHz／SNR／六 practical 不变。新增噪声比不是总 SNR。

## 源判定、E50和科学否证

冻结最终学生后只使用完整V27000 clean 和原 V1800×2固定信道draw×3practical。只有两项晋级判据：

1. AS_KEEP fullVclean−NOAS_KEEP ≥−0.1个绝对百分点；
2. pooled mid/low urban AS_KEEP−原AS30 ≥＋0.5个百分点。

任一失败即关闭固定配方，不扫描moment子集／LR／betas／seed救援。源决定先于owner读取本轮正式分数保存；所有已训练行仍自动七视图预测及独立truth-last。只有源PASS才可在后续独立run延长至总E50，不是追加50；不按正式结果晋级。两个信道draw不是两个模型seed。正式clean≤0.1pp阶段目标与可重复clean＋星地双提升分开报告。

## 成本、执行和停止

每新行新增10epoch／2220更新，复用父20／4440，累计30／6660。两新行合计20model-epochs／4440更新；每行native3162962＋source12600＋formal1176002，共享真实父／ownEMA smoke8NN，总新增8703136NN。两行LNN合计3409920。额外NN=0、额外教师训练0，本地真实CVSNN0；源／正式评估、父既有成本、optimizer状态存储和墙钟分别实报。各父20只计一次，四逻辑条件已发生训练合计80model-epochs，不冒称4×30全新成本。预算耗尽不代表收敛。

CWD=PROJECT/releases/cvs_FcrossRX_optimizer_keep_E30_20261018_r01；command=CVS-RFFI/python -u experiments/cvs_optimizer_continuity69/runner.py；输出唯一PROJECT/runs/20261018-phase1-FcrossRX-optimizerKeep-e30-s2026092701-m2-r01，owner=codex/root/optimizer-keep69。N607普通账户、现有环境FP32／CPU2，无安装；每GPU最多2个总计算进程（含其他owner）、空余≥12000MB，保护健康任务。

只对错误角色／来源／optimizer order-shape-finite／旧LR误载／runtime、非有限数、输出碰撞、非法预测等系统技术失败停止所属run，保留所有产物，不自动重试；科学低性能不中止健康训练。一次限定P0/P1、聚焦检查、一次真实父no-query smoke通过即继续，一次归档SHA比较／remotecompile、一次启动后PID/CWD/argv/GPU/log增长检查，按最小流程执行，不添加gate或签名链。

最终交付全部逐row／逐seed七环境accuracy／macroF1／最差RX、urban均值、suburban−clean差、配对错误／RX-day、源state/lock/quality/noise和固定父clean正确集合保持、默认头margin/30、完整训练曲线、真实optimizer状态／LR轨迹／覆盖和成本。单seed微增不宣称复现，空／小分层保留。无正式状态标签时不伪造物理归因；local_artifacts保存原始输出，权重／IQ／日志不入Git。


发布前实际验证：23项协议/配置负例＋40项纯张量optimizer检查PASS（命令墙钟2.857731s）；18核心Python compile PASS（.3836804s）；8项纯CPU分析toy＋compile PASS。local真实CVSNN/IQ/CP均0。唯一独立P0/P1审查PASS，无阻塞缺陷。一次普通账户direct N607预检VERIFIED：8GPU空闲、父输入两份齐、run/release不存在、diskfree3932.54GB；真实父moments与8NN smoke仍待发布首步。


实际代码 d2ddf0d48a48b7c9f8416e8c4219185155faf298 已提交、推送并由独立GitHub compare核实identical。一次发布归档SHA e432bfef1c3ca3042fa9b759f8bdb8231c4551af543afc247edb40026fc6f2cb 比较VERIFIED，22个Python文件远端compile PASS。首步两个真正父student／ownEMA／AdamW加载smoke PASS8NN、0更新；两父74个已有state均step4440，m/v逐项相等，新bodyLR.0002／SincgroupLR.2和新计数0正确。两新行GPU1/2所属PID/CWD/argv/GPU计算及日志增长已独立核实，E21后旧state步数4662、新段Sinc步数222。实际nativeargs仅output_dir/run_id变化，数据角色EXACT_MATCH，原batch_neighbor不变。Windows SCP客户端首次未产生远端文件，核对目录为空后改用既有验证客户端，归档比对成功；无训练失败或重跑。


最终：KEEP相对同父原As的urban＋0.006151pp、mid＋0.020833pp、low−0.022024pp、clean＋0.007143pp，无实质新增收益。相对匹配NoAs的urban＋28.532143pp、clean＋0.075595pp，正式单seed阶段目标通过，但不足以证明可重复双提升。low51.222024%，未达60%。源fullVclean−matchedNoAs−0.129630pp、midlow−oldAs＋0.097222pp，两项FAIL；关闭optimizerKEEP固定方案，不按正式结果救援。新增8703136NN/4440更新，端到端1185.216s；全部曲线/分层/负结果/可选cohort缺口见final_analysis.md及delivery.json。
