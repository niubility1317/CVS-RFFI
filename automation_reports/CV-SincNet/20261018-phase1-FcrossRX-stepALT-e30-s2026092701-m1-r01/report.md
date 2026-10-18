# 完整 practical：末 10 轮按训练步骤分配增强

状态：VERIFIED_RUNNING，GPU 0 训练中。固定单种子 2026092701、E20＋10＝总 E30。执行代码提交在发布前记录；当前登记中的提交为规划父提交。

唯一假设：在每步前向成本、损失尺度、clean 稳定监督和末段 As 总数相同时，将 mixed 视图的增强集中到相邻两个更新中的第一个，可能改变 clean 与 urban 识别的权衡。该假设没有正结果保证。模型没有活动 BatchNorm；本次不作 BatchNorm 解释，也不宣称每个物理 ID 的增强次数一致。

唯一新行 B01-s2026092701 从原 P43 As20 的 student 与 own EMA 严格恢复，追加 10 轮；原同种子 As30 与 NoAs30 完整产物直接复用。来源、全部 L6300／隐藏 TX 的 U56700／V27000、源 RX [1,3,4,6,8]、day [1,2,3]、split392005、现有 IQ256 和默认联合 cosine 分类头保持。没有训练 RX 留出或内部 Ldev。

原末段每步 mixed128 含 64 As；本次累计奇数 step 为 128 As，偶数为 0 As。每步仍执行 ordinary128、mixed128、raw128 三个 RDrop 对，共 6 次 L 前向／768 个样本前向；同一对使用同一份已生成 IQ。目标为 .5 RDordinary＋.5 RDmixed＋.25 RDraw＋原 .1 CROSS_RX(ordinary128)，总 L 尺度 1.25。0 As 步跳过信道生成，mixed 的两次网络前向照常执行。

每轮 222 更新，共 111 个 As-on 步；新 10 轮 2220 更新、142080 个 As、1704960 个 L 样本前向，与原末段一致。普通增强沿原 epoch13 冻结；信道和对比调度使用累计 e21 至 e30／step4441 至6660。新段 reset AdamW、lr2e-4、WD1e-4、cosine10／min1e-6，与原 As30 尾段匹配。own EMA 与原 U 规则保留，U TX 不解封。实际 teacher、验证、源读出与正式预测成本另计。

三个原训练场景 low suburban／mid urban／low urban 各占 1/3，完整传播、噪声、接收质量与内部补偿保持。另一对话正在检验 .2／.4／.4 场景比例，本次不改场景比例。历史 held-RX／8L 的时机研究、4L clean-retain 和本次 6L／全源／固定尾 10 轮不同；已失败的 PCgrad、view moment、dropout 方案不重复。

源域支持条件在训练前固定：full V27000 clean 相对 NoAs30 ≥−0.1 个绝对百分点；固定 V1800 两次观测的 mid／low 等权均值相对原 As30 >0。同时报告全部场景、draw、F1、最差 RX 和修复／新增错误。源不通过则关闭固定配方，不扫描周期、起始相位、比例或种子救跑。所有新增训练行仍自动完成正式七视图预测与独立评分。

正式阶段条件：相对同预算 NoAs30 的 clean ≥−0.1 pp；新机制增量另与原 As30 比较。报告 clean、全部六种 practical、urban 三场均值、F1、最差 RX、RX／day／TX、suburban-clean 差距和配对错误。单种子或小于 0.1 pp 的微增不能证明稳定双提升。正式分数不影响参数、晋级或扩训。

发布命令：既有 N607 普通账户／项目 Python，在唯一 release CWD 下运行 experiments/cvs_practical_step_alternate30/runner.py；唯一输出根 PROJECT/runs/20261018-phase1-FcrossRX-stepALT-e30-s2026092701-m1-r01。单 worker，资源条件为每 GPU 最多 3 个全部 owner 进程且空闲至少 12000 MB，CPU 线程 2。真实 checkpoint 无 query 烟测是 launcher 首步，通过即进入训练；冻结→源判定→clean＋六 practical 固定预测→独立 truth-last 评分。只因本 run 的非法数据／来源／头、非有限数值、成本契约、输出碰撞或系统技术故障停止，保留产物，不自动重跑；不修改健康旧任务。

预期产物：resolved args、实际导入与信道、严格初始化、全部 2220 step／10 epoch、L/U 物理序列、每步 128/0 和 6L、按 TX／RX／day／scene 的真实 As 曝光、固定 FINAL30、source/completion、七视图 prediction 与独立 scoring、配对完整分析及实际成本。

用户允许有合法源证据的方案进一步研究总 E50；当前候选仅授权执行预登记 E30。源域通过后才考虑单独固定 E50 计划并保留匹配对照，不依据正式分数扩训，也不把 E50 解释为额外 50 轮。

本地验证已完成：33 个协议负例、真实 mixed_view 的两步 mock、18 个 Python 文件与两段远端脚本 AST 均 PASS（0 NN／CP／IQ／更新）。唯一独立 P0／P1 审查 PASS（P0=0、P1=0），覆盖实际 P43／P41 调用、来源、增强份额、U、资源和自动七视图收尾；真实 checkpoint 烟测由启动器首步执行，尚未将静态通过写成训练完成。

实际训练代码 `cc9a2dac5e2667dab9d730dd8b618e8e635fc5ca` 已提交、push 并独立核对远端 OID 一致。单次 N607 普通账户预检、发布归档 SHA 比较和远端编译通过。dispatcher 1233511／worker 1233539 的 PID、CWD、argv、CUDA_VISIBLE_DEVICES 和 nvidia-smi 占用均核实；step 日志由129190增至226761字节，未见本run failure。真实原As20无query烟测2NN、双步1536NN通过后已经直接训练；ORD128实际loss／gradient与原公式精确一致。初始化source_roles=EXACT_MATCH、target_contact=false、严格ownEMA，优化器为空state／resetcosine10。实际信道配置与原As30完全一致。此状态不代表训练／评分完成，没有新性能结果。

同一行只读后处理已发布并独立核实 PID1235749、分析 CWD／argv／空 CUDA；代码 `b93b30b7f99f59046a368c3083b33ba4dcb61cc3`，7 项 focused synthetic checks与远端编译通过，WAIT_SOURCE／truth_opened=false。它不读取IQ／checkpoint、不执行模型或更新，等待原dispatcher／worker真实句柄；源结果完成后先保存源结论，全部三逻辑条件×七视图已固定预测并独立评分后才核对truth。新增训练已完成第1／10轮（累计21／30），首轮108.943s，日志841755字节，无本run failure；尚无完整源／正式结果。

固定 E30 已完成并冻结。预登记源判定失败：full V clean 相对NoAs30−0.148148pp（界限−0.1），相对原As30−0.044444pp；源mid／low相对原As30为0／+0.222222pp，等权均值+0.111111pp。固定128/0配方关闭，不扩至E50、不扫描周期/相位/seed。此决定只依据已生成源结果、早于读取正式分数，正式七视图仍继续。

只读分析r01已正常以错误退出并保留全部产物：它把新版concat/total_L（含.25raw）误作旧版base_L_before_shape比较，且假设不存在的shape/RD字段。真实首／末记录均证明concat/total_L=完整total_L，独立shape_steps保留raw损失。只修正后处理接口并使用新analysis r02根，不重训／重复预测／评分，不改阈值或容差；分析新增NN=0。

后处理定点修复就绪：真实首末标量与错误total／raw贡献的负测通过，共8项focused checks PASS；保留原容差、CE＋KL、完整L尺度、raw监督／cost／曝光检查。只将concat总L映射至完整reference total，并用差分对账.25raw；新release／output为r02，原r01保留。原训练／预测／评分进程现已退出，七视图完成文件存在，无run failure，尚待重新运行独立完整汇总。

本行已完成完整收尾，最终为VERIFIED_ANALYZED，详见[final_analysis.md](final_analysis.md)及[results.json](results.json)。源固定条件失败、方案关闭不E50；正式单seed相对NoAs clean−0.005952pp通过阶段界限，但相对原As clean−0.099405／mid−0.060119／low+0.023214／urban均值−0.036508pp，没有新增机制收益和重复双提升证据。r02完整核对378CM／63分区合并；本次新增4353102NN。原run完成文件保留其SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS状态，独立analysis_auto_r02/complete.json现为ANALYZED，全文日志已完成，不回写旧产物。所有本run进程已退出，无技术缺项。
