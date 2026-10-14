# 源 RX 对比过滤历史特征动量：固定源原型

状态 **ANALYZED / VERIFIED，固定源资格 0/4**。以下保留原预登记。根据同父、同梯度的纯 CE 状态实验，保留尺度历史有利于默认头读出，但保存矩仍使 raw 跨 RX 身份关系下降。本轮只研究一个新操作：**保留二阶尺度和原当前梯度，用源 RX 梯度对比过滤历史一阶特征动量**。这不是清空全部状态、旧完整步风险投影或已关闭损失的系数调整，尚未证明泛化收益或完整文献原创性。

对同一原 L180 CE 前向，求源 RX1/3/4/6/8 各36包的平均梯度。main仍是原whole180平均CE梯度。按完整188梯度的原clip5系数c和各坐标saved t/v计算下一步prospective AdamW分母d；历史贡献u=beta1*m/(bc1*d)，对比h_r=(1-beta1)*c*(g_r-mean(g_r))/(bc1*d)。只将u投影到五中心化h跨度的正交补，再映回m。原AdamW随后正常使用当前g更新；v/step/weightdecay/W/其他参数状态保持。五行Gram用FP64，固定eigen cutoff max(1e-30,1e-10*maxeig)，rank至多4。记录理想投影和实际FP32状态映射残差，不把数值内积当身份收益。该坐标度量只适用于固定CENTER25K chart。

对照KEEP（保存状态）、M_ZERO_F（只清feature一阶m）、RX_M_FILTER（结构过滤），全部LR2e-4及同父状态/同梯度。f_theta含原Sinc/时频/PA/reference/归一化/3融合gain的71tensor/195041coord，实际classW不滤且正常联合CE。临时188参数after完整复制，真实joint160保留after3gain；ACTUAL用afterW、PRE_W只用beforeW分析同一z，无分类器拟合或替换。

固定own BASE_CE第一seed2026092701 final20，父执行6693e4480ef990bc23c5021ff1440b547fe5fcaf；source_proof先核对实际数据角色、完整scratch来源及fixedfinal20冻结，再反序列化，不因同名权重直接继承。原L6300，7/63/30角色，6TX/5RX/3day。四currentL180采用原90cell的ordinal0/1、17/18、35/36、52/53；held180采用68/69。900包native5ID和全局baseindex互不交，held曾被父训练见过，只是独立当前步的源评价，不是盲测或未见RX。U/V/target IQ不读，U真值不读。

固定晋级要求至少2/4 cohort同时满足：FILTER对KEEP及M_ZERO_F各有实际F差norm>1e-6、heldjointRMS差>1e-5、至少1浅中层tensor变化、raw身份关系间隔增益>1e-5；FILTER自身间隔比before提高；ACTUAL与PRE_W meanCE相对before及两对照均不得恶化超过1e-6。rank须非零。raw关系定义为同TX跨RX同day余弦减异TX同RX同day余弦，不等同classmargin。只缩小步、只让投影成立、只有默认头置信度改善都不能晋级。阈值和rank规则在真实源结果前固定；失败关闭此固定机制，不扫LR/rank/强度/频率或选对照替代。

预算900uniqueL/6300metadata/3062packet网络前向（含2零IQ真实checkpoint烟测）/24临时CPUAdamW、0实际训练/头拟合/EMA/U/V/target。真实CP noquery smoke PASS立即继续，所有live模型/buffer/grad/payload/RNG保持。技术不一致或异常保存输出并仅停本run，其他owner不变。

run `20261015-phase1-SIGTIME-RX-history-momentum-sourceL-s2026092701-r01`，owner `codex/root/RX-history-momentum-source-20261015`。实际GitHEAD写release_commit.txt；普通N607账户/CVS-RFFI Python/CPU2线程/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_history_momentum_sourceL_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_history_momentum_sourceL_20261015_r01/experiments/cvs_rx_momentum20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_history_momentum_sourceL_20261015_r01/resolved_request.json`，唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-RX-history-momentum-sourceL-s2026092701-r01/source_diagnostic`，日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_history_momentum_sourceL_20261015_r01/source_probe.stdout.log`。完整环境、stop和artifacts见[experiment.json](experiment.json)。一次归档传输SHA比较、远端编译、进程/CWD/命令/日志增长核验及独立终态读回；没有新增seal/receipt/hash链。

如果源规则通过，才另行固定正式训练频率与E20+20纯CE同父同增量控制。当前没有发布正式训练或clean预测，目标测试不能反馈调参。SIG_TIME、CENTER25K、COMPACT_LINEAR、LONG_SILU历史正路线分别保留；84%默认头clean三seed目标仍active、未完成。源RX梯度差可能包含有用身份交互，过滤未必是去接收机因素，也不提供有限步风险保证。机制依据见[设计](../../../analysis/phase1_receiver_contrast_momentum_20261015.md)。


## 完整真实源结果

执行 `c91085d7f858b7ca24337294bcab409b6999e351` 已推送且remote OID一致。归档传输、远端编译、真实CP noquery烟测、所属进程/CWD/命令/CPU/GPU/日志增长及独立自然终态均VERIFIED，无failure。完整12单元、8项成对对照、60条RX记录；source wall 12.22 s。

| ordinal | 机制 | 历史移除比例 | 实际F位移 | held RMS | PRE_W CE减少 | ACTUAL CE减少 | raw身份间隔增益 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | KEEP | 0.0000% | 0.016224 | 0.014211 | +5.687e-05 | +5.118e-05 | -3.268e-04 |
| 0 | M_ZERO_F | 100.0000% | 0.002267 | 0.001738 | +4.507e-05 | +5.003e-05 | +8.935e-05 |
| 0 | RX_M_FILTER | 55.0522% | 0.013108 | 0.008781 | +1.089e-04 | +1.106e-04 | -9.602e-04 |
| 17 | KEEP | 0.0000% | 0.014506 | 0.011827 | +1.028e-04 | +1.001e-04 | -3.628e-04 |
| 17 | M_ZERO_F | 100.0000% | 0.002123 | 0.001911 | -5.568e-05 | -4.974e-05 | +9.937e-06 |
| 17 | RX_M_FILTER | 63.2379% | 0.012111 | 0.007253 | +1.313e-04 | +1.351e-04 | -9.079e-04 |
| 35 | KEEP | 0.0000% | 0.017233 | 0.015540 | +4.018e-05 | +3.265e-05 | -7.834e-04 |
| 35 | M_ZERO_F | 100.0000% | 0.005001 | 0.004515 | +1.186e-04 | +1.220e-04 | -3.597e-04 |
| 35 | RX_M_FILTER | 65.0456% | 0.012768 | 0.007519 | +2.134e-04 | +2.147e-04 | -8.859e-04 |
| 52 | KEEP | 0.0000% | 0.017192 | 0.015689 | +3.883e-06 | -7.327e-06 | -5.215e-04 |
| 52 | M_ZERO_F | 100.0000% | 0.003662 | 0.003530 | +1.019e-04 | +1.021e-04 | -7.940e-05 |
| 52 | RX_M_FILTER | 54.5856% | 0.013648 | 0.010378 | +8.063e-05 | +7.663e-05 | -4.875e-04 |

| ordinal | FILTER减对照 | F差norm | held RMS差 | PRE_W CE差 | ACTUAL CE差 | raw身份间隔增益 |
|---|---|---:|---:|---:|---:|---:|
| 0 | KEEP | 8.517e-03 | 9.285e-03 | -5.200e-05 | -5.938e-05 | -6.334e-04 |
| 0 | M_ZERO_F | 1.292e-02 | 8.769e-03 | -6.380e-05 | -6.053e-05 | -1.050e-03 |
| 17 | KEEP | 9.783e-03 | 1.011e-02 | -2.850e-05 | -3.502e-05 | -5.451e-04 |
| 17 | M_ZERO_F | 1.198e-02 | 7.149e-03 | -1.870e-04 | -1.849e-04 | -9.178e-04 |
| 35 | KEEP | 1.006e-02 | 1.020e-02 | -1.732e-04 | -1.820e-04 | -1.026e-04 |
| 35 | M_ZERO_F | 1.175e-02 | 7.103e-03 | -9.480e-05 | -9.262e-05 | -5.262e-04 |
| 52 | KEEP | 8.444e-03 | 7.929e-03 | -7.675e-05 | -8.395e-05 | +3.394e-05 |
| 52 | M_ZERO_F | 1.296e-02 | 8.371e-03 | +2.127e-05 | +2.549e-05 | -4.081e-04 |

固定source gate：{"passed": false, "qualified_ordinals": [], "cell_conditions": [{"ordinal": 0, "qualified": false, "failed_conditions": ["no_raw_identity_margin_improvement_from_before", "KEEP:raw_identity_margin_gain", "M_ZERO_F:raw_identity_margin_gain"]}, {"ordinal": 17, "qualified": false, "failed_conditions": ["no_raw_identity_margin_improvement_from_before", "KEEP:raw_identity_margin_gain", "M_ZERO_F:raw_identity_margin_gain"]}, {"ordinal": 35, "qualified": false, "failed_conditions": ["no_raw_identity_margin_improvement_from_before", "KEEP:raw_identity_margin_gain", "M_ZERO_F:raw_identity_margin_gain"]}, {"ordinal": 52, "qualified": false, "failed_conditions": ["no_raw_identity_margin_improvement_from_before", "M_ZERO_F:raw_identity_margin_gain", "M_ZERO_F:PRE_W_CE_harm", "M_ZERO_F:ACTUAL_CE_harm"]}], "controls_are_not_fallback_candidates": true, "interpretation": "New historical-momentum filtering source signal against BOTH KEEP and M_ZERO_F; not DG efficacy or a main-training authorization by target score"}

固定过滤机制未通过预登记源资格，关闭本次RX_M_FILTER；不通过rank/LR/强度/频率扫描救活，也不把KEEP或M_ZERO_F替代晋级。

实际F71/195041coord含三个aftergain，当前wholeCE梯度保持，v/step/W状态保持；每cohort三组真实AdamW之后W逐位相同。两种默认头读出共享真实afterjoint160，无拟合。FP64理想正交和FP32状态映射残差都保留，不宣称严格原子级步分解、单位不变或RX纯噪声。

实际900uniqueL/6300metadata/3062packet网络前向/24临时CPUAdamW，0实际训练/头拟合/EMA/U/V/target。heldL已被父训练见过，single eval CPU一步不能代替真实训练或目标泛化；没有新clean预测或truth评分。历史正路线SIG_TIME/CENTER25K/COMPACT_LINEAR/LONG_SILU分别保留，84%默认头clean三seed目标仍active、未达成。完整[单元](source_cells.csv)、[RX读出](source_RX_metrics.csv)、[成对对照](paired_regimes.csv)、[标量结果](result_evidence.json)及[运行核验](launch_evidence.json)已保存，私有原CP/输出/日志保留。


## 机制解释与下一步边界

FILTER并非没有作用：五RX跨度在四组均rank4，所移除的历史贡献向量norm占原历史贡献norm的54.586%至65.046%。这个比例是投影移除向量的范数比，不是剩余向量范数降低了同样百分比。实际FP32映回state的贡献舍入残差norm为1.46e-6至1.62e-6，归一化h点积残差3.36e-11至7.80e-11；不能称严格数学零。FILTER对KEEP的实际F差norm0.00844至0.01006、heldRMS差0.00793至0.01020；四组全部71个F tensor、18个浅中层tensor出现差异，Sinc核23/23/24/24变化。因此失败不是梯度为零、过滤未启用或只改变分类头。

FILTER相对before的PRE_W heldmeanCE四组减少8.06e-5至2.13e-4，ACTUAL四组减少7.66e-5至2.15e-4；相对KEEP两读出也都改善。这项真实正迹象完整保留。before heldmeanCE是0.004644、raw关系身份间隔0.320076；before及全部after的held准确率均100%，所以这里的CE改善只是置信度敏感的源证据，没有准确率增益，更没有新目标域分数。

与此同时，FILTER的raw跨RX身份关系间隔四组下降4.88e-4至9.60e-4；相对KEEP三降一升（ordinal52仅增3.39e-5），相对M_ZERO_F四组全低4.08e-4至1.05e-3。ordinal52的FILTER两种CE还比M_ZERO_F差2.13e-5/2.55e-5。固定源资格0/4是真实身份关系条件未满足，不能改成只看CE就宣布晋级，也不能把微小raw几何下降写成已证明目标识别性能下降。

M_ZERO_F单独显示一个与旧RESET_ALL不同的状态边界：保留v/step而只清m，实际F步norm0.00212至0.00500，远小于旧清全部状态约0.083；heldCE三组改善、一组恶化，raw身份关系两升两降。因此“清历史方向”也没有稳定身份收益，不能把它作为FILTER失败后的新主训练候选。该控制只帮助区分尺度历史与方向历史，不是新可调清矩比例。

新证据削弱了“源RX平均CE梯度差直接定义应删除的历史方向”这一具体假设。RX差异可能混入有用TX×RX交互、当前难例或旧W下的决策结构。数学去分量和默认头可读性改善不足以证明跨RX身份证据改善；本轮也没有识别哪一种成分导致几何变化。关闭固定RX_M_FILTER，不扫rank/LR/投影强度/频率，不复活旧完整步风险投影或CROSS_DIFF。下一机制需先说明由真实身份监督支持的可迁移学习机会，且不能重复已完成临时episodic/ridge路线、另拟合头或增加伪标签接受率来替代证据。此处尚未发布下一正式候选或宣称找到有效创新机制。

SIG_TIME、CENTER25K、COMPACT_LINEAR、LONG_SILU历史正路线独立保留，按各自配方/预算/seed解释；不把不同配方最高分相加、不按已暴露target排名选父。主84%默认头clean三seed目标仍active、未达成。
