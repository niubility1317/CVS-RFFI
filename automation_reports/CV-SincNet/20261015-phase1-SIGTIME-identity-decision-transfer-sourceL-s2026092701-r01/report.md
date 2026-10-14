# 默认头检验的跨 RX 身份残差转移：最小源原型

状态 **ANALYZED / VERIFIED，固定源资格 0/4**。下文保留原预登记。新900 L的ROW确认0/4，跨RX对应质量四增但ACTUAL/PRE_W CE四差，固定候选已关闭。这里改变训练任务，直接要求真实身份信息在另一公共RX成分下仍支持**既有默认分类功能**，不扫旧匹配参数或修改旧资格。

每RX/day/side有六个不同TX各一包，原实际四项joint为z_r,y。当前group均值mu_r=(1/6)sum_y z_r,y，转移view z_t,y−mu_t+mu_r；同时反向z_r,y−mu_r+mu_t。残差保留donor身份标签y，anchor均值提供另一当前公共成分。标签用来形成完整合法L组和真身份监督，不拟合任何参数。mu含平均身份/RX/噪声，不保证是纯RX；这是合成feature任务，不是硬件传播反事实。均值不保存为prototype、不进入部署。部署仍原actualjoint160及原defaultcos头。

辅助readout用W.detach、scale30、eps1e-4的原cos公式；loss=.1×mean relu(log(.9)−logp_true)，720synthetic decisions/control，达到.9即停压。FP64 view/分析公式回传actualjoint，记录FP32公式差但不称调用真实native logits。原主CE默认W及骨干正常联合梯度；辅助两侧group全图和三个actualgain保留，不关闭证据支路。源门槛当前CROSS至少36/720active（5%）及非zero梯度，避免为已解决任务发布正式训练；π=.9/λ=.1/mean全六类/LR2e-4先固定，失败不提高π或扫描交换强度制造学习机会。

ZERO/SAME_RX/CROSS_RX三组，FINAL20×4cohorts=12单元。CROSS20有序不同RX×3day=60unique pair，同day side0与side1双向各sixviews，共720判定；SAME5RX×3day=15unique重复4，同样60pair/720decision，每包4端点及syntheticlabel曝光。全部目标共用一个current IQ forward，ZERO graphconnected0；两个非zero目标均算相同完整预算。无BI/Sinkhorn/teacher关系。不能直接把已有PA/CROSS_DIFF/MixStyle/ROW失败当成该exact任务已被否证；有意义对照仍须actual来源相同。

复用确认900 L：current ord2/3,19/20,37/38,54/55，held66/67；current/held互不交，与更早900也不交。但它们已经用于源研究且父看过全部L，因此不声称新的独立confirmation、未见RX或DG。原roles6300L/56700hiddenU/27000V、5RX/3day/6TX不变，既有VALIDATED_ONCE复用，训练只L；不读U/V/target。此设计来源是合法source负证据，不是目标分数反馈。

父固定ownBASE_CE-s2026092701 final20（执行6693e4480ef990bc23c5021ff1440b547fe5fcaf）。加载前source_proof完整actualroles/ID/scratchancestry/fixedfinalfreeze，payloadactualargs核验；strictmodel、188参数原AdamW m/v/step/options保留，仅restartLR2e-4，globalclip5。realCP零IQ2packet smoke PASS立即继续。after全188进入临时模型，after三个gain参与actualjoint；ACTUAL afterW与PRE_W beforeW仅分析、无fit。transferview两个读出分别用afterW/beforeW，不误用livebeforeW冒充ACTUAL。

预登记资格至少2/4 cohort CROSS同时优于ZERO及SAME：held transfer **PRE_W平均CE差<=−1e-4**、720transfer错误不增，原actualjoint的ACTUAL/PRE_W CE差<=1e-6、准确率不降；actualF差norm>1e-6、heldjoint RMS差>1e-5、至少1浅中层tensor改变，并满足current36active及非zero梯度。rawcos、transfertrue概率、所有RX/perclass/classmargin、viewnorm/eps触发和均值/迁移幅度另报。不是只让生成view变简单而损害原分类路径。

900uniqueL/6300metadata/3062网络packet前向（2零IQ+180before+4×(180current+3×180after)）/24临时CPUAdamW，0实际训练/头拟合/EMA/U/V/target；synthetic view只线性运算+默认cos，不额外跑IQ网络。live模型/optimizer/CP/grad/RNG保持。source单步过门槛不等于完整训练有效，真实训练仍需同父E20+20纯CE及原hiddenU SSL/SAME/CROSS，并在冻结后逐row自动clean独立truth-last；当前尚未启动。

run `20261015-phase1-SIGTIME-identity-decision-transfer-sourceL-s2026092701-r01`，一个owner codex/root/identity-decision-transfer-20261015，actualcode/config Gitcommit写release_commit.txt。普通N607/CVS-RFFI Python/CPU2/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_decision_transfer_sourceL_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_decision_transfer_sourceL_20261015_r01/experiments/cvs_identity_decision_transfer20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_decision_transfer_sourceL_20261015_r01/resolved_request.json`。唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-identity-decision-transfer-sourceL-s2026092701-r01/source_diagnostic`、日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_decision_transfer_sourceL_20261015_r01/source_probe.stdout.log`。权限/来源/非有限技术失败保存且仅停ownrun，先核对再恢复，不影响其他owner；科学失败关闭fixedcandidate不扫π/λ/LR/meanpool/partialstrength。一个focusedcheck及P0/P1审查、preflight、归档SHA比较/remotecompile、PID-CWD-cmd-GPU-loggrowth与独立终态；详见[登记](experiment.json)。

交换特征统计已有[CrossNorm](https://openaccess.thecvf.com/content/ICCV2021/html/Tang_CrossNorm_and_SelfNorm_for_Generalization_Under_Distribution_Shifts_ICCV_2021_paper.html)先例，它交换featuremap通道均值/方差；本候选是当前合法六类平衡L的全joint公共均值、身份残差转移、默认头停止式任务。差异只是待验证研究对象，不能据此宣称理论首创或DG收益。历史正路线独立保留，84%默认头clean三seed目标active、未达成。


## 完整实际结果

实际执行 `a2a4337a3034ed01c8f3066f3149c2a516e63b81`，提交/推送/remoteOID、归档SHA、remotecompile、realCP noquery smoke、所属PID/CWD/cmd/GPU/loggrowth及自然终态均VERIFIED，无failure。源wall 11.12 s，完整12单元、8对照、60RX记录。

| ordinal | CROSS active/720 | SAME active/720 | CROSS aux norm | CROSS current CE | SAME current CE |
|---|---:|---:|---:|---:|---:|
| 2 | 12 | 16 | 0.0436851 | 0.0214535 | 0.0243138 |
| 19 | 17 | 8 | 0.0245605 | 0.0131052 | 0.0091763 |
| 37 | 18 | 20 | 0.0169038 | 0.0109178 | 0.0121823 |
| 54 | 12 | 4 | 0.0121143 | 0.00594931 | 0.00584983 |

| ordinal | CROSS−对照 | held transfer PRE_W CE差 | transfer错数差/720 | original ACTUAL CE差 | original PRE_W CE差 | actualF差norm |
|---|---|---:|---:|---:|---:|---:|
| 2 | ZERO | +8.99769e-08 | +0 | -1.4586e-05 | -1.48834e-05 | 0.000472498 |
| 2 | SAME_RX | -3.71738e-06 | +0 | +5.21355e-06 | +5.34482e-06 | 0.000343653 |
| 19 | ZERO | -1.4894e-05 | +0 | +9.49356e-06 | +9.68899e-06 | 0.000223462 |
| 19 | SAME_RX | -1.26806e-05 | +0 | +6.66704e-06 | +6.8659e-06 | 0.000230928 |
| 37 | ZERO | +3.70974e-06 | +0 | -6.26682e-06 | -6.38161e-06 | 0.000164702 |
| 37 | SAME_RX | -1.95379e-05 | +0 | -1.28554e-06 | -1.07364e-06 | 0.00019632 |
| 54 | ZERO | -3.91956e-06 | +0 | +4.49907e-06 | +4.55607e-06 | 0.000109984 |
| 54 | SAME_RX | +5.48351e-06 | +0 | -4.48799e-06 | -4.58963e-06 | 0.000124394 |

固定资格：{"passed": false, "stage_signal": null, "qualified_ordinals": [], "cell_conditions": [{"state": "FINAL20", "ordinal": 2, "qualified": false, "failed_conditions": ["insufficient_current_trueidentity_learning_opportunity", "ZERO:no_default_transfer_CE_gain", "SAME_RX:no_default_transfer_CE_gain", "SAME_RX:ACTUAL_CE_harm", "SAME_RX:PRE_W_CE_harm"]}, {"state": "FINAL20", "ordinal": 19, "qualified": false, "failed_conditions": ["insufficient_current_trueidentity_learning_opportunity", "ZERO:no_default_transfer_CE_gain", "ZERO:ACTUAL_CE_harm", "ZERO:PRE_W_CE_harm", "SAME_RX:no_default_transfer_CE_gain", "SAME_RX:ACTUAL_CE_harm", "SAME_RX:PRE_W_CE_harm"]}, {"state": "FINAL20", "ordinal": 37, "qualified": false, "failed_conditions": ["insufficient_current_trueidentity_learning_opportunity", "ZERO:no_default_transfer_CE_gain", "SAME_RX:no_default_transfer_CE_gain"]}, {"state": "FINAL20", "ordinal": 54, "qualified": false, "failed_conditions": ["insufficient_current_trueidentity_learning_opportunity", "ZERO:no_default_transfer_CE_gain", "ZERO:ACTUAL_CE_harm", "ZERO:PRE_W_CE_harm", "SAME_RX:no_default_transfer_CE_gain"]}], "controls_not_fallback": true, "interpretation": "Source availability and compatible default-decision learning; not unseenRX/DG"}

固定公共均值转移候选关闭，不进入E20+20、不扫π/λ/LR/meanpool或partialstrength。当前CROSS只有12/17/18/12个未满足判定（1.667%至2.5%），均不足预登记36/720；不是全零图，aux梯度norm.0121至.0437、所有71实际featuretensor和18浅中层tensor有非zero控制差，但实际F差只有.000110至.000472。

held transfer PRE_W CE相对ZERO仅两组改善、幅度最大1.489e-5，另两组小幅变差，所有组都没有达到1e-4明确收益；相对SAME三组改善、一组变差。原actualjoint ACTUAL/PRE_W CE相对ZERO两正两负；相对SAME也是两正两负，held原默认准确率都100%。不能只摘出个别CE正单元晋级。RAW身份间隔相对ZERO三组正、一组负，仍不能替代默认分类收益。threegain辅助norm/isNone在逐cell证据完整保存，主要未学习机会和效应小，不能归咎于头被冻结或feature图断开。

这与ROW结果构成清楚区别：ROW任务有大量身份对应约束，能够改变特征关系，却未复现默认头CE收益；默认头公共均值转移任务大部分已经满足，剩余约束又不足且方向不稳定。当前证据支持“该晚期公共均值任务没有足够新分类信息”，不证明全部RX因素已被处理，不证明均值是纯RX，更不能证明所有证据转移或早期训练无效。上述0/4只关闭本次FINAL20/savedAdamW/固定任务，不能把一次临时步骤包装成完整训练/DG负结果。

下一候选应明确检验**异TX参照所测的接收变化能否跨身份使用**，而不是提高停止阈值。建议最小形式：同day真实L取身份y、参照c≠y，在RX-r的z_r,y上加入同参照TX-c从r到s的变化，即 z_r,y−z_r,c+z_s,c，由原defaultcos头判断真实y；同步构造反向与同RX跨物理包控制。与公共均值不同，它保留具体TX参照的接收变化，检验RX×TX交互能否被训练成可迁移的差分证据，均值平均可能隐藏这类变化。与旧CROSS_DIFF不同，它不只对齐差分方向，而要求重构后的证据仍支持原默认判别。c遍历全部5异类，所有参数/端点来自当前L，W只在aux detach，三个actualgain仍在图；不需要RX/身份新分类头、持久prototype或U真值。它仍是有待检验的latent feature合成假设，参照噪声/类相关RX作用可能使假设错误，必须保留同RX控制，不能宣称物理反事实或创新已成立。

该差分候选尚未实现、发布或决定正式训练时序。源诊断应同时观察ownINITIAL和FINAL20，区别早期形成特征的机会与晚期大多数任务已满足；不加载final教师到早期、不扫描U日历/强度。单步CE用于局部诊断，不能单凭极小confidence差推断完整20epoch泛化；正式source门槛和scratch20/E20+20阶段需要在新真实结果之前明确预登记。已有两个固定候选失败保持原资格，不事后改阈值救活。

真实900uniqueL/6300metadata/3062packet网络前向/24临时CPUAdamW，0实际训练/头拟合/EMA/U/V/target。复用ROW确认900，无新独立confirmation claim，held父已见L。两readout保留afteractualgains，ACTUAL transfer用afterW，PRE_W用beforeW；FP64/FP32仅解析公式比较，eps触发与view/commonmean/迁移RMS保留。完整[12单元](source_cells.csv)、[60RX记录](source_RX_metrics.csv)、[8对照](paired_regimes.csv)、[标量证据](result_evidence.json)、[运行核验](launch_evidence.json)。历史SIG_TIME/CENTER25K/COMPACT_LINEAR/LONG_SILU正路线独立保留，84%默认头clean三seed目标active、未达成。
