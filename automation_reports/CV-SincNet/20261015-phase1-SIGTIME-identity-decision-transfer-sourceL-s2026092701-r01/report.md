# 默认头检验的跨 RX 身份残差转移：最小源原型

状态 **PLANNED**。新900 L的ROW确认0/4，跨RX对应质量四增但ACTUAL/PRE_W CE四差，固定候选已关闭。这里改变训练任务，直接要求真实身份信息在另一公共RX成分下仍支持**既有默认分类功能**，不扫旧匹配参数或修改旧资格。

每RX/day/side有六个不同TX各一包，原实际四项joint为z_r,y。当前group均值mu_r=(1/6)sum_y z_r,y，转移view z_t,y−mu_t+mu_r；同时反向z_r,y−mu_r+mu_t。残差保留donor身份标签y，anchor均值提供另一当前公共成分。标签用来形成完整合法L组和真身份监督，不拟合任何参数。mu含平均身份/RX/噪声，不保证是纯RX；这是合成feature任务，不是硬件传播反事实。均值不保存为prototype、不进入部署。部署仍原actualjoint160及原defaultcos头。

辅助readout用W.detach、scale30、eps1e-4的原cos公式；loss=.1×mean relu(log(.9)−logp_true)，720synthetic decisions/control，达到.9即停压。FP64 view/分析公式回传actualjoint，记录FP32公式差但不称调用真实native logits。原主CE默认W及骨干正常联合梯度；辅助两侧group全图和三个actualgain保留，不关闭证据支路。源门槛当前CROSS至少36/720active（5%）及非zero梯度，避免为已解决任务发布正式训练；π=.9/λ=.1/mean全六类/LR2e-4先固定，失败不提高π或扫描交换强度制造学习机会。

ZERO/SAME_RX/CROSS_RX三组，FINAL20×4cohorts=12单元。CROSS20有序不同RX×3day=60unique pair，同day side0与side1双向各sixviews，共720判定；SAME5RX×3day=15unique重复4，同样60pair/720decision，每包4端点及syntheticlabel曝光。全部目标共用一个current IQ forward，ZERO graphconnected0；两个非zero目标均算相同完整预算。无BI/Sinkhorn/teacher关系。不能直接把已有PA/CROSS_DIFF/MixStyle/ROW失败当成该exact任务已被否证；有意义对照仍须actual来源相同。

复用确认900 L：current ord2/3,19/20,37/38,54/55，held66/67；current/held互不交，与更早900也不交。但它们已经用于源研究且父看过全部L，因此不声称新的独立confirmation、未见RX或DG。原roles6300L/56700hiddenU/27000V、5RX/3day/6TX不变，既有VALIDATED_ONCE复用，训练只L；不读U/V/target。此设计来源是合法source负证据，不是目标分数反馈。

父固定ownBASE_CE-s2026092701 final20（执行6693e4480ef990bc23c5021ff1440b547fe5fcaf）。加载前source_proof完整actualroles/ID/scratchancestry/fixedfinalfreeze，payloadactualargs核验；strictmodel、188参数原AdamW m/v/step/options保留，仅restartLR2e-4，globalclip5。realCP零IQ2packet smoke PASS立即继续。after全188进入临时模型，after三个gain参与actualjoint；ACTUAL afterW与PRE_W beforeW仅分析、无fit。transferview两个读出分别用afterW/beforeW，不误用livebeforeW冒充ACTUAL。

预登记资格至少2/4 cohort CROSS同时优于ZERO及SAME：held transfer **PRE_W平均CE差<=−1e-4**、720transfer错误不增，原actualjoint的ACTUAL/PRE_W CE差<=1e-6、准确率不降；actualF差norm>1e-6、heldjoint RMS差>1e-5、至少1浅中层tensor改变，并满足current36active及非zero梯度。rawcos、transfertrue概率、所有RX/perclass/classmargin、viewnorm/eps触发和均值/迁移幅度另报。不是只让生成view变简单而损害原分类路径。

900uniqueL/6300metadata/3062网络packet前向（2零IQ+180before+4×(180current+3×180after)）/24临时CPUAdamW，0实际训练/头拟合/EMA/U/V/target；synthetic view只线性运算+默认cos，不额外跑IQ网络。live模型/optimizer/CP/grad/RNG保持。source单步过门槛不等于完整训练有效，真实训练仍需同父E20+20纯CE及原hiddenU SSL/SAME/CROSS，并在冻结后逐row自动clean独立truth-last；当前尚未启动。

run `20261015-phase1-SIGTIME-identity-decision-transfer-sourceL-s2026092701-r01`，一个owner codex/root/identity-decision-transfer-20261015，actualcode/config Gitcommit写release_commit.txt。普通N607/CVS-RFFI Python/CPU2/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_decision_transfer_sourceL_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_decision_transfer_sourceL_20261015_r01/experiments/cvs_identity_decision_transfer20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_decision_transfer_sourceL_20261015_r01/resolved_request.json`。唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-identity-decision-transfer-sourceL-s2026092701-r01/source_diagnostic`、日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_decision_transfer_sourceL_20261015_r01/source_probe.stdout.log`。权限/来源/非有限技术失败保存且仅停ownrun，先核对再恢复，不影响其他owner；科学失败关闭fixedcandidate不扫π/λ/LR/meanpool/partialstrength。一个focusedcheck及P0/P1审查、preflight、归档SHA比较/remotecompile、PID-CWD-cmd-GPU-loggrowth与独立终态；详见[登记](experiment.json)。

交换特征统计已有[CrossNorm](https://openaccess.thecvf.com/content/ICCV2021/html/Tang_CrossNorm_and_SelfNorm_for_Generalization_Under_Distribution_Shifts_ICCV_2021_paper.html)先例，它交换featuremap通道均值/方差；本候选是当前合法六类平衡L的全joint公共均值、身份残差转移、默认头停止式任务。差异只是待验证研究对象，不能据此宣称理论首创或DG收益。历史正路线独立保留，84%默认头clean三seed目标active、未达成。
