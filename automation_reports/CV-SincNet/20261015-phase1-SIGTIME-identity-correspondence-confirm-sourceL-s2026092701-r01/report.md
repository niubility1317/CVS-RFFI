# 跨 RX 真实身份对应：新样本独立确认

状态 **ANALYZED / VERIFIED，独立 ROW 确认 0/4，固定候选关闭**。下文保留原预登记。上一固定双边容量候选 INITIAL/FINAL20 各0/4，保持关闭。其独立行列匹配控制在 FINAL20 的四组源样本上同时改善自身身份质量和默认头 CE，本轮以新的物理包及同 RX 对照核实这一探索迹象，不将旧控制冒充已通过候选。

仍用实际四项 joint160：Dij=||a_i-b_j||²/160，B=D-rowmean-colmean+grandmean，C=B/max(RMS(B),1e-8)，分母可微。独立行/列 logsoftmax(-C/.1) 给真实 TX 对应的 q；loss=.1×(mean relu(log.9-logqrow)+mean relu(log.9-logqcol))/2。只在损失成本表消除公共 RX 特征偏移及纯全局幅度项，部署特征不减均值。数学范围只涵盖这种偏移/幅度不变性，不能保证全部物理 RX 失真可辨识。无共享列容量、BI 或 Sinkhorn；τ=.1、π=.9、λ=.1、RMSfloor1e-8 沿用旧控制，不扫描。

主 CE 仍正常联合训练默认 cosine 头；辅助损失双端点穿过实际 F71/195041coord 及三个 gain，W 不在辅助图。无新头、头拟合、原型、老师关系复制或梯度删除。网络浅中层的实际变化、Sinc 核、aux/main 梯度方向、默认头 CE/准确率/分类间隔及 raw cosine 身份间隔全部报告；不把这些量互换。

矩阵 FINAL20 ×4 cohort×ZERO/SAME_ROW/CROSS_ROW，共12单元。新 L 当前 ordinal2/3、19/20、37/38、54/55，held66/67，各 bank 为6TX×5RX×3day×2side=180。900包及完整native5ID/baseindex与旧900不交，原始 L 角色不变；held被父训练见过，仅当前步骤输入不交，不能声称盲测、未见RX或DG。CURRENT 两端同 day/side0→side1。CROSS20 ordered differentRX×3day=60 unique六乘六，SAME5RX×3day=15unique各重复4，共60。每包在两种目标内各4端点曝光，均720双向真实身份判定。每次共120 ROW匹配、0 BI/Sinkhorn，ZERO复用诊断且graphconnected0。

父固定 ownBASE_CE-s2026092701 final20（父执行6693e4480ef990bc23c5021ff1440b547fe5fcaf），完整source_proof核对实际roles/IDs、scratch祖先与固定20冻结后加载，actualargs再核验。保留原188参数AdamW m/v/step和options，仅stageLR2e-4，globalclip5，无reset/filter。真实CP零IQ烟测2packet PASS立即继续，源rolehandles既有VALIDATED_ONCE复用，无重复数据验证。

预登记资格：至少2/4 cohort CROSS_ROW同时优于ZERO及SAME_ROW；held自身无容量ROW true质量均值差>1e-4、720对应错数不增、ACTUAL及PRE_W held CE差<=1e-6、两读出准确率不降。actualF差norm>1e-6、heldjoint RMS差>1e-5、至少1浅中层tensor差，当前目标active且aux梯度非零。PRE_W仅把分析读出的 W 换成before W，不拟合，after三个gain仍参与同一actualjoint。rawcos间隔另报不作gate，因为loss允许公共RXoffset；这不是变更旧候选阈值救活失败机制。通过后才能另行固定同父E20+20纯CE/原hiddenU SSL/同RX/跨RX真实训练；失败关闭固定ROW确认，不扫强度、时序或替换控制晋级。

预算900uniqueL/6300metadata/3062packet网络前向（2零IQsmoke+180before+4×(180current+3×180after)）/24临时CPUAdamW。0实际训练、头拟合、EMA、U/V/target读取。所有live模型、optimizer、CPpayload、grad和RNG保持。L真实标签仅用于合法训练信号，目标预测始终独立默认头，不运行TX配额或batch全局指派。

run `20261015-phase1-SIGTIME-identity-correspondence-confirm-sourceL-s2026092701-r01`，一个 owner codex/root/identity-correspondence-confirm-20261015；actualcode/config Gitcommit写release_commit.txt。普通N607/CVS-RFFI Python/CPU2线程/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_correspondence_confirm_sourceL_20261015_r01`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_correspondence_confirm_sourceL_20261015_r01/experiments/cvs_identity_correspondence20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_correspondence_confirm_sourceL_20261015_r01/resolved_request.json`，唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-identity-correspondence-confirm-sourceL-s2026092701-r01/source_diagnostic`，日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_correspondence_confirm_sourceL_20261015_r01/source_probe.stdout.log`。技术权限/来源/ID/非有限错误保存输出且仅停该run；先核对后恢复，不触碰其他owner。一个focusedcheck、一个P0/P1审查、一个preflight/归档SHA比较/remotecompile/PID-CWD-cmd-GPU-loggrowth及独立终态读回；详细matrix/stop/artifacts见[登记](experiment.json)。

这里待验证的贡献是实际部署特征上的真实跨RX身份对应学习、成本项中性化和满足对应质量即停压，不宣称softmax或中心化数学本身首次提出，也尚无完整首创性证明。SIG_TIME/CENTER25K/COMPACT_LINEAR/LONG_SILU历史正收益按原配方独立保留；84%默认头clean三seed目标仍active、未达成。


## 完整实际结果

执行 `1f65c70457bc2e85218a2512f2b1df8d1fb2ff2b`，提交/推送/remoteOID、归档SHA、remotecompile、realCP noquery smoke、所属PID/CWD/cmd/GPU/loggrowth及自然终态均VERIFIED，无failure。源wall 11.76 s，完整12单元、8对照及60RX记录。

| ordinal | CROSS−对照 | trueROW质量差 | 对应错数差/720 | ACTUAL CE差 | PRE_W CE差 | accuracy差 |
|---|---|---:|---:|---:|---:|---:|
| 2 | ZERO | +0.002490 | -1 | +0.000290412 | +0.00029179 | +0 |
| 2 | SAME_ROW | +0.001722 | -2 | +0.000274993 | +0.000275653 | +0 |
| 19 | ZERO | +0.002105 | -2 | +0.000184022 | +0.000185219 | +0 |
| 19 | SAME_ROW | +0.001115 | -2 | +0.000104428 | +0.000104811 | +0 |
| 37 | ZERO | +0.003512 | -2 | +0.000304369 | +0.000306836 | +0 |
| 37 | SAME_ROW | +0.002516 | -2 | +0.000255862 | +0.000257973 | +0 |
| 54 | ZERO | +0.002025 | -2 | +0.000295581 | +0.00029708 | +0 |
| 54 | SAME_ROW | +0.001833 | -1 | +0.000175395 | +0.000176196 | +0 |

跨RX身份对应信号已独立复现：相对ZERO，trueROW质量四增.002025至.003512，720匹配错数少1/2/2/2；相对SAME_ROW，质量四增.001115至.002516、错数少2/2/2/1。aux梯度norm.3114至.4677，actualF差.00406至.00651，68至70个实际featuretensor及17至18个浅中层tensor改变，不是空图或默认W换头。

但是，“对应更好”没有稳定带来默认头收益。CROSS_ROW相对ZERO的ACTUAL heldCE四组均高.000184至.000304，PRE_W也高；相对SAME_ROW的CE也四高.000104至.000275。所有读出held准确率仍100%，所以这里只是confidence/CE损害，不能把它改称源分类错误增加。固定1e-6 CE非损害条件全部未通过，资格0/4。rawcos身份间隔也四组下降。上一组四个CE正结果仍完整保留，但无法跨新的physical子集复现，不能据此发布E20+20正式矩阵；不松CE容限、不扫λ/τ/π/LR、不筛掉困难held包。

判断分两层：新真L身份对应任务确实提供了与原主CE不同的特征学习信号；但当前成本表归一化匹配目标没有要求这些变化继续支持既有默认分类器，因此独立匹配改善与默认decision confidence存在稳定分离。这是源工作假设，不是target泛化因果证明。不能宣称所有对比学习无效，也不能将ROW自身指标提升当成84%目标进展。

下一步优先研究**直接由默认头检验的跨RX身份证据转移**，而不是继续制造更大的匹配梯度或再删除RX方向。一个最小候选是：用当前合法L、同day、六TX平衡group的实际joint均值表示公共RX成分，构造donor的身份残差+anchor的公共成分，即 z_t,y−mean_y(z_t,y)+mean_y(z_r,y)，由真实身份标签和现有默认cos头给停止式监督；同RX两包对应作等预算对照。所有均值仅当前L loss内计算，不进入部署或持久prototype，不复制教师关系、不用U真值。它要检验class条件证据在另一公共接收成分下能否保持原分类功能，不要求先把两个RX原特征几何全部对齐。尚未实施/发布，source可用性、actualCE方向、均值估计偏差及梯度捷径必须实测；不预称创新成立或有效。

真实900uniqueL/6300metadata/3062packet网络前向/24临时CPUAdamW，0实际训练/头拟合/EMA/U/V/target。旧900与新900完整ID/baseindex不交，held均属父已见L；不是独立DG或target盲测。sourcecode/config与完整标量分析已存，[12单元](source_cells.csv)、[60RX记录](source_RX_metrics.csv)、[8对照](paired_regimes.csv)、[结果](result_evidence.json)、[运行核验](launch_evidence.json)。历史正路线SIG_TIME/CENTER25K/COMPACT_LINEAR/LONG_SILU仍独立保留；84%默认头三seed目标active、未完成。
