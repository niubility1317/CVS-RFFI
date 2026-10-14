# 异 TX 参照的 RX 差分迁移：初期与晚期源原型

状态 **ANALYZED / VERIFIED，INITIAL 1/4、FINAL20 1/4，固定候选未通过**。下文保留原预登记。上一ROW新900确认对应质量四增但defaultCE四差，固定0/4；公共均值转移仅12/17/18/12 active（1.67%至2.5%），held/defaultCE方向不稳定，固定0/4。旧结果和资格保持。这里不提高π或λ，而提出一个不同的功能假设：**另一个TX测得的接收变化，能否用于当前TX而保持身份判断？**

同day，a为RX-r/side0、b为RX-s/side1的六TX组。对每个真实身份y及每个异TX参照c!=y，构造 v_a→b=a_y−a_c+b_c，反向v_b→a=b_y−b_c+a_c。它把TX-c测得的RX变化b_c−a_c加到a_y上，目标仍origin真实y。c遍历全部5异类，逐view分类后再平均，不能先平均δ而回到已关闭公共均值任务。所有3endpoint都来自当前可见L、同day，physical包不假设相同报文。若理想joint=TX_y+RX_r，则view恰为同y的目标RX feature；真实非线性/类相关RX作用和packetnoise可能破坏此假设，必须由真实源证据检验，不称硬件反事实。

辅助使用原defaultcos W.detach/scale30/eps1e-4，π=.9、λ=.1，loss=.1×mean relu(log.9−logp_true)，到π即停压。rawactualjoint三个端点和三个gain全图回传，无新头/prototype/learnedstatistics、没有强迫δ或完整几何相等。部署仍原joint160和原默认头，主CE正常联合更新W；auxW不学习。FP64分析公式/同viewFP32差完整记录，不称额外native前向。

矩阵INITIAL/FINAL20×4currentbank×ZERO/SAME_RX/CROSS_RX，共24单元。每pair两方向×6y×5c=60synthetic decisions，CROSS20orderedRX×3day=60unique pair→3600判定；SAME15unique同RX两physicalside重复4→同样3600。每packet作为classificationorigin/source-reference/destination-reference各20次，共60triple角色曝光；groupendpoint各4。两nonzero目标共7200decision/call，固定各3600分母，ZEROgraphconnected0复用CROSSdiagnostics。active_origin_packet_count在180真实origin物理包中去重，不能把5references或重复SAME当成五倍数据覆盖。目标身份/参照身份/destination及originRX分层、角色曝光、viewnorm/eps、reference差与真实destination同y的重构差只作诊断。

原L900复用current ord2/3,19/20,37/38,54/55和held66/67；原rolesL6300/U56700hidden/V27000、RX1/3/4/6/8/day1/2/3/6TX不变，current与held物理ID不交，父FINAL20见过held。此为源探索数据复用，不冒充新独立confirmation、未见RX或DG。原VALIDATED_ONCE数据handles复用，不做新datahash/ledger；只检查loss输入和父CP合法性，不访问U/V/target。

固定ownBASE_CE firstseed2026092701的原initial_smoke随机模型和final20，父执行6693e4480ef990bc23c5021ff1440b547fe5fcaf；source_proof在两个load前核对完整actualroles/IDs/scratch ancestry/fixedfinalfreeze，initial scratch_only及finalactualargs严格核验。INITIAL保留原groupoptions但moments为空；FINAL保留真实m/v/step，两个阶段同LR2e-4/globalclip5/188parameters，没有历史动量过滤。两个真实CP零IQ烟测各2packet PASS立即继续。源模型eval，一步optimizer只在临时副本，live状态不改。

源资格每state至少2/4 CROSS同时优于ZERO和SAME：current至少180/3600active（5%）、至少18/180不同origin（10%）、实际aux非zero；heldtransfer PRE_W meanCE差<=−1e-4、3600transfer错数不增；originalACTUAL/PRE_W准确率不降，originalCE相对控制增加<=1%。actualF差norm>1e-6、heldjoint RMS差>1e-5及至少1浅中层tensor改变。这里以新任务的**相对1%原CE guard**代替早先候选的绝对1e-6要求，明确在本候选真实结果前固定；旧ROW/mean不会按新规则重判。单步confidence小差不是完整20epoch证据，源门槛仅决定是否值得固定真实训练，不能保证未见RX收益。INITIAL通过优先scratch20，只有FINAL通过支持同父E20+20；未通过不扫π/λ/LR/referenceclass/partialstrength救活。

after全188参数进入真实临时eval模型，after三个gain保留；ACTUAL afterW与PRE_W beforeW分析同一actualjoint，无fit。transferview分别用afterW/beforeW，避免livebeforeW误作ACTUAL。原分类分组CE/accuracy/classmargin、rawcos、F/浅中层/aux-main方向/三个gain梯度、Sinc核变化完整保存；kernel snapshot比较whole CE+aux versuspre，不冒充pureaux核差。没有headonly更新或theta冻结后训头。

预算900uniqueL/6300metadata/6124网络packet前向（4零IQ+2×(180before+4×(180current+3×180after))）/48临时CPUAdamW，0实际训练/头拟合/EMA/U/V/target。额外3600解析view不跑IQ网络，固定synthetic判定预算单列。通过后真实scratch20或E20+20必须同路径纯CE/原hiddenU旧门控SSL/SAME/CROSS，默认头正常joint、finalonly、逐rowauto clean prediction/独立truth-last；当前尚未发布主训练。

run `20261015-phase1-SIGTIME-identity-contrast-transfer-sourceL-s2026092701-r01`，一个owner codex/root/identity-contrast-transfer-20261015，actualcode/config Gitcommit写release_commit.txt。普通N607/CVS-RFFI Python/CPU2/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_contrast_transfer_sourceL_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_contrast_transfer_sourceL_20261015_r01/experiments/cvs_identity_contrast_transfer20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_contrast_transfer_sourceL_20261015_r01/resolved_request.json`，唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-identity-contrast-transfer-sourceL-s2026092701-r01/source_diagnostic`、日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_contrast_transfer_sourceL_20261015_r01/source_probe.stdout.log`。权限/来源/非有限技术失败保存且只停ownedrun，核对后恢复；科学失败关闭fixedcandidate，无targetfeedback，不触碰健康任务。一个focusedcheck/唯一P0/P1 review、一个preflight/归档SHA比较/remotecompile、PID-CWD-cmd-GPU-loggrowth及独立终态。详细matrix/stop/artifacts见[登记](experiment.json)。

较近历史[RX drift](../20261013-phase1-repeated-rx-margin-clean-manysig-m12-r01/final_analysis.md)用同TX=y三折均值估计RX差、单位joint、detach/q/半径约束且辅助W学习，E30三seedMEASURED_MEAN仅+.0242pp/2正、MEASURED_ROBUST−.1302pp/0正，保留为相关弱/负证据。旧CROSS_DIFF只比较类差余弦，不重构默认判别；PA只交换一分支；meantransfer先平均六类。新公式尚无exact已实现重复结果，但库内检索不能证明全局首创。候选价值是具体异TX接收变化的**跨身份功能迁移**，不是feature相减本身创新。历史SIG_TIME/CENTER25K/COMPACT_LINEAR/LONG_SILU正路线独立保留，84%默认头三seed目标active、未完成。


类内变形跨类别迁移已有[Delta-encoder](https://arxiv.org/abs/1806.04734)先例，它通过autoencoder提取/应用类内delta并生成新类训练样本。因此不能把“借另一TX的变化”本身当首创。本候选待检验差异是合法RX/day关系限定的实测参考差、没有learned生成器、actualjoint三端反传及默认判别停止监督。跨RX射频指纹分离也已有[RIEI](https://arxiv.org/abs/2411.03636)研究；这里只做功能约束，不声称已经实现或证明其可分离条件。文献定位不作为额外发布gate。


## 完整真实源结果

执行 `ed5980ab9b5627c6aed80adf74a82365b8249c76`，提交/推送/remoteOID、归档SHA、remotecompile、两个realCP noquery smoke、PID/CWD/cmd/GPU/loggrowth及独立自然终态均VERIFIED，无failure。wall 21.30 s，24cells/16paired/120RX记录。

| state | ordinal | CROSS active/3600 | SAME active/3600 | CROSS distinct/180 | SAME distinct/180 | auxF/main cos |
|---|---|---:|---:|---:|---:|---:|
| INITIAL | 2 | 3600 | 3600 | 180 | 180 | 0.999757 |
| INITIAL | 19 | 3600 | 3600 | 180 | 180 | 0.999771 |
| INITIAL | 37 | 3600 | 3600 | 180 | 180 | 0.999744 |
| INITIAL | 54 | 3600 | 3600 | 180 | 180 | 0.999804 |
| FINAL20 | 2 | 429 | 320 | 106 | 52 | 0.407106 |
| FINAL20 | 19 | 370 | 264 | 90 | 34 | 0.474565 |
| FINAL20 | 37 | 462 | 320 | 107 | 44 | 0.567116 |
| FINAL20 | 54 | 316 | 244 | 81 | 36 | 0.688160 |

| state | ordinal | CROSS−对照 | held transfer PREW CE差 | transfer错数差/3600 | original ACTUAL CE差 | PREW CE差 | ACTUAL CE相对差% |
|---|---|---|---:|---:|---:|---:|---:|
| INITIAL | 2 | ZERO | -0.000447712 | -1 | -0.000422755 | -0.000439648 | -0.0165 |
| INITIAL | 2 | SAME_RX | -9.92267e-05 | +0 | -8.87757e-05 | -9.34924e-05 | -0.0035 |
| INITIAL | 19 | ZERO | -0.000499731 | +1 | -0.000448441 | -0.000467358 | -0.0174 |
| INITIAL | 19 | SAME_RX | -0.000245972 | +0 | -0.000211192 | -0.000221476 | -0.0082 |
| INITIAL | 37 | ZERO | -0.000402381 | +0 | -0.000278624 | -0.000286884 | -0.0109 |
| INITIAL | 37 | SAME_RX | -0.000222222 | +0 | -0.000152754 | -0.000160379 | -0.0060 |
| INITIAL | 54 | ZERO | -0.000233461 | +1 | -0.000197932 | -0.000204663 | -0.0077 |
| INITIAL | 54 | SAME_RX | -9.69078e-05 | +1 | -7.51893e-05 | -7.84878e-05 | -0.0029 |
| FINAL20 | 2 | ZERO | -0.000138251 | -1 | +0.000104704 | +0.000105212 | +1.2331 |
| FINAL20 | 2 | SAME_RX | -9.10043e-05 | +0 | +2.11968e-05 | +2.13044e-05 | +0.2472 |
| FINAL20 | 19 | ZERO | -0.000105525 | +0 | -8.96622e-06 | -8.9848e-06 | -0.1071 |
| FINAL20 | 19 | SAME_RX | -0.000115258 | +0 | -3.99944e-05 | -4.02702e-05 | -0.4761 |
| FINAL20 | 37 | ZERO | -1.31845e-05 | +0 | -1.33903e-05 | -1.3871e-05 | -0.1634 |
| FINAL20 | 37 | SAME_RX | -0.000131892 | +0 | -9.97897e-06 | -9.44772e-06 | -0.1218 |
| FINAL20 | 54 | ZERO | -6.4661e-05 | -1 | +8.11543e-05 | +8.17524e-05 | +0.9425 |
| FINAL20 | 54 | SAME_RX | -9.88254e-05 | +0 | -2.15945e-05 | -2.14724e-05 | -0.2478 |

固定资格：{"passed": false, "stage_signal": null, "qualified_ordinals_by_state": {"INITIAL": [37], "FINAL20": [19]}, "cell_conditions": [{"state": "INITIAL", "ordinal": 2, "qualified": false, "failed_conditions": ["SAME_RX:no_functional_transfer_CE_gain"]}, {"state": "INITIAL", "ordinal": 19, "qualified": false, "failed_conditions": ["ZERO:transfer_error_harm"]}, {"state": "INITIAL", "ordinal": 37, "qualified": true, "failed_conditions": []}, {"state": "INITIAL", "ordinal": 54, "qualified": false, "failed_conditions": ["ZERO:transfer_error_harm", "SAME_RX:no_functional_transfer_CE_gain", "SAME_RX:transfer_error_harm"]}, {"state": "FINAL20", "ordinal": 2, "qualified": false, "failed_conditions": ["ZERO:ACTUAL_relative_CE_harm", "ZERO:PRE_W_relative_CE_harm", "SAME_RX:no_functional_transfer_CE_gain"]}, {"state": "FINAL20", "ordinal": 19, "qualified": true, "failed_conditions": []}, {"state": "FINAL20", "ordinal": 37, "qualified": false, "failed_conditions": ["ZERO:no_functional_transfer_CE_gain"]}, {"state": "FINAL20", "ordinal": 54, "qualified": false, "failed_conditions": ["ZERO:no_functional_transfer_CE_gain", "SAME_RX:no_functional_transfer_CE_gain"]}], "controls_not_fallback": true, "interpretation": "Source opportunity and compatible functional identity transfer, not fulltraining or unseenRX/DG"}

INITIAL有充足数量，但主要是普通分类压力：3600/3600均未达到.9、180/180origin有约束，auxF/main cos四组.999744至.999804，几乎共线。CROSS相对ZERO及SAME的heldtransferCE四组均降低，原ACTUAL/PRE_W CE也四降，但改善很小；vsSAME两组只降低9.92e-5/9.69e-5，未达到固定1e-4，vsZERO另两组transfer错数各多1，固定只ordinal37通过。before原CE3.353994/accuracy19.4444%，三组after原accuracy都23.8889%；不能把共同CE一步的accuracy改善归给新aux。

FINAL20显示明确不同的接收变化学习机会。CROSS当前316至462/3600active（8.78%至12.83%），81至107/180不同origin（45%至59.44%），SAME仅244至320active、34至52origins；CROSS124/165/138/82个错误与SAME40/140/76/44个错误不同（顺序按对应ordinal表），不只是更多重复视图。auxF/main cos .4071至.6882，没有初期的近共线放大。heldtransferCE相对ZERO和SAME四组都降低，但效果1.318e-5至1.383e-4；原sourceCE对ZERO两组改善、两组变差，原accuracy始终100%。只有ordinal19同时满足两控制/CE幅度/错误/原分类健康，ordinal2原CE相对ZERO高1.23%、并且transfervsSAME不够；ordinal37与54仍不足固定transferCE1e-4。固定FINAL仅1/4，不凑组、不把接近阈值当通过，也不改阈值扩到正式训练。

本次资格INITIAL1/4、FINAL1/4均不足2，固定绝对概率停止候选关闭，无source或target选择性重跑。真实异TX参照确实产生了比公共均值更广、晚期更不同的学习信号，这一正证据保留；但source单步没有证明可稳定改善原defaultdecision，更没有完整20epoch或DG证据。before heldtransferFINAL CE.0980904、错112/3600，original CE.0105142、accuracy100%；一个已被原头正确分类的样本在借来的referenceRXdelta下也可能失败，原分类准确不等于这种差分可迁移。

第二优先假设调整**监督内容**，不扫π或λ：只训练迁移新增的风险。新目标拟 h=relu(CE_W(v,y)−stopgrad(CE_W(actual_destination_y,y))−1e-6)，loss=.1×mean h，W继续auxdetach。真实destination同y来自当前L，无teacher伪标签/完整关系复制，三个virtual端点与gain留图，factual风险只作停止门槛。即使原分类尚不可信，也只要求借来的RX变化不额外破坏真实身份；原mainCE负责形成分类。1e-6为预先固定数值容差，relu在相等时也停止梯度。它不把所有低置信样本都强压到.9，针对INITIAL几乎ordinaryCE的诊断；晚期仍要防止通过降低factual身份置信度逃避任务，所以原defaultCE/accuracy必须一起检查。新规则不会拿来重判这次1/4候选。

这一额外风险形式与旧特征sharpness“扰动CE−原CE”的gap思想有联系，不能重新命名成新理论；区别是三个真实source端点定义的异TX接收变化及对应真实destination身份参照，没有parameter adversary、没有新头，也不要求精确重构几何。其可用性/方向/科学资格和训练阶段仍需独立固定在新真实结果前。新候选暂在实现，尚未访问真实源数据或发布；当前24cell产物完整保留。历史RXdrift/PA/CROSS_DIFF/ROW/mean相关负结果不覆盖本functional任务。

900uniqueL/6300metadata/6124网络packet前向/48临时CPUAdamW，0真实训练/头拟合/EMA/U/V/target。原live模型/optimizer/CP/grad/RNG保持；afteractual3gains/正常mainW/auxWNone和每layer/gain数值梯度、Sinc wholeproposal核差完整保存。source900复用，父FINAL看过heldL，无新独立confirmation或未见RX主张。完整[24单元](source_cells.csv)、[120RX](source_RX_metrics.csv)、[16对照](paired_regimes.csv)、[证据](result_evidence.json)、[运行核验](launch_evidence.json)。历史正收益SIG_TIME/CENTER25K/COMPACT_LINEAR/LONG_SILU保留，84%默认头clean三seed目标active、未完成。
