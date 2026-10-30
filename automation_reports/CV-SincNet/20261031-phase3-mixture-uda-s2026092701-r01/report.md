# PAMF 背景混合锚定的无标签域偏移

Run `20261031-phase3-mixture-uda-s2026092701-r01`。固定P1 signal r03、原F0、DURA v1.0、source known90/OE32及原两套六阈值。只用现有星地无标签U384估计目标接收机的公共偏移，检验能否改善跨RX已知接收并保护未知拒识。唯一主 `DA_M_ratio`；不是prepared条件密度候选，不重训网络、不重校准阈值。

## 矩阵与已完成控制

三状态 DA0（无适应）、DA_K（仅known90软责任控制）、DA_M（known90＋OE32混合软责任主）均报告 known_support 和 density_ratio。固定六列顺序为 `DA0_known,DA0_ratio,DA_K_known,DA_K_ratio,DA_M_known,DA_M_ratio`，主索引5。两适应状态共享同一个U导出、源prior、半径和20轮预算，不按Q换主。

DA0精确复用dura-best r01 / actualcode `9fc23a4cb78111aa96cc72fde0d35c036a876e5d` 的source_DURA、clean_DURA、star_DURA三条主测试。原主ratio宏A_K/C_K/R_U/H分别为源89.770370/97.107407/60.888889/72.561330%、clean68.923611/85.416667/99.652778/81.487441%、星地63.020833/84.895833/76.388889/69.063927%。clean/star原F0闭集80.034722/72.916667%，原损失11.111111/9.895833pp；仍不合格。本轮只读父prediction，不在producer读取其truth派生score/results。

原F0为Tri-SIG-TIME SSL E50/11100/default6、232993参数；P1 signal为signal-open r03 / code `dc01a402582f93e452dbaa3ef59a9fbb44151818` 的headless232033＋MLP102688＝334721，特征祖先11100＋已有2220、MLP已有2220。DURA固定R69 v1.0、E20/4940/266951、reuse=True/graph=True/native=False、FP32/TF32off。source合同392005及原L/OE合法祖先保持；不加载另训P0网络或目标污染权重。

## 只由源聚合定义的偏移空间

原mode顺序class→RX→day，六类×15域，均值已位于原共享白化坐标。令mu[c,d]为mode均值，先按类中心化，再对六类等权平均形成15个域偏移b[d]，最后减其域均值以去除浮点中心化残差。其SVD非零方向给出正交行基B，方差lambda[j]=s[j]^2/15，半径r=max_d ||b[d]||。rank用FP64 eps×max(shape)×最大奇异值，不由U/Q挑维数。

当前source11聚合的设计期只读检查得到rank14、r约2.370508；未读target U/Q。B、lambda、r在U前冻结，只来自已部署的L聚合，不新读L/OE/V，不用V或query拟合prior。未知真实TX/K/配额不进入prior或任何target算法。

目标坐标w=(unit(signal_z)−origin)W，偏移delta=aB；适应后h=w−delta。只改变开放分数坐标，不改原F0 logits/身份；不对h再次单位化或白化。公共平移保持白化空间内目标两点之差，但不保证cosine、真实身份或拒识性能不变。后续匿名结构未在本轮识别或注册。

## 固定无标签目标与优化

DA_K只用既有known90 Gaussian混合；DA_M用0.5×known90＋0.5×OE32。模式均值、方差和组内权重不更新，背景模式不是未知真实类别。0.5/0.5为事前固定中性组先验，不读取U旧/新真实比例，不估未知类数或quota，不使用原F0硬伪类门控。

最大化完整U的sum log mixture likelihood（含固定白化Jacobian）减0.5 sum_j a[j]^2/lambda[j]，约束||a||≤r。a从0开始，固定20次EM更新，无U/Q早停或追加。E步对90或122个固定Gaussian求软责任。M步的Hessian为S×I＋diag(1/lambda)，其中S为责任/variance的总和；向量项为B乘责任加权残差。若无约束解超源半径，以一个非负Lagrange乘子、固定60次二分精确解球约束，不使用任意坐标裁切。

每状态保存iteration0至20的21条objective、系数/偏移norm、known posterior mass和约束乘子，要求有限及数值非降。posterior mass是模型的无标签估计，不是已知类比例/纯度。拟合结束仅部署delta160；保存coefficients及源prior作可检查聚合，U逐行特征不部署、不Git。source模型参数、密度、prior和原τ始终不变。

源统计只读、目标adapter拟合与NN训练严格区分：本轮NN/source密度/normalizer/阈值更新0，但实际目标统计拟合为两个状态各20次，不称“完全无拟合”。旧hard core正交、方向bias及四类source proxy Huber平移不是本算法，不能继承它们的性能或重复改名。

## 数据、访问顺序与三类 DURA 测试

合法U为既有G10 RX24-1、2021-03-08的384条discovery/public，schema phase3_min_v1/VALIDATED_ONCE，capsule `p3dev-star-g10-6e1292b62ab09286ebf1a0df`、split `83c5cc1fe2f2a1512dbc1287`。只提供manifest、opaque IDs和FP32[384,2,256] received；不提供真实TX、角色、K、quota或场景GT。U入口只接受该discovery角色，不接受Q或clean输入。

实际原F0＋DURA＋P1signal的两条syntheticG10无query smoke先执行，state保持，PASS立即继续。source11、两τ、prior先冻结，再打开U，经同DURA→F0/signal导出384行；core拟合只用signal z160，不用所附F0 logits。两个adapter全部冻结后，才首次打开父source V/NV及任一个Q prediction NPZ。Q的全数组可以为固定推理批量读取，但不计算Q均值/尺度/责任/τ或更新状态。

| 测试条件 | DA0 | DA_K / DA_M |
| --- | --- | --- |
| source_DURA，已见5RX | 精确父source_DURA V27000/novel3150 | 这些source RX未注册目标adapter，delta=0；六列精确重复两父列 |
| clean_DURA，跨RX24-1 | 精确父clean_DURA Q864 | 用同一star-U学得的target delta，不读clean U或重新拟合 |
| star_DURA，跨RX24-1 | 精确父star_DURA Q864 | 用对应相同target delta，逐记录开放分数重算 |

三条件均真实包含冻结DURA路径，parent中的DURA身份与本次smoke核对；不以绕过G的clean替代。原F0 closed/defaultlogits/baseline在全部状态逐值保持，闭集保护基线来自同DURA输出。source域用delta0是按实际公开接收机绑定adapter，不按TX/known/unknown真值路由。两个target Q同RX跨日，均使用同star-U状态，不能按测试条件再选adapter。

源V旧六TX各4500，曾用于父阈值校准，本轮不再校准；novel19-15/19-16/7-1各1050。source RX1/3/4/6/8为1-19/18-2/19-2/2-19/3-19，日期03-08/15/23均已见。原L6300/OE6134/Us56700仅作祖先，本轮新读0。source novel历史曝光，source已知风险不是独立盲测。

两个target Q为同物理864条、RX24-1(index26)、2021-03-15，旧六类各96/未知三类各96仅builder/scorer知。U与Q物理互斥且跨日；日期已在源RX出现，不称严格未见日期。clean/star是相同物理角色的两条件，三状态不增加独立记录。开发Q已历史曝光，正式RX24-18未访问。所有新学习IQ仍为星地received；clean仅冻结测试。

G10沿MIX01 600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、郊区城区各0.5及原六面积权重，receiver2027/scene2026103001/channel2026104001，一次输入unitRMS、输出不再RMS。已VALIDATED_ONCE角色/capsule/split复用，不重造信道或全IQ重验。

三状态×两读出×三DURA条件全部预测完成后，一个CUDAempty独立scorer先核固定source/τ、DA0精确控制、source六列重复、delta与源subspace/半径和21条trace、全部raw/margin，然后分别连接source novel/cleanQ/starQ truth各一次；不传U truth、不重新EM/NN/τfit。无目标真值、query统计或成绩回流。source失败不取消已登记Q，控制不能替唯一主。

## 指标、判定与成本

主报宏A_K、R_U、H-score，同时C_K、同DURA原F0闭集及损失pp、AUROC/OSCR、逐TX/RX/day/scene/覆盖和描述性区间。主DA_M_ratio配对DA0_ratio及DA_K_ratio，分别报告正确接收/接收/未知拒对gain/loss/net和宏变化；原身份完全相同，不能将新增接受全算正确。

source known12沿父保护、sourceRU90仅开发诊断。跨RX clean整体R_U≥95%，跨RX星地整体及六预登记主景R_U≥90%；两者C_K≥90%、原F0 closed−A_K≤2pp、必需TX覆盖。主DA适应成功还需star A_K较DA0正增且上述目标保护全部满足；objective改善、非零delta或仅接收提高不代表成功。当前小Q的high场景已知缺必需未知TX，保持N/A，不能认证六景；不事后重抽、改主景或放宽目标。

本轮仅REG0的无标签域适应，不产生匿名新类；A_N/H_OW为N/A。单开发RX/模型、历史曝光及缺覆盖均保留，完整多RX/seed、未知发现和未来匿名注册目标继续ACTIVE；任何窄子项不替完整验收。

新增G/F0/signal真实前向各384、synthetic各2；所有Q/source IQ新读取/NN前向0，复用现有固定DURA features。源L/OE/Us/正式RX0，source11/两τ不refit。实际目标fit两个状态各20步；源prior仅已有聚合。活跃神经参数834665（G＋F0＋P1signal）不变，一份密度11数组364340B、两套六FP64τ96B、两个delta各1280B；另报prior/coeff/诊断实际字节及本轮批量时长，不冒充B1或新物理记录。

N607单GPU producer、独立CPU scorer，每次启动free≥4096MiB/每GPU新增本任务≤2，保留其它owner。一次实际code commit、focused角色/真值隔离/EM解析/20步/先freeze后Q/真实CPsmoke、一次直接P0/P1review、最小登记/不可覆盖root/单preflight/archive比较/compile、PID/CWD/argv/GPU/log增长及终态。非法角色/祖先、非有限、明显objective下降、输出碰撞或无法合法预测时保留，不partial评分/自动重启；科学无收益不停止健康任务，UNKNOWN先核原PID/root。

## 实际发布预登记

Actualcode `d72a7b7c301fec2ff3736e4221dfe5047ab78efc`，唯一owner `codex-phase3-mixture-uda-20261031`。环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-mixture-uda-s2026092701-r01-release`，不可覆盖root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-mixture-uda-s2026092701-r01`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_mixture_uda_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-mixture-uda-s2026092701-r01-release/runtime.json`。六列×三DURA条件固定，唯一主DA_M_ratio，不按Q改主或τ。

Focused验证：{"core": {"unique_tests": 6, "initial_passed": 5, "targeted_source_centering_fix_passed": 1, "targeted_seconds": 0.58, "initial_seconds": 1.48, "repeated_full_suite": false}, "producer": {"tests": 3, "subtests": 4, "seconds": 8.22, "stdout": "run_tests.log"}, "scorer": {"tests": 4, "subtests": 9, "seconds": 8.04, "stdout": "score_tests.log", "legacy_numpy_trapz_deprecations": 4668}, "launcher": {"tests": 7, "seconds": 3.01, "tool_stdout_observed": true}, "test_items_total": 20, "additional_subtests": 13, "AST_files": 8, "mirror_matches": 10, "preview_runtime_validate": true, "registry_fields_validate": true, "real_IQ_weights_truth_reads": 0}；一次直接P0/P1独立审查PASS。单次N607preflight核既有DURA父prediction、三模型、合法U384 capsule/split及输出无碰撞；无IQ/private truth实值读取，复用VALIDATED_ONCE。

首步三真实checkpoint的两条syntheticG10无query smoke，PASS立即继续。源prior/原密度/原τ先冻结；starU384经DURA导出，仅signal用于两个20轮MAP-EM适配。两个adapter先冻结再读父source/Q特征；source delta0，clean/star target同starUdelta。三条件六列全完成后独立CUDAempty scorer连接三份truth各一次，无Utruth。新G/F0/signal各384真实行，source/Q新前向0；NN/sourcefit/τfit0，实际目标统计fit2×20。当前未启动，无新性能。原始IQ/权重/U逐行特征/预测/原始日志不Git。
