# 固定 SIG_TIME：语义身份排序 E50 单种子完整结果

**结论：未实现 clean 与 low urban 双增益，不扩展 D3/D4 种子。** D3/D4 已完成 E50、七环境默认头预测及独立 truth-last 评分。low urban 分别为53.4768%/53.3542%，相对普通增强D1仅+0.2054/+0.0827个百分点；clean相对无星地增强D0为−1.3667/−1.3690个百分点。low≥60%未达成。

本报告为实际单seed2026092701，不将720源开发样本或HQ反事实成绩当作正式测试。固定测试基准此前已暴露；本轮从零训练仅用源数据，不声称新的独立盲测。逐seed差、全部七视图及接收机分数保存在实际artifact；未按目标选择父模型、重排、扩种子或重跑。

## 实际基线与实现

固定single SIG_TIME／原内部门控与参考响应／默认cosine6×160、scale30；骨干、原分类头和融合增益按原CE＋旧U流程联合训练。没有新部署分支、分类器或另训诊断头。总模型863730参数，实际身份梯度范围195041参数。SIG使用t2 postDropout→6维local projection→四窗112描述→160投影，原局部和上游参数实际更新。

训练发布commit `63f6bd461834b4def6e3d7cfe4d8e8f00389b13e`；源归因发布commit `f172f98d67b19c65d7246b025a7ba2af770d197a`。实现位于experiments/cvs_urban_semantic50及cvs_urban_semantic_mechanism；逐行配置与命令唯一维护于本目录experiment.json。起始权重为scratch，父训练成本0；D1/D2是预定义的同折原A0/A-J完整产物复用，未新训练、预测或评分它们。

实际源角色L_s6300拆为L_train5580/L_dev720，U_s56700的TX真值隐藏，V27000仅健康观测。源RX1/3/4/6/8、day1/2/3，split/data392005，内部fold2026101503；每TX×RX×day留8/70进L_dev。旧full-L6300权重没有作为初始化；两控制的实际source/completion/frozen/inner契约一致。

采样率25MHz、IQ中心裁剪256点（不是减均值），native RMS后模拟器输入单位RMS，无输出RMS。实际导入experiments/adv3b02_xuc/code/leo_practical/channel.py和residual.py，已在源resolved config记录__file__。residual/post_sync，EQ关闭G=1，quality adaptation、phase tracking、IQ compensation开启，reference_snr_db=30，reflect边界及24点数值FIR延迟保留。正式clean＋六practical既有IQ/信道/SNR/困难样本不变。

每模型从头E50，33轮L＋17轮旧门控U，222步/轮、11100更新；AdamW lr2e-4→1e-6 cosine、wd1e-4、label_smoothing=.01，原generic增强保留。D1–D4相同三训练场景low suburban/mid urban/low urban，clean:satellite=1:1分别均值CE；TX×源RX×状态覆盖完整，RX/day硬件session不由TX标签定义，信道跨epoch/view刷新。U/EMA原流程和额外clean EMA前向保持；不增加U身份关系损失。

D3/D4从epoch6启用真实L关系：同TX正参照与异TX负参照共享RX/day，anchor跨RX；margin=.1，满足即停；stop-grad同次前向keys，每anchor triplet均值后全batch均值。D4为两方向半和；q仅调额外urban约束，不阻断任何有标签satellite CE。固定A0 epoch5源全局梯度0.1规则给系数D3=.6863719838、D4=1.1577435581；2160校准前向在原r01原样复用，r02新增校准前向0。没有同时加入bridge、课程或其他一致性损失。

## 固定正式测试：全部五条件

|条件|内容|clean %|low urban %|mid urban %|urban均值 %|urban Macro-F1 %|最差urban RX %|suburban均值 %|
|---|---|---|---|---|---|---|---|---|
|D0|无星地增强，保留原配方|81.9798|36.7976|44.9696|45.9857|45.0757|35.3681|56.3798|
|D1|普通拼接增强，原 A0 复用|79.8060|53.2714|63.4143|64.1504|63.6986|58.2903|76.8667|
|D2|正配对 J，原 A-J 复用|80.5048|53.5726|63.7738|64.5524|64.1112|58.7611|77.3234|
|D3|单向真实标签身份排序|80.6131|53.4768|63.5893|64.3349|63.7475|58.7194|77.0069|
|D4|双向真实标签身份排序|80.6107|53.3542|63.3482|64.1081|63.5489|58.1333|76.7131|

urban和suburban均值分别为各三环境等权均值，最差RX为三urban平均准确率的接收机最小值。下列差使用同seed，单位为百分点：

|比较|clean|low urban|mid urban|urban均值|最差urban RX|suburban均值|
|---|---|---|---|---|---|---|
|D1−D0|-2.1738|+16.4738|+18.4446|+18.1647|+22.9222|+20.4869|
|D2−D1|+0.6988|+0.3012|+0.3595|+0.4020|+0.4708|+0.4567|
|D3−D1|+0.8071|+0.2054|+0.1750|+0.1845|+0.4292|+0.1403|
|D4−D1|+0.8048|+0.0827|-0.0661|-0.0423|-0.1569|-0.1536|
|D4−D3|-0.0024|-0.1226|-0.2411|-0.2268|-0.5861|-0.2938|

普通增强D1相对无星地D0：low +16.4738、mid +18.4446、urban +18.1647个百分点，clean −2.1738个百分点。新身份排序相对D1有小量clean回升，但没有使clean超过无星地模型；双向D4也没有优于单向D3。D2仅作为预定义旧控制保留，不因本轮目标表而选为新父权重。

### 七视图逐行结果


D0，model seed2026092701，复用=False：

|固定视图|准确率 %|Macro-F1 %|
|---|---|---|
|clean|81.9798|81.3738|
|practical_high|58.8167|57.9490|
|practical_mid|56.3940|55.6281|
|practical_low_suburban|53.9286|53.1641|
|practical_high_urban|56.1899|55.3188|
|practical_mid_urban|44.9696|44.0655|
|practical_low_urban|36.7976|35.8427|

D1，model seed2026092701，复用=True：

|固定视图|准确率 %|Macro-F1 %|
|---|---|---|
|clean|79.8060|78.8236|
|practical_high|78.7982|78.1417|
|practical_mid|76.9119|76.3630|
|practical_low_suburban|74.8899|74.3580|
|practical_high_urban|75.7655|75.2148|
|practical_mid_urban|63.4143|63.0687|
|practical_low_urban|53.2714|52.8123|

D2，model seed2026092701，复用=True：

|固定视图|准确率 %|Macro-F1 %|
|---|---|---|
|clean|80.5048|79.4834|
|practical_high|79.3649|78.6236|
|practical_mid|77.3262|76.7142|
|practical_low_suburban|75.2792|74.7098|
|practical_high_urban|76.3107|75.6884|
|practical_mid_urban|63.7738|63.4426|
|practical_low_urban|53.5726|53.2025|

D3，model seed2026092701，复用=False：

|固定视图|准确率 %|Macro-F1 %|
|---|---|---|
|clean|80.6131|79.7363|
|practical_high|78.9994|78.3359|
|practical_mid|77.0298|76.4547|
|practical_low_suburban|74.9917|74.4160|
|practical_high_urban|75.9387|75.3719|
|practical_mid_urban|63.5893|63.1167|
|practical_low_urban|53.4768|52.7538|

D4，model seed2026092701，复用=False：

|固定视图|准确率 %|Macro-F1 %|
|---|---|---|
|clean|80.6107|79.7145|
|practical_high|78.6702|77.9754|
|practical_mid|76.7381|76.1461|
|practical_low_suburban|74.7310|74.1463|
|practical_high_urban|75.6220|75.0291|
|practical_mid_urban|63.3482|62.9070|
|practical_low_urban|53.3542|52.7106|

实际7视图每view168000包；五条件共35视图、5880000个固定prediction、630个metric记录，其中新3行3528000个prediction/378记录，旧2行原产物2352000/252原样复用。每row七视图全部prediction固定后独立scorer连接truth，independent recount VERIFIED；没有训练后另拟合头。逐接收机原分数在artifact final/formal_rows中，保留原A0/A-J标识。

## 源域机制：改善了什么，仍缺什么

独立冻结分析仅使用同一L_dev720与共同六源practical视图，固定D1/A0 clean正确集合703；未打开U，V前向0、query读取0，五student/ownEMA state_dict未变，36722前向、83.073758秒、optimizer/EMA/head-fit更新均0。源随机流2026101402不同于HQ诊断2026101504，不能混合两次样本实现的状态计数。

源low urban的D1/D3/D4为66.9444%/67.6389%/68.1944%；固定703集保持率68.2788%/68.8478%/69.2745%。D3修复14、新增10错误；D4修复14、新增7错误，净增4/7个。全部720样本都参与准确率；固定集只用于保持率和修复/破坏。

源low BLOCKED（386包）D1/D3/D4为167/169/174正确，43.2642%/43.7824%/45.0777%；新增噪声<3dB组（206包）为72/70/75正确，34.9515%/33.9806%/36.4078%。D4只缓解少量BLOCKED／低质量错误，不能视为强恢复。状态、lock、质量分层重叠，不相加解释为独立作用。

同RX/day正负参照的joint平均身份差由D1的.178083升至D3 .258420、D4 .259873；然而原头low margin/30由.095764降至.089000/.088870。最近错误TX相似度、分支方差和gain贡献均单独报告，不把平均几何差当作默认头改善。两个方向全体均值近乎相同，来自平衡集合下的对称聚合，不能用它区分两个方向的训练效果。

实际低urban reference贡献范数/joint范数D1=.237、D3=.149、D4=.141；SIG为.130/.103/.104，而平均绝对gain基本保持。这是表征幅度和融合使用量的变化，不能证明每条分支更可靠；比例不是可相加的贡献百分数。

实际训练epoch6–50抽取批次的extra/satellite梯度范数比中位数D3=.062416、D4=.069959，cosine=.116883/.145012；额外W/三gain梯度0，原CE/U仍更新原头和gain。冻结eval配对梯度中D3对clean cosine中位数−.040148、D4+.080656；eval探针不是实际Dropout/clip/AdamW写回，正cosine并不保证clean增益。

预登记final50源规则：clean相对D0≥+.5pp，low相对D1≥+1pp，urban≥0、worstRX≥−1pp。D3/D4的clean均−.5556pp，low为+.6944/+1.2500pp，两者FAIL。source_family_decision.json先于本汇总读取目标分数固定，未事后降门槛；不扩seed，跨seed仍NOT_RUN。

## 全量日志、成本和技术负结果

|条件|新训练epoch/更新|原训练秒|本轮训练秒|主训练前向|源开发前向|clip次数/11100|
|---|---|---|---|---|---|---|
|D0|50/11100|2620.404159|2620.404159|4769400|70560|640|
|D1|0/0（复用）|4593.410037|0.000000|6190200|70560|155|
|D2|0/0（复用）|4811.223438|0.000000|6190200|70560|193|
|D3|50/11100|4703.145112|4703.145112|6190200|70560|203|
|D4|50/11100|4797.166805|4797.166805|6190200|70560|220|

主训练前向不含源健康V、source-dev面板、smoke和测试：每row原native V为52×27000=1404000；初始/最终checkpoint smoke各2，source-dev70560，formal预测1176000另列。D0无satellite前向，因此其前向数低于D1–D4，不能宣称完全等计算量；更新数和原训练配方相同。全部计数/原始完整50轮/梯度/参数写回见training_audit.json；实际代码读取新33300步，复用原控制22200步，不冒充55500步新训练或新读取。

全新3行numeric step指标全有限、无跳过更新/隐藏U真值；50轮native V完整27000，cache106hits/0miss。stdout里250次inactive诊断nan（未启用sat/domain/query项），已与实际非有限训练分开；无技术错误、恢复、回滚或early-stop。实际t1/t2/SIG local/PA/reference/默认头/三gain组更新范数均非零。E50预算结束不等于已收敛。

r01 D3/D4在首步CPU metadata/CUDA labels mask错误退出，0完整epoch，保留产物，失败setup成本未完整计时，不写成零墙钟。修复后发布新r02；原r01 D0健康且自然完成E50/11100、2327.915211秒，因为关闭generic增强不匹配而在读目标分数前排除，仅独立诊断，额外成本保留。没有停止/热改/重启任何健康任务。

一次原P0/P1及对应问题定向复核通过；匹配恢复8检查PASS6.717s；冻结源5检查PASS.831s；完整日志3负测PASS.066s。远端compile、单archive传输比较和真实CP smoke通过。源job PID3116403，独立PID/CWD/argv及日志81→218字节、complete/results与进程自然退出已核实。图首次物理RX ID整数排序错误仅影响出图，改用原物理字符串排序后成功，未重跑训练/预测/评分。

## 实际图、结果文件和命令

![formal_per_seed_accuracy.png](E:/type10-7/local_artifacts/cvs_semantic_augmentation_recovery_20261015/figures/formal_per_seed_accuracy.png)

![formal_receiver_strata.png](E:/type10-7/local_artifacts/cvs_semantic_augmentation_recovery_20261015/figures/formal_receiver_strata.png)

![complete_training_curves.png](E:/type10-7/local_artifacts/cvs_semantic_augmentation_recovery_20261015/figures/complete_training_curves.png)

![source_learning_progress.png](E:/type10-7/local_artifacts/cvs_semantic_augmentation_recovery_20261015/figures/source_learning_progress.png)

![source_default_head_retention_margin.png](E:/type10-7/local_artifacts/cvs_semantic_augmentation_recovery_20261015/figures/source_default_head_retention_margin.png)

![source_identity_geometry.png](E:/type10-7/local_artifacts/cvs_semantic_augmentation_recovery_20261015/figures/source_identity_geometry.png)

![source_state_quality_strata.png](E:/type10-7/local_artifacts/cvs_semantic_augmentation_recovery_20261015/figures/source_state_quality_strata.png)

![source_gain_contribution.png](E:/type10-7/local_artifacts/cvs_semantic_augmentation_recovery_20261015/figures/source_gain_contribution.png)

![source_paired_gradient_angles.png](E:/type10-7/local_artifacts/cvs_semantic_augmentation_recovery_20261015/figures/source_paired_gradient_angles.png)

9图均使用实际完整artifact生成并逐图查看。全部七视图CSV、逐RX分数、完整审计、derived_results.json及summary/resources/source_family_decision保存在 E:/type10-7/local_artifacts/cvs_semantic_augmentation_recovery_20261015/final；源明细与resolved config在 E:/type10-7/local_artifacts/cvs_semantic_mechanism_20261015。不提交模型、IQ、原始日志或生成图到Git。

实际本地执行（CWD为本工作树，F:/App/miniconda3/python.exe）：
```text
-X utf8 -m experiments.cvs_urban_semantic50.full_log_audit --output E:/type10-7/local_artifacts/cvs_semantic_augmentation_recovery_20261015/final --legacy-audit E:/type10-7/local_artifacts/cvs_branch_alignment_20261015/final/training_audit.json
-X utf8 -m experiments.cvs_urban_semantic_mechanism.figures --source-root E:/type10-7/local_artifacts/cvs_semantic_mechanism_20261015 --training-root E:/type10-7/local_artifacts/cvs_semantic_augmentation_recovery_20261015/final --output E:/type10-7/local_artifacts/cvs_semantic_augmentation_recovery_20261015/figures
```
第二条首次在最后一图失败，修正物理ID后以同参数去掉--source-root重跑训练图分支成功；源5图未改变。远端实际训练和source命令/CWD/GPU分别在两登记experiment.json与preflight/readback，source发布CPU2/CUDA空。

## 未完成与判断

本轮复现了普通星地增强的显著urban适应收益，同时确认clean代价；新增相对身份目标仅改变几何并修复少量源困难错误，固定正式low仅+0.08至+.21pp，未突破60%，未达双增益。没有跨seed证据。不能从源HQ70.97%预告训练/测试可达相同数字。

Bridge-CE/Bridge-S尚未实现或发布；独立cross-RX教师草稿未发布。当前语义家族按源规则结束单seed小试，不扩种子、不按目标表选父权重。后续如继续HQ局部教学，必须独立预登记其源证据、同成本控制和默认头验收，保留本负结果，不把失败家族悄然串成主方案。

追踪表analysis/cvs_semantic_augmentation_traceability_20261015.md：verified15/pending1/deferred2/rejected0/blocked0。S14保留跨seed缺项；S16/S17 deferred；源推进FAIL是已观察的科学负结果，不把它改写成需求已实现。实现严格对应真实标签/同条件参照/双向半和/原W-gain边界；近似项是预定源系数规则和eval梯度，不将这些近似视为真实更新或性能保证。
