# CORE90扩展检索与性能差异解释

本报告补充并更正首轮检索。范围仍为2026-08-28至2026-09-11；本轮将对话检索扩展至基线、复现、对照、RIEI/DRIFT及D0/P0等关键词，不再用CORE90命名作为必要条件。检查359个会话文件入口，156个文件命中882条消息，含子任务、续接与本任务；这些不是独立实验数量。另核对N607相关run目录、完整评分、冻结release代码、checkpoint参数及6次训练的全部1200条epoch记录、6份完整stdout。没有重训、重新推理、调参或干预流水线。

## 1.首轮漏列与本轮新增

首轮的“3份完成目标测试的独立重训基线”口径过窄，未完整展开同名基线的其他seed、源域对照和中止记录。本轮确认至少6次CORE90名义基线已完成E200：FCR、U0、H0、B0/392005、B0/392002、B0/392003。其中3次已有完整target评分；两个新增B0只有source V评分；H0已进入TARGET_PREDICTIONS，但尚无完整四场景独立评分。不能将“训练完成”“source评分完成”“target评分完成”混算。

|新增/补充记录|实际状态|测试/验证边界|
|---|---|---|
|core90_game_source_20260911_d5f7e5db/runs/B0_seed392002|scratch、E200、final_only，完整200轮及source评分|V=27000/scene；未找到本行target评分|
|同批B0_seed392003|scratch、E200、final_only，完整200轮及source评分|V=27000/scene；未找到本行target评分|
|phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_20260901_r1/ADV3B02_ECRS_R0|latest checkpoint为E21；用户取消共享R0|未完成E200/最终测试，不能写为“没运行过”|
|BiCAD Quick24 r3的D0×fold1/8×seed392001/2/3|6行U5000，实际E97，四场景source结果完整|ADV3B02-compatible对照，训练4RX；不是SRC5目标基线|
|PairBiCAD P0–P4 r2的P0×fold1/8×seed392001/2/3|6行U4000，source结果完整|源LORO，非168000目标测试|
|RIEI_ALLSRC_S392002、DRIFT_ALLSRC_S392002|2行E200，source best选模后目标测试完整|目标168000/scene，但实际训练day索引[0,1,2]|
|RIEI/DRIFT fold1/8旧批次|4行仅107/107/120/118轮，无final字段|不能将预登记200轮误作完成|
|FCR-V2的C0|同一FCR权重再次评分|四场景整数正确数与原FCR完全一致；另一次评测，不是另一次训练|

两组新增B0的源验证结果如下，单位%。不要把约98%的source V与约74%的target分数放进同一排名。

|基线|source clean|clear|low-elev|rain|LEO均值|
|---|---|---|---|---|---|
|B0_392002|98.0407|89.4519|85.4852|85.5815|86.8395|
|B0_392003|98.0185|89.4852|85.0481|84.6704|86.4012|
|B0_392005|98.2148|87.9630|84.8259|84.5259|85.7716|

这两个B0来自同一“制定ADV3B02 CORE90实现计划”任务中的先行双seed矩阵，之后才有seed392005批次。两个版本的U采样实现不同，三seed数据不能直接拼成“固定同一实现的多seed统计”。

D0的6行source clean均值55.5565%、LEO均值29.1867%；P0分别54.2361%、29.1978%（精确逐行值见附件）。这些是留一源接收机和较短update预算的结果，不是当前7目标RX×4天的测试。早期失败版本、同权重C0/A1引用以及原报告已列的FastTrust CONTROL八seed继续保留关联，分别标记，避免重复计算。

## 2.先量化：3份目标基线到底差多少

所有行先统一至7个target RX×4天×6TX×1000=168000/scene。

|基线|clean|clear|low-elev|rain|LEO均值|
|---|---|---|---|---|---|
|FCR_ADV3B02|76.2268|61.3679|59.5875|59.4637|60.1397|
|GAME_B0|74.3643|62.9375|61.3351|61.1339|61.8022|
|CROSS_U0|74.7321|60.1482|58.3518|58.7274|59.0758|

统一口径后，clean最大跨度1.8625个百分点；LEO均值最大跨度2.7264个百分点。这是值得追查的差异，但不是十几个百分点的同条件重复波动。若将旧source验证、不同日期训练或混合198000总体一起看，视觉上的差距会更大。

U0的198000总体clean77.3005%，剔除30000条已见RX/未见日后为74.7321%，仅总体构成就抬高2.5684个百分点；LEO均值从59.0758%抬到61.6369%，抬高2.5611个百分点。这一部分由整数计数直接解释，并非模型性能改变。

## 3.已确认：同CORE90名称下存在数学实现差异

FCR与U0完整200轮日志和最终checkpoint参数都确认：

|有效训练项|FCR|U0|为什么影响比较|
|---|---|---|---|
|proxy component radius|core_quantile，q=.80|three_sigma|负样本约束使用的类簇边界不同|
|source episode radius|min_three_sigma_core|three_sigma|episode接受/间隔几何不同|
|prototype loss|当时当前实现|隔离历史实现|历史memory项与当前batch可微中心的梯度语义不同|
|历史loss精度保护|当时当前训练实现|legacy代数强制FP32，模型可AMP|半精度角度链路与跳步行为不同|

本轮直接读取FCR的8f1de797冻结release和U0的54afc2bc冻结release，确认FCR的domain/push项由当前z计算可微中心；U0对应项使用memory prototype，update置于no_grad。原型项梯度语义差异已由实际发布代码确认，源码节选保存在expanded_evidence.json。

U0的基线恢复审计明确记载：旧loss共有7个函数/类发生过变化，其中proxy/source radius与PrototypeMemoryBank会改变基线。U0专门绑定历史loss；FCR运行的是9月3日release。故“结构名、参数量、seed和lambda相同”不能推出目标函数与优化轨迹相同。上述是已确认混杂因素，不能把最终1.49pp clean或1.06pp LEO差距全部定量归给某一个损失。

完整曲线也表明分歧始于E1：FCR source V=30.7185%，U0=33.2704%；当时proxy/source episode尚未激活。因此后期几何差异不是全部起因，早期原型、精度和随机执行差异也必须保留。

CRRA是反例：FCR虽残留非零lambda_crra默认值，但use_crra=false，相关实际加权loss在200轮均为0；不能把未激活配置当性能原因。wisig_train_ratio的.20/.10字面差异也不能替代实际L/U/V成员证据：该路径的实际成员和6300/56700/27000计数已经核对一致。

## 4.B0的训练过程也与FCR/U0不同

B0/392005 checkpoint明确amp=false；FCR/U0为amp=true。B0使用独立game_tracking训练入口和历史模型/loss包，普通simultaneous对照关闭额外审计/控制；同一模型seed并不保证新入口生成相同初始化、batch顺序、随机增强或BN/EMA更新轨迹。

冻结release代码进一步确认：FCR/U0的U DataLoader按shuffle=True采样；B0/392005把U分为连续窗口，置乱窗口顺序后跨epoch轮转。两者改变样本邻接关系与伪标签temporal gate输入；B0日志E131记录6268个唯一U样本，FCR/U0每轮记录6272个U消费量。这里“唯一记录数”和“消费量”指标不可直接等同，采样实现差异则已由代码确认。

更早的B0/392002和392003每epoch重建不shuffle的U loader，再随49个L batch只取49×128=6272条；按冻结代码顺序会反复使用U前缀，而后续392005版本已改成跨epoch轮转。因此不能将这三个seed间差异全部叫随机seed波动。这是历史执行差异记录，本次不修改或补跑。

完整日志显示FCR/U0/H0均尝试9800次主更新，分别接受9789/9790/9790次，非有限梯度保护跳过11/10/10次，无非有限loss跳步；三个B0均接受9800次。六份stdout均无Traceback、RuntimeError、OOM或Killed指纹。它们没有明显持续无更新故障；不能把约0.1%的跳步直接换算成准确率损失。三个B0的实际L/U/V成员集合也已逐项核对相同，变化发生在成员被送入训练的顺序/覆盖方式。

## 5.LEO评测随机实现未匹配

三个基线都使用同名leo_*_weak，但U0保存的子流seed为2027/3036/4045；B0目标manifest的augmentation_seed=392002；FCR predictor按包含clean的场景序号偏移seed。因此同一物理样本在不同评测入口接收到的LEO扰动并未证明相同。评测batch切分、随机数消费及生成实现也要匹配，不能只改一个seed字段就宣称received IQ一致。

该因素只能解释LEO差异的一部分，不能解释clean差距。本次未重新生成IQ或推理，无法估计它贡献了多少个百分点。与之相对，FCR原评测与FCR-V2/C0复测的正确数128061/103098/100107/99899完全一致，支持这条固定权重/评测路径可重复，未发现评分器随机漂移的证据。

## 6.总体差距集中在哪些接收机

每目标RX权重都是1/7，因此可以准确拆分总体百分点差值。

|比较|RX|该RX clean差值pp|对总体差值贡献pp|
|---|---|---|---|
|GAME_B0-FCR_ADV3B02|2|-5.4500|-0.7786|
|GAME_B0-FCR_ADV3B02|9|-8.8375|-1.2625|
|GAME_B0-FCR_ADV3B02|11|-4.1250|-0.5893|
|CROSS_U0-FCR_ADV3B02|2|-2.6958|-0.3851|
|CROSS_U0-FCR_ADV3B02|7|-4.0583|-0.5798|

B0相对FCR的clean总差值−1.8625pp，仅RX9就贡献−1.2625pp，约占净下降的67.8%；同时有其他RX改善。这不是所有接收机等幅退化。U0相对FCR在RX7的clean下降约4.06pp，对总体贡献约−0.58pp。完整逐场景贡献见receiver_gap_contributions.csv。它定位失效分布，但不等于已证明某个物理硬件因素导致差距。

三份基线source V clean约97.83%—98.21%，target约74.36%—76.23%；source验证仍是已见RX/已见日期中的另一些signal记录，与未见RX泛化是不同难度。source分数接近并不能保证target一致。

## 7.新增RIEI/DRIFT揭示了日期标签歧义

两方法全source批次完整日志的CONFIG-DOMAINS明确为train_days_idx=[0,1,2]，实际日期2021_03_01/03_08/03_15，heldout为索引3（03_23）。CORE90则训练索引[1,2,3]，实际日期03_08/03_15/03_23，heldout为索引0（03_01）。文档都可能写“day1—day3”，物理契约却不同；同样的角色比例和数量不足以证明可比。

|方法|source日期索引|checkpoint|target clean|target LEO均值|
|---|---|---|---|---|
|RIEI全source|[0,1,2]|source V最佳E180|54.1780%|53.7637%|
|DRIFT全source|[0,1,2]|source V最佳E191|69.1470%|59.3190%|

这两行确实是相关的已完成基线复现，首轮只描述拟议矩阵不够完整；但它们不是CORE90同源成员、同预算更新、同选模规则的严格对照。D0/P0的LORO结果、FastTrust四日期源验证、旧CORE90 joint_safe选择结果也须分别保留测试口径，不混作同一列。

## 8.结论与下一步验证边界

目前能确认的解释依次是：测试总体构成不同；同名基线的损失实现与精度不同；U采样和训练随机流不同；LEO输入扰动未固定；跨RX泛化误差集中于部分接收机。更广的复现集合还存在真实训练日期、source覆盖、预算和checkpoint选择差异。没有证据支持“仅因为seed不同”或“仅因为GPU并发”这种单因解释；并发首先影响墙时，本次没有证明它造成准确率偏差。

严格分离原因需要另行固定一种基线实现、相同source成员与批次/RNG，再使用同一份预生成received IQ评测固定checkpoint；只读配对预测可量化错误翻转，单因素source诊断可检查loss梯度。当前target已经读过，不应用本次结果挑选实现/超参数后再将同target作为新的无偏确认集。本报告只解释和归档既有结果，没有启动任何新实验。

## 附件与原始来源

- [新增实验逐行记录](expanded_experiment_records.csv)
- [B0源验证总体/RX/day/TX明细](source_baseline_metrics.csv)
- [D0/P0逐行、逐场景、逐TX结果](additional_controls_metrics.csv)
- [RIEI/DRIFT目标LEO总体](additional_target_metrics.csv)、[RIEI完整卫星分组表](riei_satellite_detailed_metrics.csv)、[DRIFT完整卫星分组表](drift_satellite_detailed_metrics.csv)
- [6次训练1200轮提取表](training_epochs_curated.csv)、[实际checkpoint配置对照](config_comparison.csv)、[接收机差值贡献](receiver_gap_contributions.csv)
- [扩展证据：完整评分、冻结代码节选、日志解析、checkpoint配置](expanded_evidence.json)
- [原三基线完整目标指标与物理记录目录](report.md)
- [U0历史loss恢复审计](E:/type10-7/code/snapshots/core90_cross_response_20260911_wt/analysis/CORE90_CROSS_RESPONSE_BASELINE_AUDIT.md)
- [BiCAD完整报告](E:/type10-7/automation_reports/CV-SincNet/phase1_adv3b02_bicad_xr_quick24_seed3_u5000_20260831_r3/report.md)
- [PairBiCAD完整报告](E:/type10-7/automation_reports/CV-SincNet/phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r2/report.md)
- [RIEI/DRIFT全source原报告](E:/type10-7/automation_reports/CV-SincNet/phase1_riei_drift_newsplit_allsource_s392002_20260901_r1/report.md)
- [ECRS共享R0取消记录](E:/type10-7/automation_reports/CV-SincNet/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_20260901_r1/report.md)

远端最后完整证据读回时间：2026-09-11T11:14:28.894424；H0状态为该时点快照。
