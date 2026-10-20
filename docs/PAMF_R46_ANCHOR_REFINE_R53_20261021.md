# R53：持续IQ教师约束的正式性能与推理成本

VERIFIED/ANALYZED：三行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及三套独立权重CPU严格重载均已核实。prune_decay 273765参数，七景较父提高6/下降1，lowurban 68.527381%（较父+1.017262pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版44.1931/43.3225ms，G+F B1父/本版56.2200/54.1512ms；retain_anchor 350521参数，七景较父提高2/下降5，lowurban 67.636310%（较父+0.126190pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版44.2953/44.1067ms，G+F B1父/本版55.4548/54.3477ms；prune_anchor 273765参数，七景较父提高4/下降3，lowurban 68.347619%（较父+0.837500pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版43.4629/42.8090ms，G+F B1父/本版54.6030/54.1270ms。retain_anchor相对同宽R49/retain_refine七景提高7/下降0，lowurban差+0.814286pp；prune_anchor相对同宽R53/prune_decay七景提高0/下降7，lowurban差-0.179762pp。原指定350521直接父保持，全部正负结果保留；不把容差通过、减参、源域或解析关联当作正式提升或逐细节TX真实性。原完整目标ACTIVE。

状态：VERIFIED / ANALYZED。三固定行完整E20/4940、source1800×7及target168000×7；逐行全预测固定后独立truth-last，所属controller/worker已退出。全部结果保留，不回流调参或选择性重跑。

用户指定350521父保持唯一直接来源；现有350521/273765两宽度，同原接收链/3prior/6CG。retain_anchor/prune_anchor的冻结父三stage IQ教师系数持续为1，prune_decay控制仍1→.1；其余pairedclean IQ+.05三尺度相对功率STFT、全部coarse/action/prior联合、E20/global256micro256均保持。350521衰减控制复用R49 retain_refine，不重复训练；273765衰减控制在R53新训，R48micro64仅历史对照。0TX/F/V/target优化，L126000/U1134000；原F50/defaulthead只冻结评测。属于有clean模拟配对的TX无标签源域自监督。

方法commit `df36c3d3da4e026b747776b65b58d4921dc1a71e`。完整登记、数据与祖先、命令和停止规则见[登记](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-anchor-refine-s2026092701-m3-r53/experiment.json)；全部20轮与TX/RX/day见[结果](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-anchor-refine-s2026092701-m3-r53/results.json)及[逐组CSV](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-anchor-refine-s2026092701-m3-r53/row_metrics.csv)。

|行|完整参数|较350521减参|R17逐景1pp|父350521逐景1pp|lowurban（%）|
|---|---:|---:|---|---|---:|
|prune_decay|273765|21.897689%|True|True|68.527381|
|retain_anchor|350521|0.000000%|True|True|67.636310|
|prune_anchor|273765|21.897689%|True|True|68.347619|

逐景1pp容差、性能提升与原lowurban70%/其他五景距rawclean84.6761904762%≤5pp分列；不能用均值掩盖单景退化。R44只作完整描述对照。单seed、相关视图、已暴露且人选父的工程基准，不称新独立盲测或干净泛化证明。

|行|场景|准确率（%）|MacroF1（%）|较350521父（pp）|较R44（pp）|较原R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|prune_decay|clean|84.528571|84.243911|+0.072024|+0.022619|-0.058333|-248|
|prune_decay|practical_high|82.408929|81.971496|-0.043452|-0.009524|+1.089881|+70016|
|prune_decay|practical_mid|81.348810|80.929018|+0.126786|+0.338690|+1.770833|+69606|
|prune_decay|practical_low_suburban|79.858333|79.494004|+0.238095|+0.482143|+1.861905|+68592|
|prune_decay|practical_high_urban|81.411905|80.991570|+0.025000|+0.094643|+1.058333|+70079|
|prune_decay|practical_mid_urban|75.061310|74.831407|+0.614286|+0.834524|+1.376190|+68171|
|prune_decay|practical_low_urban|68.527381|68.572403|+1.017262|+1.754762|+1.310714|+64223|
|retain_anchor|clean|84.543452|84.243584|+0.086905|+0.037500|-0.043452|-223|
|retain_anchor|practical_high|82.301190|81.851466|-0.151190|-0.117262|+0.982143|+69835|
|retain_anchor|practical_mid|80.898214|80.485980|-0.323810|-0.111905|+1.320238|+68849|
|retain_anchor|practical_low_suburban|79.405952|79.073483|-0.214286|+0.029762|+1.409524|+67832|
|retain_anchor|practical_high_urban|81.206548|80.795877|-0.180357|-0.110714|+0.852976|+69734|
|retain_anchor|practical_mid_urban|74.374405|74.274720|-0.072619|+0.147619|+0.689286|+67017|
|retain_anchor|practical_low_urban|67.636310|67.936572|+0.126190|+0.863690|+0.419643|+62726|
|prune_anchor|clean|84.490476|84.194383|+0.033929|-0.015476|-0.096429|-312|
|prune_anchor|practical_high|82.341071|81.911280|-0.111310|-0.077381|+1.022024|+69902|
|prune_anchor|practical_mid|81.207143|80.808851|-0.014881|+0.197024|+1.629167|+69368|
|prune_anchor|practical_low_suburban|79.764286|79.463331|+0.144048|+0.388095|+1.767857|+68434|
|prune_anchor|practical_high_urban|81.269643|80.863416|-0.117262|-0.047619|+0.916071|+69840|
|prune_anchor|practical_mid_urban|75.030952|74.898483|+0.583929|+0.804167|+1.345833|+68120|
|prune_anchor|practical_low_urban|68.347619|68.578503|+0.837500|+1.575000|+1.130952|+63921|

同宽度教师系数消融：以下只比较预登记对应控制，保留七景全部差值；不按正式分数重新选择父或学生。350521控制来自R49，273765控制来自本轮prune_decay。

|持续约束行|衰减控制|场景|准确率变化（pp）|MacroF1变化（pp）|
|---|---|---|---:|---:|
|retain_anchor|R49/retain_refine|clean|+0.102381|+0.099986|
|retain_anchor|R49/retain_refine|practical_high|+0.419643|+0.484957|
|retain_anchor|R49/retain_refine|practical_mid|+0.472024|+0.533882|
|retain_anchor|R49/retain_refine|practical_low_suburban|+0.469048|+0.523330|
|retain_anchor|R49/retain_refine|practical_high_urban|+0.444643|+0.498154|
|retain_anchor|R49/retain_refine|practical_mid_urban|+0.643452|+0.663369|
|retain_anchor|R49/retain_refine|practical_low_urban|+0.814286|+0.802679|
|prune_anchor|R53/prune_decay|clean|-0.038095|-0.049528|
|prune_anchor|R53/prune_decay|practical_high|-0.067857|-0.060216|
|prune_anchor|R53/prune_decay|practical_mid|-0.141667|-0.120166|
|prune_anchor|R53/prune_decay|practical_low_suburban|-0.094048|-0.030672|
|prune_anchor|R53/prune_decay|practical_high_urban|-0.142262|-0.128155|
|prune_anchor|R53/prune_decay|practical_mid_urban|-0.030357|+0.067076|
|prune_anchor|R53/prune_decay|practical_low_urban|-0.179762|+0.006100|

|行|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|
|---|---|---:|---:|
|prune_decay|106138/79477/88150|1378332|30761512|
|retain_anchor|136890/101037/112594|1687260|39081004|
|prune_anchor|106138/79477/88150|1378332|30761512|

|行|推理口径|父/本版median ms|本版p95 ms|本版sample/s|父/本版增量峰值字节|
|---|---|---|---:|---:|---|
|prune_decay|G_only_B1|44.1931/43.3225|65.2033|20.2011|67194880/1084928|
|prune_decay|G_only_B32|94.5927/89.7459|98.4629|385.7593|276216832/227691520|
|prune_decay|G_plus_F_B1|56.2200/54.1512|75.3409|16.9726|67194880/1201152|
|prune_decay|G_plus_F_B32|105.8081/93.7408|99.0755|385.0358|276216832/227691520|
|retain_anchor|G_only_B1|44.2953/44.1067|45.4732|22.6591|67194880/67194880|
|retain_anchor|G_only_B32|58.3138/58.2915|59.2950|548.8869|276216832/276216832|
|retain_anchor|G_plus_F_B1|55.4548/54.3477|57.0154|18.2470|67194880/67194880|
|retain_anchor|G_plus_F_B32|67.9548/67.4781|69.9772|472.5757|276216832/276216832|
|prune_anchor|G_only_B1|43.4629/42.8090|43.7215|23.3329|67194880/1084928|
|prune_anchor|G_only_B32|57.9682/52.2611|53.6952|611.0934|276216832/227691520|
|prune_anchor|G_plus_F_B1|54.6030/54.1270|55.5921|18.4777|67194880/1201152|
|prune_anchor|G_plus_F_B32|67.7278/60.9471|62.6041|522.1784|276216832/227691520|

同卡64条sourceV，G与G+原冻结F分别测B1/B32，多轮预热和3轮交替顺序；全部轮次、p90/p95和驻留/峰值见JSON。计时为同步预加载GPU输入，排除传输/预处理/loader。驻留为三G+一F共享suite，增量峰值对应当前前向，不称独立部署显存；MAC只含G的Conv/Linear，FFT/CG/norm/activation/memory及F不计。参数减少不推定时延或显存同比减少。

prune_decay首轮→末轮：loss 0.13370979→0.023484957；iq_nmse 0.079269947→0.02289074；spectral_error 0.019420509→0.0060005；teacher_stage_nmse 0.053468816→0.0029419191；中位每轮50.954s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
retain_anchor首轮→末轮：loss 0.025517714→0.022345617；iq_nmse 0.02415097→0.021504429；spectral_error 0.0063670601→0.0056991055；teacher_stage_nmse 0.0010483912→0.00055623289；中位每轮56.365s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
prune_anchor首轮→末轮：loss 0.13370979→0.025510478；iq_nmse 0.079269968→0.02297493；spectral_error 0.019420515→0.0060431003；teacher_stage_nmse 0.053468798→0.0022333932；中位每轮49.658s。lowurban70=False；其他五景gap5pp=False。日志类别：[]

全部训练记录、全帧/主体232/尾24 NMSE、FFT误差、F配对clean特征几何、逐TX/RX/day和repair/damage均保留。低重建误差、识别率或F几何都不单独证明每个恢复细节来自输入TX。恢复器0TX/F训练与独立评分使用真值分别说明。

继承120串行pass+本次20=140；每组件累计100、组件轮次和300，原F50训练另计。三行同预算、同全局batch256/micro256/directbuffer，教师系数按固定行分别为恒1或1→.1，均重置优化器并以指定350521父作IQ教师；与旧父相比增加了训练预算及可能的通道裁剪，不把追加训练收益全部归因于减参。

独立恢复器已取回，CPU严格加载、synthetic零IQ有限输出通过。路径local_artifacts/pamf_restoration/r46_anchor_refine_r53_20261021/models/<arm>/final.pt；使用experiments.pamf_r46_anchor_refine.runtime.load_checkpoint，float32[B,2,256]输入/同形输出；无需teacher/F即可运行恢复器，接原F50/defaulthead完成识别。权重/数据/原始日志不入Git。总目标继续active，是否性能达标按实际逐景数据判断。


## 固定方案与历史发布记录

# R53：持续 IQ 教师约束的压缩性能优化

状态：**VERIFIED / RUNNING**，两行训练、一行排队。性能优先；不把更小参数、容差通过或源域小幅涨分当作正式性能优化完成。

研究差异来自完整源域目标审计：IQ 重建误差改善可能伴随固定识别器源准确率下降。原衰减1→0.1教师约束可能不足以保持父模型的逐输入响应；这里只检验持续权重1的波形教师约束，不用TX标签、F输出或测试分数设置新损失。该假设可能无效，全部负结果保留。R52两个相位参考均不足，未据此施加C4等变或不变损失。

固定三行：prune_decay273765保留线性衰减；retain_anchor350521和prune_anchor273765教师系数恒为1。其余仍是原pairedclean IQ NMSE＋0.05多尺度相对功率STFT＋三stage IQ教师NMSE，coarse/action/三个prior全部联合训练，原3prior/6CG/GN4结构和输入输出不变，无额外推理分支或参数。两宽度均来自已登记R48完整两点矩阵，直接父继续唯一为人指定R46 prune_refine350521 E20/4940（bbb8daf817de050b958d217b70d4c33f4aefe341），不加载R48/R49派生学生。

同宽350521原衰减控制复用R49 retain_refine，micro256、seed、数据、预算相同，不重复训练。27.38万衰减控制本次新训以匹配micro256；R48同宽micro64结果保留，仅作历史/吞吐对照。原八行全部正式测试已完成。所有新行固定最终E20/4940，无源选模或提前结束，训练后立即source1800×7、同卡G/G+F成本、正式168000×7预测；每row七预测固定后独立CPU truth-last评分，不因其他行未训完等待整批，也不据结果调参、重排或选择性重跑。

原6300L_s＋56700U_s全部使用，TX标签对恢复器隐藏；原7/63/30物理划分/R07缓存epoch01..20、model2026092701、split392005、data2026102001、augmentation2026101912、formal2027均不变。每行新增20轮/4940更新/global256micro256/直接float32buffer/ResetAdamW，LR2e-4暖247至2e-5、WD1e-4、clip1、FP32/TF32关闭。每行L126000/U1134000曝光、1260000学生和父教师IQ，读取TX/F/V/target训练信息为0。继承每组件80轮，新增20轮后每组件100、组件和300、串行累计140；F50预训练成本另计。

源source_cache READY/EXACT_MATCH和R46/R44/R38/R17/R07/R04源祖先契约复用；R46checkpoint来源已核实原source-only final，F50默认头仅训练完成后读取评价，不另训头。属于有clean模拟配对的TX无标签自监督，不能称无clean盲学习。正式基准已暴露，保留工程比较背景，不声称新的独立盲测或逐细节TX真实性。

按当前两卡约束仅物理GPU2/3，各一个本任务worker；prune_decay和retain_anchor并行，prune_anchor在GPU2排队，GPU worker最多2。每行CPU2，共享原score.lock最多1个本任务scorer。2026-10-21只读资源检查全卡空闲；实际一次N607 preflight和启动后PID/CWD/argv/GPU/日志另存。新不可覆盖release/run/output，唯一owner codex/root/R46-anchor-refine，其他owner及历史产物保持。低准确率不触发技术停止；直接权限/输出碰撞/错误checkout或split/无法合法预测等系统技术失败保留受影响输出、无自动重试/热改/回退，不停止健康同伴。

32项聚焦检查与CLIhelp通过，唯一一次限P0/P1审查见实际review.json；首步真实R46/F50/2Ls配对IQ noquery smoke，PASS立即训练。Git commit固定代码/config；一次预检、单归档两端SHA比较和compile，禁止额外设计审批/签名/成员哈希等门槛（REJECTED_EXTRA_GATE）。精确命令、环境/CWD、路径、三行矩阵与预期产物以experiment.json为准。

正式主要指标是七景accuracy/Macro-F1、相对R17/R46的逐景差、修复/损伤与逐TX/RX/day；原R17逐景最多下降1pp为退化容差，不是性能优先的成功定义。low urban≥70%、其他五受损场景距rawclean84.6761904762%≤5pp以及更强TX细节真实性目标继续ACTIVE。成本次级：同卡64源V，G和G+F的B1/B32三轮交替同步计时、median/P90/P95、吞吐、峰值/增量/共享驻留显存、完整参数/序列化大小/ConvLinear MAC；不含传输/前处理/loader，MAC不含FFT/CG/norm/激活/F，驻留是3G＋1F共享suite，不冒充单模型显存。


VERIFIED/RUNNING：真实350521父/F50/2Ls noquery smoke PASS，三行全部coarse/action/prior有限非零梯度、父/F不更新、checkpoint严格重载一致。单归档SHA/compile与实际controller、GPU2 prune_decay、GPU3 retain_anchor的PID/CWD/argv/CUDA/GPU-PID匹配、首轮曝光、micro256/directbuffer和日志增长核实。prune_anchor按固定计划在GPU2等待；各卡仅一条本任务GPU worker，其他owner不动。尚无正式性能结果；每行完整E20/4940后自动源七视图/成本/正式七视图及独立truth-last，不以负分提前停止。原性能/TX保真目标ACTIVE。

## 首两行完整训练与推理成本观察

以下两行已完整E20/4940、源1800×7独立评分及同卡G/G+F/B1/B32三轮测量，完整源阶段已结束，正式预测按固定管线自动继续；第三行仍排队。这是训练和成本证据，尚无正式识别结论。全部20轮损失/曝光/LR和profile范围由独立汇总工具核对。

|行|参数|每轮中位s|父/本版G B1 median ms|父/本版G+F B1 median ms|父/本版G+F B32 sample/s|
|---|---:|---:|---|---|---|
|prune_decay|273765|50.954|44.1931/43.3225|56.2200/54.1512|327.637/385.036|
|retain_anchor|350521|56.365|44.2953/44.1067|55.4548/54.3477|467.163/472.576|

完整p90/p95、各轮中位数、显存和Conv/Linear MAC见[source阶段与成本汇总](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-anchor-refine-s2026092701-m3-r53/source_cost_progress.json)。参数、实际时延与吞吐分别比较，任何变慢都保留，不预设压缩必然加速；训练吞吐提高不能混为推理加速。模型仍执行三prior/六CG，不能仅据这些时间断定具体瓶颈。

成本在64条sourceV、预加载GPU的FP32 eval上同步测量，不含传输/预处理/loader；三G一F共享suite驻留不是独立部署显存，增量峰值对应当前前向。GPU负载/温度与相关重复观测限制保留，尚无独立性能置信区间。原70%/5pp/输入TX细节要求继续ACTIVE。

## 已完成行的正式测试（2/3）

以下均完整168000条×七景、原冻结F50默认头、预测固定后独立truth-last；其余行继续固定训练/预测，不能把部分行冒充完整矩阵。性能优先，所有负结果保留，不回流选模、参数或父。

|场景|指定父350521|prune_decay（273765）|retain_anchor（350521）|
|---|---:|---:|---:|
|clean|84.456548|84.528571|84.543452|
|郊区高|82.452381|82.408929|82.301190|
|郊区中|81.222024|81.348810|80.898214|
|郊区低|79.620238|79.858333|79.405952|
|urban高|81.386905|81.411905|81.206548|
|urban中|74.447024|75.061310|74.374405|
|urban低|67.510119|68.527381|67.636310|

准确率单位为%。逐TX/RX/day、Macro-F1与repair/damage见[逐行正式结果](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-anchor-refine-s2026092701-m3-r53/partial_formal_results.json)。

|行|R17逐景1pp容差|指定父逐景1pp参考|lowurban≥70%|其他五景raw-clean差≤5pp|
|---|---|---|---|---|
|prune_decay|True|True|False|False|
|retain_anchor|True|True|False|False|

retain_anchor相对预登记同宽衰减控制的七景差（百分点，顺序同上）：+0.102381、+0.419643、+0.472024、+0.469048、+0.444643、+0.643452、+0.814286。

原350521直接父不变；原完整识别与更强输入TX细节真实性目标ACTIVE。单seed、已暴露工程基准；识别率和波形误差不单独证明逐细节身份。完整模型和时延/吞吐/显存随最终交付保留，源域分数不替代以上正式结果。


## 已交付前两行独立权重

prune_decay273765与retain_anchor350521已取回本地，CPU严格加载、原生float32输入形状与有限输出、完整参数和ConvLinearMAC均核实；只使用合成零IQ验证，0实际IQ/F/标签读取。完整七景正式测试已经结束，但第三行尚未结束，不能称全矩阵完成。

权重位于项目根的 `local_artifacts/pamf_restoration/r46_anchor_refine_r53_20261021/models/<arm>/final.pt`，同目录 `README.md` 提供独立G加载用法；直接使用 `experiments.pamf_r46_anchor_refine.runtime.load_checkpoint`，无需教师或源缓存。接同一冻结F50默认头时，从同一恢复IQ一致计算FFT/PA/时域/统计分支，不另训头。数据与权重不提交Git。参数/文件大小/严格加载信息见[交付元数据](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-anchor-refine-s2026092701-m3-r53/partial_model_delivery.json)。

保存全部正负结果和原指定350521直接父，不以单行正式分数重排后续候选。原70%/其他五景gap5pp/更强TX细节保真目标仍未完成。
