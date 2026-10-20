# R57：既有压缩拓扑的源GN响应校正正式结果

VERIFIED/ANALYZED：两行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及两套独立权重CPU严格重载均已核实。gn_full 273765参数，七景较父提高4/下降3，lowurban 67.330357%（较父-0.179762pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版45.8344/45.3467ms，G+F B1父/本版57.1810/57.1344ms；gn_half 273765参数，七景较父提高4/下降3，lowurban 67.588095%（较父+0.077976pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版47.8450/47.4498ms，G+F B1父/本版58.4878/58.0378ms。gn_full相对同宽full-fold控制R55/fold_full七景提高6/下降1，lowurban差+1.005952pp；gn_half相对同宽full-fold控制R55/fold_full七景提高7/下降0，lowurban差+1.263690pp。原指定350521直接父保持，全部正负结果保留；不把容差通过、减参、源域或解析关联当作正式提升或逐细节TX真实性。原完整目标ACTIVE。

状态：VERIFIED / ANALYZED。两固定行完整E20/4940、source1800×7及target168000×7；逐行全预测固定后独立truth-last，所属controller/worker已退出。全部结果保留，不回流调参或选择性重跑。

唯一直接初始化和冻结IQ教师仍为用户指定R46 prune_refine350521，不加载R55训练后学生。原273765拓扑、GN4通道图、R55 source-only full-fold输入权重和3prior/6CG均保持；两行只把同源校准的GN响应校正按1/0.5写入168原GN的weight/bias，没有新参数或分支。固定同宽控制复用R55 fold_full，同E20/global256micro256/directbuffer/pairedclean IQ+.05谱误差+1→.1三stage IQ教师，coarse/action/prior全部联合训练；全L126000/U1134000，0TX/F/V/target优化。原F50/defaulthead仅冻结评测，属于有clean模拟配对的TX无标签源域自监督。方案在读取R55正式分数前固定，不因该分数修改或取消新行。

方法commit `38a1b4da6dd2964967cebb67935ef877df11c837`。完整登记、数据与祖先、命令和停止规则见[登记](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-GN-response-s2026092701-m2-r57/experiment.json)；全部20轮与TX/RX/day见[结果](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-GN-response-s2026092701-m2-r57/results.json)及[逐组CSV](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-GN-response-s2026092701-m2-r57/row_metrics.csv)。

|行|完整参数|较350521减参|R17逐景1pp|父350521逐景1pp|lowurban（%）|
|---|---:|---:|---|---|---:|
|gn_full|273765|21.897689%|True|True|67.330357|
|gn_half|273765|21.897689%|True|True|67.588095|

逐景1pp容差、性能提升与原lowurban70%/其他五景距rawclean84.6761904762%≤5pp分列；不能用均值掩盖单景退化。R44只作完整描述对照。单seed、相关视图、已暴露且人选父的工程基准，不称新独立盲测或干净泛化证明。

|行|场景|准确率（%）|MacroF1（%）|较350521父（pp）|较R44（pp）|较原R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|gn_full|clean|84.258333|83.942644|-0.198214|-0.247619|-0.328571|-702|
|gn_full|practical_high|82.595833|82.153044|+0.143452|+0.177381|+1.276786|+70330|
|gn_full|practical_mid|81.457143|81.056675|+0.235119|+0.447024|+1.879167|+69788|
|gn_full|practical_low_suburban|79.936905|79.619458|+0.316667|+0.560714|+1.940476|+68724|
|gn_full|practical_high_urban|81.495833|81.079585|+0.108929|+0.178571|+1.142262|+70220|
|gn_full|practical_mid_urban|74.386310|74.257762|-0.060714|+0.159524|+0.701190|+67037|
|gn_full|practical_low_urban|67.330357|67.627841|-0.179762|+0.557738|+0.113690|+62212|
|gn_half|clean|84.397619|84.090510|-0.058929|-0.108333|-0.189286|-468|
|gn_half|practical_high|82.458929|81.991808|+0.006548|+0.040476|+1.139881|+70100|
|gn_half|practical_mid|81.229762|80.795673|+0.007738|+0.219643|+1.651786|+69406|
|gn_half|practical_low_suburban|79.757143|79.406731|+0.136905|+0.380952|+1.760714|+68422|
|gn_half|practical_high_urban|81.258929|80.801077|-0.127976|-0.058333|+0.905357|+69822|
|gn_half|practical_mid_urban|74.392262|74.126817|-0.054762|+0.165476|+0.707143|+67047|
|gn_half|practical_low_urban|67.588095|67.644491|+0.077976|+0.815476|+0.371429|+62645|

同宽度GN响应对照：以下比较预登记R55 fold_full控制，保留全部七景差值；额外共用源校准256父+256初始子G输入另计，不以正式结果重新选择父、宽度或筛选方式。

|GN响应校正行|固定full-fold控制|场景|准确率变化（pp）|MacroF1变化（pp）|
|---|---|---|---:|---:|
|gn_full|R55/fold_full|clean|-0.130357|-0.138071|
|gn_full|R55/fold_full|practical_high|+1.591667|+1.749882|
|gn_full|R55/fold_full|practical_mid|+1.666667|+1.814956|
|gn_full|R55/fold_full|practical_low_suburban|+1.723214|+1.839999|
|gn_full|R55/fold_full|practical_high_urban|+1.650000|+1.805557|
|gn_full|R55/fold_full|practical_mid_urban|+1.257738|+1.392616|
|gn_full|R55/fold_full|practical_low_urban|+1.005952|+1.189079|
|gn_half|R55/fold_full|clean|+0.008929|+0.009795|
|gn_half|R55/fold_full|practical_high|+1.454762|+1.588647|
|gn_half|R55/fold_full|practical_mid|+1.439286|+1.553954|
|gn_half|R55/fold_full|practical_low_suburban|+1.543452|+1.627273|
|gn_half|R55/fold_full|practical_high_urban|+1.413095|+1.527050|
|gn_half|R55/fold_full|practical_mid_urban|+1.263690|+1.261671|
|gn_half|R55/fold_full|practical_low_urban|+1.263690|+1.205728|

|行|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|
|---|---|---:|---:|
|gn_full|106138/79477/88150|1389276|30761512|
|gn_half|106138/79477/88150|1389276|30761512|

|行|推理口径|父/本版median ms|本版p95 ms|本版sample/s|父/本版增量峰值字节|
|---|---|---|---:|---:|---|
|gn_full|G_only_B1|45.8344/45.3467|48.2875|21.7862|67194880/1084928|
|gn_full|G_only_B32|80.1159/74.6931|81.3163|434.4778|276216832/227691520|
|gn_full|G_plus_F_B1|57.1810/57.1344|62.7310|17.1655|67194880/1201152|
|gn_full|G_plus_F_B32|92.2725/82.5670|97.3876|382.4236|276216832/227691520|
|gn_half|G_only_B1|47.8450/47.4498|51.6146|20.9397|67194880/1084928|
|gn_half|G_only_B32|84.6823/74.7406|86.2459|427.0246|276216832/227691520|
|gn_half|G_plus_F_B1|58.4878/58.0378|61.6967|17.1366|67194880/1201152|
|gn_half|G_plus_F_B32|93.3887/81.5528|92.8026|386.7736|276216832/227691520|

同卡64条sourceV，G与G+原冻结F分别测B1/B32，多轮预热和3轮交替顺序；全部轮次、p90/p95和驻留/峰值见JSON。计时为同步预加载GPU输入，排除传输/预处理/loader。驻留为三G+一F共享suite，增量峰值对应当前前向，不称独立部署显存；MAC只含G的Conv/Linear，FFT/CG/norm/activation/memory及F不计。参数减少不推定时延或显存同比减少。


源GN校准独立列账：同128物理样本（64L_s/64U_s），received01/20；额外256冻结父与256未校正子G前向、共512次输入，只做一次供两行共用，耗时8.0063s。复用R55同源256父G输入的folding统计，其原3.9369s和曝光不重复计算。0clean/TX/F/V/target/optimizer，父和初始子图状态在收集过程中保持不变。

168个GN分别在16固定空间位置拟合保留通道post-affine/pre-SiLU响应；斜率固定identity-centered ridge1e-3及[0.25,4]界、自由截距，再将alpha1/0.5写入原GN参数。全部拟合来自未校正子图同一遍配对收集；同时应用后下游输入变化，局部MSE不增不能证明完整G、身份真实性或识别收益。局部各module等权MSE 0.37688262→0.27831363；拟合现有affine标量6880，斜率触界24。源索引、系数及投影不作推理输入。

gn_full首轮→末轮：loss 0.1586363→0.023878081；iq_nmse 0.091058261→0.023252929；spectral_error 0.023750381→0.0061110677；teacher_stage_nmse 0.066390516→0.0031959818；中位每轮70.699s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
gn_half首轮→末轮：loss 0.090263959→0.023574033；iq_nmse 0.057485741→0.022979127；spectral_error 0.013450611→0.0060329847；teacher_stage_nmse 0.032105687→0.0029325685；中位每轮68.966s。lowurban70=False；其他五景gap5pp=False。日志类别：[]

全部训练记录、全帧/主体232/尾24 NMSE、FFT误差、F配对clean特征几何、逐TX/RX/day和repair/damage均保留。低重建误差、识别率或F几何都不单独证明每个恢复细节来自输入TX。恢复器0TX/F训练与独立评分使用真值分别说明。

继承120串行pass+本次20=140；每组件累计100、组件轮次和300，原F50训练另计。两行同E20训练预算、同batch256/micro256/directbuffer、同1→.1教师；额外共用256父+256初始子G的GN校准、0optimizer更新，另复用R55既有256父G的源投影统计，不重复计算为两份或隐藏为0数据初始化，均重置优化器并以指定350521父作IQ教师；与旧父相比增加了训练预算及可能的通道裁剪，不把追加训练收益全部归因于减参。

独立恢复器已取回，CPU严格加载、synthetic零IQ有限输出通过。路径local_artifacts/pamf_restoration/r46_gn_response_r57_20261021/models/<arm>/final.pt；使用experiments.pamf_r46_gn_response.runtime.load_checkpoint，float32[B,2,256]输入/同形输出；无需teacher/F即可运行恢复器，接原F50/defaulthead完成识别。权重/数据/原始日志不入Git。总目标继续active，是否性能达标按实际逐景数据判断。


## 固定方案与历史发布记录

# R57：既有 27.38 万参数恢复器的源 GN 响应校正

状态：**VERIFIED / RUNNING**，GPU3/2 两行实际训练。33项聚焦检查、CLIhelp及一次独立P0/P1审查通过。性能优先；主目标仍为 low urban ≥70%、其余五个受损场景距 raw clean ≤5 个百分点，以及输入 TX 细节保真。相对 R17 每景 1 个百分点是退化容差，不代表性能提升。指定 R46 `prune_refine350521` 仍为唯一直接权重与冻结 IQ 教师来源，采用原 7/63/30 物理划分。属于有 clean 模拟配对、使用 U_s、训练不读 TX 的源域自监督，不能称无 clean 盲学习。

本固定方案在读取 R55 正式分数前形成，只针对既有剪枝图的 GroupNorm（GN）响应失配，不建立其他恢复网络家族。两行均为完整 273765 参数、context 185615、原 3 prior/6 CG、GN4、权重筛选通道图和 full-fold 输入权重。候选 `gn_full`/`gn_half` 只分别将源域拟合的 GN 校正按 1/0.5 写入既有 168 个 GN 的 weight/bias；不增加推理参数、分支或校准输入。固定匹配控制复用 R55 `fold_full` 的同宽度、同输入折合及同训练预算结果，不重复训练或继承该训练后的权重。

复用 R55 已有的源通道投影与同一 128 条 source 样本（64 L_s、64 U_s，seed 2026102156，received epoch01/20）；复用的 256 次冻结父 G 前向及其成本单列。本轮额外在冻结、未校正的 full-fold 子图和指定父图各前向 256 条 received IQ，共 512 次 G 输入，一次 CPU2 校准供两行共用。0 clean/TX/F/V/target 读取，0 optimizer 更新；只读取源 received 和角色，保留父、初始子图不变后再应用校正。训练仍完整使用 paired clean 及 U_s，不混淆校准与训练权限。

每个 GN 按原通道图保留的输出通道及 16 个固定不同空间位置配对，拟合父 GN 的 post-affine、pre-SiLU 输出 `y=a*x+b`。斜率使用固定 identity-centered ridge：`a=clip((cov(x,y)+λ)/(var(x)+λ),0.25,4)`，`λ=1e-3*max(mean_channel(var(x)),1e-12)`，`b=mean(y)-a*mean(x)`。全部拟合都在同一个未校正子图上收集；随后把 `1+α*(a-1)` 和 `α*b` 合并进原 GN 的缩放、偏置。先后作用的下游输入会变化，因此局部拟合误差不增不能证明完整 G、身份细节或识别率改善。共享 prior 的长卷积仍共享，各 stage GN 保持独立。

两行均重置 AdamW，原 E20/4940、global256/micro256/direct float32 buffer、paired IQ NMSE＋0.05 relative-power MRSTFT＋冻结指定父的三阶段 IQ 教师（线性 1→0.1），coarse/action/prior 全部联合训练。每行新增 L126000/U1134000、1.26M student 和 teacher 训练输入；继承组件80＋新增20＝100、组件和300、串行140 pass；F50训练另计。GN统计校准是额外的现有参数初始化拟合，不冒充 data-free 或零初始化成本。

每行固定最终权重后自动执行 source1800×7、同卡配对 G/G+F 的 B1/B32 三轮时延/吞吐/显存和正式168000×7：clean、郊区高/中/低及 urban 高/中/低。逐行全部预测固定后独立 truth-last；不按源 F、正式分数或身份诊断筛行、调参、另选父或选择性重跑。恢复器训练/校准不加载 F；真实 no-query smoke 与冻结后评测可用原 F50 默认联合训练头，不重训或替换头。输入同一 restored IQ，由原 F 一致生成 FFT/PA/时域/统计分支。

GPU2/3 每卡至多一个本任务训练/预测 worker，R55 两行实际退出后才开始本轮 GPU 工作，跨两轮最多两条 GPU worker。CPU smoke、校准和独立评分使用 CPU2；正式评分复用唯一 R41 score.lock。普通账户、独立不可覆盖 run/output/release；一次预检/单归档 SHA/compile、真实 checkpoint no-query smoke 通过后继续，核实实际 PID/CWD/argv/GPU/log。精确命令、commit、环境及路径在 experiment.json 预登记后发布。

技术停止规则：错误输入/祖先/物理 ID、源校准非有限或不完整、输出冲突、训练或预测非有限及运行技术失败时保留产物，无自动重试；不终止、热改或重启健康任务。所有正负正式结果保留。单 seed、已暴露工程基准及人选父背景完整披露；关联和识别率均不能单独证明逐个细节属于输入 TX，原完整目标继续 ACTIVE。

REJECTED_EXTRA_GATE：泛化 brainstorming 的另行设计审批与固定 planning 审批不适用；当前用户明确的八项最小流程和既有压缩优化授权优先。本报告直接记录方法、固定矩阵及停止规则，不另设审批、签名、成员哈希、报告哈希、重复审查或数据重验。


VERIFIED/RUNNING：同128源received样本额外256父/256初始子G输入的一次GN校准完成（8.0063s），168原GN拟合的斜率/截距/岭系数/局部MSE独立NumPy复算通过；源calibration 0clean/TX/F/V/target/optimizer，父及未校正子图保持不变后才应用GN初始化。两行实际R46/F50/2Ls noquery smoke PASS、全组件有限非零梯度及严格独立权重重载一致。单归档SHA/compile、controller与两worker实际PID/CWD/argv/CPU2/GPU及首轮日志增长核实，GPU3gn_full/GPU2gn_half各一条；R55两行实际退出保护已通过。保持273765参数，不根据已完成R55正式分数修改本固定方案；仅说明实现运行正常，无R57正式识别收益结论。E20/4940后自动全七景/独立truth-last，原完整目标ACTIVE。
