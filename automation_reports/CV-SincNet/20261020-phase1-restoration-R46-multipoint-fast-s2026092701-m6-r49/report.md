# R49：35.05 万基准的多点压缩与训练吞吐优化

状态：**VERIFIED / RUNNING**，已发布且六版实际训练。40项聚焦检查、CLIhelp和一次限P0/P1审查通过。

用户最新要求“发布多点压缩优化版本，提高显卡利用率，提升效率”。本轮直接从用户指定的 R46 `prune_refine` 350521 源最终权重出发，固定一个同宽吞吐优化控制和五个不同压缩点。原 R48 两行保持健康运行，不热改或重启。普通账户只读核实当前八张3090中GPU0/1为R48，GPU2至7无计算进程后，新增六行各占一张空闲卡，每卡一条本任务worker；新矩阵最多6条，与R48合计最多8条。本扩展来自用户最新多点并行要求，不影响其他owner。

|行|GPU|完整参数|context参数|prior唯一参数|主要差异|
|---|---:|---:|---:|---:|---|
|retain_refine|2|350521|237927|112594|精确复制指定父，训练吞吐优化对照|
|prune_near|3|329885|223191|106694|较保守的全链通道裁剪|
|prune_balanced|4|293069|197727|95342|中等全链裁剪|
|prune_small|5|215049|146755|68294|更小的全链裁剪|
|context_slim|6|268257|155663|112594|优先裁剪context，保留父prior宽度|
|prior_slim|7|310905|237927|72978|保留父context，只缩prior|

仍为原接收链/context/三步共享长权重prior/六次CG架构；按每个GN4组内权重重要性裁剪，正确映射encoder、decode、skip、GN及action全局头。五个裁剪点的宽度在新target访问前固定，和R48原273765点形成更多固定观察点；本轮全部行均正式测试，不依据源或target成绩重排、取消或选择性重跑。每个组件实际宽度由design.configuration唯一维护，不另拟合身份头。

源数据、教师及目标函数与R48相同：原L6300/U56700、20轮received缓存；配对clean IQ NMSE＋0.05三尺度相对功率log1p频谱误差＋三阶段冻结父IQ教师误差线性1→0.1。全部coarse/action/prior联合优化，恢复器训练0TX标签、F权重/输出、V或target。两角色都利用，90%曝光来自U；正确称为TX无标签的模拟配对源自监督，不称无clean盲学习。

固定每行20epochs/4940更新、全局batch256、L126000/U1134000曝光；重置AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/关闭TF32，末批24用真实分母。微批从R48的64改为256，每次全局更新只执行一次微批；以float32原始内存buffer传给Torch，绕过旧tolist和Torch2.1/NumPy2不可用的C桥，不安装或改环境。真实远端合成buffer逐值一致检查通过；梯度累积对照使用同样真实分母核对。数学目标和数据曝光不变，FP32归约顺序可能引入小数值差异，报告同宽控制完整结果及实际每轮时间，不预先承诺加速倍数。

直接父与冻结IQ教师唯一来源：`20261020-phase1-restoration-R44-pruned-refine-s2026092701-m2-r46/prune_refine/native_map_prior/final.pt`，方法commit `bbb8daf817de050b958d217b70d4c33f4aefe341`，固定源E20/4940、参数350521。继承全部上游原7/63/30物理源划分，load前核对实际launch/方案/源训练/严格checkpoint契约，复用READY/EXACT_MATCH数据。每组件继承80轮/本次20/累计100，组件轮数和300、串行训练总140；F50原训练另计。用户在旧七场景揭示后人工指定父，本轮属于暴露工程基准，不称新独立盲测或干净泛化证明。

每行最终固定E20，不源早停/跨行筛选；完成即自动进入源1800×7、同卡推理成本、独立源评分、正式168000×7及独立truth-last评分。正式场景为clean、郊区high/mid/low、urban high/mid/low；全七视图固定后独立CPU scorer才连接truth。原F50/defaulthead只冻结评测，FFT/PA/time/stats由原流程计算，不换头或另训头。原R17和350521父逐景1pp分别报告，lowurban70%、其他五景距rawclean84.6761904762%≤5pp及更强TX真实性目标保持，不用参数量替代性能目标。

同卡分别测G和G＋F的batch1、batch32。batch1每轮预热16、计时64、重复3轮；batch32预热4、计时16、重复3轮；候选顺序按轮交替。报告median/p90/p95、各轮中位数、吞吐量、驻留/峰值/增量显存、完整参数/文件及Conv＋Linear MAC。64条sourceV重复计时为相关观测，预加载GPU时间不含传输/预处理/loader；三G一F共享suite驻留不称独立部署显存，MAC排除FFT/复数CG/norm/activation/memory/F。新增六卡并发及其他负载可能影响计时，报告局限，不把训练吞吐增加混为单样本推理加速。

每GPU至多一个本run worker，按free≥18GB/每卡总计算进程<2获取，GPU2至7新root独占。CPU2线程、共用R41 score.lock，R48/R49同一任务最多一个scorer；不会启动额外GPU tester。一次归档SHA/远端compile、launcher首步真实父/F50/两条Ls no-query smoke PASS后立即执行。技术失败保留受影响产物、不自动重试，健康行和其他owner不动。单run一个launch owner，原R48自动测试不重复。

最小聚焦检查覆盖来源/预算/权限拒绝、别名冲突、全行自动正式测试、truth晚于全预测、全部组件梯度/精确父控制、新buffer逐值/只读/非连续数组和不同微批梯度等价；一次独立P0/P1审查。精确矩阵、commit、命令、环境/CWD、路径、停止规则和预期产物写入同run登记，不增加其他gate。


方法commit `e96397bc3c7036f085317f9cc0e59074fb6ee24f`。精确命令、环境/CWD、路径、矩阵、预期产物见[登记](experiment.json)。


VERIFIED/RUNNING：实际350521父/F50/两Ls配对IQ noquery smoke PASS，六行全部组件有限非零梯度、teacher/F不更新、checkpoint严格重载一致。归档单SHA/compile及controller/六worker实际PID/CWD/argv/空controllerCUDA/各预登记物理GPU2至7和GPU-PID匹配、首轮曝光、micro256/directbuffer和日志增长已核实。GPU2/3发布前已有各一条其他owner轻负载，按free≥18GB/总进程<2共存；GPU4至7预检空闲；无旧进程改动。无本轮正式性能结果；每行E20后自动七正式测试和独立truth-last。


VERIFIED启动效率观察：六行真实E1/247/L6300+U56700、micro256/directbuffer、0F训练输入；retain_refine E1 78.52s/802.33sample/s；prune_near E1 80.04s/787.13sample/s；prune_balanced E1 53.93s/1168.26sample/s；prune_small E1 50.76s/1241.11sample/s；context_slim E1 53.97s/1167.36sample/s；prior_slim E1 54.58s/1154.17sample/s。GPU4至7五次5s间隔采样平均利用率58.6%至67.4%；GPU2/3为94.8%/89.4%且含其他owner负载。同宽控制E1相对R48的耗时比为2.0600，属于不同物理卡/负载/首轮条件的观察，不是隔离方法加速倍数，不能替代正式识别率或完整推理成本。

[首轮效率观察与局限](startup_efficiency.json)。

## 已完成源阶段的推理成本观察

以下四行已完整E20/4940、源1800×7独立评分及同卡G/G+F/B1/B32三轮测量，正式测试正在运行；这是源训练和成本证据，尚无正式识别结论。全部20轮损失/曝光/LR和profile范围由独立汇总工具核对。

|行|参数|每轮中位s|父/本版G B1 median ms|父/本版G+F B1 median ms|父/本版G+F B32 sample/s|
|---|---:|---:|---|---|---|
|prune_balanced|293069|51.976|42.3437/41.2085|52.3321/51.3496|486.907/529.142|
|prune_small|215049|49.592|42.4602/43.3987|52.2932/53.3125|484.142/577.724|
|context_slim|268257|52.471|42.8052/43.6513|53.1800/53.8204|475.661/507.277|
|prior_slim|310905|52.018|43.4693/43.8420|53.4682/53.8428|480.865/527.205|

完整p90/p95、各轮中位数、显存和Conv/Linear MAC见[source阶段与成本汇总](source_cost_progress.json)。减参未稳定带来batch1加速，下降和变慢全部保留；训练吞吐提高不能混为推理加速。模型仍执行三prior/六CG，不能仅据这些时间断定具体瓶颈。

成本在64条sourceV、预加载GPU的FP32 eval上同步测量，不含传输/预处理/loader；三G一F共享suite驻留不是独立部署显存，增量峰值对应当前前向。GPU负载/温度与相关重复观测限制保留，尚无独立性能置信区间。原70%/5pp/输入TX细节要求继续ACTIVE。
