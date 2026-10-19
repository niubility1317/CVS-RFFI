# 冻结原 F 引导恢复的 observed 困难度加权

状态：RUNNING / VERIFIED。目标仍是目标测试集 low-urban ≥70%，其他五环境接近原 clean 85.113889%（Agent 暂按差距≤2个百分点解释）。恢复 IQ 与同条 clean 的差距、接入原 F/default head 的真实 TX 识别分开报告。源改善、NMSE 或检查通过不能替代目标验证。

前序两个 joint30 及固定 IQ 平均均未同时改善源 low/mid-urban。已完成源汇总显示 RX/TX 差异：posterior03 源 RX3 low-urban79.722222%，其他RX86.111111%至90.833333%；其clean也较低96.111111%。这是关联诊断，不证明RX加权有效。本轮直接从合法原R03独立分叉，不继承失败精修或目标成绩挑出的权重。

三组 mean_observed、cvar_observed、rxdro_observed 的基础损失完全相同：observed＋clean两半等权的IQ NMSE＋0.1 logpower＋semantic，其中semantic为三套原F独立标量loss的均值，各自末端160维cosine＋0.3×T² KL（T=2）。原F及原默认头的参数/统计状态完全冻结，不平均不同teacher的坐标，不训练新分类/诊断头。

三组仅额外observed项不同，系数均0.5：普通对照取全部observed的semantic均值；困难样本取当前observed损失最高25%的均值（ceil，至少1条）；RX组使用源RX1/3/4/6/8的组均值，logq从0开始、每步加0.01×detach组均值并减最大值，使用当前存在组的softmax权重，缺失组不获得本batch梯度。q不参与求导，仅在训练更新。额外项绝不使用clean半。所有记录仍参与原基础损失，不改变物理采样或U曝光；q与选择统计只作训练审计，推断仅普通恢复网络，输入只有当前received IQ，不需要RX标签。

三臂均从原R03 coarse20/correction20（method b01b534f52daf9cae9e940ed6ec6f9243bccbdbe）原样加载，reset AdamW，联合10轮/2470步，coarse/correction各累计30轮，共60组件epoch、继承串行50pass。原F仍为原50轮CROSS_RX三seed，未改骨干或头。新增训练L/U曝光63000/567000、IQ1260000；原63000物理记录、缓存epoch1..10、.15clean/.60residual/.25full路由、batch256、modelseed2026092701/data2026101951、FP32、LR3e-5/warm247/cos至3e-6、WD1e-4、clip1均匹配。不生成新IQ，不读取L/U TX真值；训练RX元数据每RX12600条，经一次预检核实。额外focus实际曝光、每RX计数、最终q及更新数逐轮保存；CVaR预计157500/630000额外observed曝光，另两组630000，原U基础曝光均567000。参数量相同，实际耗时/显存另报。

三臂固定最终E10，不用V更新或早停。源V仍为同一1800条、原F1、clean＋六practical；每row全七视图预测固定后独立无CUDA CPU才接入V真值。更高的既有posterior03源指标只作发布参考，不作初始化或额外新对照：low87.055556%、mid93.444444%。候选须两urban各提高≥0.5个百分点、其他四最多下降0.5个百分点、相对原clean下降≤0.1个百分点、六环境平均NMSE≤1.15×原feature0.028326367493718863（不复利）。三个新臂互为同预算机制比较，只有它们支持归因于额外项聚合；相对posterior03的差异还包含额外observed项。全三臂完成后按源urban均值最高、NMSE更低、登记顺序选中通过者；都失败不访问新目标。

只对选定通过臂自动测试原168000条不可变目标，clean＋六practical，三套原F50 CROSS_RX seed2026092701/02/03分别argmax。全七视图恢复IQ/物理ID/决策固定后独立无CUDA CPU才接入目标truth和评分clean。practical推断不读取paired clean/其他视图/目标真值，不更新权重或q。报告accuracy/Macro-F1、逐F/TX/RX/day、修复/损坏/净修复、距原clean差值，以及无oracle相位/增益/时延对齐的full256/prefix232/tail24 NMSE、log FFT power误差均值/中位数/P90。一个恢复器训练seed、三套既有F读出，不称三恢复器种子、新盲测、整个系统完全无标签或在轨验证。

30项聚焦检查PASS；一次独立P0/P1审查PASS，无发现。N607普通账号，一次预检时GPU4/5/6各24124MB空闲无PID，固定每卡一条本run worker，全owner进程<2且获取时空闲≥18GB；CPU2线程/独立scorer最多1个。源候选完成退出后仅选定臂在其卡做target，获取空闲≥12GB。其他owner任务不动。复用原VALIDATED_ONCE数据与F/原R03来源证据，方法变化不重验数据、不加哈希链。一个代码配置commit、一个不可覆盖root、一个发布归档本地/远端SHA比较及远端compile。launcher首步两条实际L_s IQ、原R03、三F的no-query smoke，验证三个focus策略、两恢复组件有限非零梯度、原默认头到IQ梯度、F冻结、新checkpoint往返，0V/0TX/0target/0optimizer update，PASS立即续训。随后核实实际PID/CWD/完整命令/GPU/日志增长。非法权限/来源/代码、输出冲突、非有限/OOM/预算或预测缺失按技术停止规则保留产物并停止，不自动重试、不重启/热改健康任务；具体命令、环境、路径及产物见experiment.json。


## 实际启动与首轮核验

方法 commit `63c9c3d2b81d92ebfee1623b54d942011e410f0a` 已推送且独立远端 OID 与本地一致。30 项聚焦检查和唯一一次 P0/P1 审查 PASS；一次发布归档比较、远端 compile/import、真实 R03＋三 F 的两条 L_s no-query smoke 均 PASS。三个 focus 策略实际运行，两个恢复模块有有限非零梯度，默认头能向 IQ 反传，F 全冻结，新 checkpoint 往返一致，0V/0TX/0target/0optimizer update。PASS 后直接训练。

已独立核实三条实际 worker 的 PID/CWD/完整命令、物理 GPU4/5/6 对应及日志从0到首轮增长。三个源参考缓存各63000条×三套独立F坐标，未读取TX真值。首轮/当前进度与真实 focus 状态如下：

- mean_observed：1/10轮、247/2470步，observed额外选择63000/63000，q更新0，按RX1/3/4/6/8的全局权重[0.200000, 0.200000, 0.200000, 0.200000, 0.200000]。
- cvar_observed：1/10轮、247/2470步，observed额外选择15750/63000，q更新0，按RX1/3/4/6/8的全局权重[0.200000, 0.200000, 0.200000, 0.200000, 0.200000]。
- rxdro_observed：1/10轮、247/2470步，observed额外选择63000/63000，q更新247，按RX1/3/4/6/8的全局权重[0.190633, 0.212999, 0.206959, 0.198159, 0.191249]。

真实训练记录与预登记一致：CVaR额外选择25%，其余策略额外项使用全部observed；原基础损失仍覆盖所有L/U和clean配对。当前没有源V评分或新目标结果，运行检查不证明识别收益。
