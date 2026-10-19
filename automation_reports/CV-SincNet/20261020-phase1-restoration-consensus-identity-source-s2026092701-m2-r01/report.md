# 冻结原 F 的一致伪身份与跨 RX 特征恢复

状态：ANALYZED / VERIFIED。用户目标仍为目标 low-urban ≥70%，其余五 practical 接近原 clean 85.113889%（Agent 暂按差距≤2个百分点解释）。分别报告同条 clean 的 IQ 差距、恢复器接入原基座/default head 后的真实 TX 识别和净纠错，不能用源分数或 NMSE 替代目标成绩。

前序 observed 均值、top25% 困难样本和 RX 加权均未通过其固定源规则，没有新增目标访问。困难组相对 posterior03 源 low/mid 为 +0.388889/−0.388889 个百分点，波形误差也增加；它说明单纯重加权尚未同时保住身份与波形。本轮改为源 clean 的高置信一致身份约束，直接从合法原 R03 独立分叉，不继承失败精修或按目标成绩挑出的权重。

两个新臂 consensus_native、consensus_crossrx 的原基础损失完全相同：observed＋clean 两半等权的 IQ NMSE＋0.1 logpower＋三套原 F 各自 final160 cosine＋0.3×T² KL（T=2）的标量均值。原 F 和原默认头参数/统计状态冻结，不平均不同教师的特征坐标，不拟合新头。

一次源 L/U clean 参考前向中，三套原默认头 T=1 的类别必须完全一致且每套最大概率≥0.9，才产生停止梯度的伪身份。这不是 L/U TX 真值，未通过的记录仍参与原基础损失。两臂额外 0.3×native CE 只作用于 observed 半，三头各自 CE 取均值；分母是全部 observed 条数，被拒记录置零，不改变采样或偷偷增加 U 曝光。

crossrx 臂额外 0.3×cosine：每套教师在自己的160维坐标中，按伪身份/源 RX 对通过的单位 clean 特征求均值，再将其他 RX 的组均值等权平均并单位化，排除本条记录自己的 RX。参考不按每 RX 样本数加权；若同伪身份没有其他 RX 支持，只屏蔽该 prototype 项，不回退到本 RX 或删掉原基础/CE。三教师标量均值仅用于训练损失。该参考是无参数算术均值，不是拟合分类头，也不用于任何身份预测或候选打分。训练 bank、接受率、每RX/伪类计数、支持情况、L/U 接受数和实际额外曝光保存在私有输出；推断 checkpoint 只需要普通恢复网络，输入仅当前 received IQ，不读取 RX/伪标签/bank/pairedclean/其他视图或更新状态。

原 R03 coarse20/correction20（method b01b534f52daf9cae9e940ed6ec6f9243bccbdbe）是唯一恢复初始化。两新臂 reset AdamW 联合 E10/2470，coarse/correction 各累计30轮，共60组件 epoch、串行继承50pass，参数2140840。新增 L/U 优化曝光63000/567000、IQ1260000；63000物理记录、缓存epoch1..10、.15clean/.60residual/.25full路由、batch256、modelseed2026092701/data2026101951、FP32、LR3e-5/warm247/cos至3e-6、WD1e-4、clip1完全匹配，不生成新 IQ。三套原 F 为原50轮 CROSS_RX seeds2026092701/02/03。额外源 clean 参考前向、继承成本、真实训练时间和显存均另报；一个恢复器训练 seed 不能称三个种子。

固定最终 E10，不用 V 更新或早停。新源规则在任何本候选 V/目标访问前固定：相对已完成 posterior03，low-urban 至少+0.5个百分点，mid-urban 不下降，其他四最多下降0.5个百分点，原 clean 降幅≤0.1个百分点，六环境平均 NMSE≤1.15×原 feature 0.028326367493718863，不复利。按 low 优先的研究顺序，本轮将此前要求 mid 同时+0.5 改为保持；用户对实际目标的70%与接近clean要求没有降低。所有过往失败仍失败，不追认或重新访问目标。两新臂互为同成本机制对照；posterior03 仅是历史源发布参考。全部两臂完成后，合格者按源 low 最高、mid 最高、NMSE 更低、登记顺序选一；都失败则不访问目标。

源 V 为同一1800条、原F1、clean＋六practical；每row全部七视图预测固定后独立无CUDA CPU才接入 V 真值。仅选定合格臂自动完整测试原168000条目标 clean＋六 practical，三套原 F/default heads 各自 argmax。全七视图恢复IQ/ID/决策固定后，独立无CUDA CPU才读取目标 truth 与评分 clean。报告 accuracy/Macro-F1、逐F/TX/RX/day、修复/损坏/净修复、距原clean差值及无oracle对齐的full256/prefix232/tail24 NMSE、log FFT power均值/中位数/P90。已暴露基准，不称新盲测、整个系统全无标签或在轨性能。

29项聚焦检查已 PASS，一次独立 P0/P1 候选审查在发布前完成。launcher 首步用实际源 L_s、小范围多 RX 参考、原 R03/三原 F 验证一致原头 CE/原 posterior/跨 RX 特征到 IQ 的有限非零梯度、两个恢复组件更新归属、F 无梯度、新 checkpoint 往返；0 V、0 TX真值、0 target、0 optimizer update，PASS立即训练。测试只服务下一次实际运行，不增加签名、哈希链或重复审查。

N607 普通账号的一次预检已 VERIFIED：原 R03 来源、既有 source control/前序完成；GPU4/5 各24124MB空闲无PID。固定 native GPU4、crossrx GPU5，每卡一条本run worker、计入其他owner且获取时空闲≥18GB/总进程<2；CPU2线程、独立scorer最多1。两source worker退出后，选定臂在对应卡获取空闲≥12GB再做target。其他健康进程不变。代码/配置一个Git commit，一个不可覆盖 run root，一个发布归档本地/远端SHA比较及远端compile。启动后核实真实PID/CWD/完整命令/GPU/日志增长；来源/权限/代码错误、输出冲突、非有限/OOM、预算/预测缺失依预登记技术停止规则保存产物、不自动重试、不停止或热改健康其他行。命令、环境/CWD、路径、GPU和产物见experiment.json。原始 IQ、预测、权重、bank和日志不进Git，仅提交方法、登记和派生汇总。未向另一对话发指令。


## 实际启动与首轮证据

方法 commit `623b3f3cb93f56fe77bf680e71570def9170ed2e` 已推送并独立核实远端 OID 一致。29 项聚焦检查、一次独立 P0/P1 审查 PASS；一次发布归档比较、远端 compile/import、真实原 R03＋三 F no-query smoke 均 PASS。smoke 只用160条实际 L_s 参考、其中2条实际 IQ 验证原 posterior/一致 native CE/其他 RX cosine 到 IQ 的有限非零梯度、两恢复组件梯度、F 完全冻结及新 checkpoint 往返；0V/0TX真值/0target/0optimizer update，PASS 立即训练。

已核实两条 worker 的实际 PID/CWD/完整命令与物理 GPU4/5 对应，日志从0增长到完整首轮。两组源 bank 均63000条，其中59402条满足三原头一致且各≥0.9，接受率94.288889%，L/U接受6074/53328；这不是正确率，也未读取TX真值。全部30个源 RX/伪类组合都有其他 RX 支持。两臂 bank 统计一致，推断不使用 bank/RX。

- consensus_native：1/10轮、247/2470步，observed63000、一致原头CE曝光59402、跨RX特征曝光0。
- consensus_crossrx：1/10轮、247/2470步，observed63000、一致原头CE曝光59402、跨RX特征曝光59402。

实际执行与固定配方一致；原基础损失仍覆盖全部 L/U 及 clean 配对，未通过伪身份的记录没有被删除。当前没有新源 V 或目标成绩，运行/接受率不证明用户目标达成。


## 完整源结果

两臂均完成固定10轮/2470步，原F与默认头参数/统计状态保持不变，0L/U TX真值及目标训练。以下为同一1800条源V、原F1/default head的真实TX评分，不是目标成绩。

|源场景|原IQ|既有posterior03参考|一致原头CE|增加跨RX特征|
|---|---:|---:|---:|---:|
|clean|98.500000%|98.500000%|98.500000%|98.500000%|
|high|54.277778%|98.333333%|98.055556%|98.222222%|
|mid|51.722222%|97.500000%|97.444444%|97.555556%|
|low-suburban|51.000000%|96.722222%|96.833333%|96.888889%|
|high-urban|51.000000%|97.944444%|97.777778%|97.722222%|
|mid-urban|43.388889%|93.444444%|93.333333%|93.222222%|
|low-urban|36.666667%|87.055556%|86.888889%|87.388889%|

|候选|相对posterior03 low-urban（百分点）|mid-urban（百分点）|其他四最差（百分点）|原clean降幅（百分点）|平均practical NMSE|固定源资格|
|---|---:|---:|---:|---:|---:|---|
|consensus_native|-0.166667|-0.111111|-0.277778|0.000000|0.03277007|未通过|
|consensus_crossrx|+0.333333|-0.222222|-0.222222|0.000000|0.03295989|未通过|

新low优先规则在本候选源V/target之前固定，未追认旧失败：low≥+0.5个百分点、mid不下降、其他四≥−0.5个百分点、clean降幅≤0.1个百分点、NMSE≤1.15×原feature0.028326367493718863。全部两臂完成后按源low、mid、NMSE、登记顺序选定。两新臂只差额外跨RX cosine；相对既有posterior03还包括一致伪身份CE，不能单独归因于跨RX项。

匹配跨RX臂相对本轮native臂的源low/mid差值为+0.500000/-0.111111个百分点。

## 实际伪身份曝光与计算成本

两臂均以同一59402/63000条源记录通过一致条件，L/U通过6074/53328，接受率94.288889%不是正确率；全部30个RX/伪类组合均有其他RX支持。原基础损失保留全部630000次observed及其clean配对；没有将被拒记录从训练删除。

- consensus_native：一致CE曝光594020，其中L/U60740/533280；跨RX特征曝光0。训练1105.474 s，含准备/容量等待1124.704 s，峰值分配5537740800 bytes。
- consensus_crossrx：一致CE曝光594020，其中L/U60740/533280；跨RX特征曝光594020。训练1110.233 s，含准备/容量等待1129.299 s，峰值分配5538072576 bytes。

每臂新增L/U优化曝光63000/567000、IQ1260000，coarse/correction各累计30轮，共60组件epoch、串行继承50pass，参数2140840。三原F/默认头不更新，新bank无拟合参数、只算源clean均值；不作为预测分类器，推断不读取bank/RX/伪身份。每轮真实曝光与总loss按IQ+0.1logpower+feature+0.3posterior+0.3nativeCE+各臂prototype权重×cosine重构均核实。完整10轮和全部stdout已解析，未见NaN/Inf、OOM或技术异常。

两臂均未通过固定源规则，没有target目录或新增目标IQ/truth访问；源负结果完整保留。未测目标不能写成目标负收益。实际目标low-urban最好仍65.309127%，用户完整目标未达成。

## 验证与边界

29项聚焦检查、一次独立P0/P1审查PASS；实际原R03/三F no-query smoke验证真实nativeCE/posterior/otherRX到IQ梯度、两恢复组件梯度、F冻结与新checkpoint往返，PASS立即训练。一次发布归档比较、远端compile和实际PID/CWD/完整命令/GPU/日志增长均有证据。每row全七视图预测固定后独立CPU接入对应truth；如有目标行，同样全部七视图/三F决策完成后才读取目标truth与评分clean。无L/U TX真值、新拟合头、目标训练/调参/选模、practical pairedclean输入或推断更新。原始IQ/预测/权重/bank/完整日志保留私有输出，仅派生汇总提交Git。保留其他owner文件/任务，未向另一对话发送指令；公开基准已暴露，不称新盲测、完全无标签系统或在轨验证。
