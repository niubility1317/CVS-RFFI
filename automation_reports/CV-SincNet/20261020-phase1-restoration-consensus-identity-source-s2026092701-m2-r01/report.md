# 冻结原 F 的一致伪身份与跨 RX 特征恢复

状态：PLANNED。用户目标仍为目标 low-urban ≥70%，其余五 practical 接近原 clean 85.113889%（Agent 暂按差距≤2个百分点解释）。分别报告同条 clean 的 IQ 差距、恢复器接入原基座/default head 后的真实 TX 识别和净纠错，不能用源分数或 NMSE 替代目标成绩。

前序 observed 均值、top25% 困难样本和 RX 加权均未通过其固定源规则，没有新增目标访问。困难组相对 posterior03 源 low/mid 为 +0.388889/−0.388889 个百分点，波形误差也增加；它说明单纯重加权尚未同时保住身份与波形。本轮改为源 clean 的高置信一致身份约束，直接从合法原 R03 独立分叉，不继承失败精修或按目标成绩挑出的权重。

两个新臂 consensus_native、consensus_crossrx 的原基础损失完全相同：observed＋clean 两半等权的 IQ NMSE＋0.1 logpower＋三套原 F 各自 final160 cosine＋0.3×T² KL（T=2）的标量均值。原 F 和原默认头参数/统计状态冻结，不平均不同教师的特征坐标，不拟合新头。

一次源 L/U clean 参考前向中，三套原默认头 T=1 的类别必须完全一致且每套最大概率≥0.9，才产生停止梯度的伪身份。这不是 L/U TX 真值，未通过的记录仍参与原基础损失。两臂额外 0.3×native CE 只作用于 observed 半，三头各自 CE 取均值；分母是全部 observed 条数，被拒记录置零，不改变采样或偷偷增加 U 曝光。

crossrx 臂额外 0.3×cosine：每套教师在自己的160维坐标中，按伪身份/源 RX 对通过的单位 clean 特征求均值，再将其他 RX 的组均值等权平均并单位化，排除本条记录自己的 RX。参考不按每 RX 样本数加权；若同伪身份没有其他 RX 支持，只屏蔽该 prototype 项，不回退到本 RX 或删掉原基础/CE。三教师标量均值仅用于训练损失。该参考是无参数算术均值，不是拟合分类头，也不用于任何身份预测或候选打分。训练 bank、接受率、每RX/伪类计数、支持情况、L/U 接受数和实际额外曝光保存在私有输出；推断 checkpoint 只需要普通恢复网络，输入仅当前 received IQ，不读取 RX/伪标签/bank/pairedclean/其他视图或更新状态。

原 R03 coarse20/correction20（method b01b534f52daf9cae9e940ed6ec6f9243bccbdbe）是唯一恢复初始化。两新臂 reset AdamW 联合 E10/2470，coarse/correction 各累计30轮，共60组件 epoch、串行继承50pass，参数2140840。新增 L/U 优化曝光63000/567000、IQ1260000；63000物理记录、缓存epoch1..10、.15clean/.60residual/.25full路由、batch256、modelseed2026092701/data2026101951、FP32、LR3e-5/warm247/cos至3e-6、WD1e-4、clip1完全匹配，不生成新 IQ。三套原 F 为原50轮 CROSS_RX seeds2026092701/02/03。额外源 clean 参考前向、继承成本、真实训练时间和显存均另报；一个恢复器训练 seed 不能称三个种子。

固定最终 E10，不用 V 更新或早停。新源规则在任何本候选 V/目标访问前固定：相对已完成 posterior03，low-urban 至少+0.5个百分点，mid-urban 不下降，其他四最多下降0.5个百分点，原 clean 降幅≤0.1个百分点，六环境平均 NMSE≤1.15×原 feature 0.028326367493718863，不复利。按 low 优先的研究顺序，本轮将此前要求 mid 同时+0.5 改为保持；用户对实际目标的70%与接近clean要求没有降低。所有过往失败仍失败，不追认或重新访问目标。两新臂互为同成本机制对照；posterior03 仅是历史源发布参考。全部两臂完成后，合格者按源 low 最高、mid 最高、NMSE 更低、登记顺序选一；都失败则不访问目标。

源 V 为同一1800条、原F1、clean＋六practical；每row全部七视图预测固定后独立无CUDA CPU才接入 V 真值。仅选定合格臂自动完整测试原168000条目标 clean＋六 practical，三套原 F/default heads 各自 argmax。全七视图恢复IQ/ID/决策固定后，独立无CUDA CPU才读取目标 truth 与评分 clean。报告 accuracy/Macro-F1、逐F/TX/RX/day、修复/损坏/净修复、距原clean差值及无oracle对齐的full256/prefix232/tail24 NMSE、log FFT power均值/中位数/P90。已暴露基准，不称新盲测、整个系统全无标签或在轨性能。

29项聚焦检查已 PASS，一次独立 P0/P1 候选审查在发布前完成。launcher 首步用实际源 L_s、小范围多 RX 参考、原 R03/三原 F 验证一致原头 CE/原 posterior/跨 RX 特征到 IQ 的有限非零梯度、两个恢复组件更新归属、F 无梯度、新 checkpoint 往返；0 V、0 TX真值、0 target、0 optimizer update，PASS立即训练。测试只服务下一次实际运行，不增加签名、哈希链或重复审查。

N607 普通账号的一次预检已 VERIFIED：原 R03 来源、既有 source control/前序完成；GPU4/5 各24124MB空闲无PID。固定 native GPU4、crossrx GPU5，每卡一条本run worker、计入其他owner且获取时空闲≥18GB/总进程<2；CPU2线程、独立scorer最多1。两source worker退出后，选定臂在对应卡获取空闲≥12GB再做target。其他健康进程不变。代码/配置一个Git commit，一个不可覆盖 run root，一个发布归档本地/远端SHA比较及远端compile。启动后核实真实PID/CWD/完整命令/GPU/日志增长；来源/权限/代码错误、输出冲突、非有限/OOM、预算/预测缺失依预登记技术停止规则保存产物、不自动重试、不停止或热改健康其他行。命令、环境/CWD、路径、GPU和产物见experiment.json。原始 IQ、预测、权重、bank和日志不进Git，仅提交方法、登记和派生汇总。未向另一对话发指令。
