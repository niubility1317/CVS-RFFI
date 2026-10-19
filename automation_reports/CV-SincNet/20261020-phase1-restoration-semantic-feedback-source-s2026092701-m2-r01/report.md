# 原 F 特征反馈的纠错结构对比

状态：ANALYZED / VERIFIED。用户完整目标仍为目标low-urban≥70%，其他五practical接近原clean85.113889%（Agent暂按≤2个百分点解释），并分开报告恢复IQ与同条clean的差距、接入原F/defaulthead后的真实TX识别及净纠错。源分数、波形或检查PASS均不能替代目标成绩。

此前多个损失/重加权/伪身份变体没有同时改善源low/mid。最新一致身份native源low/mid86.888889%/93.333333%，crossRX87.388889%/93.222222%，均未通过固定源规则，0新目标访问，负结果保留。本轮将冻结原F特征显式加入恢复结构，不追认失败精修或按目标结果选择权重。

两个新臂iq_condition、semantic_condition都从合法原R03 coarse20/correction20（method b01b534f52daf9cae9e940ed6ec6f9243bccbdbe）复制原网络权重。仅新增FiLM调制MLP：160→128 SiLU→640，末层weight/bias为0，拆成四个encoder/skip层（32/64/96/128）的gamma/beta，feature×(1+0.1tanh gamma)+0.1tanh beta。初始输出应逐点等于原R03。两个新臂可训练参数均2244008，其中新FiLM103168；相同数据/初始权重/预算下只改变条件代码。

IQ对照将本帧received＋coarse IQ的4通道各自adaptiveavgpool至40，拼成160维并逐条单位化。semantic臂用原冻结F1（seed2026092701）对本帧coarse IQ提取160维特征并单位化。F1只提供原监督先验，不拟合新头；参数与统计状态始终冻结、优化器排除，梯度可经F1输入回传到coarse。该特征只由当前IQ产生，不读取clean/其他帧/RX/TX/伪身份/bank，不在推断优化权重或状态。语义臂独立恢复checkpoint包含原F1；当前source/target代码与外部原读出共享F1实例。部署若已用F1可共享权重，使用其他F时需携带F1；额外特征前向与独立恢复模型状态不同，不能把2244008相同说成总状态或计算成本相同，实际耗时/显存另报。

两臂基础损失恢复为既有posterior03：observed＋clean等权的IQ NMSE＋0.1logpower＋三套原F各自final160cosine＋0.3×T²KL（T=2）的标量均值。没有新增失败CE/跨RX prototype/困难项，不平均教师坐标。原F均为原50轮CROSS_RX seeds2026092701/02/03及原联合训练默认头；原有参数/统计状态逐一保持，0新头拟合/0F优化更新。称冻结F引导自监督恢复，不称整个系统全无标签。

reset AdamW联合E10/2470，旧coarse/correction LR3e-5、新FiLM LR2e-4；共同warm247、cos至各自0.1，WD1e-4、全可训练梯度clip1、FP32/TF32关闭。加载三教师后重置modelseed2026092701，确保新增MLP的初始随机权重匹配；dataseed2026101951。原63000物理L/U、缓存epoch1..10、.15clean/.60residual/.25full、observed＋clean配对、batch256、L/U新增优化曝光63000/567000、IQ1260000全部匹配，无新IQ与L/U TX真值。coarse/correction各累计30轮，共60原组件epoch、串行继承50pass；新增FiLM单独10轮，F原50轮仅继承不更新，预算不混称。新增参数、冻结运行F参数、独立恢复模型总参数和实际时间/显存分别报告，逐轮记录FiLM末层权重/偏置范数及两种LR。

固定最终E10，不用V更新或早停。源V同一1800条、原F1、clean＋六practical，每row七视图预测固定后独立无CUDA CPU才接入真值。沿用最新low优先规则：相对已完成posterior03，low≥+0.5个百分点、mid不下降、其他四≥−0.5个百分点、相对原clean降幅≤0.1个百分点、六环境平均NMSE≤1.15×原feature0.028326367493718863（不复利）。两个新臂是可训练参数/预算/损失匹配的条件代码比较；旧posterior03没有新FiLM，是历史源发布参考，不是相同参数结构对照。全部两臂完成后按源low最高、mid最高、NMSE更低、登记顺序选一通过者；全失败不访问新目标，不追认旧失败。

只对选定合格者自动完整测试原168000条目标clean＋六practical。三套原默认头分别argmax，对同一恢复IQ各自评分，不投票、不拼接按环境选出的模型。全七视图IQ/ID/决策固定后，独立无CUDA CPU才接入目标truth和评分clean。报告accuracy/Macro-F1、逐F/TX/RX/day、修复/损坏/净修复与原clean差距；无oracle相位/增益/时延对齐的full256/prefix232/tail24 NMSE及logFFTpower均值/中位数/P90。一个恢复器seed、三套既有F读出；不称三恢复器种子、新盲测、Phase2部署或在轨性能。

25项聚焦检查、一次独立P0/P1审查PASS。launcher首步实际原R03/三F/2条Ls no-query smoke：两臂初始输出逐点等于原R03，共同参数初始值相同，coarse/correction/FiLM有限非零梯度、原默认头到IQ梯度、F无梯度、真实新checkpoint往返；0V/0TX/0target/0optimizer更新，PASS立即训练。新增层缺失是受控方法变化，父checkpoint数据/继承来源已核实，不以部分加载绕过契约。复用VALIDATED_ONCE数据，方法变化不重验IQ或加哈希链。

一次N607普通账号预检VERIFIED：父R03和源参考合法、前序完成，新run/release不存在；GPU4/5各24124MB空闲无PID，其他owner进程保留。固定GPU4 IQ、GPU5 semantic，每卡一条本run worker，计入全owner进程<2/获取空闲≥18GB；CPU2线程/独立scorer最多1。源worker全部退出后选中臂在对应卡获取≥12GB再跑target。一个代码配置commit、一个不可覆盖root、一个发布归档本地/远端SHA比较与远端compile；随后核实实际PID/CWD/完整命令/GPU/日志增长。非法来源/权限/代码、输出冲突、非有限/OOM、预算或预测缺失按技术停止规则保存产物，不自动重试、不热改/重启健康任务；具体环境/CWD/命令/路径/GPU/产物见experiment.json。原始IQ/权重/预测/日志私有，仅方法、登记与派生汇总进Git。未给另一对话发指令。


## 实际发布与首轮核验

方法 commit `1c869980ae9059e613ec29915d2b92db7444ed20` 已推送且独立远端OID一致。25项聚焦检查、一次独立P0/P1审查PASS；一次发布归档本地/远端比较、远端compile/import、真实原R03/三F/2条Ls no-query smoke均PASS。两臂初始输出逐点等于直接加载的原R03，共同参数初值一致；coarse/correction/新增FiLM均有有限非零梯度，原默认头能向IQ反传，F无梯度，两种新checkpoint往返一致。两臂可训练参数实际均2244008，0V/0TX真值/0target/0optimizer更新，PASS立即训练。

已核实两个实际worker的PID/CWD/完整命令与物理GPU4/5对应，日志从0增长至完整首轮；三教师参考缓存各63000条，无TX真值。新增FiLM末层权重从0变为非0，两组LR比例与预登记一致：

- iq_condition：1/10轮、247/2470步，FiLM末层weight/bias范数0.35935473/0.03062300，旧LR3e-05，新FiLM LR0.0002。
- semantic_condition：1/10轮、247/2470步，FiLM末层weight/bias范数0.40294698/0.03038993，旧LR3e-05，新FiLM LR0.0002。

这证明新增层在实际训练中参与；尚未产生新源V或目标准确率，运行检查不证明用户目标达成。后续按固定源规则决定是否自动完整目标测试，独立恢复模型/冻结运行F参数、耗时和显存按实际结果报告，不把可训练参数匹配当成总成本相同。


## 完整源结果

两臂均完成固定10轮/2470步，原F与默认头参数/统计状态保持不变，0L/U TX真值及目标训练。以下为同一1800条源V、原F1/default head的真实TX评分，不是目标成绩。

|源场景|原IQ|既有posterior03参考|IQ摘要条件|冻结F语义条件|
|---|---:|---:|---:|---:|
|clean|98.500000%|98.500000%|98.500000%|98.500000%|
|high|54.277778%|98.333333%|98.166667%|98.166667%|
|mid|51.722222%|97.500000%|97.500000%|97.555556%|
|low-suburban|51.000000%|96.722222%|96.888889%|96.833333%|
|high-urban|51.000000%|97.944444%|97.611111%|97.666667%|
|mid-urban|43.388889%|93.444444%|93.055556%|92.944444%|
|low-urban|36.666667%|87.055556%|86.833333%|86.777778%|

|候选|相对posterior03 low-urban（百分点）|mid-urban（百分点）|其他四最差（百分点）|原clean降幅（百分点）|平均practical NMSE|固定源资格|
|---|---:|---:|---:|---:|---:|---|
|iq_condition|-0.222222|-0.388889|-0.333333|0.000000|0.03120911|未通过|
|semantic_condition|-0.277778|-0.500000|-0.277778|0.000000|0.03114994|未通过|

新low优先规则在本候选源V/target之前固定，未追认旧失败：low≥+0.5个百分点、mid不下降、其他四≥−0.5个百分点、clean降幅≤0.1个百分点、NMSE≤1.15×原feature0.028326367493718863。全部两臂完成后按源low、mid、NMSE、登记顺序选定。两新臂有相同可训练参数/预算/损失和FiLM，只改变当前帧条件代码；旧posterior03没有新FiLM，不是相同参数结构对照。冻结运行F与额外前向成本另报。

匹配语义条件相对IQ条件的源low/mid差值为-0.055556/-0.111111个百分点。

## 实际条件模块与成本

|候选|可训练参数|冻结运行F参数|独立恢复模型总参数|训练/末次保存（s）|含准备等待（s）|峰值分配（bytes）|
|---|---:|---:|---:|---:|---:|---:|
|iq_condition|2244008|0|2244008|1098.309|1116.354|5547268096|
|semantic_condition|2244008|232993|2477001|1107.328|1123.912|6162735104|

iq_condition 的FiLM末层weight范数首/末轮为0.35935473/1.61546731，bias为0.03062300/0.11112026；最后旧/新LR为3e-06/2e-05。

semantic_condition 的FiLM末层weight范数首/末轮为0.40294698/3.92723250，bias为0.03038993/0.11063375；最后旧/新LR为3e-06/2e-05。

两个新臂可训练参数2244008（原coarse/correction2140840＋新FiLM103168）相同。语义恢复器携带原冻结F1并额外计算coarse特征，与外部读出共享F1实例时无需重复权重；独立恢复模型参数量不等于全pipeline独立权重或显存，不能据此声称总成本匹配。实际峰值含三原F及当前训练图，耗时按完整执行报告，不预先承诺加速。

每臂新增L/U优化曝光63000/567000、IQ1260000；原coarse/correction各累计30轮，60原组件epoch、串行继承50pass；新增FiLM独立10轮/2470更新，原F50仅继承不更新。所有记录与clean配对仍参与基础损失，无CE/prototype/focus额外项。完整10轮、全部stdout、逐轮FiLM非零权重与两种LR、总loss按IQ+0.1logpower+feature+0.3posterior重构均核实，无NaN/Inf、OOM或技术异常；F参数/统计状态保持一致。

两臂均未通过固定源规则，没有target目录或新增目标IQ/truth访问；源负结果完整保留。未测目标不能写成目标负收益。实际目标low-urban最好仍65.309127%，用户完整目标未达成。

## 验证与边界

25项聚焦检查、一次独立P0/P1审查PASS；实际原R03/三F no-query smoke验证默认头posterior到IQ梯度、coarse/correction/新FiLM梯度、零FiLM输出与直接原R03逐点相同、共同初始参数、F冻结及两种新checkpoint往返，PASS立即训练。一次发布归档比较、远端compile和实际PID/CWD/完整命令/GPU/日志增长均有证据。每row全七视图预测固定后独立CPU接入对应truth；如有目标行，同样全部七视图/三F决策完成后才读取目标truth与评分clean。无L/U TX真值、新拟合头、目标训练/调参/选模、practical pairedclean输入或推断更新。原始IQ/预测/权重/bank/完整日志保留私有输出，仅派生汇总提交Git。保留其他owner文件/任务，未向另一对话发送指令；公开基准已暴露，不称新盲测、完全无标签系统或在轨验证。
