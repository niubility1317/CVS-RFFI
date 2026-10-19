# 波形与原F识别梯度的单向投影

状态：ANALYZED / VERIFIED。目标仍是完整目标测试 low urban ≥70%，其余五practical接近原F clean85.113889%（Agent暂用≤2个百分点口径）；同时报告同物理clean波形差距和冻结原默认头真实TX识别/净纠错。当前已验证最好low urban65.309127%，目标未完成。

源域依据：既有source V的精修NMSE约0.0748，原R030.068199，误差增加遍及有效频带；末端time/frequency的source low urban87.055556/86.888889，相对posterior0387.055556无明显突破。source_rationale.json保存聚合证据，0新增模型前向/更新/目标访问。这只能提出“波形与识别梯度可能冲突”的假设，不能证明原因；实际conflict/applied比例将在训练中记录。本设计和代码固定时没有读取新basis候选的目标准确率，不继承它或任何失败精修权重，不改变其正在运行的预测。

两臂scalar_sum/waveform_projection从原合法R03（coarse20/corr20，method b01b534f52daf9cae9e940ed6ec6f9243bccbdbe）严格继承相同完整state，均2140840参数，无新增模块。原两任务数值不变：wave=paired IQ NMSE＋0.1logpower；identity=三原F各自final160cosine＋0.3×T²KL（T2）的均值。均对当前恢复IQ的每帧2×256实坐标求两个任务sum的梯度gw/gi。control直接相加；projection仅在dot(gw,gi)<0且norm²(gw)>1e-20时，用gi'=gi-dot(gw,gi)/norm²(gw)×gw，否则gi'=gi。合并gw+gi'后除以当前observed+clean实际批量，再向原恢复器反传一次、clip1并AdamW更新。第一任务求梯度保留共享中间图，兼容两loss共享中间计算；无Hessian或教师参数更新。

本方法借鉴[Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)的冲突投影思想，但为单向逐IQ输出空间投影，不是论文的完整参数空间PCGrad/随机任务顺序。共享网络Jacobian、AdamW和clip意味着输出空间的投影性质不保证参数步波形改善、最终NMSE下降或身份保真；全部由独立评分检验。控制行也计算分离梯度及冲突统计，标量和参数梯度/一步AdamW等价已验证。记录calls/IQ/conflict/applied、observed/clean冲突数、前后dot、波形/身份/移除分量norm²；没有冲突属于科学结果，不作为技术停止或候选资格门槛。

原63000 L/U、R02缓存epochs1..10、.15clean/.60residual/.25full路由、observed+clean等权、batch256、modelseed2026092701/dataseed2026101951保持。两行固定jointE10/2470、L63000/U567000优化曝光、1260000IQ；原coarse/corr各累计30轮、60组件epoch、串行继承50pass，无新网络epoch。重置AdamW3e-5、warm247/cos至3e-6、WD1e-4、clip1、FP32/TF32关闭。原F50 CROSS_RX三seed2026092701/02/03及原默认头（method4df7b94d0fd8ce29056bd8236364e45afa92e07f）冻结参数/统计，0L/U TX真值、0新头拟合。复用已验证同数据契约和来源，不重验不变数据或增加哈希/签名链。

推断仍是原R03网络当前单帧receivedIQ的一次coarse/corrector前向，无运行F或梯度、信道oracle、RX/场景、clean、其他帧/视图、迭代或状态更新。source V原1800条/原F1/七视图只作诊断，无早停、资格/NMSE gate、排序或持久状态更新。两个固定最终E10都自动完整168000物理目标clean+六practical/三原默认头，各自argmax，不投票或按环境拼接。每行七视图IQ/ID/预测全部固定后，独立无CUDA CPU才连接truth与配对clean；报告逐F/TX/RX/day accuracy/MacroF1、修复/损坏/净修复、距clean差距、无oracle对齐full256/prefix232/tail24 NMSE/logFFTpower均值/中位数/P90及耗时/显存。目标成绩不回流配置、预算、权重、顺序或选择性重跑；一个恢复器seed和三读出不冒充三个恢复器seed、新盲测、Phase2或在轨结果。

44项聚焦检查及唯一一次独立P0/P1审查PASS（审查者独立复跑44项通过）。launcher首步实际R03/三F/四source Ls no-query smoke，检查初始IQ/state相同、合并梯度到coarse/correction有限非零、原头到IQ梯度、F无梯度及新checkpoint往返，0V/TX/target/optimizer；PASS立即训练。一轮普通账号N607预检VERIFIED，原R03合法、新root不存在、GPU6/7各24124MB空闲无PID、空间≥8GB；实际GPU4/5上前一basis的target PID/CWD/命令已核实LIVE并保留。新GPU6scalar/GPU7projection各一worker，与前一预测重叠期间最多4条本owner GPUworker，单卡最多1条本owner、计入allowner<2、source获取≥18GB/target≥12GB。CPU2线程/scorer最多1；新source scorer等待前basis complete/failure后，source/target共用本run score.lock，避免跨run CPU评分重叠。实际source进程退出才同卡target，不等待另一行。

停止规则：非法来源/权限/代码、冲突、非有限/OOM、预算/梯度合并执行或预测失败时保存产物并技术失败该行，无自动重试，健康行继续；不停止/重启/热改其他任务。一个代码配置commit、不可覆盖root、一次归档本地/远端SHA比较、remotecompile/import，随后核实PID/CWD/命令/GPU/日志增长。命令、环境/CWD、矩阵、路径、资源和预期产物见experiment.json；原IQ、权重、逐样本预测、日志与私有运行状态不进Git。未向另一用户对话发消息或改动其文件/任务。


## 实际发布与首轮核验

方法commit `9f2d8f65e507809de396af5fdb89f11b749759b1`已推送并独立核实远端OID，一次归档本地/远端SHA比较及远端compile/import均PASS。实际R03/三F/四Ls no-query smoke PASS：两臂完整初始IQ/state等于原R03，coarse/correction合并梯度及原头到IQ梯度有效，F无梯度，新checkpoint往返一致，0V/TX/target/optimizer更新。两臂实际2140840参数，无新增模块；PASS立即训练。

两条实际source worker的PID/CWD/完整命令/GPU6/7及持续日志增长已核实，三F参考缓存各63000条。实际冲突/合并计数：

- scalar_sum：1/10轮、247/2470步，梯度合并调用247、IQ126000条、eligible冲突22614（observed10011/clean12603），实际投影0；dot前/后总和208.49425/208.49425，移除梯度norm²总和0。
- waveform_projection：1/10轮、247/2470步，梯度合并调用247、IQ126000条、eligible冲突22555（observed10017/clean12538），实际投影22555；dot前/后总和209.95659/225.39494，移除梯度norm²总和21.84372。

首轮核验时尚无新源V或目标准确率。实际冲突/投影不代表性能改善或参数步波形保证；两臂固定E10后分别自动完整目标测试，source V不作资格筛选，用户目标仍未达成。最终状态与结果以完整结果节为准。前一basis的GPU4/5目标进程保留，CPU评分等待其完成后执行。


## 完整固定目标比较

两个登记行均完成E10/2470、source V诊断、168000物理目标样本的七视图预测和独立CPU truth-last评分。实际source/target/controller进程均已退出。没有用源分数筛掉一组，也没有根据目标成绩更改/重排/选择性重跑。以下是三套原F各自argmax的均值±样本标准差，三套原读出不是三次独立恢复器训练。原clean参考均值85.113888889%；历史完整特征约束候选low urban为65.309126984%，仅作已暴露基准参照，不作为初始化。

|目标环境|原R03|标量和|单向投影|
|---|---:|---:|---:|
|clean|84.743056%|85.062500±0.447442%|85.063889±0.443389%|
|high|77.575000%|77.240873±0.210815%|77.228968±0.244251%|
|mid|76.491071%|76.230556±0.162641%|76.241270±0.203339%|
|low suburban|75.284127%|75.187302±0.070686%|75.187698±0.069442%|
|high urban|76.330952%|76.157738±0.150138%|76.173016±0.153077%|
|mid urban|70.116865%|70.735913±0.110889%|70.727579±0.096225%|
|low urban|63.392460%|64.989484±0.239284%|64.952778±0.199827%|

|识别判据|标量和|单向投影|
|---|---:|---:|
|low urban≥70%|未达到|未达到|
|其余五环境距原clean≤2个百分点|未达到|未达到|

“接近”采用本对话Agent预先记录的≤2个百分点口径，不冒充用户指定数值。未通过者保持负结果，未因实现/训练完成或波形误差而称用户目标实现。

|逐原F的low urban|标量和|单向投影|
|---|---:|---:|
|2026092701|65.258929%|65.156548%|
|2026092702|64.801786%|64.757143%|
|2026092703|64.907738%|64.944643%|

|距原clean差距（百分点）|标量和|单向投影|
|---|---:|---:|
|clean|0.051389|0.050000|
|high|7.873016|7.884921|
|mid|8.883333|8.872619|
|low suburban|9.926587|9.926190|
|high urban|8.956151|8.940873|
|mid urban|14.377976|14.386310|
|low urban|20.124405|20.161111|

## 同物理clean的波形差距与净纠错

配对clean仅在七视图预测全部固定后的独立CPU评分读取，不做oracle增益/相位/时延对齐。逐环境full256/prefix232/tail24 NMSE和logFFTpower均值/中位数/P90、逐F/TX/RX/day与混淆矩阵完整保存在results.json。

|目标low urban|原received|原R03|标量和|单向投影|
|---|---:|---:|---:|---:|
|full_nmse均值|1.987855210|0.076996611|0.080557630|0.080239212|
|prefix232_nmse均值|1.998446002|0.068349912|0.071845541|0.071588297|
|tail24_nmse均值|1.928131431|0.157928481|0.163219265|0.162200019|
|log_power_mse均值|0.160772502|0.023813475|0.024496098|0.024435903|

|low urban修复/损坏/净修复|相对received|相对原R03|
|---|---:|---:|
|标量和|185498/27532/157966|38917/30868/8049|
|单向投影|185471/27690/157781|38813/30949/7864|

计数为三原头的决策合计，504000个重复视图决策只有168000个物理样本，不称504000个独立样本。

## 源诊断、实际执行和成本

|源V（固定F1）|clean|low urban|mid urban|
|---|---:|---:|---:|
|标量和|98.500000%|87.111111%|93.111111%|
|单向投影|98.500000%|86.777778%|93.166667%|

源V只诊断，无候选资格/排序/早停/参数或统计更新。两臂各63000训练物理L/U、E10/2470、L63000/U567000优化曝光、1260000IQ，原coarse/correction各累计30轮、60原组件epoch、串行继承50pass；梯度合并安排同10轮，无新模块。原F和默认头冻结，0新头拟合、0L/U TX真值、0目标训练更新。

- 标量和：2140840可训练参数，无新增模块；梯度合并调用2470次、IQ1260000条、eligible冲突225835（observed108004/clean117831）、实际投影0；dot前/后总和1447.13234/1447.13234，wave/identity/removed norm²总和10008.8848/2071228.32/0。训练/保存1127.013s，含准备/等待1145.103s，峰值分配5544580096bytes；七视图目标预测合计794.433s。
- 单向投影：2140840可训练参数，无新增模块；梯度合并调用2470次、IQ1260000条、eligible冲突225921（observed107767/clean118154）、实际投影225921；dot前/后总和1459.71725/1607.19203，wave/identity/removed norm²总和10007.3374/2100557.08/208.601508。训练/保存1142.795s，含准备/等待1161.320s，峰值分配5544579584bytes；七视图目标预测合计797.264s。

梯度计数和norm²是逐IQ未除batch前的任务sum-loss输出梯度统计，不是参数梯度/FLOP；两行均计算两个输出梯度，只有合并不同。逐轮实际调用/曝光/冲突/投影/obs-clean/dot/norms、匹配LR/路由/角色预算及两项标量loss值已核对。投影更新一般不是原标量和的梯度；共享网络Jacobian/AdamW/clip意味着输出空间操作不保证参数步或最终波形收益。全部source/target日志已完整解析，无技术错误。方法commit`9f2d8f65e507809de396af5fdb89f11b749759b1`；44项聚焦检查、唯一一次独立P0/P1审查、真实原R03/三F与新checkpoint no-query smoke、归档比较/编译和PID/CWD/命令/GPU/日志增长证据保留。未给另一用户对话发消息。

输出梯度机制的源训练诊断：

|源训练累计统计|标量和|单向投影|
|---|---:|---:|
|eligible冲突IQ占比|17.923413%|17.930238%|
|identity/wave输出梯度累计RMS比|14.385373|14.487985|
|移除的identity梯度norm²占比|0.000000%|0.009931%|

冲突出现次数不代表投影改变量大。上述是累计逐IQ任务sum-loss输出梯度统计，不能解释成逐帧/step/参数梯度比或因果证明；源统计不替代目标识别结论。

本轮仍未达到用户完整目标。单向逐IQ输出梯度投影不等于完整参数空间PCGrad，也不证明最终波形或身份保真保证，不能把单恢复器种子或已暴露基准当作三seed恢复结论、新独立盲测、Phase2部署或在轨性能。
