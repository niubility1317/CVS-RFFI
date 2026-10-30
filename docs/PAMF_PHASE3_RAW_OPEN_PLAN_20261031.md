# PAMF P0开放表征与P1身份路径：匹配训练单候选

Run `20261031-phase3-star-raw-open-s2026092701-r01`，唯一新arm `E_RAW`，唯一主 `P0_known12`。保留原F0/default6身份路径P1不变，研究开放分支从同一星地received P0学习是否优于已完成的P1开放分支。P0不是clean；不能由已有拒识结果预先断言DURA损坏未知身份。

## 固定研究因子与矩阵

旧P1信号分支来自signal-open r03：同原F0初始化、headless232033＋zero-last MLP102688，2220更新、L192/OE64、crossset KL＋OE软结构权1。新P0分支使用相同初始化、结构、预算、采样和目标，改变开放分支及其冻结原F0 OE teacher的输入路径为P0。两路径teacher均取该路径下原冻结F0的u，不将P1 teacher用于P0蒸馏。新分支不能从已训练的P1分支warmstart。

四列固定为 `P1_known12`（精确复用macro12主控制）、`P1_ratio12`（父固定P1 ratio仅新known12校准）、`P0_known12`（唯一主）、`P0_ratio12`（诊断）。四列均按相同known12规则校准／复用，旧身份一律原F0 P1默认head；不使用P0较低闭集准确率作为保护基线。P1两分数来自同一已冻结P1 NN，P0两分数来自同一新NN；path×readout对照固定，不按novel结果换主。

## 数据、观测与来源

已有star-representation缓存根`/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-representation-s2026092701-r01/cache`，四role为L6300/OE6134/V27000/novel3150。`received.npy`为FP32[N,2,256]的P0，`restored.npy`是相同ID顺序的P1。新训练/新投影明确只读received值，不把P0重命名为restored，不读clean/P1实值，不重抽信道、额外RMS或EQ，不运行G。旧P1身份与控制复用已完成的P1预测，不是把clean当对照。

L旧六TX各1050、每TX×RX×day70，真实旧标签可见；OE来自21辅助设备，算法只见outside-six成员角色与公共RX/day，无真实OE TX/K。source RX1/3/4/6/8为1-19/18-2/19-2/2-19/3-19；日期2021-03-08/15/23均已见。四角色物理互斥共42584既有记录，没有新增物理观测。novel为19-15/19-16/7-1各1050，有历史开发曝光；不能称新RX/TX盲测。

训练只打开L/OE；V旧六各4500（每TX×RX×day300）只在NN/全部密度冻结后用于known风险阈值；novel只在全部新旧τ冻结后预测，独立scorer最后连接trueTX。原Us56700、target U/Q、正式RX24-18不读，不回收历史query。

全部沿固定G10MIX01：600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、郊区/城区各0.5及原六面积权重；receiver2027/scene2026103001/channel2026104001，输入一次unitRMS、输出不再RMS。P1是同received经DURA v1.0输出。新P0分支和旧P1控制共享同一物理记录／信道实现，只改变合法计算路径。

原F0为Tri-SIG-TIME SSL E50/11100/232993，sourcecontract392005与DURA祖先、既有VALIDATED_ONCE角色结论复用。P1训练控制config/checkpoint来源固定signal-open r03 actualcode `dc01a402582f93e452dbaa3ef59a9fbb44151818`；其原L/OE steps.jsonl用于逐步匹配采样。P1 known12控制取macro12-risk r01 actualcode `26fbecff1df7e5476bcc5f459d8367c6fa49fd36` 的signal_known12列3；P1 ratio固定state/raw取r03 signal_density_ratio列3，同ID/模型来源核对。没有新目标训练、其它父模型或更高控制替代。

## 教师与真正可训练的分支

首先通过现有make_trainable_model严格重建合法原F0，实际checkpoint两行synthetic G10无query smoke PASS；随后先从该可微reader导出独立学生，保留真实SequenceReference/features/components和全部信号模块，仅删W并旁路被丢弃的内部logits，沿用已验证HeadlessSignal/SignalOpen。原身份模型未修改。先完成学生8行合成梯度probe再读取真实L/OE，避免复制冻结部署reader或把teacher缓存误带成断梯度路径。

学生导出后，原F0 eval/冻结，分批对OE6134原received计算teacher u160（raw fused FP64单位化→FP32），参数/buffers保持；teacher stopgrad、无V/NV/目标，保存私有训练teacher及其原source身份、P0路径、opaque IDs/公共metadata。不得复用P1 teacher或把student更新后的输出当teacher。原F0新增真实teacher前向6134单列；teacher逐行bank不写入部署checkpoint或Git。

学生结构与旧P1一致：信号232033＋MLP102688＝334721参数，MLP640→128GELU→160、seed2701、末层精确0。每次P0 forward捕获raw fused160及原四贡献，FP64规范单位化后FP32为u_theta/c_theta，z=unit(u_theta+g(c_theta))，学生u/c不detach。NoExplicitCFO坐标27保持0。全部BN/Dropout eval、buffers不更新，原F0/default6身份仍独立P1。

新信号祖先11100＋新增2220＝13320、新MLP2220与原识别器native0/E50分别报告。fresh AdamW base lr1e−4/wd1e−4/eps1e−8，原CENTER_UNITS scale25000/multiplier1000/Sinc lr.1/eps1e−11/group decay0及物理衰减/global clip5不变；记录实际2220 Sinc steps/clipcalls。原F0＋单个P0开放分支活跃567714，加既有G算术834665；不由参数量推定B1时延。

## 完全匹配的目标、采样与预算

复用Sampler L RNG2026103017/OE RNG2026103018；L192＝六类×15域各2＋每类两轮转格，OE64＝15域各4＋四轮转格，batch无重复。全部2220步L/OE索引与父P1 r03保存trace逐值一致，不重新采样或改变配额。

T=.1 crossset KL仅Lanchor，候选须跨RX且跨day，L/OE各0.5先验且各自公共有效域等权／域内均分；positive仅L同旧TX。另加OE frozen-original-P0-u软邻域KL权1，teacher stopgrad，同跨RX/day与域均衡。没有OE–OE硬同/异类、未知K、P0/P1一致性、VICReg、logit KD、EMA、新augmentation或目标状态。固定2220终点，无V/NV早停、追加epoch或根据其它row结果中断。

首两步记录学生L/OE VJP、Sinc/浅中层/四贡献/MLP梯度，后续保存全局物理norm、实际索引、loss和optimizer计数/有限性；未记录的逐层漂移不补造。所有输入值路径明确P0，不通过P1字段别名改变旧helper语义。

## 终点冻结、四列预测与评分

新checkpoint保存完整学生state/source身份/P0 teacher来源，不携teacher逐行bank；eval严格重载及两行synthetic signal smoke后，先对L/OE received投影，拟合同族D160 known90/background32/25Lloydseed3019/covshrink.1。新P0 known8和ratio11完全共享known统计。旧P1 known8/ratio11从两个已完成控制root精确复用，所有统计在V前冻结。

V读取后一次P0投影，计算P0 known/ratio两个raw，连同父固定P1 ratio raw分别执行相同known12 common-tail校准；P1 known12旧τ/model/raw/margin逐值复用。known12保overall＋5RX＋6scene的宏C_K90和原F0 P1闭集−open≤2pp；30RX×scene完整报告，不能把known12通过写成known42通过。真实已知y只作源风险，部署按原F0 P1预测类路由六τ。

四套τ全部固定后才打开novel received及两个父novel预测；保存新P0 unit160，并精确保留父P1 signal_features/defaulthead/ID/publicmetadata。独立CUDAempty CPU scorer从四聚合state、两unit160和τ重建全部raw/margin，验证父known12控制和P1 ratio已冻结来源、P0统计共享及路径身份后才唯一读novel trueTX，不重载NN、重fit或调τ。主与P1_known12、主与P0_ratio12配对，报告完整TX/RX/day/scene、12硬组和30诊断、AUROC/OSCR、真实新teacher/学生前向/训练/参数/祖先成本。

唯一主P0_known12通过known12及source整体宏R_U≥90%才另登记跨未见RX星地验证，控制不能替主。source是历史开发池早筛，不是最终总体/每主景C_K90/R_U90/A_K损失2pp证书。完整零目标TX标签DA、未知发现和未来匿名注册（A_N80/旧类损失2pp/后续未知保护）继续，A_N/H_OW本轮N/A。

N607单GPU teacher/train随后单GPU eval、一个独立CPU scorer；每次启动actualfree≥4096MiB、每卡新增本任务≤2，允许与其他owner共卡，不停止/修改/重启其进程。一次实际代码commit、聚焦路径/teacher冻结/采样/真实合成CP/真值隔离负测、一次直接P0/P1审查、最小登记、不可覆盖root/单preflight/archive比较/remotecompile、PID/CWD/argv/GPU/log增长及终态。非法来源/角色、非有限、碰撞或无法合法prediction保留，不评分partial、不自动重启；UNKNOWN先核原PID/root，科学低分不停止健康任务。
