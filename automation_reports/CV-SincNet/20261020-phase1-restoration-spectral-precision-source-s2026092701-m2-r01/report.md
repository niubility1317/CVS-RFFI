# 有界源频谱精度加权的复数恢复误差

状态：ANALYZED / VERIFIED。用户目标仍为完整目标low urban≥70%、其余五practical接近原F clean85.113889%；“接近”暂按≤2个百分点，是Agent操作解释，目标未达成。本次矩阵、权重规则和预算在spatial/scratch新目标分数读取前固定。

源依据是已有完整source-only observability（86aa3179）：L6300/U56700在lag16/32/64/128的平均相干最高约.143，不能假定周期补全有效；oldUNet源low urban最低功率FFT四分位的有界分母复误差约.981。该旧模型诊断既不证明当前R03仍有同样缺陷，也不证明低能量频点就是TX指纹。当前pair_loss的logpower不保留相位，因此检验有限权重下的复误差分配，而非无限放大高频或依据目标涨跌调参。

|新固定行|复杂谱残差权重|对照|
|---|---|---|
|spectral_uniform|全部256频点为1|新匹配控制|
|spectral_whitened|同source L/U全局clean PSD的有界倒数，均值1|新加权比较|

两行wave均为IQNMSE+.1logpower+.1complexspectralNMSE；后者为orthonormal FFT(estimate-clean)的复数模方加权均值，除mean|complexclean|²（下界2e-4）。该下界与原I/Q均方1e-4一致，均匀权重由Parseval等于IQNMSE。whitened先对同一63000 source L/U clean全部样本平均FFT功率，再除全频平均功率，分母加.01；w=clip(lambda/(PSD+.01),.25,4)，lambda用固定64步二分使平均权重1。两行都在首个optimizer步骤前计算并固定全源谱，uniform输出全1；不按单包、类别、RX、环境或目标调整，0TX/V/target读取。

严格只加载原合法R03（methodb01b534f、coarse20/correction20）完整权重，2140840参数joint训练，不继承任何失败或refined/spatial/scratch权重。E10/2470、modelseed2026092701/dataseed2026101951、原R02 cache1..10、.15clean/.60residual/.25full、observed+clean、batch256物理、AdamW3e-5/warm247/cos3e-6、WD1e-4/clip1、FP32/noTF32保持。L63000/U567000优化曝光、1260000IQ；coarse/corr各累计30轮，60组件epoch、50串行继承pass。三原F50/defaulthead seeds01/02/03/method4df7b94d整体冻结，mean finalcos+.3T²KL(T2)/per_frame_cap不变，实际任务sum-gradient合并后除本批IQ数一次。无新身份/RX头，0分类头拟合、0L/U TX真值。

两个新行是匹配比较，仅谱权重不同。旧per_frame_cap没有额外.1complexspectral项，仅保留历史背景，不冒充匹配第三行。固定训练谱向量和PSD为私有训练损失状态，不进入恢复模型或推理；实际mean/min/max/sourcecount、逐轮spectral loss/原损失/cap/曝光/预算和成本保存。有限输出梯度限制不保证AdamW参数步或最终身份收益。

source V1800/固定F1/七视图只诊断，无资格/排序/早停/状态更新。两fixedfinalE10均自动完整168000 clean+六practical，三原默认头独立argmax，不投票。每行全部七视图IQ/ID/预测固定后，独立无CUDA CPU读取目标truth/配对clean，报告逐F/TX/RX/day accuracy/MacroF1、修复/损坏/净纠错、距原clean差距与full256/prefix232/tail24 NMSE/logpower。推理只输入当前IQ，不读权重/PSD/F/clean/sourcecache、真实TX/RX/场景、其他帧或更新状态。源/波形检查不替代目标识别，不称三恢复器seed、新独立盲测、Phase2或在轨性能。

33项聚焦检查PASS，覆盖有界均值1/scale不变、source全人口权限、Parseval与能量floor/梯度、相位敏感、原R03 state/新checkpoint、谱状态和source行负例、actualhandle队列及truth-last前置拒绝。唯一一次独立P0/P1审查PASS，审查者独立复跑33项通过，未发现确定P0/P1。launcher首步实际R03/三F/四Ls no-query smoke，实际复谱loss和原头到IQ梯度、coarse/correction梯度、F冻结、完整初始state与checkpoint往返；0TX/V/target/optimizer，smoke权重只用四Ls、不是训练PSD；PASS立即资源队列。

新不可覆盖run/root/release，一次N607普通账号预检、单实际代码配置commit、一次归档本地/远端SHA比较和远端compile。CPUcontroller等spatial根终态且旧owned source/target真实handles退出，才在GPU6/7创建新worker，不只凭marker、不读分数放行；scratchGPU4/5和其他owner全部保留，总最多4ownGPU，每卡一个本owner，source≥18GB/target≥12GB、allowner<2。CPU2线程/本owner scorer最多1；新source scorer等scratch根终态，再共用本run score.lock，无循环。权限/代码/冲突、非有限/OOM、PSD/weight/预算/梯度/F冻结或预测失败保留产物并失败该行，无自动重试，健康行继续。

launch后实际PID/CWD/完整命令/GPU/日志增长独立核实。精确命令/CWD/路径/数据与父来源、矩阵、stop rule及产物见experiment.json。IQ/训练权重/训练PSD与向量/逐样本预测/原始日志/私有机器状态不提交。没有向另一用户对话发消息。

一次N607普通账号预检VERIFIED：新root/release不存在，原R03合法来源核实；当前spatialGPU6/7实际target和scratchGPU4/5健康任务全部保留，GPU6/7将在旧owned handles真实退出后使用，磁盘空间充足。无额外签名、数据重验证或审批流程。正式训练尚未启动。


## 实际发布、smoke 与资源队列

方法commit`ba2dbf4de934ae785330877166601a969ece65c6`已推送并独立核实远端OID；一次归档本地/远端SHA比较、远端compile/import PASS。launcher首步实际原R03/三F/四Ls no-query smoke PASS：两行完整初始state/IQ等于原R03，actual复谱loss与coarse/correction梯度有效，原默认头到IQ梯度、F无梯度和checkpoint往返有效；0TX/V/target/optimizer。smoke中的谱权重仅由四Ls FFTpower生成，不作为正式63000source训练谱。

CPUcontroller实际PID/CWD/完整命令LIVE，日志持续记录等待spatial旧owned GPU handles，读取目标分数0；此时两条新GPU模型worker尚未创建。已经独立核实旧spatial target仍LIVE并保持，scratchGPU4/5健康任务保持，未抢占/停止/热改。队列满足旧run终态且实际oldowned handles退出后才创建GPU6/7source；后续PID/GPU/训练日志增长会单独核实。当前没有新训练/源V/目标结果，不能据发布或smoke宣称恢复收益。


## 实际新GPU训练与首轮核验

spatial旧owned source/target实际handles退出后，新GPU6/7两source实际启动，释放记录scores_read0；scratchGPU4/5保持，总最多4ownGPU。新PID/CWD/完整命令/GPU及持续日志增长核实，原R03完整state初始化、2140840joint参数与三原F冻结不变。两行训练前均完成全63000source L/U clean全局谱计算，TX/V/target访问0，实际有界/均值1的256频点权重固定：

- spectral_uniform：1/10轮、247/2470步；sourcecount63000，precision min/max/mean=1/1/1；IQNMSE/complexspectralNMSE=0.0123897474/0.0123897463。梯度合并247次/126000IQ、scaled51029（observed41567/clean9462），平均scale0.745471665、范围0.00277244998至1；wave/identity/缩放后identity norm²=1002.99234/684206.208/535.747385。
- spectral_whitened：1/10轮、247/2470步；sourcecount63000，precision min/max/mean=0.25/3.40890503/0.999999999；IQNMSE/complexspectralNMSE=0.012390079/0.00335737353。梯度合并247次/126000IQ、scaled50546（observed41575/clean8971），平均scale0.748304896、范围0.00277070329至1；wave/identity/缩放后identity norm²=998.993883/683880.822/525.496988。

两行三F参考缓存各63000；均匀行precision实际全1，复杂谱NMSE与IQNMSE一致；逆谱行权重仅来自训练源全局谱，范围和均值均实际核实。没有新源V或目标准确率，不能把loss、频谱权重或实现成功当作身份收益。固定finalE10后两行自动全168000/七视图/三原头truth-last，sourceV仅诊断；CPU scorer等scratch根终态，再本run lock。旧无额外谱项的cap不是本run匹配控制。


## 完整固定目标比较

两个登记行均完成E10/2470、source V诊断、168000物理目标样本的七视图预测和独立CPU truth-last评分，实际source/target/controller进程均已退出。所有预登记行都保留，没有源分数筛选、目标反馈或选择性重跑。以下为三套原F各自默认头argmax的均值±样本标准差，一次恢复器训练的三个读出不等于三个独立恢复器种子，不做投票。原clean参考85.113888889%。

|环境|原R03|均匀复谱误差|有界逆谱精度误差|逆谱相对均匀（百分点）|
|---|---:|---:|---:|---:|
|clean|84.743056%|85.063294±0.459885%|85.061706±0.461668%|-0.001587|
|high|77.575000%|77.946230±0.219656%|77.941468±0.225029%|-0.004762|
|mid|76.491071%|77.018651±0.193624%|77.000595±0.193509%|-0.018056|
|low suburban|75.284127%|75.971230±0.090402%|75.948810±0.093342%|-0.022421|
|high urban|76.330952%|76.976190±0.144578%|76.962103±0.138283%|-0.014087|
|mid urban|70.116865%|72.023413±0.205792%|72.008532±0.193087%|-0.014881|
|low urban|63.392460%|66.378373±0.292331%|66.359127±0.307803%|-0.019246|

|判据|均匀复谱误差|有界逆谱精度误差|
|---|---:|---:|
|low urban≥70%|未达到|未达到|
|其余五环境距原clean≤2个百分点|未达到|未达到|

“接近”的≤2个百分点是本对话Agent在实验前记录的解释，用户没有指定该数值。

|距原clean差距（百分点）|均匀复谱误差|有界逆谱精度误差|
|---|---:|---:|
|clean|0.050595|0.052183|
|high|7.167659|7.172421|
|mid|8.095238|8.113294|
|low suburban|9.142659|9.165079|
|high urban|8.137698|8.151786|
|mid urban|13.090476|13.105357|
|low urban|18.735516|18.754762|

|逐原F的low urban|均匀复谱误差|有界逆谱精度误差|
|---|---:|---:|
|2026092701|66.698214%|66.693452%|
|2026092702|66.125000%|66.087500%|
|2026092703|66.311905%|66.296429%|

## 同物理clean波形差距与净纠错

配对clean与目标TX真值只由独立CPU scorer在该行全部七视图预测固定后读取。未使用oracle增益、相位或时延对齐。每个环境的full256/prefix232/tail24 NMSE、logFFTpower均值/中位数/P90及逐F/TX/RX/day、混淆矩阵保存于results.json。

|目标low urban|原received|原R03|均匀复谱误差|有界逆谱精度误差|
|---|---:|---:|---:|---:|
|full_nmse均值|1.987855210|0.076996611|0.076305022|0.076381874|
|prefix232_nmse均值|1.998446002|0.068349912|0.067879160|0.067945504|
|tail24_nmse均值|1.928131431|0.157928481|0.155220509|0.155420242|
|log_power_mse均值|0.160772502|0.023813475|0.023519029|0.023524457|

|low urban修复/损坏/净修复|相对received|相对原R03|
|---|---:|---:|
|均匀复谱误差|190464/25498/164966|32985/17936/15049|
|有界逆谱精度误差|190355/25486/164869|32962/18010/14952|

计数为168000个物理样本在三原F上的504000次决策合计，不能视为504000个独立样本。

## 源诊断、实际执行与成本

|源V（固定F1）|clean|low urban|mid urban|
|---|---:|---:|---:|
|均匀复谱误差|98.555556%|83.888889%|92.222222%|
|有界逆谱精度误差|98.555556%|83.888889%|92.166667%|

源V只诊断，无候选资格、排序、早停或持久状态更新。仅原R03 coarse20+correction20初始化，joint E10后各累计30轮，60组件epoch/50串行继承pass；2140840可训练参数，没有新增网络/分类头或失败/refined权重继承。原三F50/defaultheads整体冻结，0头拟合、0L/U TX真值与目标训练读取。

- 均匀复谱误差：全63000sourceL/U clean全局PSD，TX/V/target读取均0；实际precision min/max/mean为1/1/1，source globalpower mean为1；训练前固定，向量不用于推理。
- 均匀复谱误差：E10/2470，L63000/U567000物理优化曝光、1260000IQ；实际梯度合并2470次/1260000IQ，scaled441971（observed389469/clean52502），平均scale0.781994127、范围0.00202711136至1；训练/保存1120.000s，含准备/容量等待1138.177s，峰值分配5550880768bytes；七视图目标预测合计804.305s。
- 有界逆谱精度误差：全63000sourceL/U clean全局PSD，TX/V/target读取均0；实际precision min/max/mean为0.25/3.40890503/0.999999999，source globalpower mean为1；训练前固定，向量不用于推理。
- 有界逆谱精度误差：E10/2470，L63000/U567000物理优化曝光、1260000IQ；实际梯度合并2470次/1260000IQ，scaled444063（observed389688/clean54375），平均scale0.781203902、范围0.0017409533至1；训练/保存1128.855s，含准备/容量等待1146.373s，峰值分配5550880768bytes；七视图目标预测合计796.243s。

两个新行具有相同原R03初始化、2140840参数、E10/2470、原R02数据/缓存/路由/observed+clean/持续顺序、AdamW3e-5/warm247/cos3e-6/WD1e-4/clip1/FP32和三F指导/cap。两者wave均增加.1complexspectralNMSE，均匀项由Parseval等价IQNMSE，逆谱项权重由同源全局PSD生成，范围[.25,4]且均值1。唯一比较项是均匀或有界逆谱权重；旧per_frame_cap没有额外谱项，不能冒充匹配第三行。全部source日志的谱/波形/原F分项及预算已核对，均匀谱NMSE与IQNMSE一致。源旧UNet低能量复误差不是TX指纹证据，不预设逆谱必然有效；输出梯度cap不保证AdamW有限步或最终效果。

恢复推理只有当前receivedIQ，不读取PSD/权重/F/clean/sourcecache/真实TX/RX/环境、其他帧或投票，不更新状态。GPU6/7新worker只在spatial根终态且旧owned source/target真实handles退出后启动，放行scores_read0；scratchGPU4/5健康任务保留。CPU scorer等scratch根终态再共用本runlock。33项聚焦检查、一次独立P0/P1审查及审查者独立33项、真实原R03/三F/四Ls no-query smoke、单归档SHA比较、远端编译和实际PID/CWD/cmd/GPU/日志增长均通过；source/target/controller完整日志解析无技术错误。方法commit`ba2dbf4de934ae785330877166601a969ece65c6`。未向另一用户对话发送消息。

本轮仍未达到用户完整目标。单恢复器seed的三个读出不是三个恢复器训练，已暴露基准不称新独立盲测、Phase2或在轨性能；波形和源分数不替代完整目标识别。
