# 原 F 多尺度汇总与时序特征辅助恢复

状态：ANALYZED / VERIFIED。用户目标仍为完整目标low urban≥70%、其余五practical接近原F clean85.113889%；“接近”暂按≤2个百分点，是Agent操作解释。登记时已测最好66.384127%，目标未完成。本轮结构、输入、预算与顺序在当前局部敏感度候选的新目标结果返回前固定；不依靠目标分数制定或筛选。

源代码依据：旧semantic_feedback把原F1的全局160维向量用于FiLM；layerwise_refine只在训练增加中间层距离。本轮检验原F的局部时序特征能否在恢复前向中提供信息，不复用或晋级任何旧失败权重。源逐F梯度的局部冲突只能说明引导方向有差异，不能证明新结构有效。另一对话保持独立，没有发送指令。

|新固定行|冻结F1条件|对比|
|---|---|---|
|pooled_features|各native map归一化后沿时间平均，再广播|新匹配控制|
|spatial_features|保留各native map时间位置，再插值|新时序条件|

均冻结原合法R03完整coarse/correction（methodb01b534f52daf9cae9e940ed6ec6f9243bccbdbe，各20轮）及原F1/defaulthead（seed2026092701，method4df7b94d0fd8ce29056bd8236364e45afa92e07f）。从当前received IQ计算R03anchor，再用原F1提取t1的72×64及t2的96×32原生maps。每map/frame除自身RMS，平方范数下界1e-4；之后分别池化或保留时序。通过相同learned1×1 projection和fusion，按t1/t1/t2/t2注入新Core四尺度32/64/96/128。线性插值align_corners=False是架构上的特征网格对应，不是oracle物理时延对齐或信道参数恢复。

新残差场输入concat(received,anchor)/anchorRMS（norm²下界1e-4），输出anchor＋anchorRMS×field；两臂相同零输出2通道head，初始IQ严格原R03。925378可训练参数、3299211总参数，含2140840冻结R03和232993冻结F1。只有新field的IQ编码/解码、projection/fusion训练；原R03/F1前向no_grad并保持eval，完整参数/缓冲状态冻结。没有新身份/RX分类头，没有运行后验选类、投票或目标更新。F1是明确的运行条件，增加参数/延迟；三套原F读出不等于未见识别器泛化或三次恢复器训练。

原63000源L6300/U56700、R02缓存epochs1..10、.15clean/.60residual/.25full、observed+clean、batch256、modelseed2026092701/dataseed2026101951保持。新field E10/2470，L63000/U567000优化曝光、1260000 IQ；原coarse/corr各20不更新，加field10，共50组件/串行epoch。AdamW2e-4/warm247/cos至2e-5、WD1e-4/clip1、FP32/noTF32。wave仍IQNMSE＋0.1logpower，identity为三原F finalcos＋0.3T²KL(T2)均值；沿用逐帧per_frame_cap，输出任务sum-gradient合并后除实际observed+clean数量一次，再仅更新field。记录梯度cap、field head/encoder/projection/fusion实际非零梯度步数、运行F训练调用/IQ、准备/训练/预测成本。零head首步仅head收到非零梯度是预期，不将source分数或激活比例作为资格gate。

两新臂匹配相同父/F、全部field参数/初始state、数据/预算/loss/optimizer，仅F map时间池化不同。旧joint梯度候选、旧直接残差场的架构、优化参数和预算不同，不称三行匹配对照。每行均保留received/原R03原头读出作基础参照，历史最好仅作已暴露目标背景，不从其初始化。

source V1800/原F1/七视图只诊断，无源资格/早停/排序。两固定最终E10均自动完整目标168000、clean＋六practical、三原默认头独立argmax。每行七视图IQ/ID/预测全部固定后，独立无CUDA CPU才连接truth/pairedclean，报告逐F/TX/RX/day accuracy/MacroF1、修复/损坏/净纠错、距clean差距及同物理full256/prefix232/tail24 NMSE/logpower。运行输入仅当前IQ及其anchor/F maps，无clean/sourcecache、真实信道/TX/RX/场景、其他帧/视图或状态更新。目标不回流配置/顺序/预算/重跑，不称新盲测、Phase2或在轨性能。

58项聚焦检查与唯一一次独立P0/P1审查PASS（审查者独立58项）。launcher首步实际R03/三F/四Ls no-query smoke，初始IQ/state匹配、native posterior到IQ与新field head梯度、F/parent无梯度、F完整状态及checkpoint重建往返，0TX/V/target/optimizer，PASS立即进入已登记资源队列。

一次N607普通账号预检VERIFIED，原R03合法、新root/release不存在，当前局部J在GPU6/7的真实target PID/CWD/cmd LIVE，agreement在GPU4/5，全部保留。新CPUcontroller先等局部J根终态并核实所有旧owned source/target真实handle退出，才创建GPU6/7新worker，不能只凭终态文件；不读取分数来放行。后续source≥18GB/target≥12GB、allowner<2、每卡一个本owner、最多4个本owner GPUworker。CPU2线程，scorer最多1；新source评分等agreement根终态，再与本run target共用score.lock；source实际退出才同卡target。非法输入/权限/代码、冲突、非有限/OOM、预算/梯度执行/冻结state或预测失败时保留并失败该行，无自动重试，健康行继续。

一个实际代码配置commit、不可覆盖root、一次归档本地/远端SHA比较及remotecompile、真实smoke、controller/worker PID/CWD/cmd/GPU/日志增长随后核实。精确命令、环境/CWD、路径、数据与父来源、seed、矩阵及产物见experiment.json。IQ/权重/逐样本预测/原始日志和私有机器状态不进Git。


## 实际发布与首轮核验

方法commit `79088f493863d81f5408d2577ff858cf27b44922`已推送并独立核实远端OID，一次归档本地/远端SHA比较及远端compile/import均PASS。实际R03/三F/四Ls no-query smoke PASS：两臂完整初始IQ/state等于原R03，新field零输出head及原头到IQ梯度有效，F无梯度，新checkpoint往返一致，0V/TX/target/optimizer更新。两臂实际925378可训练/3299211总参数，原R03和232993参数的原F1冻结，新增field/projection/fusion；PASS立即训练。

两条实际source worker的PID/CWD/完整命令/GPU6/7及持续日志增长已核实，三F参考缓存各63000条。实际field前向/非零梯度与逐帧cap计数：

- pooled_features：2/10轮、494/2470步，梯度合并调用494、IQ252000条，实际缩放97993（observed82946/clean15047），平均scale0.75141345、范围0.0025452143至1，wave/identity/缩放后identity norm²总和1991.3193/1476179.7/1009.9166。
- spatial_features：2/10轮、494/2470步，梯度合并调用494、IQ252000条，实际缩放98117（observed82892/clean15225），平均scale0.75181723、范围0.0022333772至1，wave/identity/缩放后identity norm²总和1991.0933/1461570.7/1011.3796。
- pooled_features field：实际调用494、IQ252000；head/encoder/projection/fusion非零梯度步数494/493/493/493。原R03/F1冻结，仅field参与optimizer。
- spatial_features field：实际调用494、IQ252000；head/encoder/projection/fusion非零梯度步数494/493/493/493。原R03/F1冻结，仅field参与optimizer。

首轮核验时尚无新源V或目标准确率。实际缩放不代表性能改善或参数步波形保证；两臂固定E10后分别自动完整目标测试，source V不作资格筛选，用户目标仍未达成。最终状态与结果以完整结果节为准。局部J旧GPU6/7的source/target实际handles退出后才创建新worker，放行不读分数，释放记录与退出证据已核实。agreement GPU4/5任务保持，CPU评分等其终态后执行。两新行只比较pooled/spatial原生maps；旧不同结构/预算行不称匹配对照。


## 完整固定目标比较

两个登记行均完成E10/2470、source V诊断、168000物理目标样本的七视图预测和独立CPU truth-last评分，实际source/target/controller进程均已退出。所有预登记行都保留，没有源分数筛选、目标反馈或选择性重跑。以下为三套原F各自默认头argmax的均值±样本标准差，一次恢复器训练的三个读出不等于三个独立恢复器种子，不做投票。原clean参考85.113888889%。

|环境|原R03|原F全局池化条件|原F多尺度时间条件|时间条件相对池化（百分点）|
|---|---:|---:|---:|---:|
|clean|84.743056%|84.984325±0.444428%|85.045437±0.485970%|+0.061111|
|high|77.575000%|77.827976±0.268339%|77.444643±0.433308%|-0.383333|
|mid|76.491071%|76.988095±0.234154%|76.534921±0.339950%|-0.453175|
|low suburban|75.284127%|75.872222±0.264117%|75.463690±0.202181%|-0.408532|
|high urban|76.330952%|76.913095±0.163459%|76.457540±0.348147%|-0.455556|
|mid urban|70.116865%|71.949802±0.150308%|71.102778±0.088497%|-0.847024|
|low urban|63.392460%|65.980556±0.256688%|65.112897±0.140864%|-0.867659|

|判据|原F全局池化条件|原F多尺度时间条件|
|---|---:|---:|
|low urban≥70%|未达到|未达到|
|其余五环境距原clean≤2个百分点|未达到|未达到|

“接近”的≤2个百分点是本对话Agent在实验前记录的解释，用户没有指定该数值。

|距原clean差距（百分点）|原F全局池化条件|原F多尺度时间条件|
|---|---:|---:|
|clean|0.129563|0.068452|
|high|7.285913|7.669246|
|mid|8.125794|8.578968|
|low suburban|9.241667|9.650198|
|high urban|8.200794|8.656349|
|mid urban|13.164087|14.011111|
|low urban|19.133333|20.000992|

|逐原F的low urban|原F全局池化条件|原F多尺度时间条件|
|---|---:|---:|
|2026092701|66.276786%|65.275000%|
|2026092702|65.823810%|65.020238%|
|2026092703|65.841071%|65.043452%|

## 同物理clean波形差距与净纠错

配对clean与目标TX真值只由独立CPU scorer在该行全部七视图预测固定后读取。未使用oracle增益、相位或时延对齐。每个环境的full256/prefix232/tail24 NMSE、logFFTpower均值/中位数/P90及逐F/TX/RX/day、混淆矩阵保存于results.json。

|目标low urban|原received|原R03|原F全局池化条件|原F多尺度时间条件|
|---|---:|---:|---:|---:|
|full_nmse均值|1.987855210|0.076996611|0.077199061|0.077246379|
|prefix232_nmse均值|1.998446002|0.068349912|0.068576240|0.068668792|
|tail24_nmse均值|1.928131431|0.157928481|0.158074903|0.157625854|
|log_power_mse均值|0.160772502|0.023813475|0.023818740|0.023857655|

|low urban修复/损坏/净修复|相对received|相对原R03|
|---|---:|---:|
|原F全局池化条件|188945/25984/162961|32207/19163/13044|
|原F多尺度时间条件|186318/27730/158588|32332/23661/8671|

计数为168000个物理样本在三原F上的504000次决策合计，不能视为504000个独立样本。

## 源诊断、实际执行与成本

|源V（固定F1）|clean|low urban|mid urban|
|---|---:|---:|---:|
|原F全局池化条件|98.444444%|84.222222%|91.222222%|
|原F多尺度时间条件|98.500000%|84.611111%|91.333333%|

源V只作诊断，无候选资格、排序、早停或参数/统计更新。原R03 coarse20+correction20整体冻结，新增IQ场10轮；共50组件/串行继承epoch。原F1原始50轮/defaulthead整体冻结，运行时由当前R03 anchor计算原生t1/t2特征图，增加232993个冻结参数与每次前向成本。新增field及投影/融合925378个可训练参数，模型合计3299211个参数。既有三F指导器及三个独立读出保持。0新身份/RX头拟合、0L/U TX标签、0目标训练或运行时更新。

- 原F全局池化条件：E10/2470，L63000/U567000物理优化曝光、1260000IQ；field实际调用2470次/1260000IQ，head/encoder/projection/fusion非零梯度步数分别为2470/2469/2469/2469；原R03 coarse/correction更新均0，F/默认头更新0。运行时F训练调用2470次/1260000IQ，来自实际field调用计数，不含源特征缓存和预测前向。
- 原F全局池化条件：逐帧cap实际缩放450633（observed394373/clean56260），平均scale0.778464799，范围0.0012082241至1；训练/保存1178.124s，含准备/容量等待1196.185s，峰值分配3333905408bytes；七视图目标预测合计968.399s。
- 原F多尺度时间条件：E10/2470，L63000/U567000物理优化曝光、1260000IQ；field实际调用2470次/1260000IQ，head/encoder/projection/fusion非零梯度步数分别为2470/2469/2469/2469；原R03 coarse/correction更新均0，F/默认头更新0。运行时F训练调用2470次/1260000IQ，来自实际field调用计数，不含源特征缓存和预测前向。
- 原F多尺度时间条件：逐帧cap实际缩放451576（observed393839/clean57737），平均scale0.778689192，范围0.00200625136至1；训练/保存1240.170s，含准备/容量等待1259.641s，峰值分配3349289984bytes；七视图目标预测合计1054.117s。

两新行相同初始化、925378可训练参数/3299211总参数、运行时F、训练数据与逐轮路由/曝光、E10、AdamW2e-4/warm247/cos2e-5、损失和逐帧cap，仅原生图是否保留时间位置不同。池化条件是本run新训练的匹配控制；旧joint更新和旧FiLM/全局160条件架构及预算不同，仅保留历史背景，不冒充匹配控制。输出梯度norm-cap不是AdamW有限参数步或最终收益保证。零输出头第一步屏蔽上游field梯度，后续实际非零梯度计数如上。

运行时只接收当前received IQ、冻结R03 anchor及其冻结F1特征图，不访问pairedclean、source缓存、真实TX/RX/环境、其他帧或投票；指导与读出均使用已核实原F，因此不声称未见过的识别器迁移。新GPU worker启动前，前序tangent run的run-owned source/target实际PID均退出；已记录PREVIOUS_OWNED_GPU_HANDLES_EXITED且scores_read=0。全部source/target/controller日志完整解析无技术错误。58项聚焦检查、一次独立P0/P1审查、真实原R03/三F/新checkpoint no-query smoke、单归档SHA比较、编译和PID/CWD/命令/GPU/日志增长均已验证。方法commit`79088f493863d81f5408d2577ff858cf27b44922`。未给另一用户对话发消息。

本轮仍未达到用户完整目标。单恢复器种子和已暴露目标基准不作为三seed恢复结论、新独立盲测、Phase2部署或在轨性能；波形误差与源准确率不替代完整目标识别。
