# 从零学习恢复映射时加入冻结原 F 指导

状态：ANALYZED / VERIFIED。用户目标为目标low urban≥70%，其余五practical接近原clean85.113889%。“接近”暂按≤2个百分点，是Agent操作解释。当前最好66.406548%，完整目标未实现。本次假设和两行预算在teacheragreement的新目标分数读取前固定；代码依据为原R03先以波形目标训练coarse20+correction20，近期方案再做E10后置F约束。检验从原逆映射学习开始加入身份保真是否不同，不根据目标成绩选起点、调参或选择性重跑。

|固定行|恢复器初始化|训练目标|训练F开销|
|---|---|---|---|
|scratch_wave|相同新ComplexTFUNet随机权重|IQNMSE+.1logpower|0教师缓存/前向|
|scratch_F_guided|相同新ComplexTFUNet随机权重|同波形+三原F finalcos+.3T²KL(T2)，per_frame_cap|三F参考缓存/训练前向|

两行均1305958可训练参数、相同原始ComplexTFUNet架构，零temporal输出初始严格等于当前receivedIQ。无任何恢复器初始化/resume；原R03 b01b534f只作已经核实的数据契约一致的源/目标读出参照，不载入新模型。禁止晋级任何失败或按测试挑选权重。三原F seeds2026092701/02/03/method4df7b94d/defaulthead整体冻结，无新身份/RX分类头或单独头拟合，L/U TX真值均不读取。

原source L6300/U56700、R02缓存epochs1..20按1..20顺序重复一次，随机shuffle由dataseed2026101951持续生成40次，原.15clean/.60residual/.25full路由不变。observed+samephysicalclean，batch256物理→512IQ，末批按实际数归一化。modelseed2026092701，E40/9880，L252000/U2268000优化曝光、5040000IQ。两臂AdamW2e-4/warm247/cos2e-5、WD1e-4/clip1、FP32/noTF32。guided从首步计算mean3F finalcos+.3T²KL；逐IQ wave/identity sum-loss输出梯度，经既有per_frame_cap后除实际IQ数一次。wave臂不建训练教师缓存、不执行教师前向，输出梯度严格等于波形mean-loss梯度。损失/实际cap/路由/LU曝光/三F训练调用和IQ、成本均保留。输出梯度限制不保证AdamW有限步或最终身份收益。

两新行的40轮从零训练是匹配比较，旧R03复合架构/40+10继承训练不冒充相同训练成本或结构。原F50训练已固定，不更新，三原F独立读出不是三个恢复器seed。恢复推理只输入当前receivedIQ，不调用F、clean、缓存、TX/RX/环境标签、其他帧或投票，不更新模型/运行状态。

source V1800/固定F1/七视图仅诊断，固定finalE40，无源资格/早停/排序/状态更新。每行训练冻结后自动168000物理目标clean+六practical，三原头各自argmax。每行七视图IQ/ID/预测全部固定后独立无CUDA CPU连接目标truth和同物理clean，报告逐F/TX/RX/day、准确率/MacroF1、修复/损坏/净纠错、距原clean差距及full256/prefix232/tail24 NMSE/logpower。波形、源分数或实现检查不替代目标结果；已暴露基准不称新独立盲测、Phase2或在轨性能。

31项聚焦检查PASS，覆盖scratch/state/参数量/零输出/批独立/checkpoint与权限预算、wave梯度等价、guided原头到IQ和冻结F、缓存轮次/LR、source行负例及truth-last包。唯一一次独立P0/P1审查PASS，审查者独立复跑31项通过，未发现确定P0/P1。launcher首步真实原F/四Ls no-query smoke：相同随机初态、原头到IQ与新head梯度、F完整状态/新checkpoint往返，0TX/V/target/optimizer；PASS立即进入资源队列。

新run/root/release不可覆盖。一次N607普通账号预检核实合法原F/缓存/基线、空间、现有owner和GPU4/5；CPUcontroller先等agreement根终态且其owned source/target真实PID均退出，再创建两worker，不能只看终态marker，放行不读分数。spatial GPU6/7全部保留，总最多4个本owner GPUworker，每卡一个本owner，source≥18GB/target≥12GB、allowner<2。CPU2线程/本owner scorer≤1，新source scorer等spatial根终态，再共用本run score.lock；source实际退出才同卡target。权限/代码/冲突、非有限/OOM、预算/梯度/F冻结或预测失败则保留产物、失败该行、不自动重试；健康行继续。

代码配置一个Gitcommit；一次归档本地/远端SHA比较、远端compile；launch后实际PID/CWD/完整命令/GPU/日志增长独立核实。精确命令、CWD、原始来源、路径、矩阵和stop rule见experiment.json。IQ/权重/逐样本预测/日志和私有机器状态不提交。未向另一用户对话发送消息。

一次N607普通账号预检VERIFIED：新root/release不存在，合法原R03只用作baseline；agreement已终态，当前其他owner与spatial健康任务完整保留，GPU4/5可用且启动队列仍核实旧owned handles真实退出，磁盘空间充足。无额外签名、数据重验证或审批流程。正式训练尚未启动。


## 实际发布与首轮核验

方法commit`933e93c88668cd41ad5fe2969394e194ffb9f705`推送及远端OID已核实，单归档SHA比较、远端compile/import PASS。launcher首步实际原三F/四Ls no-query smoke PASS：两臂完整随机初态一致、初始输出等于receivedIQ，新temporal零head收到梯度，原默认头到IQ梯度有效，F无梯度且完整state不变，新checkpoint往返一致；0TX/V/target/optimizer、0恢复器权重加载。PASS立即继续，agreement根终态且旧owned handles实际退出后两新GPU4/5source启动，spatialGPU6/7保持。

两source实际PID/CWD/完整命令/GPU及日志持续增长已核实：

- scratch_wave：2/40轮，cacheepoch2，494/9880步，梯度调用494、IQ252000，scaled0（observed0/clean0），平均scale1、范围1至1；wave/identity/缩放后identity norm²3078.16158/0/0。
- scratch_F_guided：1/40轮，cacheepoch1，247/9880步，梯度调用247、IQ126000，scaled79115（observed56325/clean22790），平均scale0.528934262、范围0至1；wave/identity/缩放后identity norm²2043.74062/567944.192/1760.45887。

wave行无训练教师缓存，feature/posterior loss、identity norm²与scaled均实际0；guided行三F sourceclean缓存63000，标签和target读取0。两行1305958可训练参数，R03未初始化新模型。当前只有训练执行证据，无新源V或目标分数；原预算E40不变，完成后自动各行全168000/七视图/三原头truth-last。CPU评分先等spatial根终态，再本run lock；输出梯度性质不当作性能收益，用户完整目标仍未实现。


## 完整固定目标比较

两个登记行均完成E40/9880、source V诊断、168000物理目标样本的七视图预测和独立CPU truth-last评分，实际source/target/controller进程均已退出。所有预登记行都保留，没有源分数筛选、目标反馈或选择性重跑。以下为三套原F各自默认头argmax的均值±样本标准差，一次恢复器训练的三个读出不等于三个独立恢复器种子，不做投票。原clean参考85.113888889%。

|环境|原R03|从零纯波形|从零三F指导|三F指导相对波形（百分点）|
|---|---:|---:|---:|---:|
|clean|84.743056%|85.151984±0.389777%|85.099802±0.410792%|-0.052183|
|high|77.575000%|77.388095±0.356218%|78.607540±0.285997%|+1.219444|
|mid|76.491071%|76.321230±0.260892%|77.634524±0.289793%|+1.313294|
|low suburban|75.284127%|74.984325±0.175752%|76.400000±0.243570%|+1.415675|
|high urban|76.330952%|76.237897±0.242131%|77.697421±0.244616%|+1.459524|
|mid urban|70.116865%|70.256349±0.171724%|72.568452±0.134251%|+2.312103|
|low urban|63.392460%|63.684325±0.383174%|66.818056±0.174447%|+3.133730|

|判据|从零纯波形|从零三F指导|
|---|---:|---:|
|low urban≥70%|未达到|未达到|
|其余五环境距原clean≤2个百分点|未达到|未达到|

“接近”的≤2个百分点是本对话Agent在实验前记录的解释，用户没有指定该数值。

|距原clean差距（百分点）|从零纯波形|从零三F指导|
|---|---:|---:|
|clean|-0.038095|0.014087|
|high|7.725794|6.506349|
|mid|8.792659|7.479365|
|low suburban|10.129563|8.713889|
|high urban|8.875992|7.416468|
|mid urban|14.857540|12.545437|
|low urban|21.429563|18.295833|

|逐原F的low urban|从零纯波形|从零三F指导|
|---|---:|---:|
|2026092701|64.009524%|66.969643%|
|2026092702|63.261905%|66.627381%|
|2026092703|63.781548%|66.857143%|

## 同物理clean波形差距与净纠错

配对clean与目标TX真值只由独立CPU scorer在该行全部七视图预测固定后读取。未使用oracle增益、相位或时延对齐。每个环境的full256/prefix232/tail24 NMSE、logFFTpower均值/中位数/P90及逐F/TX/RX/day、混淆矩阵保存于results.json。

|目标low urban|原received|原R03|从零纯波形|从零三F指导|
|---|---:|---:|---:|---:|
|full_nmse均值|1.987855210|0.076996611|0.074371861|0.075688056|
|prefix232_nmse均值|1.998446002|0.068349912|0.065216069|0.066355540|
|tail24_nmse均值|1.928131431|0.157928481|0.159914689|0.164050258|
|log_power_mse均值|0.160772502|0.023813475|0.022806051|0.023197293|

|low urban修复/损坏/净修复|相对received|相对原R03|
|---|---:|---:|
|从零纯波形|180034/28646/151388|38153/36682/1471|
|从零三F指导|192491/25309/167182|51795/34530/17265|

计数为168000个物理样本在三原F上的504000次决策合计，不能视为504000个独立样本。

## 源诊断、实际执行与成本

|源V（固定F1）|clean|low urban|mid urban|
|---|---:|---:|---:|
|从零纯波形|98.500000%|82.222222%|89.000000%|
|从零三F指导|98.500000%|84.833333%|91.888889%|

源V只诊断，不作候选资格、排序、早停或持久状态更新。新恢复器均为原ComplexTFUNet架构、1305958可训练参数、相同随机初态，未加载任何R03或失败恢复器权重；原R03只作合法源/目标读出参照。原三F50/defaultheads全冻，未拟合新身份/RX头，L/U TX标签和目标训练读取均0。训练40轮，原R02缓存epochs1..20重复一次，shuffle持续。

- 从零纯波形：E40/9880，L252000/U2268000物理优化曝光、5040000IQ；实际梯度合并9880次/5040000IQ，scaled0（observed0/clean0），平均scale1、范围1至1。三F训练前向0次/0IQ；不含launcher smoke、参考缓存和源/目标预测开销。
- 从零纯波形：训练/保存2258.512s，含准备/容量等待2260.111s，峰值分配2382391808bytes；七视图目标预测合计673.849s。
- 从零三F指导：E40/9880，L252000/U2268000物理优化曝光、5040000IQ；实际梯度合并9880次/5040000IQ，scaled1827573（observed1585900/clean241673），平均scale0.772777351、范围0至1。三F训练前向29640次/15120000IQ；不含launcher smoke、参考缓存和源/目标预测开销。
- 从零三F指导：训练/保存3302.114s，含准备/容量等待3320.206s，峰值分配4207366656bytes；七视图目标预测合计664.068s。

两新行相同初始权重、架构、40轮/9880步、63000物理L/U、原缓存/路由/observed+clean、持续shuffle和AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32。唯一比较项为从首步加入mean3F finalcos+.3T²KL/per_frame_cap；纯波形行无训练教师缓存/前向，所有feature/posterior loss、identity norm²与scaled计数实际为0。指导行sourceclean参考63000，训练计数来自每步执行；全部原F完整参数/缓冲保持冻结。旧R03复合结构与40+10继承训练不同，不冒充相同成本/结构控制。逐帧norm-cap不是AdamW有限参数步或最终收益保证。

恢复推理只有当前receivedIQ，不运行F条件、不读取clean/sourcecache/真实TX/RX/场景、其他帧或投票，不更新模型或持久状态。新GPU4/5启动前，agreement root终态且其owned source/target实际PID均退出，释放不读分数；spatialGPU6/7健康进程保留。全部source/target/controller日志完整解析无技术错误。31项聚焦检查、一次独立P0/P1审查及审查者独立31项通过、launcher首步实际三原F/四Ls no-query smoke、单归档SHA比较/编译和PID/CWD/cmd/GPU/日志增长均已验证。方法commit`933e93c88668cd41ad5fe2969394e194ffb9f705`，没有向另一用户对话发送消息。

本轮仍未达到用户完整目标。一个恢复器seed的三原F读出不等于三恢复器训练，已暴露基准不称新独立盲测、Phase2部署或在轨性能；波形误差和源分数不替代完整目标识别。
