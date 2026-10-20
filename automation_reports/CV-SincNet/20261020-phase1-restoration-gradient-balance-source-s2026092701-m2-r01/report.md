# 整批与逐帧的识别梯度范数上限

状态：ANALYZED / VERIFIED。目标仍是完整目标测试low urban≥70%，其余五practical接近原F clean85.113889%（Agent暂用≤2个百分点口径）；分别报告同物理clean的波形差距和冻结原默认头真实TX识别/净纠错。登记时已测最好low urban为65.309127%；本轮逐帧约束将已测最好提高至66.384127%，完整目标仍未达成。

source依据：当前gradient_projection的scalar_sum控制行E5/1235、630000条训练IQ，identity梯度norm²总和1191929.8158283，wave为5008.57419455，累计RMS范数比15.426531。该统计来自合法L/U训练，不是逐帧或逐step比，不证明幅度失衡就是性能原因，也不是目标证据。source_gradient_evidence.json仅含聚合统计，0新模型前向/更新/目标读取。新范数约束在该控制行新目标准确率被访问前固定，不热改其或投影行。

|科学比较行|执行|来源|
|---|---|---|
|scalar_sum|复用已登记健康行，不重复训练|20261020-phase1-restoration-gradient-projection-source-s2026092701-m2-r01/scalar_sum，method9f2d8f65e507809de396af5fdb89f11b749759b1，固定最终E10/2470|
|global_cap|本run新训练，GPU4|每个observed+clean整批的范数约束|
|per_frame_cap|本run新训练，GPU5|每个恢复IQ帧的范数约束|

两个新行都从原合法R03（coarse20/corr20，method b01b534f52daf9cae9e940ed6ec6f9243bccbdbe）完整相同state初始化，2140840参数，无新模块；不从当前控制或任何失败精修权重初始化。原wave=paired IQ NMSE＋0.1logpower，identity=三原F各自final160cosine＋0.3×T²KL（T2）的均值，损失数值不变。两者先求当前恢复IQ2×256实坐标上的任务sum-loss梯度gw/gi，s=min(1,sqrt(norm²(gw)/norm²(gi)))，然后(gw+s×gi)/实际IQ批量向原恢复器反传一次。global用整批norm²总和、同一个s；per-frame各帧独立计算s。identity norm²≤1e-20则s=1；非零identity而wave为0则s=0。s停止梯度且只缩小不放大identity方向，不作冲突投影。第一任务求导保留共享图，无Hessian/教师更新。

本方法是source输出梯度的上限，不是参数裁剪、论文GradNorm或PCGrad。global约束批量总体，不保证各帧identity≤wave；per-frame可能更大幅削弱总体identity。共享Jacobian、AdamW及clip也不保证参数步或最终波形/识别改善。记录calls/IQ/conflict/scaled、observed/clean缩放数与scale和、scale均值/最小/最大、前后dot及wave/identity/identity-after norm²。零实际缩放是科学结果，不作资格或技术失败门槛。

scalar控制已在本轮设计中明确固定为前run的scalar_sum E10/2470，与本轮相同原R03、数据/缓存、seed/顺序/路由、paired loss/三F、AdamW日程和分离输出梯度计算。仅在它完成后复用已有完整源与目标结果进行匹配比较，不重新预测或按分数选控制，不把它的权重传给新模型。若控制技术失败，保留并报告缺项，不能静默声称完整匹配比较。target_matrix明确新目标预测仅两新行，external_control为只读比较引用。

原63000 L/U与R02缓存epochs1..10、.15clean/.60residual/.25full、observed+clean等权、batch256、modelseed2026092701/dataseed2026101951保持。新两行各jointE10/2470、L63000/U567000优化曝光、1260000IQ；原coarse/corr各累计30轮、60组件epoch、串行继承50pass，无新增网络epoch。重置AdamW3e-5、warm247/cos至3e-6、WD1e-4/clip1、FP32/TF32关闭。原F50 CROSS_RX三seed2026092701/02/03及原默认头（method4df7b94d0fd8ce29056bd8236364e45afa92e07f）冻结参数/统计，0L/U TX真值/F更新/新头拟合。复用同数据契约与来源，不重复数据验证或增加哈希/签名链。

推断仍为原网络当前单帧receivedIQ一次coarse/corrector，无运行F或梯度、normcap、信道oracle、RX/场景、clean、其他帧/视图、迭代或状态更新。source V原1800条/原F1/七视图只诊断，无早停/资格/NMSE gate/排序/状态更新。两新固定最终E10都自动完整168000物理目标clean+六practical/三原默认头，各自argmax，不投票或按环境拼接；每行七视图IQ/ID/预测全部固定后独立无CUDA CPU才连接truth/pairedclean。报告逐F/TX/RX/day accuracy/MacroF1、修复/损坏/净修复、距clean差距、无oracle对齐full256/prefix232/tail24 NMSE/logFFTpower均值/中位数/P90及耗时/显存、相对复用标量控制的差值。目标成绩不回流配置/预算/权重/顺序/选择性重跑，一个恢复器seed与三读出不冒充三个恢复器seed、新盲测、Phase2或在轨性能。

50项聚焦检查与唯一一次独立P0/P1审查PASS（审查者独立复跑50项）。launcher首步实际R03/三F/四source Ls no-query smoke：初始完整IQ/state相同、有限非零merged coarse/corr及原头到IQ梯度、F无梯度、新checkpoint往返，0V/TX/target/optimizer，PASS立即训练。一次N607普通账号预检VERIFIED，原R03合法、新root不存在、GPU4/5各24124MB空闲无PID、空间≥8GB；GPU6/7现有source PID/CWD/cmd LIVE并保留。每卡一条本owner、计入allowner<2，重叠最多4本owner GPUworkers，source≥18GB/target≥12GB。CPU2线程/scorer最多1，新source scorer先等前gradient_projection complete/failure，再与本run target共用score.lock；实际source进程退出才同卡target，不等另一新行。

停止规则：非法来源/权限/代码、冲突、非有限/OOM、预算/缩放执行或预测失败时保留产物并技术失败该行，无自动重试，健康行继续，不停止/重启/热改他人任务。一个代码配置commit、不可覆盖root、一次归档本地/远端SHA比较及remotecompile/import，随后核实PID/CWD/cmd/GPU/日志增长。命令、环境/CWD、矩阵、路径、资源、停止规则和产物见experiment.json；原IQ、权重、逐样本预测、日志及私有运行状态不进Git。另一用户对话未被发送指令或改动。


## 实际发布与首轮核验

方法commit `9866f24dab7097294d585069bfcfe529d3346ade`已推送并独立核实远端OID，一次归档本地/远端SHA比较及远端compile/import均PASS。实际R03/三F/四Ls no-query smoke PASS：两臂完整初始IQ/state等于原R03，coarse/correction合并梯度及原头到IQ梯度有效，F无梯度，新checkpoint往返一致，0V/TX/target/optimizer更新。两臂实际2140840参数，无新增模块；PASS立即训练。

两条实际source worker的PID/CWD/完整命令/GPU4/5及持续日志增长已核实，三F参考缓存各63000条。实际范数约束/缩放计数：

- global_cap：1/10轮、247/2470步，梯度合并调用247、IQ126000条，实际缩放126000（observed63000/clean63000），平均scale0.046522301、范围0.029634306至0.079243258，wave/identity/缩放后identity norm²总和996.19665/503684.15/996.19664。
- per_frame_cap：1/10轮、247/2470步，梯度合并调用247、IQ126000条，实际缩放50262（observed41542/clean8720），平均scale0.74880064、范围0.00275951至1，wave/identity/缩放后identity norm²总和995.95361/683227.22/521.45933。

首轮核验时尚无新源V或目标准确率。实际缩放不代表性能改善或参数步波形保证；两臂固定E10后分别自动完整目标测试，source V不作资格筛选，用户目标仍未达成。最终状态与结果以完整结果节为准。前一梯度投影的GPU6/7目标进程保留，CPU评分等待其完成后执行；复用其scalar_sum固定控制，不重复训练/预测。


## 完整固定目标比较

两个登记行均完成E10/2470、source V诊断、168000物理目标样本的七视图预测和独立CPU truth-last评分。实际source/target/controller进程均已退出。没有用源分数筛掉一组，也没有根据目标成绩更改/重排/选择性重跑。以下是三套原F各自argmax的均值±样本标准差，三套原读出不是三次独立恢复器训练。原clean参考均值85.113888889%；历史完整特征约束候选low urban为65.309126984%，仅作已暴露基准参照，不作为初始化。

|目标环境|原R03|整批范数上限|逐帧范数上限|
|---|---:|---:|---:|
|clean|84.743056%|84.939286±0.480025%|85.061111±0.456708%|
|high|77.575000%|77.463294±0.228572%|77.926984±0.225846%|
|mid|76.491071%|76.530357±0.203656%|76.995238±0.202889%|
|low suburban|75.284127%|75.509722±0.113183%|75.948016±0.102274%|
|high urban|76.330952%|76.479365±0.158088%|76.960119±0.143041%|
|mid urban|70.116865%|71.363095±0.213722%|72.014881±0.204059%|
|low urban|63.392460%|65.755357±0.334615%|66.384127±0.296731%|

|识别判据|整批范数上限|逐帧范数上限|
|---|---:|---:|
|low urban≥70%|未达到|未达到|
|其余五环境距原clean≤2个百分点|未达到|未达到|

“接近”采用本对话Agent预先记录的≤2个百分点口径，不冒充用户指定数值。未通过者保持负结果，未因实现/训练完成或波形误差而称用户目标实现。

|逐原F的low urban|整批范数上限|逐帧范数上限|
|---|---:|---:|
|2026092701|66.141071%|66.702976%|
|2026092702|65.542857%|66.116071%|
|2026092703|65.582143%|66.333333%|

|距原clean差距（百分点）|整批范数上限|逐帧范数上限|
|---|---:|---:|
|clean|0.174603|0.052778|
|high|7.650595|7.186905|
|mid|8.583532|8.118651|
|low suburban|9.604167|9.165873|
|high urban|8.634524|8.153770|
|mid urban|13.750794|13.099008|
|low urban|19.358532|18.729762|

## 同物理clean的波形差距与净纠错

配对clean仅在七视图预测全部固定后的独立CPU评分读取，不做oracle增益/相位/时延对齐。逐环境full256/prefix232/tail24 NMSE和logFFTpower均值/中位数/P90、逐F/TX/RX/day与混淆矩阵完整保存在results.json。

|目标low urban|原received|原R03|整批范数上限|逐帧范数上限|
|---|---:|---:|---:|---:|
|full_nmse均值|1.987855210|0.076996611|0.076123152|0.076425473|
|prefix232_nmse均值|1.998446002|0.068349912|0.067788513|0.067989468|
|tail24_nmse均值|1.928131431|0.157928481|0.154080082|0.155436056|
|log_power_mse均值|0.160772502|0.023813475|0.023490732|0.023530512|

|low urban修复/损坏/净修复|相对received|相对原R03|
|---|---:|---:|
|整批范数上限|188350/26524/161826|34930/23021/11909|
|逐帧范数上限|190439/25444/164995|33097/18019/15078|

计数为三原头的决策合计，504000个重复视图决策只有168000个物理样本，不称504000个独立样本。

## 源诊断、实际执行和成本

|源V（固定F1）|clean|low urban|mid urban|
|---|---:|---:|---:|
|整批范数上限|98.500000%|85.277778%|92.944444%|
|逐帧范数上限|98.555556%|84.000000%|92.111111%|

源V只诊断，无候选资格/排序/早停/参数或统计更新。两臂各63000训练物理L/U、E10/2470、L63000/U567000优化曝光、1260000IQ，原coarse/correction各累计30轮、60原组件epoch、串行继承50pass；梯度合并安排同10轮，无新模块。原F和默认头冻结，0新头拟合、0L/U TX真值、0目标训练更新。

- 整批范数上限：2140840可训练参数，无新增模块；梯度合并调用2470次、IQ1260000条、实际缩放1260000（observed630000/clean630000），平均scale0.0562537934、范围0.0292518623至0.182469398；wave/identity/缩放后identity norm²总和9950.39562/3428259.57/9950.39562。训练/保存1109.400s，含准备/等待1126.749s，峰值分配5544580608bytes；七视图目标预测合计785.962s。
- 逐帧范数上限：2140840可训练参数，无新增模块；梯度合并调用2470次、IQ1260000条、实际缩放443937（observed390000/clean53937），平均scale0.781080569、范围0.00122285495至1；wave/identity/缩放后identity norm²总和9945.45079/5789442.09/4621.47332。训练/保存1113.850s，含准备/等待1131.865s，峰值分配5544582144bytes；七视图目标预测合计786.248s。

梯度计数和norm²是逐IQ未除batch前的任务sum-loss输出梯度统计，不是参数梯度/FLOP；两行均计算两个输出梯度，只有范数上限作用范围不同。逐轮实际调用/曝光/缩放/obs-clean/scale/dot/norms、匹配LR/路由/角色预算及两项标量loss值已核对。停止梯度的norm-cap更新一般不是原标量和的梯度；共享网络Jacobian/AdamW/clip意味着输出空间操作不保证参数步或最终波形收益。全部source/target日志已完整解析，无技术错误。方法commit`9866f24dab7097294d585069bfcfe529d3346ade`；50项聚焦检查、唯一一次独立P0/P1审查、真实原R03/三F与新checkpoint no-query smoke、归档比较/编译和PID/CWD/命令/GPU/日志增长证据保留。未给另一用户对话发消息。

本轮仍未达到用户完整目标。输出梯度范数上限不等于参数空间GradNorm/PCGrad，也不证明最终波形或身份保真保证，不能把单恢复器种子或已暴露基准当作三seed恢复结论、新独立盲测、Phase2部署或在轨性能。


## 预登记复用的固定标量控制

控制run `20261020-phase1-restoration-gradient-projection-source-s2026092701-m2-r01`/`scalar_sum`在范数上限试验发布前已独立固定，不依据本试验或目标分数选择；不从其初始化，不重训或重预测。比较状态：VERIFIED_REUSED。

控制与两新行的逐轮epoch/update/路由/L-U曝光/IQ/LR相同，168000/七视图/三原F及truth-last完成，原received/R03读出逐环境一致。方法与父权重/缓存相同已在预登记固定。

|环境|复用标量控制|整批上限相对控制（百分点）|逐帧上限相对控制（百分点）|
|---|---:|---:|---:|
|clean|85.062500%|-0.123214|-0.001389|
|high|77.240873%|+0.222421|+0.686111|
|mid|76.230556%|+0.299802|+0.764683|
|low suburban|75.187302%|+0.322421|+0.760714|
|high urban|76.157738%|+0.321627|+0.802381|
|mid urban|70.735913%|+0.627183|+1.278968|
|low urban|64.989484%|+0.765873|+1.394643|
