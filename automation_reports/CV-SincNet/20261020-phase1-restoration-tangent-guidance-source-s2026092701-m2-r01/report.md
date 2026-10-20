# 源 clean 局部敏感度约束恢复

状态：ANALYZED / VERIFIED。用户目标为完整目标 low urban≥70%，其余五种 practical 接近原 F clean85.113889%。“接近”暂按≤2个百分点记录，是 Agent 的操作解释。登记时最好 low urban66.384127%，目标仍未完成。

源训练已有 identity/wave 输出梯度累计 RMS 比15.426531；本轮检验固定 clean 参考点的局部敏感度是否能约束身份损伤。这个源证据不证明局部近似有效，也不决定最终性能。当前参数梯度对照尚在目标预测，本轮结构、预算及顺序不依据其目标分数制定。

|固定行|语义约束（先各 F，再三 F 平均）|资源|
|---|---|---|
|既有 per_frame_cap|exact finalcos＋0.3 exact T²KL|仅复用既有完整结果|
|feature_tangent|local feature quadratic＋0.3 exact T²KL|新 GPU6|
|posterior_tangent|exact finalcos＋0.3 local posterior quadratic|新 GPU7|

三套原 F50 CROSS_RX（2026092701/02/03，method4df7b94d0fd8ce29056bd8236364e45afa92e07f）及原默认头全部冻结，不拟合新头。对63000条合法源 L/U clean，逐物理样本计算六坐标对2×256 IQ的 Jacobian。feature 坐标为 normalized160feature 乘固定 Rademacher6×160（符号/√6），再除√2；每 F 用独立 CPU generator seed2026102081＋Fseed，不依赖标签、不拟合参数。posterior 坐标为√2×T×sqrt(softmax(native logits/T).clamp_min(1e-12))，T2，六类原默认输出对称处理。

local=Σ6[Jclean×(restored−clean)]²，三 F 平均，替换且只替换对应一个语义项，另一项及系数0.3保持。feature 是六探针近似，不是完整160维 Jacobian；posterior 是带数值下界的局部 KL 二次近似，不是有限扰动的原 KL。J、clean 及 reference feature 全部停止梯度，无 Hessian、高阶优化或 F 更新。固定缓存只用于源训练；推理仍为原网络单帧 received IQ 的 coarse/corrector，无 J、缓存、F、clean、真实信道/TX/RX/场景、其他帧或状态更新。

每行缓存 J 为63000×3×6×2×256 float32，2322432000 bytes，另有120960000 bytes源feature。493个128样本batch，1479次F前向/189000个F输入，8874次坐标反传/1134000个样本坐标。此额外准备工作与存储如实计入，不能把相同模型训练epoch称为相同总成本。NumPy2/Torch2.1使用经测试的可写frombuffer输入桥；输出缓存沿用已验证tolist桥，无numpy ABI依赖。

两新行严格从原合法 R03（method b01b534f52daf9cae9e940ed6ec6f9243bccbdbe，coarse20/corr20）完整相同state开始，不继承失败精修、control或另一对话权重；2140840参数，无新增NN。jointE10/2470，coarse/corr各累计30轮，60组件epoch，串行继承50pass。仍用原R02 source63000 L6300/U56700、epochs1..10、.15clean/.60residual/.25full、observed+clean、batch256、modelseed2026092701/dataseed2026101951；新优化曝光L63000/U567000，1260000 IQ。

wave仍为 IQ NMSE＋0.1logpower，输出空间逐帧cap复用既有operator：分别求任务sum-loss对估计IQ的gw/gi，s=min(1,sqrt(norm²gw/norm²gi))，i2≤1e-20取1，wave0且identity非零取0，停止scale梯度，(gw+s gi)/实际observed+clean数，再一次网络反传。无冲突投影或放大。AdamW3e-5/warm247/cos3e-6、WD1e-4/clip1/FP32/noTF32保持。记录实际逐IQ scale/conflict/norms、训练loss、成本及源诊断。

复用控制按per-frame cap结构固定为gradient_balance/per_frame_cap（method9866f24dab7097294d585069bfcfe529d3346ade），不是按其目标分数选取，不从它初始化，不重训/重预测。源J的额外成本及损失几何差异明确保留。

source V1800/原F1/七视图只作诊断，无源分数/NMSE gate/排序。两新最终E10都自动完整目标168000/clean＋六practical/三原默认头各自argmax；每行七视图全部IQ/ID/预测固定后，独立无CUDA CPU才连接truth/pairedclean。分别报告逐F/TX/RX/day accuracy、MacroF1、修复/损坏/净纠错、clean差距与同物理full256/prefix232/tail24 NMSE/logpower。不按环境拼接或投票，不以单恢复器seed冒充三个独立训练、新盲测、Phase2或在轨性能。目标分数不回流结构/参数/预算/顺序/选择性重试。

50项聚焦检查PASS；唯一一次独立P0/P1审查PASS，审查者独立复跑50项通过。launcher首步实际R03/三F/四Ls no-query smoke，核实局部J、完整初始IQ/state一致、native posterior IQ梯度、coarse/correction合并梯度、F无梯度、checkpoint往返，0TX/V/target/optimizer，PASS立即训练，不增审批或重复数据验证。

一次N607普通账号预检VERIFIED，原R03合法、新run/release不存在，GPU6/7各24124MB空闲无PID，参数实验GPU4/5实际target PID/CWD/cmd匹配并保留。每卡一个本owner、allowner<2，重叠最多4个本owner GPUworker，source≥18GB/target≥12GB，CPU2线程/scorer最多1。新source评分等parameter_balance终态再与本run共用score.lock；source实际退出才同卡target。非法输入/权限/代码、冲突、非有限/OOM、预算/梯度执行或预测失败时保留产物并失败该行，无自动重试，健康行继续。

一个实际代码配置commit、不可覆盖root、单归档本地/远端SHA比较及remotecompile、真实smoke与实际PID/CWD/cmd/GPU/日志增长随后核实。精确命令、环境/CWD、数据/父来源、路径、seed与逐行矩阵在experiment.json。J/IQ/权重/逐样本预测/原始日志和私有机器状态不进Git。没有向另一用户对话发送指令。


## 实际发布与源约束准备核验

方法commit`2afa1cdbd31e1b2d56214499a797bac00ab64c5c`已推送且远端OID独立匹配；一次归档本地/远端SHA比较、remotecompile/modulehelp PASS。launcher首步实际R03/三F/四Ls no-query smoke PASS：两臂原R03完整初始IQ/state一致，真实源J有限非零、形状8×3×6×2×256，native posterior至IQ梯度及coarse/correction合并梯度有效，F无梯度，新checkpoint往返一致，0TX/V/target/optimizer更新。

GPU6/7两条实际source worker的PID/CWD/完整命令/GPU和日志持续增长已独立核实。此时正在生成63000条源clean局部约束缓存，进度为：

- feature_tangent：实际完成12800/63000条源物理样本的三F坐标Jacobian准备。
- posterior_tangent：实际完成12800/63000条源物理样本的三F坐标Jacobian准备。

缓存进度不冒充完成epoch或性能改善。准备完成后按原固定E10/2470训练、源V诊断及完整目标测试，无额外确认/选模；尚无本轮新目标表现。前序参数梯度run健康GPU4/5任务保持，CPU评分等其终态再共用本run score.lock；复用旧per_frame_cap控制，不重新训练/预测。最终实际梯度统计、准备与训练成本在完整结果节记录。


## 完整固定目标比较

两个登记行均完成E10/2470、source V诊断、168000物理目标样本的七视图预测和独立CPU truth-last评分。实际source/target/controller进程均已退出。没有用源分数筛掉一组，也没有根据目标成绩更改/重排/选择性重跑。以下是三套原F各自argmax的均值±样本标准差，三套原读出不是三次独立恢复器训练。原clean参考均值85.113888889%；历史完整特征约束候选low urban为65.309126984%，仅作已暴露基准参照，不作为初始化。

|目标环境|原R03|特征局部敏感度|后验局部敏感度|
|---|---:|---:|---:|
|clean|84.743056%|85.055357±0.457277%|85.045040±0.456453%|
|high|77.575000%|77.946825±0.214706%|78.069444±0.298538%|
|mid|76.491071%|77.010714±0.184083%|77.130556±0.251912%|
|low suburban|75.284127%|75.979960±0.090336%|76.033333±0.136802%|
|high urban|76.330952%|77.002579±0.134890%|77.118849±0.196991%|
|mid urban|70.116865%|72.116865±0.238591%|71.972619±0.184547%|
|low urban|63.392460%|66.406548±0.389762%|66.097817±0.276413%|

|识别判据|特征局部敏感度|后验局部敏感度|
|---|---:|---:|
|low urban≥70%|未达到|未达到|
|其余五环境距原clean≤2个百分点|未达到|未达到|

“接近”采用本对话Agent预先记录的≤2个百分点口径，不冒充用户指定数值。未通过者保持负结果，未因实现/训练完成或波形误差而称用户目标实现。

|逐原F的low urban|特征局部敏感度|后验局部敏感度|
|---|---:|---:|
|2026092701|66.842262%|66.395238%|
|2026092702|66.091071%|65.848810%|
|2026092703|66.286310%|66.049405%|

|距原clean差距（百分点）|特征局部敏感度|后验局部敏感度|
|---|---:|---:|
|clean|0.058532|0.068849|
|high|7.167063|7.044444|
|mid|8.103175|7.983333|
|low suburban|9.133929|9.080556|
|high urban|8.111310|7.995040|
|mid urban|12.997024|13.141270|
|low urban|18.707341|19.016071|

## 同物理clean的波形差距与净纠错

配对clean仅在七视图预测全部固定后的独立CPU评分读取，不做oracle增益/相位/时延对齐。逐环境full256/prefix232/tail24 NMSE和logFFTpower均值/中位数/P90、逐F/TX/RX/day与混淆矩阵完整保存在results.json。

|目标low urban|原received|原R03|特征局部敏感度|后验局部敏感度|
|---|---:|---:|---:|---:|
|full_nmse均值|1.987855210|0.076996611|0.076291459|0.076266751|
|prefix232_nmse均值|1.998446002|0.068349912|0.067899559|0.067838119|
|tail24_nmse均值|1.928131431|0.157928481|0.154851981|0.155584294|
|log_power_mse均值|0.160772502|0.023813475|0.023535518|0.023519541|

|low urban修复/损坏/净修复|相对received|相对原R03|
|---|---:|---:|
|特征局部敏感度|190412/25304/165108|32779/17588/15191|
|后验局部敏感度|189289/25737/163552|33255/19620/13635|

计数为三原头的决策合计，504000个重复视图决策只有168000个物理样本，不称504000个独立样本。

## 源诊断、实际执行和成本

|源V（固定F1）|clean|low urban|mid urban|
|---|---:|---:|---:|
|特征局部敏感度|98.555556%|83.833333%|91.777778%|
|后验局部敏感度|98.555556%|83.277778%|90.777778%|

源V只诊断，无候选资格/排序/早停/参数或统计更新。两臂各63000训练物理L/U、E10/2470、L63000/U567000优化曝光、1260000IQ，原coarse/correction各累计30轮、60原组件epoch、串行继承50pass；梯度合并安排同10轮，无新模块。原F和默认头冻结，0新头拟合、0L/U TX真值、0目标训练更新。

- 特征局部敏感度源J准备：240.013s，2443392000bytes，1479次F前向/189000输入，8874次坐标反传/1134000样本坐标，全部63000源物理clean、0TX/V/target、0F更新。
- 特征局部敏感度：2140840可训练参数，无新增模块；梯度合并调用2470次、IQ1260000条、实际缩放485639（observed431217/clean54422），平均scale0.758390039、范围0.000406669307至1；wave/identity/缩放后identity norm²总和9950.62222/7205879.67/4914.18268。训练/保存1178.721s，含准备/等待1419.920s，峰值分配5568017408bytes；七视图目标预测合计815.145s。
- 后验局部敏感度源J准备：254.241s，2443392000bytes，1479次F前向/189000输入，8874次坐标反传/1134000样本坐标，全部63000源物理clean、0TX/V/target、0F更新。
- 后验局部敏感度：2140840可训练参数，无新增模块；梯度合并调用2470次、IQ1260000条、实际缩放441382（observed390239/clean51143），平均scale0.793472393、范围0.000848815893至1；wave/identity/缩放后identity norm²总和9948.68837/902571.211/4603.91496。训练/保存1187.227s，含准备/等待1443.467s，峰值分配5568611328bytes；七视图目标预测合计820.999s。

梯度计数和norm²是逐IQ未除batch前的任务sum-loss输出梯度统计，不是参数梯度/FLOP；两行均沿用逐帧cap，但各替换一个语义项；额外源J准备与缓存成本不同于复用控制。逐轮实际调用/曝光/缩放/obs-clean/scale/dot/norms、匹配LR/路由/角色预算及两项标量loss值已核对。停止梯度的norm-cap更新一般不是原标量和的梯度；共享网络Jacobian/AdamW/clip意味着输出空间操作不保证参数步或最终波形收益。全部source/target日志已完整解析，无技术错误。方法commit`2afa1cdbd31e1b2d56214499a797bac00ab64c5c`；50项聚焦检查、唯一一次独立P0/P1审查、真实原R03/三F与新checkpoint no-query smoke、归档比较/编译和PID/CWD/命令/GPU/日志增长证据保留。未给另一用户对话发消息。

本轮仍未达到用户完整目标。固定clean局部二次近似及六特征探针不保证有限扰动或最终身份保真，不能把单恢复器种子或已暴露基准当作三seed恢复结论、新独立盲测、Phase2部署或在轨性能。


## 预登记复用的固定逐帧cap控制

控制run `20261020-phase1-restoration-gradient-balance-source-s2026092701-m2-r01`/`per_frame_cap`在局部敏感度试验发布前已独立固定，不依据本试验或目标分数选择；不从其初始化，不重训或重预测。比较状态：VERIFIED_REUSED。

控制与两新行的逐轮epoch/update/路由/L-U曝光/IQ/LR相同，168000/七视图/三原F及truth-last完成，原received/R03读出逐环境一致。方法与父权重/缓存相同已在预登记固定。

|环境|复用逐帧cap控制|特征局部项相对控制（百分点）|后验局部项相对控制（百分点）|
|---|---:|---:|---:|
|clean|85.061111%|-0.005754|-0.016071|
|high|77.926984%|+0.019841|+0.142460|
|mid|76.995238%|+0.015476|+0.135317|
|low suburban|75.948016%|+0.031944|+0.085317|
|high urban|76.960119%|+0.042460|+0.158730|
|mid urban|72.014881%|+0.101984|-0.042262|
|low urban|66.384127%|+0.022421|-0.286310|
