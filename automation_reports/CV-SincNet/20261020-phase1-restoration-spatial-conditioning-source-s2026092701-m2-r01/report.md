# 原 F 多尺度汇总与时序特征辅助恢复

状态：RUNNING / VERIFIED。用户目标仍为完整目标low urban≥70%、其余五practical接近原F clean85.113889%；“接近”暂按≤2个百分点，是Agent操作解释。登记时已测最好66.384127%，目标未完成。本轮结构、输入、预算与顺序在当前局部敏感度候选的新目标结果返回前固定；不依靠目标分数制定或筛选。

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
