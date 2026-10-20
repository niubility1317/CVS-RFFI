# R44：低秩卷积与重要性剪枝的正式结果

状态：VERIFIED / ANALYZED。两方法均完成固定E20/4940、source1800×7及target168000×7；每行全部预测固定后独立CPU truth-last，所属controller/worker均已退出。全部结果保留，测试不回流训练或选择性重跑。

唯一直接父/冻结IQ教师为用户指定R38 dense_half794529；low_rank保留通道、长卷积半秩SVD并保持共享因子，group_pruned按GN4组内encoder权重重要性选通道。固定两方法各新增20轮，paired-clean IQNMSE+10到1三stageIQ教师约束。全部coarse/action/prior联合，L126000/U1134000曝光、0TX/F/V/target优化；原F50/defaulthead仅冻结评测。不是无clean盲学习。

方法commit：`2f5247dbd5fe50f28defa38a0252d70e692934af`。完整矩阵、命令、数据契约、祖先来源、预算和停止规则见[登记](../automation_reports/CV-SincNet/20261020-phase1-restoration-R38-compression-methods-s2026092701-m2-r44/experiment.json)；统计、完整20轮曲线和逐TX/RX/day见[结果JSON](../automation_reports/CV-SincNet/20261020-phase1-restoration-R38-compression-methods-s2026092701-m2-r44/results.json)及[逐组CSV](../automation_reports/CV-SincNet/20261020-phase1-restoration-R38-compression-methods-s2026092701-m2-r44/row_metrics.csv)。

|方法|完整参数|较79.45万减参|相对R17逐景1pp通过|相对R38逐景1pp通过|low urban（%）|
|---|---:|---:|---|---|---:|
|low_rank|530161|33.2735%|False|False|65.771429|
|group_pruned|442805|44.2682%|True|False|66.772619|

1pp是逐场景退化容差，不等于性能优化成功；另列原lowurban≥70%、其他五景距rawclean84.6761904762%≤5pp。单seed、用户指定的已暴露工程基准，不称新独立盲测或干净泛化证明。

|方法|场景|准确率（%）|Macro-F1（%）|相对R38（pp）|相对R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|
|low_rank|clean|84.551786|84.259375|-0.077976|-0.035119|-209|
|low_rank|practical_high|81.379167|80.927807|-0.768452|+0.060119|+68286|
|low_rank|practical_mid|79.918452|79.455084|-1.054762|+0.340476|+67203|
|low_rank|practical_low_suburban|78.248810|77.805332|-1.244643|+0.252381|+65888|
|low_rank|practical_high_urban|80.151190|79.704297|-1.004167|-0.202381|+67961|
|low_rank|practical_mid_urban|72.865476|72.522713|-1.910714|-0.819643|+64482|
|low_rank|practical_low_urban|65.771429|65.693770|-2.498214|-1.445238|+59593|
|group_pruned|clean|84.505952|84.216444|-0.123810|-0.080952|-286|
|group_pruned|practical_high|82.418452|82.021788|+0.270833|+1.099405|+70032|
|group_pruned|practical_mid|81.010119|80.608682|+0.036905|+1.432143|+69037|
|group_pruned|practical_low_suburban|79.376190|79.028723|-0.117262|+1.379762|+67782|
|group_pruned|practical_high_urban|81.317262|80.922358|+0.161905|+0.963690|+69920|
|group_pruned|practical_mid_urban|74.226786|73.977900|-0.549405|+0.541667|+66769|
|group_pruned|practical_low_urban|66.772619|66.819605|-1.497024|-0.444048|+61275|

|方法|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|同卡R38/本版median ms|resident/峰值字节|
|---|---|---:|---:|---:|---|
|low_rank|213902/144869/171390|2508604|56799808|41.8769/51.9555|19421696/185215488|
|group_pruned|176962/125509/140334|2056092|49266224|45.1313/50.3344|19108352/336296960|

延迟为同卡16条sourceV、batch1、2次预热、完整context/三prior/六CG；顺序及并存负载影响计时。MAC仅Conv/Linear，不含FFT/复数CG/归一化/激活/访存。参数减少不预先推定加速或峰值显存降低。

low_rank完整20轮首轮→末轮：loss 2.0783148→0.028578833；iq_nmse 0.21699548→0.025384925；teacher_stage_nmse 0.18613193→0.0031939085。中位每轮183.045s。lowurban70通过=False；其他五景gap5pp通过=False。日志发现：[]

group_pruned完整20轮首轮→末轮：loss 1.7560273→0.028025675；iq_nmse 0.18583916→0.024819454；teacher_stage_nmse 0.15701881→0.003206221。中位每轮163.245s。lowurban70通过=False；其他五景gap5pp通过=False。日志发现：[]

全帧/主体232/尾24的paired-clean NMSE、FFT log-power差距、配对clean身份特征几何及全部TX/RX/day保留在JSON。识别改善或这些几何指标均不能证明每个新增细节属于输入TX。恢复器0TX训练与评分时独立打开标签分别说明。

祖先60串行pass+R38joint20+新增joint20=100；每组件累计60，组件轮次和180，原F50预训练另计。与R13E10/constant teacher1/LR1e-4并非同预算单因素比较。

两份独立恢复器已取回并CPU严格加载、有限synthetic IQ输出。路径：local_artifacts/pamf_restoration/r38_compression_methods_r44_20261020/models/<arm>/final.pt。使用experiments.pamf_r38_compression_methods.runtime.load_checkpoint，原生float32[B,2,256]输入/同形输出；接原F50/defaulthead。权重不入Git。长期总目标继续active，不由本轮产物完整代替总目标达成。


## 固定方法与历史发布记录

以下为结果揭示前的原方法和启动记录，不代表当前仍在训练。


预登记时状态：RUNNING。用户在“继续压缩79.45万版”的基础上进一步明确“探索另外的不同压缩，优化提升”。本轮只压缩同一 R38 恢复器，比较两种权重压缩方法，不研发其他恢复网络、不加同参数量变体。刚登记的 R43 延长训练方案尚未发布、没有远端进程，暂存为 DEFERRED_BEFORE_LAUNCH，不占 GPU；已在运行的 R41/R42 保留按原计划收尾。

|固定方法|完整参数|上下文参数|较 794529 减参|结构变化|
|---|---:|---:|---:|---|
|low_rank|530161|358771|33.27%|保留所有通道宽度，将成本确实降低的长卷积分解为低秩长卷积与1×1投影|
|group_pruned|442805|302471|44.27%|各 GroupNorm 组内按权重重要性选通道，粗恢复、信道估计和先验一起剪枝|

唯一直接父权重与冻结 IQ 教师为用户指定 R38 dense_half794529、固定源 E20/4940，方法 commit `e0868f0a0c3cf056a1753a5b4e5ce4f0bff84381`。同一原物理 L/U/V 划分及 R17/R07/R02 合规源训练祖先，R38 训练及固定最终轮次0TX/F/V/target优化。不继承 R13/R41 学生。用户在历史测试已暴露后指定该工程基底，R13 的既有压缩负结果公开保留；本轮不称新的独立盲测或干净泛化证明。新训练及初始化均不读取目标 IQ、truth 或指标，两方法固定均跑，不按源或目标分数取消一行。

低秩方法对每个普通 Conv1d 长核权重 W[out,in×k] 做 CPU SVD，rank=floor(min(out,in×k)/2)，仅在 factor 参数更少时替换。用 U×sqrt(S) 与 sqrt(S)×Vh 初始化两个无中间激活的卷积，保持 stride/padding/dilation 与独立输出 bias。三步 prior 原共享长权重只分解一次，左右因子继续共享，逐步 bias/GN/pointwise 保持独立；持有原权重引用避免对象 id 重用。不是剪掉一半通道，不声称函数完全等价。

剪枝方法对每个 Core encoder 层的四个输出卷积计算各输出 filter 的均方权重，并分别除以本卷积所有输出的均值后求和。每个原 GN4 组内按此分数选所需 top 通道，稳定排序处理并列，选中通道按原序排列；不读 IQ、标签或 F。encoder、decoder、skip、GN、输出层和 action global head 使用一致通道映射；decoder 的 skip 偏移使用旧完整 upstream 宽度。先验长卷积重建共享，step-specific bias/GN/pointwise 保持。宽度为主分支12/24/36/48、谱分支8/16/24/36；区别于 R13 直接取每组前部通道。

两方法均固定新增 E20/4940，从原 R07 全 L6300/U56700 和20个已缓存 received epoch 学习，新增 L126000/U1134000，学生 context 和教师 IQ 各1260000。所有 coarse/action/prior 联合优化，最终 paired-clean IQ NMSE 加冻结 R38 三 stage IQ NMSE 均值；教师权重 E1至E20线性10→1，clean能量detach/floor1e-4。重置 AdamW、LR2e-4/warm247/cos2e-5、WD1e-4、clip1、B256/micro64、尾批24按实际分母，FP32/noTF32，model2026092701/data2026102001。固定最终轮次，无早停或选行。原祖先60串行pass+R38joint20+本轮joint20=100；每组件累计60，组件轮次和180，原F50预训练E50另计。与 R13 的 E10、常数教师约束1、LR1e-4不属于同预算单因素比较。

恢复器训练 0 TX 标签、0 F 权重/输出、0 V/target。合法源 clean 配对和 IQ 教师参与，术语为“无TX标签的模拟配对源域自监督”，不称无clean盲学习。推理只读当前原生尺度float32 `[B,2,256]` IQ，输出同形恢复IQ；原冻结F50/defaulthead统一计算FFT、PA、时域和统计输入，0另训头。三prior/六CG保持，低秩增加卷积调用数，实际速度和显存须实测，不能由参数下降预先推定。

每行 E20 后自动 source1800×7诊断、同卡原R17/R38/学生完整链batch1成本，然后七个正式场景：clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。每景相同168000物理样本，每行全部预测IQ/logits/features/IDs固定后独立CPU连接truth。报告 accuracy/Macro-F1、TX/RX/day、波形/FFT/身份特征误差、配对修复与损伤、相对原R17和79.45万版逐景差；原R17每景最多下降1pp保留，并保守同时报告R38的1pp。lowurban≥70%、其他五景距rawclean84.6762%≤5pp单列，输入TX逐细节真实性仍未证明。所有固定行及负结果保留。

N607 GPU0/1每卡一own、总计最多二own；按卡等待 R41 对应row终态且实际PID/cmd句柄退出，不停止或热改健康任务。容量free≥18GB/all-owner进程<2，共享R41 score.lock/CPU2/max1本任务scorer。不可覆盖release/run/output；聚焦协议负测、一次独立P0/P1、launcher首步实际R38/F50/两条L_s配对IQ noquery smoke，PASS立即排队。单archiveSHA/远端compile和PID/CWD/argv/日志读回。技术失败保留受影响输出，健康行继续，不自动重试。精确方法commit、命令、路径、环境/CWD、矩阵与停止规则见本run的experiment.json；当前没有R44正式性能结果。

已发布方法 `2f5247dbd5fe50f28defa38a0252d70e692934af`；实际R38/F50/两Ls配对IQ smoke、单archiveSHA/compile及controller/CWD/argv/commit/等待日志增长 VERIFIED。已按卡开始接续训练，实际首轮进展见登记；无R44识别结论。

GPU实际训练VERIFIED：group_pruned (GPU1,E1)；实际PID/CWD/argv/commit/GPU和首轮曝光/日志增长核实，0TX/F训练。其余行按既定队列接续，无正式识别结论。

GPU实际训练VERIFIED：low_rank (GPU0,E1)；实际PID/CWD/argv/commit/GPU和首轮曝光/日志增长核实，0TX/F训练。其余行按既定队列接续，无正式识别结论。
