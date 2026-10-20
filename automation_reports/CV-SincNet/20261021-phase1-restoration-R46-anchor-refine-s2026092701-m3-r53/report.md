# R53：持续 IQ 教师约束的压缩性能优化

状态：**VERIFIED / RUNNING**，已发布，两行训练、一行排队。性能优先；不把更小参数、容差通过或源域小幅涨分当作正式性能优化完成。

研究差异来自完整源域目标审计：IQ 重建误差改善可能伴随固定识别器源准确率下降。原衰减1→0.1教师约束可能不足以保持父模型的逐输入响应；这里只检验持续权重1的波形教师约束，不用TX标签、F输出或测试分数设置新损失。该假设可能无效，全部负结果保留。R52两个相位参考均不足，未据此施加C4等变或不变损失。

固定三行：prune_decay273765保留线性衰减；retain_anchor350521和prune_anchor273765教师系数恒为1。其余仍是原pairedclean IQ NMSE＋0.05多尺度相对功率STFT＋三stage IQ教师NMSE，coarse/action/三个prior全部联合训练，原3prior/6CG/GN4结构和输入输出不变，无额外推理分支或参数。两宽度均来自已登记R48完整两点矩阵，直接父继续唯一为人指定R46 prune_refine350521 E20/4940（bbb8daf817de050b958d217b70d4c33f4aefe341），不加载R48/R49派生学生。

同宽350521原衰减控制复用R49 retain_refine，micro256、seed、数据、预算相同，不重复训练。27.38万衰减控制本次新训以匹配micro256；R48同宽micro64结果保留，仅作历史/吞吐对照。原八行全部正式测试已完成。所有新行固定最终E20/4940，无源选模或提前结束，训练后立即source1800×7、同卡G/G+F成本、正式168000×7预测；每row七预测固定后独立CPU truth-last评分，不因其他行未训完等待整批，也不据结果调参、重排或选择性重跑。

原6300L_s＋56700U_s全部使用，TX标签对恢复器隐藏；原7/63/30物理划分/R07缓存epoch01..20、model2026092701、split392005、data2026102001、augmentation2026101912、formal2027均不变。每行新增20轮/4940更新/global256micro256/直接float32buffer/ResetAdamW，LR2e-4暖247至2e-5、WD1e-4、clip1、FP32/TF32关闭。每行L126000/U1134000曝光、1260000学生和父教师IQ，读取TX/F/V/target训练信息为0。继承每组件80轮，新增20轮后每组件100、组件和300、串行累计140；F50预训练成本另计。

源source_cache READY/EXACT_MATCH和R46/R44/R38/R17/R07/R04源祖先契约复用；R46checkpoint来源已核实原source-only final，F50默认头仅训练完成后读取评价，不另训头。属于有clean模拟配对的TX无标签自监督，不能称无clean盲学习。正式基准已暴露，保留工程比较背景，不声称新的独立盲测或逐细节TX真实性。

按当前两卡约束仅物理GPU2/3，各一个本任务worker；prune_decay和retain_anchor并行，prune_anchor在GPU2排队，GPU worker最多2。每行CPU2，共享原score.lock最多1个本任务scorer。2026-10-21只读资源检查全卡空闲；实际一次N607 preflight和启动后PID/CWD/argv/GPU/日志另存。新不可覆盖release/run/output，唯一owner codex/root/R46-anchor-refine，其他owner及历史产物保持。低准确率不触发技术停止；直接权限/输出碰撞/错误checkout或split/无法合法预测等系统技术失败保留受影响输出、无自动重试/热改/回退，不停止健康同伴。

32项聚焦检查与CLIhelp通过，唯一一次限P0/P1审查见实际review.json；首步真实R46/F50/2Ls配对IQ noquery smoke，PASS立即训练。Git commit固定代码/config；一次预检、单归档两端SHA比较和compile，禁止额外设计审批/签名/成员哈希等门槛（REJECTED_EXTRA_GATE）。精确命令、环境/CWD、路径、三行矩阵与预期产物以experiment.json为准。

正式主要指标是七景accuracy/Macro-F1、相对R17/R46的逐景差、修复/损伤与逐TX/RX/day；原R17逐景最多下降1pp为退化容差，不是性能优先的成功定义。low urban≥70%、其他五受损场景距rawclean84.6761904762%≤5pp以及更强TX细节真实性目标继续ACTIVE。成本次级：同卡64源V，G和G+F的B1/B32三轮交替同步计时、median/P90/P95、吞吐、峰值/增量/共享驻留显存、完整参数/序列化大小/ConvLinear MAC；不含传输/前处理/loader，MAC不含FFT/CG/norm/激活/F，驻留是3G＋1F共享suite，不冒充单模型显存。

方法commit `df36c3d3da4e026b747776b65b58d4921dc1a71e`。精确命令、环境/CWD、路径与矩阵见[登记](experiment.json)。


VERIFIED/RUNNING：真实350521父/F50/2Ls noquery smoke PASS，三行全部coarse/action/prior有限非零梯度、父/F不更新、checkpoint严格重载一致。单归档SHA/compile与实际controller、GPU2 prune_decay、GPU3 retain_anchor的PID/CWD/argv/CUDA/GPU-PID匹配、首轮曝光、micro256/directbuffer和日志增长核实。prune_anchor按固定计划在GPU2等待；各卡仅一条本任务GPU worker，其他owner不动。尚无正式性能结果；每行完整E20/4940后自动源七视图/成本/正式七视图及独立truth-last，不以负分提前停止。原性能/TX保真目标ACTIVE。

## 首两行完整训练与推理成本观察

以下两行已完整E20/4940、源1800×7独立评分及同卡G/G+F/B1/B32三轮测量，完整源阶段已结束，正式预测按固定管线自动继续；第三行仍排队。这是训练和成本证据，尚无正式识别结论。全部20轮损失/曝光/LR和profile范围由独立汇总工具核对。

|行|参数|每轮中位s|父/本版G B1 median ms|父/本版G+F B1 median ms|父/本版G+F B32 sample/s|
|---|---:|---:|---|---|---|
|prune_decay|273765|50.954|44.1931/43.3225|56.2200/54.1512|327.637/385.036|
|retain_anchor|350521|56.365|44.2953/44.1067|55.4548/54.3477|467.163/472.576|

完整p90/p95、各轮中位数、显存和Conv/Linear MAC见[source阶段与成本汇总](source_cost_progress.json)。参数、实际时延与吞吐分别比较，任何变慢都保留，不预设压缩必然加速；训练吞吐提高不能混为推理加速。模型仍执行三prior/六CG，不能仅据这些时间断定具体瓶颈。

成本在64条sourceV、预加载GPU的FP32 eval上同步测量，不含传输/预处理/loader；三G一F共享suite驻留不是独立部署显存，增量峰值对应当前前向。GPU负载/温度与相关重复观测限制保留，尚无独立性能置信区间。原70%/5pp/输入TX细节要求继续ACTIVE。

## 已完成行的正式测试（2/3）

以下均完整168000条×七景、原冻结F50默认头、预测固定后独立truth-last；其余行继续固定训练/预测，不能把部分行冒充完整矩阵。性能优先，所有负结果保留，不回流选模、参数或父。

|场景|指定父350521|prune_decay（273765）|retain_anchor（350521）|
|---|---:|---:|---:|
|clean|84.456548|84.528571|84.543452|
|郊区高|82.452381|82.408929|82.301190|
|郊区中|81.222024|81.348810|80.898214|
|郊区低|79.620238|79.858333|79.405952|
|urban高|81.386905|81.411905|81.206548|
|urban中|74.447024|75.061310|74.374405|
|urban低|67.510119|68.527381|67.636310|

准确率单位为%。逐TX/RX/day、Macro-F1与repair/damage见[逐行正式结果](partial_formal_results.json)。

|行|R17逐景1pp容差|指定父逐景1pp参考|lowurban≥70%|其他五景raw-clean差≤5pp|
|---|---|---|---|---|
|prune_decay|True|True|False|False|
|retain_anchor|True|True|False|False|

retain_anchor相对预登记同宽衰减控制的七景差（百分点，顺序同上）：+0.102381、+0.419643、+0.472024、+0.469048、+0.444643、+0.643452、+0.814286。

原350521直接父不变；原完整识别与更强输入TX细节真实性目标ACTIVE。单seed、已暴露工程基准；识别率和波形误差不单独证明逐细节身份。完整模型和时延/吞吐/显存随最终交付保留，源域分数不替代以上正式结果。
