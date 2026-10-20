# R15：保持先验通道的低秩压缩

状态：ANALYZED。

本轮结论：779745参数版通过原R17逐场景1pp，但相对原794529父模型七场景均下降，最大1.191667pp；750035参数版仅clean提高0.425pp，六种practical均下降。两版均没有实现相对用户79万基底的联合压缩/性能优化，不替换原父模型。lowurban分别67.269048%/65.779167%，70%目标未达。

本次同卡顺序实测，779745版完整链中位45.6035ms/父42.3066ms，750035版47.8737ms/父43.1524ms；相对慢约7.79%/10.94%。16条sourceV/batch1测量且存在顺序/共存负载限制，不据此断言普适速度规律；当前没有加速证据。
唯一父模型仍为原R38 dense_half、794529参数、source E20/4940、method e0868f0a0c3cf056a1753a5b4e5ce4f0bff84381。0继承R13/R14已测权重；原7/63/30数据与合法祖先复用，非新独立盲测。

R14全部30轮source日志表明，冻结context使裁剪几乎归零，但窄版末轮teacher-stage IQ NMSE仍为原宽度约17/29倍。该source证据支持进一步检验通道裁剪造成的函数缺口，不证明GN为唯一起因。R15只在prior内对实际节省参数的长卷积做固定截断SVD，保持所有外部通道、GN分组/参数、skip与输出维度。left/right权重均跨三stage共享、各stage bias独立；0IQ校准。原548795参数context逐位冻结。结构与rank不依据F/V分类分数或target结果选择。

|行|完整参数|可训练prior|rank比例|GPU|
|---|---:|---:|---:|---:|
|factor_three_quarter|779745|230950|0.75|2|
|factor_five_eighth|750035|201240|0.625|3|

两行同final-clean IQ NMSE+1×三stage原R38 IQ锚；reset prior-only AdamW1e-4/warm247/cos1e-5/WD1e-4/clip1、B256/micro64真实尾24，received01..10、MODELseed2026092701、DATAseed2026102001。固定E10/2470，L63000/U567000曝光、teacher/studentprior630000、共享context630000/studentcontext0；0TX标签/F权重/F输出/V/target训练。context累计40、prior累计50，component-epoch130、祖先串行pass90，原F50E50另计。

source1800×7仅诊断，每版无论源结果均自动target168000×clean/三suburban/三urban。每版全部预测固定后独立truth-last，原F50/defaulthead不重训。原R17逐场景1pp容差、原R38保持/实际收益、lowurban70/其他五场景距rawclean5pp、净纠错、TX/RX/day、NMSE/FFT/配对clean特征距离分别报告。同预算control复用R14 prior_keep全部既有结果，不重跑。0目标反馈调参/候选筛选/选择性重跑。

部署只当前原生float32[B,2,256] IQ，完整自身context/prior/3×2CG，无teacher、clean、标签、场景或状态更新。完整参数、Conv/Linear MAC（FFT/CG另列）、文件字节、16sourceV/batch1/2warmup实测完整链延迟/显存均报告，不凭因子参数减少断言加速。

GPU2/3各一本任务/max2，R14对应owned句柄终态且实际退出才交接，free>=18GB、全owner<2，其他owner保持；原R06共享锁max1own scorer/CPU2。新不可覆盖run/release；普通预检、一次归档两端SHA、远端compile。launcher首步真实父/F50/2Ls no-query smoke通过立即训练；prior梯度、精确冻结context、teacher/F/head零更新、严格重载输出一致。技术故障保留该行、健康行继续，0自动重试/回退。

REJECTED_EXTRA_GATE：项目八项最小流程覆盖通用设计/计划审批。一次独立P0/P1限于下一真实运行；具体命令、环境、CWD、路径、停止规则及产物见experiment.json。

真实发布VERIFIED：method f2580edf27efcb968829d3f29e71625210393458，归档两端一次SHA一致、远端compile通过；launcher真实权重smoke PASS。GPU2/3实际PID、CWD、完整命令、commit、GPU及首轮日志增长逐项核实，其他owner不变。14项聚焦检查和唯一独立P0/P1审查PASS；报告计算入口12项聚焦检查PASS。尚未以源结果宣布正式识别成功。

## R15 完整结果交付

两行均完成固定 E10/2470、source 1800×7 和 target 168000×7；每行全部预测固定后独立 truth-last。实际所属进程已退出，两份完整恢复器已取回并严格加载。全部负结果保留，结果不回流训练或重跑。

|版本|参数|较 R38 减少|最大下降 R38/R17（pp）|low urban（%）|R38/R17 各自 1pp|训练 F 知识|
|---|---:|---:|---:|---:|---|---|
|factor_three_quarter|779745|1.8607%|1.191667/0.363095|67.269048|False/True|False|
|factor_five_eighth|750035|5.6000%|2.490476/1.437500|65.779167|False/False|False|

进一步缩小且同时通过两个参考容差的 IQ 行：无；使用冻结 F 知识的行：无。本轮两版均为严格 F-free；只更新 prior，coarse/action 保持并冻结。


1pp 通过仅表示满足退化容差。两版训练均不加载 F，不使用 TX 标签、V 或 target。low urban≥70% 和其他五场景距 raw clean≤5pp 分别记录，不以压缩验收替代。单 seed、已暴露基准，不称独立盲测或全局最小。

|版本|场景|准确率（%）|相对 R38（pp）|相对 R17（pp）|相对R14同预算原宽度（pp）|净纠错|
|---|---|---:|---:|---:|---:|---:|
|factor_three_quarter|clean|84.620833|-0.008929|+0.033929|+0.045238|-93|
|factor_three_quarter|practical_high|80.955952|-1.191667|-0.363095|-0.845238|+67575|
|factor_three_quarter|practical_mid|79.783929|-1.189286|+0.205952|-0.769643|+66977|
|factor_three_quarter|practical_low_suburban|78.341071|-1.152381|+0.344643|-0.763690|+66043|
|factor_three_quarter|practical_high_urban|80.050595|-1.104762|-0.302976|-0.780357|+67792|
|factor_three_quarter|practical_mid_urban|73.817857|-0.958333|+0.132738|-0.770238|+66082|
|factor_three_quarter|practical_low_urban|67.269048|-1.000595|+0.052381|-0.848810|+62109|
|factor_five_eighth|clean|85.054762|+0.425000|+0.467857|+0.479167|+636|
|factor_five_eighth|practical_high|80.922024|-1.225595|-0.397024|-0.879167|+67518|
|factor_five_eighth|practical_mid|79.515476|-1.457738|-0.062500|-1.038095|+66526|
|factor_five_eighth|practical_low_suburban|77.982143|-1.511310|-0.014286|-1.122619|+65440|
|factor_five_eighth|practical_high_urban|79.741071|-1.414286|-0.612500|-1.089881|+67272|
|factor_five_eighth|practical_mid_urban|72.883333|-1.892857|-0.801786|-1.704762|+64512|
|factor_five_eighth|practical_low_urban|65.779167|-2.490476|-1.437500|-2.338690|+59606|

|版本|coarse/action/prior 参数|Conv+Linear MAC/IQ|文件字节|R38/本版中位 ms|resident/增量 peak 字节|
|---|---|---:|---:|---:|---:|
|factor_three_quarter|329782/219013/230950|85037120|3425796|42.3066/45.6035|11895296/317162496|
|factor_five_eighth|329782/219013/201240|78298688|3323940|43.1524/47.8737|11783168/317162496|

MAC 不含 FFT、复数 CG、归一化、激活、插值和访存。实测包含完整恢复链，使用 16 条 source V、batch 1、2 次预热；物理求解保持 3 个恢复阶段、每阶段 2 次 CG，共 6 次 CG 迭代。顺序和共存负载影响延迟，不从参数比例推断加速。原 R17 的同口径延迟和峰值显存保留在 results.json。


### factor_three_quarter：完整训练与波形证据

全量 10 轮损失首轮→末轮：iq_nmse 0.032036253→0.022955259；loss 0.0380979→0.023820638；teacher_stage_nmse 0.0060616465→0.00086537877。中位每轮 119.933 s。日志命中：[]。

|场景|full256 NMSE|prefix232 NMSE|tail24 NMSE|FFT log-power L1|配对 clean 特征距离|
|---|---:|---:|---:|---:|---:|
|clean|0.000605562585|0.000621382904|0.000425819861|0.043858435|0.00038613134529441595|
|practical_high|0.0175975505|0.00799410511|0.110754639|0.276833057|0.014355381950736046|
|practical_mid|0.0227778442|0.0133973127|0.113620989|0.317035764|0.018516160547733307|
|practical_low_suburban|0.0283675361|0.0191998817|0.116901956|0.350912035|0.023003440350294113|
|practical_high_urban|0.0227841679|0.0133734802|0.113799132|0.293946803|0.017410002648830414|
|practical_mid_urban|0.0475099571|0.0390130244|0.128219455|0.387857825|0.034106217324733734|
|practical_low_urban|0.0700840577|0.0622582734|0.14281255|0.457021445|0.050985608249902725|

### factor_five_eighth：完整训练与波形证据

全量 10 轮损失首轮→末轮：iq_nmse 0.047935722→0.025546941；loss 0.0654487→0.028069842；teacher_stage_nmse 0.017512978→0.0025229018。中位每轮 124.327 s。日志命中：[]。

|场景|full256 NMSE|prefix232 NMSE|tail24 NMSE|FFT log-power L1|配对 clean 特征距离|
|---|---:|---:|---:|---:|---:|
|clean|0.000681288366|0.00067721348|0.000676544034|0.0518692993|0.0008392282179556787|
|practical_high|0.0180134755|0.00796984602|0.115854867|0.29079169|0.014618056826293468|
|practical_mid|0.023224948|0.0134249032|0.11848513|0.332863122|0.018996896222233772|
|practical_low_suburban|0.0288105067|0.0192441083|0.121520288|0.368271261|0.023663682863116264|
|practical_high_urban|0.0231429245|0.0133095961|0.118595935|0.308013827|0.017817184329032898|
|practical_mid_urban|0.0478795432|0.0390374251|0.132128283|0.404690564|0.034861475229263306|
|practical_low_urban|0.0707289949|0.0626385212|0.14608635|0.476965487|0.05206456407904625|

逐 TX/RX/day 的准确率、Macro-F1、纠错/损伤、混淆矩阵、配对 clean 几何与完整训练曲线见 results.json 和 row_metrics.csv。这些指标不能证明恢复了全部真实 TX 细节。coarse/action 各累计 40 轮，prior 累计 50 轮；祖先训练共 90 串行 pass，component-epoch 合计 130，F50 预训练另计。

本地模型：local_artifacts/pamf_restoration/r38_factor_prior_20261020_r15/models/<arm>/final.pt。加载 experiments.pamf_r38_factor_prior.runtime.load_checkpoint；输入原生尺度 float32 [B,2,256] IQ，输出同形 IQ，接原 F50 默认头。权重不入 Git。目标状态依据全部识别指标保留，不以本轮完成宣称长期优化目标完成。

## 压缩失败分析与两条后续路线


本次新增五行均从固定原R38父模型独立分叉，全部F-free、E10/2470；每行完成168000物理样本×七场景与独立truth-last，共5880000条视图预测。原R13全60轮、R14全30轮与R15全20轮记录完整保留，不以首尾抽样代替完整日志。

|新增版本|完整参数|low urban（%）|相对原79万最大下降（pp）|原R17/原79万分别1pp|
|---|---:|---:|---:|---|
|prior_keep|794529|68.117857|0.419643|True/True|
|prior_three_quarter|689129|65.092857|3.176786|False/False|
|prior_half|613041|59.750595|8.519048|False/False|
|factor_three_quarter|779745|67.269048|1.191667|True/False|
|factor_five_eighth|750035|65.779167|2.490476|False/False|


已证实的主要问题是压缩后的源IQ函数拟合缺口。R13两窄版末轮teacher-stage误差仍是同预算原宽度15.56/39.58倍，且较窄版末轮72.87%更新触发裁剪；最后五轮损失仍下降。不存在NaN、OOM、漏轮或提前终止，不能将失败归为训练根本没有执行，也不能据此证明容量极限。

结构解释仍有边界：R13同时裁剪coarse/action/prior，GN组内通道统计集合改变且输出卷积丢失被删通道贡献，可能造成逐阶段偏移；它没有隔离每一模块。R14精确冻结context后裁剪几乎消失，但两窄版末轮teacher-stage误差仍为同宽约17/29倍，正式结果也未通过。它证明只保住context不足以解决问题，不能证明GN是唯一起因。

R15以源IQ拟合与结构事实检验另一种压缩：保留所有通道、GN分组/参数、skip与输出尺寸，用权重SVD降低内部秩。两版末轮teacher-stage误差为0.000865379/0.0025229，低于R14窄版0.00409466/0.00702153；仍高于原宽度0.000241619。不同压缩比例并非参数匹配，因此不能将差值全归功于归一化保持。

同预算控制、源IQ拟合、目标识别与部署成本分别判断：1pp通过只是容忍退化，不能冒充实际性能优化；IQ NMSE下降也不保证保留TX细节。卷积分解新增算子，固定FFT/6次CG成本仍在，必须看实际完整链延迟，不按参数比例宣布加速。所有原始负结果保留，不将目标分数回流rank、训练预算、父模型或选择性重跑。

原R13详细依据见 [失败分析](../20261020-phase1-restoration-r38-successor-s2026092701-m6-r13/failure_analysis.md)，R14全部七场景与成本见 [冻结上下文报告](../20261020-phase1-restoration-r38-prior-compact-s2026092701-m3-r14/report.md)。本轮完成不等于lowurban70%、其余五场景距rawclean5pp、真实TX细节保真和联合性能目标全部达成。
