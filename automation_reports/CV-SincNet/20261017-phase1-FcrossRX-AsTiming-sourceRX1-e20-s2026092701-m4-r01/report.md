# F＋跨 RX：等剂量早期与后期 As 的固定 E20 源域检验

run_id：20261017-phase1-FcrossRX-AsTiming-sourceRX1-e20-s2026092701-m4-r01。状态 ANALYZED／执行 VERIFIED／固定候选科学 FAIL。四个新模型的 E20、全部七个源域视图及独立全量复算均完成。训练运行代码 f788309f5f9e27689ec1e98ae99a6bd83237e1f9；离线分析代码 8773b1c4feab2decba85b4c8aac1bf852fead09a。逐行矩阵仍以 experiment.json 为准。

LATE7 比等剂量 EARLY7 的 clean 高 1.269841 个百分点、urban 均值高 2.619048 个百分点、low 高 1.666667 个百分点，但比 ALL 的 clean 低 2.460317、urban 均值低 0.740741 个百分点，未通过原先固定的 ALL 对照条件。固定早晚时序方案关闭；不扫描启用轮次、剂量、权重、RX、seed、预算或门槛补救，不将 ALL／EARLY7 控制事后晋级。

本轮是合法源域 held RX1 L1260、model seed 2026092701 的训练策略检验，没有访问正式 target。As 总剂量相等不代表早晚使用了完全相同的物理包与 draw；时序同时关联原学习率和 SSL 日历，不能把差异唯一归因于“先学身份”。没有证明新 Ag、Ag×As 联合收益、low urban >60% 或跨 seed 重现。

## 固定模型、数据与四个匹配控制

增强前基线继续F＋跨RX同TX浅/中层对比（SIG_TIME_X3、统一Sinc、真实joint160、NoExplicitCFO及原内部必要补偿），默认cos30头随骨干正常联合训练，无新部署模块/头/参数，不冻结骨干另训头。所有模型scratchE20、seed2026092701、4440更新、原13L＋7U、oldgate/ownEMA、AdamW与cosine保持。当前不是新Ag或Ag×As四因子矩阵。

复用split392005/VALIDATED_ONCE原sourcecarrier；本轮fit RX3/4/6/8，day1至3，TX0至5；L5040、U45360真值隐藏、V21600仅健康。held source RX1的原L1260仅在固定FINAL20自动读出七场景。L6300原carrier、fitL5040和历史L_train5580/L_dev720不能混用；不读target、不重验未变数据、不加载祖先权重。兼容smoke唯一例外严格载入同有效L/U/V来源已确认的B00 FINAL20，2个合成包no-query后丢弃，不作为学生或教师初始化。

完整原base＝.75raw＋.25ordinary＋原ordinary-only浅/中层CROSS_RX SupCon，绝对份额保持；另加.25RD(view)，RD为同一个已生成IQ的两次CE均值＋原RDrop KL1。每组8L前向、总pairweight1.25，无新跨视图一致性/对齐/课程门控。inactive额外raw时raw总权重1.00，active As时raw总权重.75，明确披露相对份额变化。所有合法有标签As均参与CE，不要求urban学生先高置信。

|行|第四对的As日历|实际As监督包数|每步额外前向|原base/SSL|
|---|---|---:|---:|---|
|RAW|无；全20轮用raw|0|2|保持|
|ALL|E1至20|568320|2|保持|
|EARLY7|E1至7；其余raw|198912|2|保持|
|LATE7|E14至20；其余raw|198912|2|保持|

四行每一步都从同一raw、原物理元数据和原命名随机流生成单次完整practical shadow（每行568320包），只由epoch确定是否送入第四对。没有连续两次完整信道，不对ordinary再加As，不独立放大扰动，不改变场景边际概率，不用状态/质量/TX真值定义session。原几何/遮挡/传播/噪声/接收质量及补偿依赖保留。

实际星地入口仍为既有backend的experiments/adv3b02_xuc/code/leo_practical，actual_channel.json/actual_modules.json保存实际__file__和全配置。25MHz、2.462GHz、IQ256；原cache RMS、进入信道normalize_input；residual/post_sync，EQoff时G=1，reference_snr_db=30，reflect，原相位跟踪、IQ补偿、AGC/人工边界/数值延迟保持。RX硬件参数按rx/day/eq session稳定，不由TX定义；训练流20261016101与source_eval20261016103分离，receiver20261016102。新增噪声SNR不是总SNR，不反演原录音未知TX硬件，也不改变正式clean＋六practical receivedIQ/信道。

## 首次固定判据与放弃规则

LATE7必须满足以下全部条件，同时完成全部源域读出：

- 对RAW：clean和clean宏F1不下降，low≥＋5pp，urban均值≥＋3pp。
- 对ALL：clean≥＋1pp，clean宏F1不降，low和urban均值各≥−0.5pp。
- 对等剂量EARLY7：clean≥＋1pp，clean宏F1不降，low及urban均值各≥＋1pp。
- 对三个控制的每个practical及seenV均≥−0.5pp；全4440更新/4546560LNN/3330SupCon步/额外1136640LNN、真实生成及消费计数正确，原完整base最终目标≤0.5。

FAIL关闭固定等剂量早晚时序，不改起点、剂量、轮次、系数、RX、seed或门槛救跑，不将控制事后晋级。PASS仅支持另行登记源域确认；不等于有效Ag、正式target收益或跨seed联合成功。当前run明确source-only，失败行不访问target。

## 预算、执行和技术停止

4个新模型×20＝80model-epochs、17760更新，无父训练/继承成本。每行原ordinary568320次、新As生成568320次（包含RAW和inactive shadow）；合计2273280次As生成，实际As监督共966144包。每行4546560LNN（额外1136640）＋实际原U/EMA/V/8820held；预期总NN23293274含dispatcher兼容smoke2，最终按真实计数核对，教师成本、物理生成、NN和墙钟分开报告。影子生成成本明确，不以减少输入构造成本冒充时序优势。

普通N607原CVS-RFFI环境，CPU2/worker、fullFP32，原容量策略计入所有owner/preCUDA，每GPU最多3进程、空闲至少12000MiB；不停止或修改任何健康任务。唯一owner codex/root/FcrossRX-AsTiming-20261017；CWD=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_AsTiming_sourceRX1_E20_20261017_r01。入口：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_AsTiming_sourceRX1_E20_20261017_r01/experiments/cvs_as_timing20/runner.py。输出=/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-AsTiming-sourceRX1-e20-s2026092701-m4-r01，日志=/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-AsTiming-sourceRX1-e20-s2026092701-m4-r01/logs。本地证据=E:/type10-7/local_artifacts/cvs_FcrossRX_training_observation_20261017。

本次Git commit为实际代码/配置身份；一次N607 preflight已PASS，发布将做一个archive本地/远端SHA比对和remotecompile。launcher首步同契约真实CP no-query smoke，PASS立即继续四行scratch；实际PID/CWD/argv/GPU/log增长独立核实后才记RUNNING。角色/来源/非有限/错误预算/输出冲突/不可启动等系统技术失败保留产物、停止新派发，不自动重试，不停止健康已运行worker。普通低分不是技术失败，不热改或选择性重跑。

本地compile和15个协议反例、7个科学反例PASS；真实CVS合成检查8216NN／1024完整物理调用／4个联合更新PASS，六/十四轮激活、同选中IQ成对、四行同shadow统计、206state/900722参数、默认头更新和导出joint160均核实。真实IQ/CP/target为0；测试首个计数断言曾混淆state张量与参数张量，已在任何NN前修正，未重跑真实实验。一轮限定独立P0/P1静态审查PASS。

## 自动收尾和交付边界

每行FINAL20后即自动冻结checkpoint内原默认头，读出clean＋六种原practical的held source预测、margin/30和传播/接收元数据，不等其他row完成后才测试。随后独立全量复算各row/TX/day/状态/质量/最差RX、宏F1、paired修复/新增错误、固定RAW clean正确集的sat保持率、原L/U顺序、实际覆盖、loss/梯度曲线和成本。单held RX1不能冒充五个RX最差值，四row也不是四个seed。所有负结果和缺项保留。

真实IQ、checkpoint、逐包ID/预测、原日志、生成图保留私有产物；Git仅代码/config/登记/报告与合规聚合。正式目标评分、跨seed重现以及low>60%/suburban接近自身clean均尚未证明；本轮源域默认头结果与clean代价见下文完整复算，总目标保持ACTIVE。


## 全部七个视图的默认头结果

每格为准确率%／六类宏 F1%；每行每视图固定 1260 包、每 TX 210 包。均使用对应 FINAL20 内原默认 cos30 头，0 次另训头。

|视图|RAW|ALL|EARLY7|LATE7|
|---|---:|---:|---:|---:|
|clean|79.841270／79.343615|83.730159／83.241273|80.000000／79.035397|81.269841／81.106673|
|practical_high|38.412698／28.303558|58.571429／56.769435|53.412698／50.130626|56.507937／53.299344|
|practical_mid|39.047619／29.361850|54.920635／53.103216|49.365079／45.685720|53.571429／50.162762|
|practical_low_suburban|36.031746／28.450928|51.746032／49.991600|47.698413／44.837749|50.317460／47.422835|
|practical_high_urban|37.619048／28.793624|52.936508／51.371149|48.333333／45.220134|52.460317／49.431707|
|practical_mid_urban|32.063492／26.669836|42.777778／42.122174|38.333333／36.382559|40.396825／39.065087|
|practical_low_urban|26.984127／23.660254|33.571429／32.818207|32.539683／30.920409|34.206349／33.124936|
|urban 三场景准确率均值|32.222222|43.095238|39.735450|42.354497|

RAW／ALL／EARLY7／LATE7 的 clean 准确率为 79.841270%／83.730159%／80.000000%／81.269841%，low urban 为 26.984127%／33.571429%／32.539683%／34.206349%。这是四个控制行，不是四个独立 seed。

|LATE7 对照差，百分点|对 RAW|对 ALL|对 EARLY7|
|---|---:|---:|---:|
|clean|1.428571|-2.460317|1.269841|
|practical_high|18.095238|-2.063492|3.095238|
|practical_mid|14.523810|-1.349206|4.206349|
|practical_low_suburban|14.285714|-1.428571|2.619048|
|practical_high_urban|14.841270|-0.476190|4.126984|
|practical_mid_urban|8.333333|-2.380952|2.063492|
|practical_low_urban|7.222222|0.634921|1.666667|
|urban 均值|10.132275|-0.740741|2.619048|
|clean 宏 F1|1.763058|-2.134600|2.071276|
|seen V|-0.023148|-0.111111|-0.143519|

固定条件独立复算与原运行决策完全一致：对 RAW、EARLY7 的全部条件通过；对 ALL 的 clean≥+1、clean 宏 F1不降、urban≥−0.5 和逐 practical 保持条件失败。对 ALL 的 low +0.634921 个百分点通过单项容忍条件；这不能抵消其他失败。健康及预算条件通过。ALL 的同 seed 局部较高结果保留为控制证据，不作为事后选中配方。

LATE7 自身 suburban 比 clean 低：practical_high 24.761905 个百分点、practical_mid 27.698413 个百分点、practical_low_suburban 30.952381 个百分点。仍远未达到与自身 clean 接近的目标。

![七视图结果](E:/type10-7/local_artifacts/cvs_FcrossRX_training_observation_20261017/final/figures/source_scene_results.png)

## 配对错误、固定正确集与间隔

|视图|对照|LATE7 修复|LATE7 新增错误|净正确数|
|---|---|---:|---:|---:|
|clean|RAW|38|20|18|
|clean|ALL|29|60|-31|
|clean|EARLY7|56|40|16|
|practical_high|RAW|324|96|228|
|practical_high|ALL|66|92|-26|
|practical_high|EARLY7|102|63|39|
|practical_mid|RAW|286|103|183|
|practical_mid|ALL|88|105|-17|
|practical_mid|EARLY7|112|59|53|
|practical_low_suburban|RAW|267|87|180|
|practical_low_suburban|ALL|82|100|-18|
|practical_low_suburban|EARLY7|102|69|33|
|practical_high_urban|RAW|276|89|187|
|practical_high_urban|ALL|92|98|-6|
|practical_high_urban|EARLY7|111|59|52|
|practical_mid_urban|RAW|204|99|105|
|practical_mid_urban|ALL|79|109|-30|
|practical_mid_urban|EARLY7|97|71|26|
|practical_low_urban|RAW|170|79|91|
|practical_low_urban|ALL|95|87|8|
|practical_low_urban|EARLY7|99|78|21|

固定 RAW clean 正确集为 1006 包，不随候选变化。下表是同一集合上的保持率%，不是重新拟合分类器。

|视图|RAW|ALL|EARLY7|LATE7|
|---|---:|---:|---:|---:|
|clean|100.000000|97.813121|97.017893|98.011928|
|practical_high|36.580517|69.681909|59.343936|65.506958|
|practical_mid|36.580517|64.314115|54.671968|60.735586|
|practical_low_suburban|34.691849|61.232604|53.479125|57.057654|
|practical_high_urban|36.580517|63.021869|53.578529|60.437376|
|practical_mid_urban|30.417495|48.906561|40.954274|44.930417|
|practical_low_urban|26.938370|38.170974|35.884692|38.369781|

|默认头平均间隔／固定 cos30 尺度|RAW|ALL|EARLY7|LATE7|
|---|---:|---:|---:|---:|
|clean|0.108450|0.119804|0.108058|0.116236|
|practical_high|-0.038559|0.055843|0.033911|0.036206|
|practical_mid|-0.036842|0.046471|0.026922|0.029856|
|practical_low_suburban|-0.040314|0.037487|0.017428|0.021280|
|practical_high_urban|-0.039237|0.044448|0.024330|0.029396|
|practical_mid_urban|-0.045126|0.017084|-0.001056|0.002279|
|practical_low_urban|-0.048652|0.000099|-0.016863|-0.013707|

low 上 LATE7 的准确率略高于 ALL，但平均 margin/30 为 −0.013707，ALL 为 +0.000099；CE 为 1.834895 对 1.656546。不能把单一间隔统计当成准确率收益。本轮没有新增 clean→As 一致性、特征或门控对齐。没有另设 gate 权重记录钩子，因而不声称门控变化已解释收益；保留真实前向和联合梯度记录。CE／margin 来自固定运行保存的标量，不冒称由未保存的全 logits 重新算出 CE；hard predictions、准确率、F1 和混淆矩阵为独立复算。

![固定正确集和配对错误](E:/type10-7/local_artifacts/cvs_FcrossRX_training_observation_20261017/final/figures/fixed_retention_and_paired_errors.png)

## low urban 的状态与质量分层

每个模型使用完全相同的物理样本、信道、接收状态和补偿参数。下表是关联分层，不是移除某种损伤后的独立因果贡献。

|维度|层|样本数|RAW%|ALL%|EARLY7%|LATE7%|
|---|---|---:|---:|---:|---:|---:|
|state|LOS|384|40.104167|57.031250|51.562500|54.947917|
|state|BLOCKED|714|18.347339|19.047619|19.747899|20.028011|
|state|PARTIAL|162|33.950617|41.975309|43.827160|47.530864|
|lock|locked|467|40.471092|55.674518|50.963597|55.460385|
|lock|degraded|228|21.052632|25.877193|29.824561|28.947368|
|lock|unlocked|565|18.230088|18.407080|18.407080|18.761062|
|quality|[20,30)|396|40.909091|57.323232|51.767677|55.808081|
|quality|[5,10)|157|19.108280|21.019108|24.840764|25.477707|
|quality|[0,5)|385|18.701299|19.480519|20.000000|21.038961|
|quality|<0|180|17.222222|16.111111|15.000000|13.888889|
|quality|[10,20)|142|31.690141|41.549296|43.661972|45.070423|

相对 RAW，LATE7 对 LOS 从 40.104167% 提高到 54.947917%，PARTIAL 从 33.950617% 到 47.530864%，locked 从 40.471092% 到 55.460385%。BLOCKED 仅从 18.347339% 到 20.028011%，unlocked 从 18.230088% 到 18.761062%；质量代理 <0 dB 反而从 17.222222% 降到 13.888889%。因此得到缓解的是仍保留较多可用身份证据的观测，最困难状态没有被有效解决，也不能把改善唯一归因于噪声、多径或门控。

TX／day／新增噪声 SNR／abs CFO／残余核长度／AGC 限幅／边界的全部 824 个分层记录保存在 source_strata.json；质量代理和 signal_to_added_noise_db 均不冒充总 SNR。这里只有一个 held RX，不能给出五 RX 的 worst-held-RX 结果。训练 RX 上 V 的最差值另列，不混同 held 性能。

![low 状态和质量](E:/type10-7/local_artifacts/cvs_FcrossRX_training_observation_20261017/final/figures/low_urban_strata.png)

## 全训练曲线与梯度健康

独立读取四行全部 17760 更新、80 个 epoch、全部 stdout／CSV／结构记录及 23976 个 L/U 序列记录，没有只取 tail 作为完整分析。所有有效更新有限，无零梯度或跳过更新。stdout 中每行 260 处 nan 文本来自关闭项的空指标；338 个禁用空字段有明确解析，不能误记为数值发散。

|行|最终 base 目标|最终总 L 目标|额外 CE|额外 KL|最终平均裁剪前梯度|全程最大裁剪前梯度|全程 >5 的步数|seen V%|V 最差 RX%|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|RAW|0.310891|0.335194|0.083361|0.013848|0.704541|10.752827|33|98.004630|95.333333|
|ALL|0.321413|0.459971|0.497991|0.056240|0.794334|14.150159|33|98.092593|95.703704|
|EARLY7|0.313540|0.338822|0.086921|0.014210|0.693692|10.602881|31|98.125000|95.814815|
|LATE7|0.320678|0.515584|0.681556|0.098070|0.875202|10.752827|32|97.981481|95.259259|

LATE7 在 E14 启用 As 后，额外 CE 为 0.930410，E20 仍为 0.681556；ALL E20 为 0.497991。LATE7 总 L 目标 0.515584，高于 ALL 的 0.459971；原健康门槛针对完整 base≤0.5，LATE7 base 为 0.320678，不能误用总目标判技术失败。原 seen V 在末五轮基本稳定，但预算耗尽不等于 As 收敛。

原 cosine LR 从 0.0002 降到 E14 的 0.000046078658，再到 E20 的 0.000001。原 SupCon 从 E6 启用，3330 步／13320 层项；原 SSL 从 E14 开始。每个 SSL epoch U 总观察数 56624，门控选中数按整轮 SUM 作图，E20 为 1080／1056／1070／1060；U 真值未读取，teacher 额外时序调用为0。后期 As 与较低 LR 和 SSL 同时出现，不能声称只检验了单一先后机制。

原完整 L／U 序列和保存的 As shadow 统计四行完全相同；所有六场景的评估 physics 参数和接收补偿状态也完全一致。尽管同 seed／初始状态／序列一致，LATE7 与 RAW 在第1步裁剪前梯度已存在微小数值差异，额外 CE 第3步、总目标第5步开始差异，不能宣称不同 GPU 的真实训练逐位一致。

![完整20轮曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_training_observation_20261017/final/figures/complete_training_curves.png)

## 实际曝光与成本

|行|As 生成|As 消费|消费唯一 L|TX×RX×day 格|再分 scene 格|NN 包前向|训练原生 elapsed 秒|进程起点至最终结果秒|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|RAW|568320|0|0|0|0|5823318|2301.330696|2319.033353|
|ALL|568320|568320|5040|72|432|5823318|2357.209996|2375.316348|
|EARLY7|568320|198912|5040|72|432|5823318|2279.515132|2297.356354|
|LATE7|568320|198912|5040|72|432|5823318|2305.957146|2323.869352|

|行|LOS|PARTIAL|BLOCKED|locked|degraded|unlocked|
|---|---:|---:|---:|---:|---:|---:|
|ALL|392475|66448|109397|438962|60995|68363|
|EARLY7|137515|23151|38246|153701|21307|23904|
|LATE7|137367|23489|38056|153704|21295|23913|

EARLY7／LATE7 均消费 198912 个 As 包，全部 5040 个 fit L 和 72 个 TX×RX×day 格、432 个 scene 格均覆盖。两者 BLOCKED 实际 38246／38056、PARTIAL 23151／23489，场景计数亦有差异；这是同总量日历比较，不是同一物理/draw 多重集合交换顺序。joint 状态覆盖采用保存的原生记录，物理 TX/RX/day 和边际计数独立复算；没有重新打开 IQ 回放物理过程。

合计 80 model-epochs／17760 联合更新／23293274 NN 包前向（含唯一 dispatcher 真实 checkpoint 兼容 smoke 2），0 次独立头训练。每行 4546560 L NN，其中额外 1136640；原 student_train 4942928、student_eval 475202、teacher_train 0、teacher_eval 396370 的计数包含原计数器中兼容探针的 2 次，独立 NN 合计时已扣除，再加 held8820。四模型新父训练成本为0，不加载历史模型；兼容 smoke 的历史 B00 E20／4440 更新／4686678 NN 仅单列来源成本，不能算本轮初始化或新增训练。

每行 568320 原 ordinary 生成及 568320 完整 As shadow；四行共 2273280 次 shadow 生成、966144 次有标签 As 监督，影子成本没有隐藏。矩阵从首 worker 起点到完成为 2380.459348 s，约 39.674322 min；并行墙钟不等于四行耗时之和。每行七视图首次至末次写入约13.2至13.4 s，属于写入跨度，不冒称完整预测计算时间。

![时序曝光和实际覆盖](E:/type10-7/local_artifacts/cvs_FcrossRX_training_observation_20261017/final/figures/timing_dose_and_calendar.png)

## 执行、验证和交付

一次普通 N607 preflight、一个 archive 本地/远端 SHA 比对、remote compile 和同契约真实 checkpoint no-query smoke均 PASS。四个原 worker PID168714/168787/169270/169349、GPU0/1/2/3，以及 dispatcher168626 已自然终止；独立 post-state 确认全部 final、七视图、无 failure.json，不依赖仅退出码。未停止、重启或修改其他健康任务。

本地15个协议反例／7个科学反例、真实CVS合成8216 NN／1024完整物理调用／4联合更新、一次限定独立P0/P1审查 PASS。离线19个协议反例／7个科学反例、四组17760步完整合成日历和8个计数反例 PASS；0新增真实 IQ/CP/NN。完成后130个固定文件取回、35280条源判决独立复算与固定条件一致；只取报告/日志/预测，没有取回IQ或权重。五张PNG＋五张PDF已实际渲染并逐张检查可读，无裁切。

本轮新增代码：experiments/cvs_as_timing20/{design,model,source,runner,dispatch,preflight,publish,harvest,cost,analyze,plot,checks_analysis,checks_timing_recount}.py 和登记配置；模型 state206、参数900722保持，默认头联合更新已合成核实。离线代码和结果报告提交不改变运行 release。

实际启动命令、环境/CWD/GPU、stop rule及路径在上文预登记段和 experiment.json 保留。离线分析命令：F:/App/miniconda3/python.exe -m experiments.cvs_as_timing20.analyze --input E:/type10-7/local_artifacts/cvs_FcrossRX_training_observation_20261017/final，CWD=E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014；制图命令为同一Python执行 E:/type10-7/local_artifacts/cvs_FcrossRX_training_observation_20261017/plot_local.py（仅提交已完成的聚合JSON到既有环境，0NN）。

聚合证据：[独立复算](<E:/type10-7/local_artifacts/cvs_FcrossRX_training_observation_20261017/final/analysis_summary.json>)、[全部分层](<E:/type10-7/local_artifacts/cvs_FcrossRX_training_observation_20261017/final/source_strata.json>)、[全部曲线](<E:/type10-7/local_artifacts/cvs_FcrossRX_training_observation_20261017/final/training_curves.json>)、[实际成本](<E:/type10-7/local_artifacts/cvs_FcrossRX_training_observation_20261017/actual_cost.json>)、[图索引](<E:/type10-7/local_artifacts/cvs_FcrossRX_training_observation_20261017/final/figures/manifest.json>)。每row冻结身份见 source/source_frozen.json；checkpoint 为相应row/source/final_ssdg.pth，仅保留远端，未另训/替换头。

最终回答：固定 F＋跨 RX 同 TX CVS 与原 practical 下，本轮 LATE7 相对匹配 RAW 的默认头 urban 均值提高10.132275、low提高7.222222个百分点，clean提高1.428571个百分点；改善主要在LOS/PARTIAL/locked，而BLOCKED/unlocked仍低。相对全程As却付出clean −2.460317和urban均值−0.740741个百分点，固定候选FAIL。low仅34.206349%，未到60%；不存在正式target／多RX最差值／跨模型seed重复证明，也不存在已验证新Ag或Ag×As因子交互。所有缺项保留，总目标继续ACTIVE。
