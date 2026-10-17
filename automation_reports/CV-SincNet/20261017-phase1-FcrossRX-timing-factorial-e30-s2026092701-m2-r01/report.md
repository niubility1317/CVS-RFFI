# E30 有界时移 × 原 practical：固定配方负结果

**ANALYZED／VERIFIED；本固定时移方案关闭，联合研究目标仍 ACTIVE。** 两条新增训练均从零完成 E30、FINAL30 冻结、clean＋六种 practical 预测和独立 truth-last 评分。四项预登记源判据全 FAIL；不扩种子，不扫描时移幅度、概率、边界、RX、seed 或预算救跑。

时移单独训练缓解了部分时序敏感性，但没有带来 clean 跨 RX 泛化收益；叠加原 practical 后 clean、urban 与 suburban 全部低于仅 practical 控制。本轮没有实现 low urban>60% 或双环境突破。

## 对照、真实执行与权限

B00/B01 原样复用 P32 的同 seed、全源、scratch E30、raw-secondary 控制，保持原配置、权重和七视图产物，不重训、不重预测、不重评分。只新增 B10/B11。四行共有 F＋跨 RX 同 TX SupCon、SIG_TIME_X3、CENTER_25K/CENTER_UNITS Sinc、joint160、900722 参数/206 state tensors、默认 cosine scale30 及原 RDrop/旧门控/ownEMA；骨干与默认头正常联合训练，0另训头/诊断分类器/新增部署参数。

|行|主视图（监督0.5）|第二视图（监督0.5）|来源|
|---|---|---|---|
|B00|原 ordinary(x)|raw|P32完整复用|
|B10|原 ordinary(x)|timing(raw)|新增 scratch E30|
|B01|原 ordinary(x)|As(raw)|P32完整复用|
|B11|原 ordinary(x)|As(timing(raw))|新增 scratch E30|

两半 CE/RDrop 分别归一化再各乘0.5；每对RDrop复用同一已生成IQ，原SupCon仅primary。全部合法有标签卫星样本参与CE。B11实际As覆盖852480观察、90个TX×RX×scene单元和90个TX×RX×state单元。全部源RX1/3/4/6/8与day1/2/3、L6300/U56700隐藏TX/V27000、split392005/7:63:30不变，无留出训练RX或新增Ldev；V不更新参数。每行20L+10U、222步/轮、6660更新、3409920个L网络前向样本。原U配方保持，没有新增U时移或urban学生高置信门控。

时移固定均匀整数[-16,16]、reflect、不循环，恢复raw原包RMS；每行实际852480视图、6300个唯一L物理样本、全部33偏移，最大相对功率误差1.307899e-7，平均相对L2变化1.390768。时移在数值波形距离上不小，不能只凭“16点”称温和。私有训练seed202610174101、源诊断202610174102，按视图/物理观察/出现序号命名，无TX依赖、跨epoch/view刷新，训练/源诊断/正式流分离。

实际__file__已逐行保存：leo_practical/channel.py、batch.py、residual.py，以及DataAugmentation.py、practical_adapter.py、SSDG/train_ssdg.py来自原不可变release的experiments/adv3b02_xuc/code，**不是顶层code/**。新增timing来自本轮release。As仍为原residual/post_sync，EQ关闭、G=1，reference_snr_db=30、fs25MHz、fc2.45GHz、reflect边界，原接收质量适配/相位跟踪/IQ补偿/AGC及其依赖保留。记录equalized=1不等于新增星地EQ。没有提高正式SNR、改信道、改received IQ或删样本。reference SNR/新增噪声比均不等于总SNR。

归一化：原float64记录转float32后在完整256点做数据集RMS（center=false、eps1e-12）；ordinary主视图保持原流程；新增timing恢复raw功率；原As按原处理规范归一化输入，输出不额外RMS。网络既有内部归一化及补偿保持。没有逆推出原录音未知TX硬件。kernel62/数值延迟24/参考80:160使±16下直接参考IQ依赖位于[3,176)，但不声称所有分支的神经感受野均避开边界。

## 正式七环境结果

下表单位为%；每视图168000原固定样本，model seed仅2026092701。B00/B01为既有同预算控制，B10/B11为本轮新增模型。

|视图|B00|B10：时移|B01：practical|B11：联合|
|---|---:|---:|---:|---:|
|clean|85.237500|83.286905|83.057143|81.718452|
|practical_high|46.350595|67.655357|81.772619|78.906548|
|practical_mid|44.589286|64.651786|79.417857|76.617857|
|practical_low_suburban|42.622024|61.844048|77.039286|74.370833|
|practical_high_urban|44.975000|64.282738|78.350000|75.717262|
|practical_mid_urban|37.426190|50.724405|64.337500|61.985714|
|practical_low_urban|31.755357|40.779762|53.394643|51.480357|
|urban三场景均值|38.052183|51.928968|65.360714|63.061111|

|指标（%）|B00|B10|B01|B11|
|---|---:|---:|---:|---:|
|clean Macro-F1|84.987751|82.944712|82.666197|81.120144|
|low urban Macro-F1|30.425385|39.082576|52.824516|50.880262|
|low urban最差RX|28.183333|34.120833|48.737500|45.895833|

B11 low最差RX为14-7：45.895833%；B01最差RX为20-1：48.737500%。完整逐RX/day/TX/F1见实际CSV。

## 收益归因与代价

|配对差（百分点）|clean|mid urban|low urban|urban均值|
|---|---:|---:|---:|---:|
|B10−B00|-1.950595|+13.298214|+9.024405|+13.876786|
|B01−B00|-2.180357|+26.911310|+21.639286|+27.308532|
|B11−B00|-3.519048|+24.559524|+19.725000|+25.008929|
|B11−B10|-1.568452|+11.261310|+10.700595|+11.132143|
|B11−B01|-1.338690|-2.351786|-1.914286|-2.299603|

仅原As相对B00使low urban提高21.639286pp，clean付出2.180357pp。新增时移相对B00使low提高9.024405pp，但clean下降1.950595pp，因此不是所需Ag泛化收益。B11相对B01 low下降1.914286pp、urban均值下降2.299603pp、clean下降1.338690pp。不能将独立收益相加预测组合。

|视图|交互B11−B10−B01＋B00（pp）|
|---|---:|
|clean|+0.611905|
|practical_high|-24.170833|
|practical_mid|-22.862500|
|practical_low_suburban|-21.890476|
|practical_high_urban|-21.940476|
|practical_mid_urban|-15.650000|
|practical_low_urban|-10.938690|

|模型|高suburban距自身clean（pp）|中suburban|低suburban|
|---|---:|---:|---:|
|B00|38.886905|40.648214|42.615476|
|B10|15.631548|18.635119|21.442857|
|B01|1.284524|3.639286|6.017857|
|B11|2.811905|5.100595|7.347619|

B11 suburban三场景全部未达首轮距clean≤2pp探索目标，而且clean本身下降；不把差距变化记作联合成功。固定匹配B00 clean正确的143199个正式样本上，low保持率B01=58.215490%，B11=56.440338%。这是固定诊断参照集合，不是训练父权重。

旧ordinary-secondary As背景参照仍原样保留：clean=82.797024%、low=53.137500%。其输入不同，不替代本轮raw-secondary因果控制。

## 源判据、退化类型与机制边界

源判据在读取新正式分数前固定；首条否定先由commit374e8ebfa记录，dispatcher完整source_matrix_decision随后由独立分析复算一致。全部4项FAIL：B10全Vclean下降0.407407pp、B11下降0.418519pp，均越过−0.25pp；B10两timing draw提升4.888889/4.777778pp，未达+5pp；B11原full urban两draw的mid增量−2.166667/−1.722222pp，low增量−0.611111/−1.611111pp，未达每格+1pp。没有阈值放宽、目标反馈或选择性重跑。

仅用原源V1800×两draw（各场景3600观察），在同一原full practical上：

|源场景准确率（%）|B00|B10|B01|B11|
|---|---:|---:|---:|---:|
|practical_low_suburban|54.750000|76.916667|90.777778|88.888889|
|practical_mid_urban|45.861111|61.972222|75.861111|73.916667|
|practical_low_urban|38.555556|48.333333|62.138889|61.027778|

low urban源分层，新增噪声比是signal_to_added_noise_db，**不是总SNR**：

|信号/新增噪声比|观察数|B00|B10|B01|B11|
|---|---:|---:|---:|---:|---:|
|<3 dB|1127|21.118012|20.053239|30.079858|29.724933|
|3-8 dB|794|27.329975|26.700252|44.332494|42.947103|
|8-15 dB|303|41.914191|52.145215|78.217822|75.247525|
|>=15 dB|1376|58.575581|83.139535|95.130814|93.968023|

时移单独训练的源收益主要出现在LOS/PARTIAL及较高新增噪声比段，未解决BLOCKED/unlocked的噪声主导困难。low源LOS/locked从60.660377%到85.094340%，PARTIAL/locked从51.465798%到76.547231%；BLOCKED/unlocked反而从22.300319%到21.533546%。已有As在这些可恢复状态上更强，新增时移没有保留增量。微小单元（如PARTIAL/unlocked n=1）仅描述，不作机制结论。正式测试无逐包状态文件，不能把源分层冒充正式LOS/lock成绩。

前序P33/P34冻结源归因保留：移除噪声的固定状态与重新计算状态效应不同；flat-H族的数值延迟贡献通过、相位贡献失败。去延迟同时改变reflect前缀，不能证明原full多径全部由延迟造成。本轮支持“时序敏感性可被部分缓解”，不支持它是新的跨RX身份泛化，或能再叠加到As。

默认头scale统一为30。源low平均真类间隔/30：B01=0.068694，B11=0.060699。Sinc/默认头/原投影均有联合更新，完整梯度健康记录无跳步；参数位移、gate接受率不作为身份收益。未保存逐包内部门控激活与逐分支梯度，不能把本次退化归咎于某个门控。

## 实际预算、验证与产物

|新增行|epoch/更新|L网络前向|源阶段总NN|源读出NN|正式预测NN|源训练墙钟s|预测墙钟s|
|---|---:|---:|---:|---:|---:|---:|---:|
|B10|30/6660|3409920|5407922|16202|1176002|1633.512086|47.255914|
|B11|30/6660|3409920|5407922|16202|1176002|2940.112291|48.232849|

每新行student train3976920/eval864002、teacher train0/eval567000；源阶段5407922 NN。新训练无父权重，父训练成本0；兼容性用历史全源FINAL20仅2个合成NN，不作初始化。另旧B00时移参考3602NN、新行各16202源读出NN。累计新增60模型epoch/13320更新/13203856NN，含所有新smoke、教师、源读出和正式预测。B00/B01只读历史成本另列60模型epoch/13320更新/13193052NN；四模型统计读回120epoch/26640更新不是本次新增训练量。

调度启动到终态3020.884386s，B10/B11并行GPU2/3，FP32/CPU2，每GPU含其他owner总进程cap3。墙钟受设备与并行负载影响，不解释成固定加速倍率。主分析进程墙钟2747.643464s**含等待训练**；实验终态到分析完成144.699988s，不能当纯CPU处理时间。源归因P33/P34既有成本单列于analysis_summary，不重复算新增。预算耗尽不等于收敛，不据尾部曲线延长轮次。

验证实际完成：7项IQ/路由/源判据/NaN比较回归、4项结果分析流程回归、2项实际增强统计回归与编译PASS；一次P0/P1检查及一次原NaN问题定向复核PASS；一个release archive对比、远端compile、启动真实CP no-query smoke和PID/CWD/argv/GPU/log增长VERIFIED。Git提交保留且已推送，最终OID按交付提交独立核对。

四模型各30轮完整日志及6660步已解析；每行1198行stdout、活动loss有限、0错误标记/跳过更新，300处NaN文字均为未启用的可选统计。五组样本序列各8880条全部相同；B10/B11的6660步timing元数据/偏移/数值统计全相同。普通增强120轮实际概率及波形MSE/maxabs均匹配。共有旧ordinary实际PA概率0.014至0.049、CFO/循环移位0.035至0.1225、AWGN0.04至0.14、多径0.018至0.063、DAC0；不据默认值误判关闭。主视图保留旧普通损伤，As从raw/timing(raw)开始，没有再叠加ordinary AWGN/多径。

每row先固定七视图1176000预测再独立truth-last；全部504个正式CM和1112个源分层CM复算一致。两条新增正式预测2352000个，0头拟合/目标训练/U真值/新查询适配。全部worker、dispatcher和两分析进程自然结束，未停止/修改健康旧实验。10组图的PNG和PDF（共20文件）实际生成、取回并查看，未编造路径。

训练执行commit：7e4af64bb887c08c6a225bcb6bee4856fdb7c588；四因子分析commit：2a1efbaeca711303083cb78c5a4645e089ed5dae；真实增强统计commit：f81590fab954f1b4d27b528410611775c556da26。实现目录experiments/cvs_timing_factorial35；固定配置design.py；入口runner.py，精确环境/CWD/GPU/停止规则见保留experiment.json/preregistration.md。

实际命令：

`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_timing_factorial_E30_20261017_r01/experiments/cvs_timing_factorial35/runner.py`

独立分析：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-timing-factorial-e30-s2026092701-m2-r01/analysis_code_r01/runner.py --truth [原固定truth.json] --output /home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-timing-factorial-e30-s2026092701-m2-r01/analysis_r01 --commit 2a1efbaeca711303083cb78c5a4645e089ed5dae --wait-dispatcher 663968`（完整解释器和truth路径保留于本地analysis_launch.json）。正式IQ/权重/逐包预测/完整细日志均保留远端不可变run root，不入Git。

本地产物：

- [逐row/seed/RX/day/TX结果](E:/type10-7/local_artifacts/cvs_FcrossRX_timing_factorial_E30_20261017/analysis_r01/per_row_seed.csv)
- [完整统计与成本](E:/type10-7/local_artifacts/cvs_FcrossRX_timing_factorial_E30_20261017/analysis_r01/analysis_summary.json)
- [训练曲线CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_timing_factorial_E30_20261017/analysis_r01/training_curves.csv)
- [源状态/质量分层](E:/type10-7/local_artifacts/cvs_FcrossRX_timing_factorial_E30_20261017/analysis_r01/source_strata.csv)
- [源时移分层](E:/type10-7/local_artifacts/cvs_FcrossRX_timing_factorial_E30_20261017/analysis_r01/source_timing_strata.csv)
- [普通增强实际日程](E:/type10-7/local_artifacts/cvs_FcrossRX_timing_factorial_E30_20261017/telemetry_r01/ordinary_probability_curves.csv)
- [分析完成记录](E:/type10-7/local_artifacts/cvs_FcrossRX_timing_factorial_E30_20261017/analysis_r01/completion.json)
- [正式结果图](E:/type10-7/local_artifacts/cvs_FcrossRX_timing_factorial_E30_20261017/analysis_r01/formal_accuracy.png)
- [训练曲线图](E:/type10-7/local_artifacts/cvs_FcrossRX_timing_factorial_E30_20261017/analysis_r01/training_curves.png)
- [源状态图](E:/type10-7/local_artifacts/cvs_FcrossRX_timing_factorial_E30_20261017/analysis_r01/source_state_lock_accuracy.png)
- [源新增噪声比分层图](E:/type10-7/local_artifacts/cvs_FcrossRX_timing_factorial_E30_20261017/analysis_r01/source_added_noise_accuracy.png)

rows/B10、B11的实际args/config/module_files/初始化/冻结/计数及源读出已取回本地产物目录；参考控制仍保留其P32原来源，未合并成伪造新训练。

## 结论及未完成项

已证实普通As在固定测试下改善urban，但clean有代价；时移单独训练缓解部分时序敏感性，未改善跨RXclean，且不与As叠加。默认头本轮B11相对仅As反而low−1.914286pp、clean−1.338690pp；没有达成low>60%和suburban接近clean。仅1个模型seed，两源draw不是两个模型seed，**不能声称跨seed复现**。

当前固定候选关闭，完整目标继续ACTIVE。无新已获源证据支持的训练候选；不从本轮正式分数选择新幅度/父权重/损失/seed。下一最小源问题应回到原默认头在噪声主导观测下的分支证据与clean/satellite梯度关系，先核实而非预设门控或梯度冲突；尚未实现或运行，不能当新方法成立。已有FIR/PSD/PA旁路训练或源否定结果继续保留，不换命名重跑。
