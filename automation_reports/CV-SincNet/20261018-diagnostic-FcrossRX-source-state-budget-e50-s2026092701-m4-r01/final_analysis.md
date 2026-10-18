# E50追加预算在源域缓解了哪些观测退化

**已完成零前向源域重分层。** 在固定同一批full practical观测上，mid urban的BLOCKED/degraded由42.530488%升至48.170732%（+5.640244个百分点），贡献37/71个净修复；low urban的BLOCKED/unlocked由30.351438%升至31.884984%（+1.533546个百分点），贡献24/59个净修复。后者仍最困难。low的PARTIAL/locked完全持平；1至10kHz残余频偏区间没有准确率改善证据。

这是原完整As配方E30→E50的源域状态关联，不是新训练机制，不证明噪声、CFO或多径的单独因果效应。全部结果仅modelseed2026092701、1800个独立源V物理样本、两次固定信道draw。新NN／IQ／checkpoint／更新／训练／正式评分均为0。没有对本轮E50其他四行调参、选模、晋级或干预。

## 同观测状态的预算增量

以下n是两draw合并的视图数，不能当作独立物理样本数。准确率为%，变化为绝对百分点；预算交互=(As50−As30)−(NoAs50−NoAs30)。所有未显示的空状态仍保留在完整CSV；LOS/degraded、LOS/unlocked、BLOCKED/locked均n=0，不能视作准确率0。

|场景|状态／锁定|n|As30|As50|NoAs50|As追加预算变化|修复／新增错误|预算交互|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|mid|LOS/locked|1899|95.471301|96.577146|54.133755|+1.105845|27／6|+1.579779|
|mid|PARTIAL/locked|306|88.235294|89.869281|53.594771|+1.633987|6／1|+0.980392|
|mid|PARTIAL/degraded|59|76.271186|77.966102|52.542373|+1.694915|1／0|+1.694915|
|mid|PARTIAL/unlocked|2|0|0|0|0|0／0|0|
|mid|BLOCKED/degraded|656|42.530488|48.170732|31.097561|+5.640244|55／18|+5.792683|
|mid|BLOCKED/unlocked|678|31.563422|32.595870|21.386431|+1.032448|34／27|−0.737463|
|low|LOS/locked|1060|94.716981|96.320755|54.716981|+1.603774|19／2|+2.358491|
|low|PARTIAL/locked|307|88.925081|88.925081|49.837134|0|6／6|−3.257329|
|low|PARTIAL/degraded|253|79.446640|81.818182|39.920949|+2.371542|11／5|+3.952569|
|low|PARTIAL/unlocked|1|0|0|0|0|0／0|0|
|low|BLOCKED/degraded|414|42.028986|44.927536|31.159420|+2.898551|23／11|+1.932367|
|low|BLOCKED/unlocked|1565|30.351438|31.884984|21.661342|+1.533546|77／53|+0.383387|

mid的BLOCKED/degraded占净修复52.112676%；low的BLOCKED/unlocked占40.677966%，LOS/locked另贡献17个净修复。PARTIAL/unlocked仅1或2个视图，不支持一般化结论。逐draw非空state_lock层未见As50相对As30的准确率退步，但预算交互存在上表两处负值；不能据此宣称所有状态的As预算收益都超过NoAs。

## 接收质量、频偏与边界

下表为signal_to_added_noise_db，包含传播后输入与新加噪声的功率比。它不包含未知原录音噪声分解，**不是总SNR**。各质量代理同样不能视作总SNR。

|场景|dB区间|n|As30%|As50%|变化（百分点）|修复／新增错误|预算交互|
|---|---|---:|---:|---:|---:|---:|---:|
|mid|<3|383|27.415144|28.720627|+1.305483|20／15|+0.522193|
|mid|3至8|718|38.579387|42.339833|+3.760446|49／22|+1.810585|
|mid|8至15|293|52.559727|56.996587|+4.436860|21／8|+6.143345|
|mid|≥15|2206|94.514959|95.693563|+1.178604|33／7|+1.541251|
|low|<3|1127|27.329193|29.281278|+1.952085|60／38|+0.709849|
|low|3至8|794|39.672544|41.057935|+1.385390|36／25|+0.503778|
|low|8至15|303|72.277228|75.247525|+2.970297|15／6|+3.960396|
|low|≥15|1376|93.386628|94.622093|+1.235465|25／8|+1.090116|

波形功率代理与white-input代理分到不同质量箱的比例：mid为116/3600=3.222222%，low为133/3600=3.694444%。这不等同已经观测到接收状态切换，本分析只使用full观测的分类结果。mid waveform_higher组n=52从34.615385%降至32.692308%（−1.923077个百分点，修复3／新增4）；waveform−white为1至3dB组n=40下降2.5个百分点。这两组重叠，不能相加。

绝对残余频率1至10kHz组：mid n=48，As30/As50均22.916667%；low n=120，均29.166667%。预算交互分别−8.333333／−4.166667个百分点，因此没有该频偏区间改善证据。残余核能量<−20dB组mid n=1271提高3.304485个百分点，low n=2012提高1.838966个百分点；该变量与遮挡、质量及状态共同变化，不能单独归因多径。

mid没有AGC限幅样本，low仅1个且As30/As50均错，不能评价限幅鲁棒性。实际EQ关闭、G=1；全部观测的噪声着色为flat。边界均reflect、数值延迟24、核长62、有效内部输出195/256（76.171875%），这些参数无样本间变化，没有足够变化支持边界或色噪的收益归因。数值延迟不是额外物理传播时延。相位跟踪记录存在true/false，本次没有其独立分层对照。未修改任何源／正式IQ或信道。

## 固定clean正确集合与默认头间隔

固定集合为NoAs30 clean正确的1772个物理样本；两draw卫星保持率分母3544。分类头ResidualPhysicalHead的scale=30.0是固定float，不可学习，分类weight仍随原网络联合训练；本次只读已保存logits。间隔定义为真类logit减最大非真类logit，下表除以固定30。

|视图|As30保持率%|As50保持率%|NoAs50保持率%|As30平均间隔/30|As50平均间隔/30|
|---|---:|---:|---:|---:|---:|
|clean|99.717833|99.604966|99.887133|0.238302803|0.238218037|
|low suburban|89.700903|91.111738|49.661400|0.138745276|0.154276673|
|mid urban|73.673815|75.564334|43.989842|0.098742350|0.110462960|
|low urban|59.593679|61.286682|36.258465|0.057351629|0.066648948|

卫星保持率提高；clean固定集合另丢失2个样本，保持率下降0.112867个百分点。与此同时clean panel总准确率由98.333333%升至98.444444%，修复4／新增2，净修复2。两种口径不同；阶段0.1个百分点约束针对匹配总准确率，不是这个固定集合的保持率。完整V27000原生指标未在本次逐包重算。

## 执行证据与交付

run=20261018-diagnostic-FcrossRX-source-state-budget-e50-s2026092701-m4-r01；分析代码commit=a5d6ba378b5f20b3d443126cb6e8d0b3793b1bc3。同一full条件的10800条physical ID／TX／RX／day／draw／scene／session／条件记录与metadata逐条一致；4组×4视图的accuracy和macro-F1共32项与旧源汇总一致。产出2904组CM／分层、2904组配对、726组预算交互，空cell和n<30标记均保留。10项限定fixture及远端compile PASS；一次独立P0/P1 PASS；一个release archive SHA本地／远端比较VERIFIED。普通账户CPU2／CUDA隐藏，PID1182435已正常结束，实际report.json终态SOURCE_STATE_ANALYZED；worker墙钟6.211521s。新训练／更新／NN／IQ／CP／正式评分均0。原父训练与预测成本见对应P43/P50报告，不再次计入。

命令：CVS-RFFI/python -u analysis/cvs_budget50_source_states56_20261018.py --project PROJECT --out PROJECT/runs/20261018-diagnostic-FcrossRX-source-state-budget-e50-s2026092701-m4-r01/result --code-commit a5d6ba378b5f20b3d443126cb6e8d0b3793b1bc3。CWD=PROJECT/releases/cvs_FcrossRX_source_state_budget_E50_20261018_r01。本次为明确source-only诊断，没有新增模型需要formal预测。技术缺项无；科学缺项为跨modelseed重复性、反事实分类因果、正式观测状态及AGC/边界变化证据。

[全部源域分层CSV](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_20261018_r01/retrieved/result/source_strata.csv>)、[配对修复与新增错误](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_20261018_r01/retrieved/result/paired_effects.csv>)、[预算交互](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_20261018_r01/retrieved/result/budget_interaction.csv>)、[完整CM](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_20261018_r01/retrieved/result/source_confusions.json>)、[执行与限制](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_20261018_r01/retrieved/result/report.json>)。

![low urban源域状态与质量](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_20261018_r01/retrieved/result/practical_low_urban_source_states.png>)

![mid urban源域状态与质量](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_20261018_r01/retrieved/result/practical_mid_urban_source_states.png>)

E50源域low urban为60.722222%，不能写成正式测试超过60%。此前核实的正式单seed low urban仍52.387500%，尚无可重复双提升结论。原定02/03的四行E50匹配训练由另一唯一owner继续，预算与配方未改变。当前唯一在训假设仍是固定原As预算收益能否跨原定seed复现；本报告不额外启动机制候选。
