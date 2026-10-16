# F＋跨 RX 同 TX：配对 practical 的源 RX1 结果

源规则 **PASS**；两组 scratch E20 完成，0 目标访问。

|场景|C0 准确率|C1 准确率|差值 pp|C0 F1|C1 F1|
|---|---:|---:|---:|---:|---:|
|clean|83.4444%|86.1667%|+2.7222|83.2275%|86.0422%|
|practical_high|38.4259%|63.4074%|+24.9815|30.1789%|61.3980%|
|practical_mid|36.1667%|60.6296%|+24.4630|29.0926%|58.6890%|
|practical_low_suburban|34.0370%|57.2037%|+23.1667|28.0013%|55.1784%|
|practical_high_urban|37.0000%|59.8333%|+22.8333|29.0694%|58.0430%|
|practical_mid_urban|30.1667%|47.8519%|+17.6852|26.3720%|47.2812%|
|practical_low_urban|25.6852%|38.4074%|+12.7222|23.1182%|38.0837%|

urban 均值 +17.746914 pp，low +12.722222 pp，urban F1 +21.616061 pp；clean 损失 -2.722222 pp，seen V 损失 +0.185185 pp。

完整 40 epoch／8880 更新／全部 stdout、native CSV、SupCon 和物理梯度轨迹已解析。全 L/U 批次 5994 条逐条一致；默认头逐包保存预测独立复算，75,600 个 held source 预测。同名实际信道元数据完全一致。

总真实网络前向 7158036，矩阵墙钟 2121.057 s；实际源六 practical 评分耗时、U/EMA/训练/验证成本见 measured_costs.csv，父模型权重成本0。本地合成检查另计。

C0=.5 RD(共同旧 ordinary)+.5 RD(同一 IQ)；C1=.5 RD(共同旧 ordinary)+.5 RD(practical(raw))。两组四次 L 前向，原浅/中层跨 RX SupCon 仅复用前两次；无新 Ag、无额外跨视图一致性、所有有标签 satellite 直接监督。旧 ordinary 不是已验证的新泛化增强。

一个已使用过的源 RX1 折和一个模型 seed；不是正式目标测试或跨 seed 证据。低 urban 超过60%、clean 净收益、suburban 接近 clean 及联合 Ag/As 目标均须独立后续证据。预算用尽不等于收敛。固定配方 FAIL 不扫描强度、权重、场景或更换 RX/seed 救援；PASS 只支持另登记源确认，不自动进入目标测试。

分层和固定 C0 clean 正确集合的保持率见 CSV。quality_snr_db 是白输入期望功率代理，signal_to_added_noise_db 只表示新增噪声，均不冒充原录音总 SNR。EQ 关闭、G=1，噪声未被 EQ 着色；固定 FIR 延迟与反射边界保持。状态分层是机制关联，不等于单因素因果归因。

门控图记录实际默认融合 gain 参数随训练的变化；本矩阵没有额外逐样本分支/门控干预，不能将参数变化写成门控因果结论。

实际状态：**VERIFIED / PASS_SOURCE_RX1_ONLY**。训练代码 `8235c71cc4a1619eade9def1371e1bdfc31b9eae`，两worker及dispatcher自然结束，无failure／重启／热改／外部进程变动。唯一限定P0/P1及锁父目录修复、单archive本远对比／remotecompile、真实own scratchCP无query smoke和PID/CWD/argv/GPU/log增长证据保留。

模型：F三路SIG_TIME/CENTER_25K/CENTER_UNITS，25 MHz／IQ256／equalized1／native RMS／NoExplicitCFO／原joint160／默认cosine30联合训练。浅层time_down和中层t2.pool的L-only同TX不同物理RX对比，温度0.2、最大权重0.1、warm5/ramp10，不新增头或部署参数，不把U作为有真值负样本。默认头、三个SIG投影、Sinc均实际更新，物理单位clip及lr映射每步核对；没有冻结骨干重训头。

实际数据角色是源RX3/4/6/8的L5040／隐藏U45360／seenV21600；held RX1的L/U均不训练，heldV5400仅FINAL。它不是完整源L6300，也不是5580/720内部合同。两模型各从零20轮=13L＋7U、4440更新，0旧父权重／EMA／optimizer／统计继承；旧E50只复用已核实代码配方。模型seed只有2026092701，一个源RX折；不声称独立新盲测。

实际practical导入依赖release的`experiments/adv3b02_xuc/code/leo_practical`，不是根`code/`。保留原residual/post_sync、EQ关闭（G=1）、reference_snr_db=30、质量自适应、相位跟踪、注入RX IQ补偿、AGC和reflect边界；完整传播／遮挡／噪声／接收动作依赖未拆散。输入为原native规范化IQ，再经原simulator normalize_input及原AGC，无额外输出RMS。未知原TX硬件不反演。正式测试IQ／SNR／信道完全未修改，实际0 target访问。

训练As先验均匀low_suburban／mid_urban／low_urban；C1每步128个原IQ生成一次full residual，跨epoch／view刷新；同RX/day/eq硬件session稳定且不按TX定义。全部6TX×4fitRX×3训练场景及3遮挡状态有效覆盖，0拒绝／删除／学生高置信门槛。训练和源验证命名流分离，C0/C1源验证逐物理样本信道元数据完全相同，C0训练As调用0、C1为4440次／568320包。

共同旧ordinary实际保留，不冒称已验证的新Ag；As直接作用原IQ，避免在完整practical前再叠其PA/AWGN/多径。C0第二对重复同ordinary IQ、C1第二对用practical(raw)，两对各权重0.5、每对同IQ两次dropout，四次L前向匹配。比较检验该完整视图替换配方的收益，不把它解释成各物理扰动的独立因果贡献。SupCon只读取共同第一对的两次原前向，无sat特征对齐／新一致性／新课程／新U增强。

|成本|C0|C1|
|---|---:|---:|
|L_forward_packets|2273280|2273280|
|student_U_forward_packets|396368|396368|
|student_source_eval_forward_packets|475202|475202|
|EMA_eval_forward_packets|396368|396368|
|held_V_forward_packets|5400|5400|
|source_practical_forward_packets|32400|32400|
|elapsed_seconds|1123.5720767974854|2046.1922359466553|
|source_practical_seconds|64.00892653502524|63.4716848381795|

两组新增40模型轮／8880更新／7158036真实网络包，矩阵墙钟2121.057396s；父权重成本0，source eval含两组各2包own scratchCP smoke。额外本地合成检查1024网络包／188信道输入／2optimizer更新／10协议负例另列，源日志分析和绘图0网络／IQ／CP读入，CPU分析墙钟未统一计时。预算耗尽不等于收敛。

逐场景准确率／F1／confusion由两组75,600个保存的默认头预测独立复算，CE沿用执行器记录。完整40轮／8880步／nativeCSV／stdout／SupCon及F物理clip已解析，全5994条L/U实际顺序逐条相同。源保持集合固定为C0 clean正确集合，不随C1重选。融合图只记录训练中的原gain参数，不提供额外拟合诊断头或逐样本干预因果证据。

实际低urban的分层修复与新增错误：

|维度|分层|样本数|修复|新增错误|净增／百分点|
|---|---|---:|---:|---:|---:|
|agc_limited|False|5399|1237|550|+12.724579|
|agc_limited|True|1|0|0|+0.000000|
|receiver_lock_status|degraded|965|187|100|+9.015544|
|receiver_lock_status|locked|2081|735|222|+24.651610|
|receiver_lock_status|unlocked|2354|315|228|+3.695837|
|state_end|BLOCKED|3016|415|298|+3.879310|
|state_end|LOS|1674|592|186|+24.253286|
|state_end|PARTIAL|710|230|66|+23.098592|

源分层可说明收益集中在哪类受损观测，不能据此宣称某种损伤的单因素因果恢复。质量是白输入功率代理、added-noise指标不等于总SNR。FIR数值延迟、人工反射前缀和残余核保留，未删边界困难包或修改正式条件；本矩阵未单独重算移除各损伤后的接收动作。

首轮处理：固定配对As通过首源折规则，仅支持另登记源确认；不自动晋级target、扩seed或宣布Ag有效。 当前完整源三seed、正式clean＋六practical、跨seed复现及真正Ag/As四格交互均未完成。lowurban>60%、clean净收益和suburban接近自身clean的联合目标保持active。

命令、环境/CWD、输入/输出及停止规则见experiment.json和preregistration.md；运行根不可重复启动。离线读取入口（不读IQ/CP）：`F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_f_crossrx_practical20.harvest --output E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/final --state E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/terminal_readback.json --cached`。

实际图表与完整数值：

- [complete_training_curves.png](E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/final/complete_training_curves.png)
- [source_scene_results.png](E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/final/source_scene_results.png)
- [low_urban_strata.png](E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/final/low_urban_strata.png)
- [fusion_and_margins.png](E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/final/fusion_and_margins.png)
- [scene_results.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/final/scene_results.csv)
- [source_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/final/source_strata.csv)
- [paired_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/final/paired_strata.csv)
- [training_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/final/training_curves.csv)
- [fusion_gain_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/final/fusion_gain_curves.csv)
- [measured_costs.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/final/measured_costs.csv)

## 收益位置、负结果与下一项检验

low urban中，LOS为37.9331%→62.1864%（n=1674），PARTIAL为30.2817%→53.3803%（n=710），BLOCKED为17.8050%→21.6844%（n=3016）；locked为36.6170%→61.2686%，unlocked仅16.7375%→20.4333%。主要改善集中在仍有较可用观测的类别，严重遮挡／失锁仍是主要缺口。这是同流源分层关联，尚未证明特定损伤的单因素恢复。

|TX|C0 low urban|C1 low urban|差值／百分点|
|---|---:|---:|---:|
|0|11.888889%|29.222222%|+17.333333|
|1|4.111111%|48.777778%|+44.666667|
|2|39.111111%|18.555556%|-20.555556|
|3|35.111111%|38.777778%|+3.666667|
|4|50.222222%|56.888889%|+6.666667|
|5|13.666667%|38.222222%|+24.555556|

TX2为明确负结果：39.1111%→18.5556%，下降20.5556个百分点；聚合改善不代表全部TX改善。low urban总体修复1237／新增错误550；固定C0 clean正确的4506包上，sat保持率25.9432%→42.4767%。clean修复321／新增错误174，C1保持原clean正确4332包（96.1385%）。seen V均值97.8843%→97.6991%，最差fit RX为95.2407%→95.1296%，分别损失0.1852和0.1111个百分点。

训练末轮的实际平均目标：

|模型|primary CE|secondary CE|SupCon原值|对称KL|clip前梯度|
|---|---:|---:|---:|---:|---:|
|C0|0.229633|0.229473|1.684119|0.034255|1.192400|
|C1|0.269383|0.680555|1.645570|0.039390|1.129340|

完整曲线已渲染并人工查看。C1的satellite第二对CE后期仍明显高于clean，不能把E20末轮或源健康通过写成收敛。clean准确率提高时，其平均归一化默认头间隔略低于C0；因此没有把间隔大小直接当作准确率改善。原fusion gain均值的训练变化也不构成门控因果证据。

本轮支持固定配对As在单源RX1折同时提高clean和practical，不证明独立Ag有效。下一项唯一值得检验的假设：该同配方收益能否在其余原源RX留出折保持。最小确认矩阵为固定model seed2026092701、held RX3/4/6/8各C0/C1 scratch E20，RX1复用已冻结结果；先单独登记、适配动态源角色并验证，尚未发布。每折沿用本轮urban≥3pp／low≥5pp／F1不降／clean和seen V损失≤0.5pp及完整健康条件；不更换失败RX或seed，任一源折失败便不把它写成稳定跨RX方案。全部健康row照预算收尾，0target，不新增Ag／一致性／课程。确认通过后才决定后续独立Ag与四格组合，不能把跨源折冒充跨模型seed。
