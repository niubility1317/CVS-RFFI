# F＋跨 RX：白输入质量代理的状态切换与边界反事实

run_id：20261017-diagnostic-FcrossRX-qualityproxy-fitL-manysig-m1-r01。执行 VERIFIED；全量独立复算完成，登记状态 ANALYZED。科学结论为 FAIL_FIXED_QUALITY_PROXY_PREMISE_CLOSED，固定假设关闭；活动总目标继续。

增强前基线保持 F＋跨 RX 同 TX 对比学习（SIG_TIME_X3、统一 Sinc 修正、NoExplicitCFO、joint160、默认 cosine 头），训练实验仍为从头 E20。本次只诊断物理接收过程，不实例化模型，不读 checkpoint，不训练或新增分类头。

唯一假设：残差实现按白输入功率近似计算接收质量，对实际短时结构化波形的估计差异，足以在 mid/low urban 造成稳定接收状态切换，并且差异主要不来自人工 FIR 前缀。

## 数据和实际运行配置

复用既有 VALIDATED_ONCE source carrier。原 L6300/U56700/V27000 契约不变；只使用当前 C0 训练 RX3/4/6/8 的 L5040（TX0至5、day1至3、72 个 TX×RX×day 单元，各70包）。源 RX1 的 L1260、U、V 和目标均不用于物理统计。已有 source_cache 整包反序列化包含未用源角色，读取器不检查这些角色的 IQ 或 U 真值。C0 仅提供实际 L_s 物理 ID 元数据，不继承任何模型权重。

采样率25 MHz，中心频率2.462 GHz，IQ256。cache 已按原协议 RMS 归一化，进入原信道 normalize_input；完整输出始终保留256点。真正导入既有 release 的 experiments/adv3b02_xuc/code/leo_practical，在 actual_engine.json 保存 channel/residual 的 __file__ 及六种场景全部实际配置。固定 residual、post_sync、EQ关闭（G=1）、reference_snr_db=30、reflect；保留原接收质量适配、相位跟踪、IQ补偿、AGC及几何/遮挡/传播依赖。RX 硬件 session 为 rx/day/eq1，绝不由 TX、scene、draw 定义。

质量分母为原新增噪声期望方差0.001，不能称为原录音与新增噪声合并后的总 SNR。真实 Hx 这里只是相同冻结 H 的 preRX 功率反事实，不声称等同完整时变 full-route 接收过程。FIR 数值延迟24点；内部功率统计使用索引 len(H)-1 至255，排除人工前缀影响，绝不裁剪送出波形或删除困难包。

## 同物理样本的五个条件

1. 原生白输入代理：mean|x|² × sum|H|² / 新增噪声期望方差。
2. 显式同质量重放，验证与原生 IQ 输出逐点完全相同。
3. 同一 Hx 全256点功率估计质量，重新计算接收状态和补偿：总效应。
4. 同一 Hx 的无人工前缀内部功率估计质量，重新计算接收状态和补偿：边界对照。
5. 条件3的质量，但固定原生 receiver_quality 返回的 status、residual_scale；其余原处理和随机抽样不变：受控零效应校验。

五个条件共享 H、RX session 和底层噪声随机流。状态改变可能改变残余补偿和最终信号，不把总效应冒充固定接收状态效应。

## 预算、判据与停止规则

两个独立物理命名随机流 × 六种原 practical × L5040，共60,480条记录、302,400次原生物理 process 调用；它们不是60,480个独立源样本。augmentation seed20261017891，receiver seed20261016102，split392005；无 model/evaluation/support seed。记录12个 scene/draw、864个 TX×RX×day 单元、12个固定硬件 session。新模型、checkpoint读取、神经网络前向、优化器更新、教师前向和头拟合均为0。

每一个 draw 的 mid_urban、low_urban 都须同时满足：质量绝对偏差中位数≥1 dB；全长估计接收 lock 切换比例≥5%；全长与内部估计差异中位数≤0.5 dB；全长切换中至少50%在内部估计上仍切换到相同状态；全部内部区间至少128点。

PASS 只支持另行登记冻结默认头可恢复性检验，不代表识别收益、有效 Ag/As 或自动开展 E20。FAIL 关闭本固定代理/状态切换假设；不改阈值、draw、seed、场景、窗口、RX 或训练预算补救。任何权限、输入、非有限值、输出冲突或启动失败均保留产物，不自动重启，不影响其他健康任务。科学结果不反馈目标调参。

## 执行和验证

实际代码/配置身份以本次 Git commit 为准，release_commit.txt 保存发布时 HEAD；不增加文件或配置哈希门槛。命令、数据句柄、独占路径、owner 和预期产物见 experiment.json。N607 普通账号、原 CVS-RFFI 环境、CPU2、CUDA隐藏；CWD为本次唯一 release 目录。启动入口：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_qualityproxy_fitL_20261017_r01/experiments/cvs_quality_proxy_source/runner.py。

已完成本地 compileall，以及22个输入/协议反例、5个科学判据反例、六场景6个合成包/30次原生物理调用、1,048,576个解析白输入观测和独立充分统计复算。真实 IQ、checkpoint、NN、更新均为0。一次独立静态 P0/P1 review 为 PASS；没有新增审批。

本阶段不消费模型/checkpoint，无实际 checkpoint smoke 所需机制；launcher 首步执行原生信道 no-query smoke（2合成包、10物理调用、0NN），PASS即继续。发布执行一次 N607 preflight、一个归档本地/远端 SHA 比较和远端 compile；启动后独立读回 PID/CWD/argv/CUDA隐藏及日志增长。

## 产物与声明边界

实际模块/config/数据角色、完整60,480条充分统计、全部场景/状态/接收切换分层、独立完整复算及4幅分层图（各PNG/PDF）均已保存。原始 IQ、权重、原始日志和生成图不提交 Git。源域诊断不能代替冻结默认头准确率或正式 clean＋六 practical 测试；跨模型 seed 重现与联合目标目前均未证明。

## 发布读回

运行代码 commit：4f7968ec32ea52ff56f63c42c1654fb4507573d6，已推送且 GitHub 独立 OID 比对 identical。N607 preflight、单归档传输和远端 compile 均 PASS。普通 CPU 进程156603的 PID/CWD/argv/CUDA隐藏及增长日志已独立核实；这是发布时的历史启动读回；最终完成证据见下文。离线复算及绘图的后续提交不热修改运行 release。

## 最终结果：原固定假设未通过

进程156603自然结束，独立 /proc 读回已无该进程；12个scene/draw及完成标记齐全，完整stdout14行，无技术异常。独立复算全部60,480条记录、864个TX×RX×day单元（每单元70条）、全部原实际L物理ID、12个硬件session、原配置及两种精确零效应控制，结果与在线统计一致。不是因漏包、增强未执行、未完成或接收session由TX定义而失败。

|物理流/场景|质量绝对偏差中位数（dB）|接收状态切换|全长/内部差中位数（dB）|全长切换在内部保留|平均绝对残余CFO变化（Hz）|
|---|---:|---:|---:|---:|---:|
|0/practical_high|0.1177|8/5040（0.1587%）|0.1331|75.00%|0.650|
|0/practical_mid|0.1249|26/5040（0.5159%）|0.1328|61.54%|1.899|
|0/practical_low_suburban|0.1293|35/5040（0.6944%）|0.1326|48.57%|2.450|
|0/practical_high_urban|0.1233|22/5040（0.4365%）|0.1343|86.36%|3.283|
|0/practical_mid_urban|0.1728|106/5040（2.1032%）|0.1352|77.36%|14.687|
|0/practical_low_urban|0.2337|122/5040（2.4206%）|0.1381|68.85%|15.491|
|1/practical_high|0.1188|3/5040（0.0595%）|0.1330|100.00%|0.410|
|1/practical_mid|0.1231|24/5040（0.4762%）|0.1333|58.33%|2.235|
|1/practical_low_suburban|0.1319|49/5040（0.9722%）|0.1327|53.06%|3.000|
|1/practical_high_urban|0.1247|14/5040（0.2778%）|0.1327|78.57%|2.357|
|1/practical_mid_urban|0.1751|133/5040（2.6389%）|0.1366|83.46%|17.736|
|1/practical_low_urban|0.2355|145/5040（2.8770%）|0.1400|75.86%|16.704|

两个流的mid/low urban全部未达到质量偏差≥1dB、状态切换≥5%的原条件；四组边界差≤0.5dB、切换保留≥50%的条件均通过。全部内部观察区间至少195点，超过预设128点；数值延迟仍24点，输出仍256点。失败主要是偏差与状态切换比例不够大，而非被人工前缀差异完全解释。

这是预登记整体假设的负结果，不是“每一包代理都准确”或“接收状态永不影响IQ”的证明。mid urban最小有符号偏差约−8.887dB，存在尾部样本，不能用中位数掩盖。状态/质量分层只作描述，不事后挑阳性子集晋级。

## 状态、RX、噪声与因果边界

以下合并同一L5040的两种物理流，各场景10,080条记录，不能当作10,080个独立物理源样本。

|场景|传播状态|记录数|质量绝对偏差中位数（dB）|接收状态切换|平均绝对CFO变化（Hz）|
|---|---|---:|---:|---:|---:|
|practical_mid_urban|LOS|5312|0.1158|0（0.0000%）|0.000|
|practical_mid_urban|PARTIAL|1006|0.1725|26（2.5845%）|2.716|
|practical_mid_urban|BLOCKED|3762|0.4109|213（5.6619%）|42.712|
|practical_low_urban|LOS|2928|0.1177|0（0.0000%）|0.022|
|practical_low_urban|PARTIAL|1426|0.1693|27（1.8934%）|3.976|
|practical_low_urban|BLOCKED|5726|0.4035|240（4.1914%）|27.336|

BLOCKED的切换更集中，但mid/low分别约5.66%/4.19%，质量绝对偏差中位数也只有0.411/0.404dB，不能将该分层当作原整体假设通过。LOS均无接收lock切换。原始locked/degraded/unlocked的完整转移九宫格见图；低urban双流合计degraded→unlocked101条、unlocked→degraded139条，方向不是单侧减弱。

各RX每场景2520条：mid urban切换RX3/4/6/8为59/61/56/63条，low为76/63/74/54条，全部约2%至3%；无某个RX达到整体5%的证据。完整TX/RX/day单元已保存，urban格最大单元绝对质量偏差中位数0.5132dB，不据此重选TX、day或RX。

五条件的原生/显式同proxy重放与固定status/scale受控输出均逐点完全一致，MSE=0；真实全长/内部质量的总效应允许重新计算相位/CFO/补偿和噪声处理。底层噪声抽样相同，不声称重新补偿后的噪声波形完全相同。EQ关闭G=1，没有均衡器FIR噪声着色；原实现仍将白热噪声经alpha·w旋转＋beta·conj(w)镜像混合，因此不能把最终噪声直接称为理想圆对称白噪声。未反演原录音未知TX硬件。

全矩阵仅low_urban流0和low_suburban流1各出现1条AGC限幅；全长质量条件和原生条件相同。保存的signal_to_added_noise_db由原preAGC信号/新增噪声功率独立重算，仍不是原录音与新增噪声合计的总SNR。

## 成本、命令与实际文件

父模型读取/训练成本本轮为0；仅复用源cache与C0角色元数据。新增模型0、网络前向0、更新0、教师0、头拟合0；真实源记录60,480×5=302,400次物理process，启动另外2个合成包/10次process。远端物理诊断墙钟448.956941s（约7.483min），非神经网络训练成本。两个物理draw不是两个模型seed；无训练曲线可报告。

运行代码4f7968ec32ea52ff56f63c42c1654fb4507573d6；独立统计及绘图代码f78551c40afa82f8d6b4acb921a0099d641098f4，两次均push后GitHub独立OID identical。本地compile、22个协议/输入反例、5个科学判据反例、6个合成包/30次物理调用及解析白输入检查PASS；最终完整独立复算PASS。一次P0/P1 review PASS，未新增重复审查。

本地验证命令（CWD=E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014）：

```text
E:/type10-7/local_envs/ssr-gpu/python.exe -m experiments.cvs_quality_proxy_source.checks E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/adv3b02_xuc/code
F:/App/miniconda3/python.exe -m experiments.cvs_quality_proxy_source.analyze --input E:/type10-7/local_artifacts/cvs_FcrossRX_qualityproxy_20261017/final
F:/App/miniconda3/python.exe E:/type10-7/local_artifacts/cvs_FcrossRX_qualityproxy_20261017/plot_local.py
```

实际实现：experiments/cvs_quality_proxy_source/{design,source,physics,run,runner,checks,metrics,analyze,plot,publish}.py；固定矩阵见registration.json。源数据读取、物理执行、总效应/受控效应和独立复算均有可执行入口。

结果与证据：[独立完整复算](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityproxy_20261017/final/independent_analysis.json)、[完成读回](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityproxy_20261017/completion_readback.json)、[登记交接](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityproxy_20261017/handoff.json)。实际图表已逐幅打开检查：[固定判据](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityproxy_20261017/final/figures/quality_criteria.png)、[urban偏差分布](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityproxy_20261017/final/figures/urban_bias_distribution.png)、[接收状态转移](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityproxy_20261017/final/figures/lock_confusion.png)、[源RX分层](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityproxy_20261017/final/figures/urban_receiver_strata.png)。同名PDF全部保留；完整充分统计quality_records.jsonl.gz、12组结果、864格分层、data_roles、实际engine/config和完整stdout均在E:/type10-7/local_artifacts/cvs_FcrossRX_qualityproxy_20261017/final，未提交原始数据、权重、日志或生成图。

## 未完成项和后续约束

固定质量代理/状态切换解释关闭，不将少量尾部切换升级为新训练方法，不改阈值、状态子集、流、seed、RX、窗口或测试信道补救。此结果没有识别出可稳定缓解的urban退化类型。默认头识别增量和clean代价本轮未测；新模型及正式目标测试均为0，不能把零更新说成已证明零clean代价。也没有跨模型seed准确率重现。

当前仍无独立确认的新Ag，无完整Ag×As四因子交互或正式target联合收益；low urban>60%和suburban接近clean均未完成。F＋跨RX同TX、默认cos联合训练与E20继续固定。新的独立假设需依据合法源域证据另行提出，本次没有自动启动补救训练；总goal保持ACTIVE。
