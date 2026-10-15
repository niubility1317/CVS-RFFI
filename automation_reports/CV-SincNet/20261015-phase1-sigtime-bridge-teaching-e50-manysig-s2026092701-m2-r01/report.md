# 固定 SIG_TIME：HQ 监督与局部 SIG 教学 E50 预登记

本轮是独立两行从零E50小试，检验同信道较清晰视图是否能提供额外训练信息，以及局部SIG教学是否超过同成本HQ监督。依据为已有合法源L_dev720的冻结HQ诊断；own EMA low64.8611%→70.9722%是0更新源域受控结果，不是正式测试提升或训练可达保证。语义D3/D4源条件FAIL保持，不扩种子、不加载其权重；控制由预定义角色确定。

实际矩阵、配置、种子、命令、环境/CWD、路径、资源和stop rule唯一维护于[experiment.json](experiment.json)；对应运行配置为experiments/cvs_urban_bridge50/registration.json。当前已发布并开始两行训练，无新准确率结果。代码身份在发布时由实际Git HEAD写入release_commit/preflight/run experiment；不增加报告或成员SHA。

## 固定边界

原single SIG_TIME、内部门控、参考响应及默认cosine6×160/scale30均保持；所有原参数继续按原CE＋旧门控U流程联合训练。局部目标沿真实t2 postDropout→6维local projection→四窗raw112计算，不加部署分支、头、诊断分类器或新U目标。骨干和默认头不冻结后另训。

复用原source角色：L_s6300拆L_train5580/L_dev720，U_s56700 TX隐藏，V27000健康观测；源RX1/3/4/6/8、day1/2/3、6TX，split/data392005，内部fold2026101503每TX/RX/day留8/70。L_dev不参与CE、EMA或归一化更新。实际L_train base indices在每批检查，开发/未登记样本拒绝；累计实际5580独立包及90个clean TX×RX×day单元、90个TX×RX×state支持在finish核对。无不匹配或来源不明父权重初始化。

采样率25MHz，中心裁剪256点（不是减均值），原native RMS后模拟器单位RMS，无模拟器输出RMS。使用native_modules解析实际leo_practical，并保存channel.py/residual.py的__file__。原residual/post_sync、EQ关闭G=1、接收质量适配/相位跟踪/IQ补偿开启、reference_snr_db30、reflect边界和24点数值FIR延迟保留。训练仍均匀nominal低郊区／mid urban／low urban三场景，无课程或独立放大全部物理扰动。

正式固定clean＋六practical共七视图：practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。复用现有VALIDATED_ONCE原数组，每view168000，不再生成、提高SNR、调弱信道或删除困难样本。Phase2/support/query适配不在本轮；相关null不是权限变化。

## 两个条件与精确损失

两行model seed均2026092701；从零E50=33轮L＋17轮原U、222步/轮、11100更新，旧generic增强、旧U门控/EMA updater及cosine日程保持。AdamW lr2e-4→1e-6、wd1e-4、label smoothing=.01沿现SIG_TIME配方，不套用旧纯CE E20预算。

|新行|共同监督|唯一实验增量|
|---|---|---|
|BRIDGE_CE|0.5 CE(clean)+0.5 CE(full)+0.25 CE(HQ)，epoch6起相同joint锚点|无局部教学|
|BRIDGE_SIG|完全相同的标签CE和joint锚点|epoch6起增加raw112局部教学|

full与独立HQ使用相同源IQ、传播实现、RX/day/eq硬件session、接收状态/质量代理、补偿和所有核、full AGC增益/限幅决策、FIR延迟与边界。HQ只将独立模拟器新增噪声幅度乘10^(-6/20)，原录音噪声与未知TX硬件不反演。独立噪声有限样本能量不同，所以逐波形实际S/N增益不强制精确+6dB；记录full/HQ实际signal_to_added_noise_db，不能冒充总SNR。

原native clean/full 1:1 concat均值CE不改；HQ另做真实网络/default头前向计算标签CE。全部合法标签full/HQ参与CE，不用困难学生置信度门控。共同joint额外项是clean ownEMA→full与clean ownEMA→HQ两个方向损失的半和；仍保留direct clean身份锚点。其eligibility只用真实L上的clean EMA正确性及urban场景，未使用U真值。HQ不是唯一参照。

SIG局部项使用实际同H、同时间对应的urban学生raw112与HQ ownEMA raw112。每窗increment6／feature-area15／time-area6，排除常量dt；坐标差先按源RMS缩放，smooth-L1/(1+smooth-L1)有界，组内均值后3组等权。权重是full物理q=sigmoid((added-noise SNR−5)/5)×clean EMA正确性×urban指示，仅作用额外项。梯度须进入原t2和local_projection；local项对末端输出投影、W和gain无梯度，原CE/U仍联合更新它们。

各模型自己的实际epoch5/1110个L步时一次源校准，绝不抄final50 HQ统计：同一L_dev720，mid/low各一次，teacher clean/HQ2880＋student full/HQ2880=5760前向。HQ EMA组RMS下限1e-3；joint/local各按全局195041特征参数梯度对0.5 satellite CE的范数中位数比目标.1确定系数，界1e-4至10，超界拒绝而非静默放大。两行都计算两项校准以匹配成本，local系数只在SIG行使用。校准0 optimizer/EMA/head-fit更新，保持模式、state_dict、pending .grad和RNG。成功真实校准尚待实际第5轮，不由合成单测保证。

前5轮transfer关闭，两行仍保留全部五个有标签视图前向以匹配已固定预算，不声称预热额外teacher前向增加监督。native labeled wrapper只调用原clean/full一次，额外HQ identity-only前向不增加L step counter；own EMA没有被额外前向更新。两行同规则分别校准；FP32/TF32关闭，CUDA非确定性仍可能使同seed前5轮数值略异，不宣称逐bit同轨迹。

## 比较、预算与验收

匹配的局部教学归因只使用BRIDGE_SIG−BRIDGE_CE。固定同折旧D0 clean-only、A0普通拼接和A-J正方向控制原产物只读复用、不再训练测试；HQ CE改变总CE幅度且新校准不同，所以与旧控制的差不能单独归因给SIG。复用控制各自原E50/11100及成本保留。

每新行parent训练成本0，增量/累计E50。5个有标签student/EMA视图各1420800，原U弱/强合计1927800，主训练前向9031800/行。另列own source校准5760、14源面板70560、native V52×27000=1404000、初始/最终smoke及正式1176000预测/行；dispatcher真实旧A0 smoke2前向单列。两行匹配前向和更新，不假设额外local反向FLOPs/墙钟完全相等，训练秒数和clip/实际写回从完整日志统计。E50预算用尽不等于收敛。

只按固定final50 L_dev推进：候选clean相对D0≥+0.5pp，low相对A0≥+1pp，urban相对A0≥0、worst urban RX≥−1pp；局部教学证据另看SIG−CE low≥+1pp且clean≥−0.5pp。源码decide已做正/负测试。先固定source_family_decision，再读取新目标分数汇总；无自动扩种子或目标回流。探索目标仍是正式low≥60%、urban/low实质改善和clean保持，不保证达成。

## 执行与最小保护

N607普通账号现有SSH路线；CPU2/loader0、单GPU worker、所有owner含pre-CUDA总进程≤3/GPU，借用空闲卡并按实时共享receiver_residual_capacity4.lock分配，启动前free≥12000MiB。实际命令是`python -u -m experiments.cvs_urban_bridge50.dispatch`（精确解释器、CWD、run/release/row路径见experiment.json）。唯一owner codex/root/bridge-teaching-e50-20261015。只读旧A0实际checkpoint/default头smoke是dispatcher首步，PASS继续，两行scratch。

每row训练/机制计数通过后冻结，自动exec预测七原视图，再exec独立CPU scorer。七视图所有prediction和logits/IDs固定、model状态不变、head-fit0后才连接truth并独立重计；每row完成即报告，不等整矩阵。只有两row均结束后保存源规则结论及整组summary/resources/curves/analysis。完整source描述/身份margin/修复破坏归因在冻结后继续，不把loss下降写成身份恢复。

新run/output不可覆盖，只有一个launch owner。源角色/CP/fold/物理冻结、非有限、固定校准系数界、更新预算、输出冲突或系统性技术错误退出本owned行，保留所有产物，不自动重试；已健康行自然完成，失败后不新增派发。观察超时不是终止证据，先读PID/CWD/argv/log/完成文件；禁止停止/迁移/重启任何健康或其他owner任务。发布只有一次archive传输SHA比较、远端compile和preflight；Git为代码身份，没有额外签名、成员hash或报告审批。

## 已执行验证与缺项

5项核心检查PASS8.159s：full nominal逐值一致、TX不定义session/跨epoch刷新、raw112有界等权/dt排除、U/非法q拒绝、真实t2/local梯度与native CE/HQ/头更新/钩子生命周期。新增校准状态测试首次在未训练合成模型触发系数37.344685>10，按成功路径预期失败；未放宽界限。将该fixture明确为拒绝路径负测后PASS17.004s，实际验证model/EMA/grad/mode/RNG保持，不声称真实正校准成功。

4项pipeline检查PASS.112s：源双增益不能互相补偿、原7视图预测完整/非法logits时truth关闭、输出不覆盖、未验证机制不加载query模型。另物理L索引拒绝＋修改后的真实native两行图定向PASS6.642s。`python -m compileall -q experiments/cvs_urban_bridge50`通过。检查日志在local_artifacts/cvs_coupled_bridge_20261015的bridge50_*文件；新登记辅助脚本最初缺少工作树import路径，修正后生成成功，未触及远端或实验输出。

一次独立P0/P1审查PASS：未发现直接使下一实跑错误、越权、破坏、无法启动或无法产生合法预测的问题。已核对训练/校准接线、默认头梯度、同折角色、预算、七视图truth-last、共享容量、输出保护及登记/入口。发布、单archive传输比较、远端compile、真实CP无query烟测和独立PID/CWD/argv/GPU/训练日志增长均已VERIFIED；真实epoch5正校准、E50、七视图结果和图仍待运行。不能把登记结构VALID当许可或科学有效证明。最高风险仍是HQ teacher误判、局部相似增强却无TX区分改善；保留负结果。追踪表analysis/cvs_bridge_teaching_e50_traceability_20261015.md继续区分已验证、运行待证和延期。

## 实际发布读回

代码commit `12bd0be93bbf295ee1c8e75156ed5f1c3855cff7` 已push，独立remote branch OID与local HEAD相同。一次不可覆盖release/run已发布，archive传输比较VERIFIED、远端compile PASS、普通账号preflight VERIFIED。dispatcher PID3151548；BRIDGE_CE PID3151567、BRIDGE_SIG PID3151644，实际均GPU1。两worker工作目录/argv与提交记录一致，resolved_config和实际channel_execution已落盘。当前GPU1两个owned进程，不停止或迁移其他健康任务。

首步实际A0 checkpoint/default头CPU两包smoke PASS、query_read=false，仅作来源及可执行核验，两行仍从零。两次独立读回之间训练step_metrics和channel_training流已持续写入；早期worker文本日志只在epoch结束打印，尚无完整epoch，不将暂时不增长的文本日志当训练停滞。证据为local_artifacts/cvs_bridge_teaching_e50_20261015/readback_initial.json、readback_training.json与delivery_r01/submission.json。source/calibration/正式测试成绩仍未产生。
