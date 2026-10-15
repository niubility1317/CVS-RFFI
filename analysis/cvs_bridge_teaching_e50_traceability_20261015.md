# HQ 桥接教学：实现与验证追踪（已发布，两行训练中）

本候选由用户17f04499及31cc8a22附件中的同信道bridge／SIG局部教学要求驱动。依据是已完成源HQ零更新诊断，不依据语义家族的正式目标排名选父权重。语义D3/D4源条件FAIL保持；本候选是两条从零E50独立小试，不加载或扩种子失败语义权重。

当前核心、发布/逐row评估/报告管线、本地检查、正式登记及一次独立P0/P1审查已完成；远端一次发布及真实CP烟测/独立运行读回已验证，真实第5轮源校准两行已PASS，完整训练及正式结果仍待运行。正式run_id为`20261015-phase1-sigtime-bridge-teaching-e50-manysig-s2026092701-m2-r01`。

## 方法和成本边界

固定single SIG_TIME／原参考响应和内部门控／原cosine6×160默认头，全部原参数按原CE和旧U机制联合训练；不新增部署模块或拟合头。源角色、模型seed2026092701、5580/720内部fold2026101503和原正式七环境IQ保持。

两条新行：BRIDGE_CE、BRIDGE_SIG。原native clean+full practical concat的CE保持0.5/0.5；两行均额外添加0.25×真实HQ标签CE。保留用户要求的direct clean→urban joint锚点，同类joint方向任务也作用于HQ，两项半和；own clean EMA在真实L上正确且属于urban时提供该锚点。HQ不是唯一身份参照，所有合法full/HQ标签样本始终参与CE。新行没有额外U特征或U urban监督。

BRIDGE_SIG仅额外加入urban raw SIG112→同H独立HQ ownEMA raw SIG112的局部教学。实际四窗每窗increment6/feature-area15/time-area6，常量dt排除；按自己的epoch5 HQ EMA组RMS（下限1e-3）冻结尺度，坐标smooth-L1/(1+smooth-L1)有界，再组内均值、三组等权。额外local权重为full-view物理q×clean-ownEMA正确性×urban指示；q不影响任何标签CE。

HQ沿用full的传播、RX/day/eq session、接收质量与lock、残余补偿/核、AGC增益和限幅决策、FIR延迟/边界；只将独立模拟器新增噪声幅度乘10^(-6/20)。原录音噪声和未知TX硬件保留，不改变原测试IQ/SNR。full训练波形与原nominal生成器逐值一致；保留三训练场景和全部状态支持。新增噪声SNR不等于总SNR。

每行从零E50=33L+17旧U、222步/轮、11100更新。每步clean/full/HQ student和clean/HQ EMA各128，共5个有标签前向视图；原U弱/强前向不变。主训练前向9031800/行，源dev面板70560、源校准5760、V健康52×27000、smoke和正式预测单列。两行前向/更新匹配，局部反向计算和实际墙钟不假定相等。parent训练成本0，不隐藏旧诊断/控制的历史训练成本。

自己的epoch5／实际1110个L步时一次校准：同一L_dev720、mid/low两场景，teacher clean/HQ2880＋student full/HQ2880=5760前向，0 optimizer/EMA/head-fit更新。两行都计算joint/local两系数以匹配校准量；各自原始global feature梯度比目标.1、系数1e-4至10，超界拒绝，不静默放大。两行common joint任务相同，SIG行增加local项，因此不声称总辅助梯度强度或反向FLOPs完全相同。所有校准模式、state_dict、pending .grad和随机流须保持；native U/EMA更新不被额外teacher前向触发。

训练/源cal随机流分离；训练源流2026101401沿epoch/step/view/physical ID刷新，cal2026101508；同seed首5轮是相同机制/样本预算，但CUDA非确定性不保证逐bit轨迹相同，因此分别使用相同校准规则，不谎称共用同一数值系数。新CE权重在访问新目标结果之前固定，不扫描比例。

科学推进只按final50 L_dev：候选clean相对匹配D0≥+.5pp、low相对普通A0≥+1pp、urban≥0、worst urban RX≥−1pp；局部教学效果另外看SIG−CE的low≥+1pp且clean≥−.5pp。源判定代码/登记已接通并通过规则负测，真实final50判定尚未发生。任何未来扩seed仍需源条件实际通过，不能按正式分数选择/重跑。

## 追踪表

|ID|需求及实现位置|状态|实际证据／缺项|
|---|---|---|---|
|B01|固定原SIG_TIME与默认头联合训练，model.py|verified|真实模型native CE+HQ CE反向，原头有梯度，无新模块|
|B02|同H/full状态/AGC及独立HQ，channel.py|verified|既有coupled物理实现复用；full传播/状态/核一致检查实际通过|
|B03|原nominal full波形与源RX session，channel.py|verified|12例逐值相等；改TX meta不变、跨epoch刷新且硬件稳定；拒绝U/训练固定scene|
|B04|源L真标签、物理q只调额外约束，objective.py|verified|隐藏-1标签/非法q拒绝；q=0样本仍保留普通/full/HQ标签CE|
|B05|native 1:1 CE与五视图捕获，features/observer.py|verified|两行真实128包native图；HQ附加前向不增加native L步，hook在U/EMA前已移除；CE加权等价|
|B06|实际raw112三组与有界损失，objective.py|verified|6/15/6、dt扰动不影响、组等权、有界及非法尺度负例|
|B07|梯度进入t2/local_projection且不只末端，features/objective.py|verified|真实梯度两处非0，末端输出投影/W/gain无local梯度；CE继续更新原头|
|B08|E50/更新/U/EMA/覆盖真实成本，observer.finish|pending|计数与守卫已实现，本地步图通过；无完整训练证据|
|B09|own epoch5正校准、冻结尺度系数，calibrate.py|verified|真实两行SOURCE_BRIDGE_CALIBRATED，各5760前向、0更新、state/grad保持；joint .569532/.609200、local .787842/.809337，无触界；CE不使用local系数|
|B10|唯一owner、运行保护与一次P0/P1|verified|一次独立限定审查PASS；archive比较VERIFIED、compile PASS、真实CP smoke PASS；PID3151548/3151567/3151644及GPU1/CWD/argv/训练流增长独立读回VERIFIED|
|B11|完整矩阵登记和最小预登记|verified|registration.json与实际两行config匹配；canonical实验登记已建立、launch-ready结构VALID，最小report完成；未冒充已发布|
|B12|逐row原头七视图和独立truth-last|pending|evaluate/queue已接线；完整性/非法logits/输出冲突负测PASS，真实七视图运行待完成|
|B13|源推进代码与目标隔离|pending|规则已实现于report.decide，规则正/负测试PASS；真实final50判定待完成|
|B14|全量曲线、逐seed/RX结果、全部成本|pending|完整日志/报告/曲线工具已接通；当前无新性能或图，不预写结果|
|B15|新机制归因：身份margin/修复破坏/S使用量|pending|待冻结后的合法源诊断；不沿用旧HQ诊断数字作为训练收益|
|B16|跨seed重现与组合扩展|deferred|先单seed两行；失败语义家族不扩种子|

计数：verified10/pending5/deferred1/rejected0/blocked0。严格实现了同H物理配对、真实IQ和默认头、direct身份锚点及现有SIG坐标。近似为新增噪声HQ、源RMS、物理q和一次eval全局梯度比，不把HQ当理想无噪声或无误teacher。

## 实际检查

最初5项检查PASS8.159s，命令为`E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_urban_bridge50.checks`，由既有invoke.py捕获到local_artifacts/cvs_coupled_bridge_20261015/bridge50_core_checks.log。新增校准测试首次按成功路径运行时，在未训练合成模型上触发固定系数37.344685>10，明确FAIL；未放宽固定界限。该合成用例改为校准拒绝的状态保护负测，定向命令`-m unittest experiments.cvs_urban_bridge50.checks.Checks.test_calibration_rejection_preserves_pending_grads_modes_states_and_RNG`实际PASS17.004s。这验证了失败路径，不证明真实epoch5正校准成功。

`F:/App/miniconda3/python.exe -X utf8 -m compileall -q experiments/cvs_urban_bridge50`通过。环境仍有已知torch/NumPy ABI导入警告，进程exit0；实现使用tolist/frombuffer路线，没有改环境。真实checkpoint无query smoke留在正式launcher首步；已完成一次独立P0/P1审查PASS；无报告审批、额外签名或重复未变测试。

新增pipeline四项测试PASS.112s；物理L拒绝及更新后的真实native两行图定向PASS6.642s。实际5580独立L_train包及90个clean TX×RX×day单元由observer计数，L_dev/未登记样本在HQ CE前拒绝。上述检查分批执行，不声称完整11项单次重跑。

最高风险：HQ teacher仍会误判、局部描述更接近未必提高真实TX分离；旧joint锚点也可能继续压缩差异。必须由匹配CE控制、真实头margin、错误修复/破坏与正式固定IQ结果判断。当前所有新性能、clean代价和跨seed结论均缺失，目标active。
