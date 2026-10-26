# PAMF 多信道实现半监督：下一轮固定计划

状态：计划已固定；三份 U 缓存已完成并独立核实，四条 F 匹配对照已实现、登记，聚焦检查／P0/P1／真实 E30 新缓存 smoke 通过，训练已独立核实运行。原联合目标保持未达成。本文件是后续实施依据，不是新的成功标准或审批门槛。

## 原目标与当前证据

同一个方法的三个学生 seed `2026092701/02/03`，F0 所有目标 RX／日期共 168000 包／seed：raw clean≥85.98174603174603%，raw 六种 practical 均值>72.0961970899471%，同一固定原 F0 G-LU 的六景均值>78.07463624338624%。同时报告 raw PAMF 和同 G 的 DURA+PAMF。不能拼接不同方法的最好项，不能把源结果或 F0 局部结果写成所有设置全面优越。

原恢复分支软 CE 已完成三种子目标负结果。固定参照、任务缩放／剂量、特征任务 CE、IQ 插值、关系约束和非对称任务均有完整源负结果。最近两批：

- RestorationBridge 的 SeparateRG／BridgeRG／BridgeHG／BridgeGG：held G3 差分别 +0.117284／+0.166667／+0.179012／+0.203704 pp。
- AsymmetricHG 的 HJoint／HAnchor／GJoint／BothJoint：+0.160494／+0.135802／+0.179012／+0.166667 pp。GJoint 的 held H3 +0.216049 pp，高于该批其他三条；H 分类任务在参与 RX 拟合更强，但未扩大 held 收益。
- 两批均未达原 held G3≥+0.3 pp、overall V raw clean≥−0.1 pp 条件，不扩其种子或目标。

`20261026-diagnostic-PAMF-DURARealizations-source-m1-r01` 在完整可见 L6300、一个冻结原 E30 默认头上核实：相同 H 的 18900 次重复 G 逐点一致；改信道随机 namespace 后，G 相对 L2 中位数约 9.4%／13.7%／19.5%，四实现间 F 预测变化包比例 10.8413%／24.9365%／38.7302%。新视图准确率有升有降，不能选最好随机实现或声称训练收益；波形差异尚未去除全局复增益／相位。这只说明真实信道重采样能提供非重复的恢复视图。

完整证据见 [非对称源结果](../../automation_reports/CV-SincNet/20261026-phase1-PAMF-AsymmetricHG-sourceHold-m4-r01/report.md)、[多实现源 L 诊断](../../automation_reports/CV-SincNet/20261026-diagnostic-PAMF-DURARealizations-source-m1-r01/report.md)。

## 先构建三份额外源 U H/G 缓存

拟登记 `20261026-diagnostic-PAMF-URealizations-cache-m3-r01`，m3 表示三个缓存任务，不是三个学生种子。只读原 `VALIDATED_ONCE` U56700 IQ／物理 ID 和同一个冻结原 F0 G-LU，不读取 U TX 真值、V、目标、F 权重或 F 输出，不训练 G/F。

数据句柄从已核实原 seed01 `segment30/complete.json` 的 config 读取，不能依赖精简 context 中 `case.source_config` 有 `source_cache`。允许读原 F 配置元数据来定位 source cache／contract，缓存生成不载入 F 权重或伪标签输出。G 使用原 `load_G` 合约检查及 `InferenceEngine(reuse=True, graph=True, native=False)`，FP32／TF32off。U batch 的标签必须全部 −1、`tx_label_visible=False`，metadata 不含真实 y。物理键沿 `eq_i/rx_i/day_i/sig_i/base_index`，顺序与原 U role IDs 一致。

三个新 namespace 在任何新训练／目标访问前固定：

- `PAMF/realization_diversity/U/fresh1`
- `PAMF/realization_diversity/U/fresh2`
- `PAMF/realization_diversity/U/fresh3`

仍是原 lowSuburban／midUrban／lowUrban 三景，原 source_channel 的 residual／post_sync／noEQ／reflect／SNR30、channelSeed20261016421、hardwareSeed20261016423 及 receiver/day hardware session 保持。仅 namespace 改变信道随机实现，不附加白噪声、IQ 插值或虚拟接收机参数，不对 G 输出另归一化。

每个 fresh 保存成对 `channel_U.npy` 与 `restored_U.npy`，均为 FP32 `[3,56700,2,256]`；另存物理索引／ID 和 private 合约／G／namespace／冻结状态完成记录。三份新缓存总 2090188800 字节，逻辑 G 包调用 510300 次，F 前向 0。旧 fixed 实现直接复用：H 来自 FeatureTask 的 `source_preparation/channel_U.npy`，G 来自 RestoredPL 的 `source_U_cache/restored_U.npy`，不重新生成或覆盖旧文件。

先做一个真实 G、隐藏 U2 的无 query smoke，通过后立即串联三个缓存任务。一次预登记／代码 commit／聚焦负例／唯一 P0/P1 检查／archive 传输比较／N607 preflight；每卡全局 CUDA+pending<2、free≥10000 MiB，最多三个本 run GPU worker，保护其他 owner。只一次共享 smoke，不增加 receipt、逐文件 hash 或数据重验。首技术失败停新派发，健康已启动任务继续，保留输出，不自动重提。

当前诊断的 source namespace 含 L，新的 U namespace 含 U；二者是同一随机物理模型的独立流，不把 L 预测缓存当成 U 缓存。所有三份 fresh 都保留、等量使用，不按识别分数挑种子。生成后核实 G 参数／全部 buffers 未变、U 完整覆盖、成对 H/G 及物理索引，不引入 F 或真实 U 标签评分。

## 然后做匹配的数据支持对照

拟登记 `20261026-phase1-PAMF-URealizations-sourceHold-m4-r01`，只有一个学生 seed01，四条 own 原 ordinary-PL E30 独立+20到 E50／11100、累计13L37LU。旧研究 F50 只作冻结 source 对照，不作父权重。四条只改变训练视图支持，沿用各自旧损失／权重／两次额外学生前向：

| 新候选 | 已完成 fixed 对照 | 两次额外输入／目标 | 固定系数 |
| --- | --- | --- | ---: |
| GTaskMulti | FeatureTask/GFeatureHard | 同一 G 两次、硬伪标签特征 CE 平均 | 0.05941532657537612 |
| GGCLMulti | RestorationBridge/BridgeGG | G_k、G_(k+1 mod3)，跨视图跨 RX CL | 0.04677712061808677 |
| HGCLMulti | RestorationBridge/BridgeHG | H_k、对应 G_k，跨视图跨 RX CL | 0.04677712061808677 |
| GJointMulti | AsymmetricHG/GJoint | H_k、对应 G_k；G CE＋双向 CL | 0.04221045401136996 |

这些是匹配数据支持的固定系数，不重新用 V 或目标挑剂量，也不声称新输入上的初始梯度比仍精确 10%。复用已有 `feature_task_training.extra`、`restoration_bridge.extra/relation_loss`、`asymmetric_hg.extra(...,'GJoint')` 的原数学及梯度路由，避免顺带更换损失。

原三 E30 raw 全同类且各 confidence≥.95 的 q／接受集合只读复用。原完整 T1 L、raw self-PL .95／weight1、默认头主任务联合训练保持；附加分支排除 RX1、分类 weight 梯度0，featurefusion norm/gain 仍训练。无 EMA／新头／G 更新，U 真值始终隐藏。

新增 U epoch `e=0..19` 的实现 `r=e mod4`：r0 读旧 fixed，r1/2/3 读三个 fresh；每份恰好使用五个完整 U epoch，分景 k 仍按原全局 step 轮换。实现选择只用整数日程，不消费原生 CPU/CUDA RNG。每 row U 曝光1134000、held226800、附加接受838780、额外学生 FW2268000；每实现 U 曝光283500、附加接受209695。四方法的接受／held 曝光应匹配；GTask 无 CL，不把它的有效 pair 数与另外三条强行视为相等。

先实际 E30/defaulthead 与新缓存的无 query 联合 backward smoke，核实 exact 原目标数学、隐藏 U、H/G 成对及相应梯度；不是新增源选模门槛。原先 controls 都保留并复用冻结 V 预测。新模型每 row final 后自动完整 V27000 八视图，报告相对普通 PL 与对应 fixed control 的逐包净纠错、TX/RX/day、held/trained 和完整训练曲线。

全部四条完成后仍按原 held RX1 G3≥+0.3 pp 且 overall V raw clean≥−0.1 pp，选合格者中 held G 最大、精确同分方法名字母排序；不放宽。通过者再单独冻结三个学生种子的 raw／同 G 七视图正式矩阵；未通过不扩种子／目标。RX1 只在附加分支留出，不称全网络未见 RX。

## 完成边界

缓存生成、源模型或源门槛通过都不是原目标达成。最终必须有同一方法三个学生种子的实际目标 raw clean、raw 六景和同 G 六景增益；报告真实 seed 数、配对差、负结果和基准已曝光背景。不得将多实现缓存转换为推理多 G ensemble，正式 DURA 仍为同一个固定 G、同一既定单次执行路径。


## 三份 U 缓存已完成

`20261026-diagnostic-PAMF-URealizations-cache-m3-r01`，实际代码`8f0c94480f8f352189531fb499794763e1006438`，三份各U56700×三景paired H/G，新增IQ float字节2090188800。全部六个数组finite FP32、原物理顺序与隐藏U metadata独立核实，G参数/全部registered buffers检查通过；控制器/worker退出。无F/真值/V/目标/optimizer。缓存不是三个学生种子或性能目标达成，后续四条匹配对照按上方固定计划继续。


## 四条匹配 F 训练已启动

`20261026-phase1-PAMF-URealizations-sourceHold-m4-r01`，实际代码`abc13af0b1e145e886eab61f2054611774cd0002`，四条单学生seed01在GPU0/2/3/4的身份/GPU/step增长已独立核实。共享smoke逐个方法在真实own E30/defaulthead/fresh3做原L128/U256联合backward及aux head shield，全部PASS；训练依计划先r0，再三fresh，各5个完整Uepoch。原+.3/−.1源门槛、全部四条结束后判断和原三学生种子联合目标不变。
