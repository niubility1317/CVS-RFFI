# 同信道 HQ 桥接：源域零更新预登记

本次只诊断已有固定 A0 final50 与同 checkpoint 内 own EMA，不训练或选择模型。父模型是已登记八位置研究的预定普通拼接基线，未按目标分数选择 S/J。完整源域数据与逐行配置见 [experiment.json](experiment.json)。

同一个源 IQ、传播实现和 RX/day session 构造 full、共享噪声 HQ、独立噪声 HQ、无新增噪声四个视图。HQ 将模拟器新增噪声降低 6 dB，冻结 full 的接收质量代理、锁定状态、残余补偿、传播／残余／均衡核、AGC 增益及限幅决策、FIR 延迟和边界。原录音未知噪声保留，新增噪声 SNR 不冒充总 SNR。实际引擎没有逐样本硬裁剪，不编造 clipping mask。

## 范围与预算

- 仅内部 L_dev 720 个物理样本，源 RX 1/3/4/6/8、day 1/2/3、6 TX；L_train 5580 与 L_dev 分离。复用现有源缓存可能物化 V 容器，但 V 前向为 0；不开 U，不调用目标加载器。
- clean 各前向 student／own EMA 一次，加六种源 practical × 四视图 × 两个模型，共 36,000 个前向样本。0 个新增 epoch、0 次 optimizer／EMA 更新、0 次另训头。父训练 E50／11,100 次更新及实际墙钟单列。
- 默认 cosine 头保持 checkpoint 的 6×160、scale30；所有诊断 eval，记录原头准确率、固定父 clean 正确集保持率、修复／新增错误、除以 30 的间隔、EMA 熵和 RX／状态／质量分层。
- joint160 比较同样本与最近不同 TX 的余弦差；不拟合读出器。实际 SIG112 按每窗 increment7／area21 分成 6／15／6 三组，排除常量 dt。源 own EMA HQ 的固定 RMS 尺度带 1e-3 下限，先组内平均。
- 源随机流 2026101504，与训练／验证／其他源诊断／正式测试分离。HQ 设置不是修改正式测试 IQ 或测试信道。

## 实现、执行与保护

代码为 `experiments/cvs_urban_bridge/`，配置为该目录 `registration.json`。代码身份使用实际发布 Git commit；本报告及源码不增加独立 SHA／签名要求。N607 普通账号、CPU 2 线程、CUDA_VISIBLE_DEVICES 为空；CWD、命令、唯一输出根及 launch owner 见 experiment.json。发布器执行一次 archive SHA 比较和远端 compile，首个真实 checkpoint 无 query smoke PASS 后立即继续；随后独立读回 PID／CWD／argv／日志增长。

遇到契约／checkpoint 来源、物理冻结、非有限数、前向预算、输出冲突或技术错误时仅退出本次 owned 诊断，保留产物。不自动重试，不改变其他健康任务。本次显式 source-only，无正式目标测试或 truth-last 评分，不能把已有父模型重测成新实验。

本地已执行 5 项针对性测试（通过，0.754 s）：full 与原引擎一致、固定状态及 AGC、EQ 噪声着色／context 边界、噪声剂量与非法值、SIG 真实坐标及梯度边界。编译通过。独立 P0/P1 审查发现并修复 Linux validation_ref 路径问题，定向复核通过；未发现其他 P0/P1。真实 checkpoint smoke 与完整源域运行现已通过，证据见下。

## 后续 D/E 设计边界

D/E 都保留 clean→urban 的直接 joint 身份锚点，另加 HQ student CE 与 clean→HQ 锚点，额外 student／teacher 前向一致。E 单独增加 urban raw SIG112→HQ own EMA raw SIG112 的分组教学。保持原默认头联合训练及旧门控 U 流程，首个比较不加新 U 特征目标。不得用 HQ-only 替换 direct anchor，也不把本次诊断变成额外可行性证书或审批门槛。

本次尚未执行 D/E 训练，尚无训练收益、clean 代价或跨 seed 重现结论。既有八位置族源域扩种子条件失败，继续保留其负结果，不扩种子或选目标最高的分支。

## 已完成源域结果

VERIFIED：实际冻结发布 commit `baae811ff715831a51c86b13a488f5a2cab3a78c`；N607 CPU 作业 PID 3029039 已正常结束。一次 release archive SHA 比较及远端 compile PASS；真实 A0／own EMA 默认头 smoke PASS 后自动完成。独立初次读回 PID／CWD／argv 匹配，日志 406→571 字节，末次读回有 complete/results、无活跃进程，GPU process 列表为空。证据位于本地 artifact 的 delivery_r01、independent_readback.json 和 results.json。

执行 36,000 个前向样本，墙钟 34.409653 s，新增 epoch／optimizer／EMA／拟合头更新均为 0。student 与 own EMA state_dict 完全未变。父 A0 的 E50 成本为 4593.410037 s、11,100 次更新；本次没有用父权重初始化任何新训练。

下表是同一批 **源 L_dev720**、同一传播／接收状态下的冻结默认头。不是正式测试成绩，也不是训练改善。新增噪声 SNR 只对 simulator-added noise 定义。

|源场景|full student %|HQ shared student %|HQ independent student %|无新增噪声 student %|full EMA %|HQ independent EMA %|
|---|---:|---:|---:|---:|---:|---:|
|practical_high|96.5278|96.8056|96.5278|96.8056|96.5278|96.5278|
|practical_mid|92.6389|94.1667|94.4444|94.5833|92.6389|94.4444|
|practical_low_suburban|91.1111|92.3611|93.4722|94.4444|90.9722|93.6111|
|practical_high_urban|92.3611|92.7778|93.1944|93.7500|92.2222|93.1944|
|practical_mid_urban|74.8611|78.8889|79.3056|80.9722|74.7222|79.1667|
|practical_low_urban|64.7222|73.3333|71.2500|73.7500|64.8611|70.9722|

low urban 的 own EMA 原头由 467/720（64.8611%）变为独立 HQ 的 511/720（70.9722%），增加 6.1111 个百分点；mid urban 由 538/720（74.7222%）变为 570/720（79.1667%），增加 4.4444 个百分点。low 的 BLOCKED 由 165/405（40.7407%）变为 205/405（50.6173%），增加 9.8765 个百分点；LOS 始终为 201/207。该受控效应支持噪声遮蔽仍有可恢复部分，不代表网络已学会恢复。

固定父 student clean 正确集为 703/720。low student 保持率由 65.8606% 变为独立 HQ 的 72.5462%；该集合内修复 93 个错误、产生 46 个新错误。HQ teacher 仍有约 29% 的 low 错误，不能把 HQ 当作无误标签，也不能以 HQ-only 取代 clean 身份锚点。

共享噪声 HQ 的 low EMA 为 73.0556%，高于独立 HQ 的 70.9722%；两者不是严格等方差的单样本 SNR 配对。共享新增噪声剂量逐实现精确 +6 dB；独立噪声的实际 S/N 增益六场景均值为 5.9823 至 6.0187 dB。数据不支持仅凭共享视图更相似来宣称教学有效。

low full student 对 clean EMA 的同样本余弦为 0.702965，最近异 TX 为 0.896917，差为 −0.193952；对独立 HQ EMA 分别为 0.975959、0.947675、+0.028285。对 HQ 的相似度更高，同时不同 TX 也更相似，且比较还共享传播和 RX；该差不是原分类头 margin，也不是身份已恢复的因果证明。

实际 SIG112 的 low 分组平均损失：clean EMA 0.622179、HQ shared EMA 0.071604、HQ independent EMA 0.257632。共享噪声明显减小误差，进一步说明需控制相关噪声捷径。固定源 RMS 尺度为 increment 0.545247、feature_area 0.448109、time_area 0.298131；这些是冻结诊断值，不是未来 E50 预热校准系数。

全部 4,320 个 sample×scene 的 full 波形重建逐值一致，接收状态与传播／残余／均衡算子检查通过；17,280 条四视图记录完整核对，同样本的 RX／state／lock／full-SNR／CFO／kernel／AGC 等元数据一致。原 FIR 数值延迟 24 samples 保留；AGC 限幅决策不重算，没有输出 RMS 或逐样本硬裁剪。U 不开，V 前向为0，原录音噪声保留，正式测试 IQ 未变。

## 分层图与复现

![冻结默认头与保持率](E:/type10-7/local_artifacts/cvs_coupled_bridge_20261015/figures/frozen_default_head_HQ.png)

![身份关系与真实 SIG112 三组](E:/type10-7/local_artifacts/cvs_coupled_bridge_20261015/figures/identity_and_SIG_groups.png)

![源域状态、锁定与 full 噪声质量区间](E:/type10-7/local_artifacts/cvs_coupled_bridge_20261015/figures/source_state_quality_HQ.png)

图由 `F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_urban_bridge.figures --root E:/type10-7/local_artifacts/cvs_coupled_bridge_20261015` 生成并逐图检查。源完整明细、derived_summary.json、results.json 和实际模块 __file__/resolved config 保存在该 artifact 根；不提交原始源明细或模型。

## 解释边界与未完成项

本次证明在固定传播／状态下减少新增噪声可改善低／中 urban 的冻结原头，主要缓解 BLOCKED／未锁定的噪声遮蔽；未证明训练方法已改善既有固定 practical 测试。没有新 clean 训练代价或跨 seed 重现结论。原正式 low urban≥60% 目标仍未达成。

D/E 继续按附件实现：从头 E50，保留 direct clean→urban joint 锚点，双方同加独立噪声 HQ CE 与 clean→HQ 锚点并匹配 extra forwards，E 仅增加正确时间对应下的 6／15／6 分组 SIG 教学。系数与尺度从该新研究的源预热得到，不抄用本次 final50 诊断值；默认头和骨干继续联合更新，旧 U 门控保持。第一版不加新 U 特征目标。当前未启动 D/E 训练；不因源诊断而擅改测试条件或重启已完成位置族。

追踪表：`analysis/cvs_coupled_bridge_traceability_20261015.md`，本源阶段 verified12／pending3／deferred1／rejected0／blocked0；pending 为 D/E 实现、预算匹配及冻结后的七视图证据。
