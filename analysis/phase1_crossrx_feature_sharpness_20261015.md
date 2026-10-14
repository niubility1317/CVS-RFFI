# 跨 RX 脆弱方向的特征学习

候选假设：一个源 RX 上的监督梯度，可能揭示其他 RX 上也敏感的特征方向；训练应抑制这种跨样本、跨 RX 的损失增量。它不依赖 U 真标签、不增加伪标签接受量，也不要求末期 AdamW 一小步 CE 增量达到贡献门槛。当前尚无该候选的训练收益或创新性证明。

## 已有证据与保留路线

- SIG_TIME 是已验证有潜力的实际时序特征路线；固定 CENTER_25K 的历史 3 种子各有正差。保留各自证据，不相加增益、不据目标分数挑父权重。
- 固定 BASE/final50 的 RAW_HZ primary Sinc 更新受 FP32 分辨率及 AdamW eps 影响。CENTER_25K 源假设步骤可改变 12 至 13 个滤波核，但跨样本 CE credit 仍小于 1e−6。该检查不支持直接扩训练，也不能否定所有早中期机制。
- 旧 contribution selector 的 functional-call 参数缓存存在版本键碰撞，归因限制已补记；原模型、预测、评分和未启动行保留。

## 明确机制

以原默认分类头 W 和部署特征参数 θ_F 定义真实 L 的 CE。θ_F 只包括 primary identity Sinc、time_fuse、t1、t2 和 SIG_TIME sequence_pool（排除 gain_raw）。其他参数在辅助分支中取 detached 当前值，原主训练仍联合更新默认头及骨干。Sinc 固定原构造 origin 与 25,000 Hz 中心坐标，不扫描单位。

对当前 L 梯度批次的每个源 RX r，计算 g_r=∇θ_F CE_r，再取停止梯度的 δ_r=0.05g_r/max(||g_r||₂,1e−12)。内步骤不执行 optimizer，也不扰动 W/gain。使用与当前批次完整物理 ID/base_index 分离的原 L bank，评价另外 4 个源 RX。令 E_r(θ_F) 为这些样本的平均 CE，辅助项为：

`L_cross = (0.1/5) Σ_r relu(E_r(θ_F + stop(δ_r)) − E_r(θ_F))`。

clean 与 perturbed 两项都对 θ_F 保留梯度；不是把 clean 当 detached 常数后重复加普通 CE。δ 停止梯度，所以没有二阶元梯度。零增量停止施压，所有 5 个 RX 维持固定分母。每个 functional state 克隆 buffer，临时 eval、保留 RNG，逐次清空并最终恢复参数派生缓存，live 参数、buffer、gradient 和 moment 保持。

固定对照 ZERO、CLEAN_CE、GLOBAL_GAP、CROSS_GAP。GLOBAL_GAP 从完整 current L 的平均 CE 产生一条共同 δ，但采用相同 5 个 other-RX masks；CLEAN_CE 使用相同 bank/masks 的普通 CE；ZERO 的辅助值为 0。各控制均执行相同 clean 和 5 次 perturb forward，披露实际 source exposure/forward/backward 成本。新增参数、头拟合、真实 inner optimizer、V/U truth 和目标访问均为 0。

held bank 仍是模型此前训练过的源 L，只在当前梯度步骤排除，不称未训练 holdout 或未见 RX。它检验跨样本可迁移的局部脆弱性，不等于真正目标域泛化。

## 与文献的关系

[SAM](https://arxiv.org/abs/2010.01412) 已提出在参数邻域内优化最坏损失；不能把权重扰动本身称作新方法。[SAGM](https://arxiv.org/abs/2303.10353) 已研究 empirical/perturbed risk 与其 gap 的梯度关系。[DISAM](https://arxiv.org/abs/2405.18861) 使用域损失方差调整 perturbation；[DGSAM](https://arxiv.org/abs/2503.23430) 研究逐域 sharpness；2026 年 [Dual Adaptive SAM](https://proceedings.mlr.press/v306/he26m.html) 进一步处理域差异与多方向 flatness。

这里待检验的区别是：只干预实际部署特征，固定当前默认读出；每个 RX 生成脆弱方向，却在排除该 RX 和当前物理样本的其他 RX 上学习损失 gap。必须用 GLOBAL_GAP 和 CLEAN_CE 验证其是否提供额外信号。文献核查不证明独创，也不证明这种定向信号有性能收益。

## 下一项有界检查

先在固定 BASE 首种子/final50 上做 source-only 原型；只是合法既有权重的诊断，不续训该 E50 checkpoint。4 个固定 L90 cohorts、原 L180 bank，固定 ρ=.05、系数=.1、4 个控制。检查跨 RX 正 gap、auxiliary 梯度进入浅中层与实际 kernel、W/gain 不接辅助梯度、CROSS 与 GLOBAL 的梯度差及 live 状态恢复。通过后才预登记实际训练。

后续新训练遵循当前 E20 预算（scratch20 或合规 E20+20），另保留同预算纯 CE 对照；不默认复制旧 E50 日程。研究变体按明确安排使用隐藏真值的 U；默认头保持联合训练，最终固定 clean 预测后独立 truth-last。旧 E50 成本及结果保留，不冒充 E20 同成本。当前默认头 clean 三种子 84% 目标未达成。


2026-10-15：跨RX feature sharpness源原型 `20261015-phase1-SIGTIME-crossRX-feature-sharpness-sourceL-manysig-s2026092701-r01` 4/4通过固定信号可用性条件。currentRX造detachedρ=.05特征方向，在otherRX／排除currentID的seenL上优化relu(pertCE-cleanCE)，两项均向17个部署参数传梯度，固定preW/defaulthead。CROSS最大RXgap6.325e−4至4.005e−3，梯度范数.001616至.007193，与GLOBAL相对差.441至3.532；actualAdamW aux增量改变2至3个Sinc核。540L IQ／14762forwards／8tempsteps，0realtrain/headfit/U/V/target。没有证明DG收益或原创性。保留SIG_TIME／固定CENTER既有正路线，已经准备新的E20匹配矩阵接口；BASE_CE／BASE_SSL／CLEAN_CE／GLOBAL_GAP／CROSS_GAP计划先首seed再固定source规则扩展，实际阶段与预算须预登记，不默认E50续训。84%默认头clean三种子仍active／未完成。


2026-10-15 后续训练接口：已实现 Observer 的标准 L callback，只返回一次辅助 scalar，不执行真实 backward/optimizer/EMA、不装永久 forward hook。原 current logits 计算 RX 梯度，不把 labeled_inputs 误称同一增强 view；scheduled 时按全局 base_index 排除当前所有 bank 包，缺 RX／空 bank 整次 skip，维持5分母。inactive 不做额外 bank 前向。独立 default W 按原 classifier 对象查名，aux 不接 W/gain。focused mock检查通过，真实 trainer 接入尚未执行。

已只读核实原 E20 三种子父基线：run `20261011-phase1-ce-epoch10-20-30-40-manysig-m12-r01`，commit `83b1903fed01854f5abb75646a02e9a5e190a90e`，label20/pseudo0、4440真实更新、use_unlabeled=False/use_ema_teacher=False、scratch/final-only/no祖先，无星地训练；本地回读尚不能确定历史use_aug实际值，不由parser默认推断。source契约与SIG_TIME完全一致，实际RX1/3/4/6/8、day1/2/3、6TX、L/U/V6300/56700/27000，split_seed392005，角色比例.07/.63/.30。原feature_recipe的source_days0/1/2不作为实际角色依据。

下一轮拟采用同架构SIG_TIME+CENTER25K的scratch20匹配矩阵 BASE_CE、BASE_SSL、CLEAN_CE、GLOBAL_GAP、CROSS_GAP ×固定3seeds，15行／66600真实更新；首seed5行22200，剩余10行44400。BASE_CE保持20L/0U并实际关闭teacher/U执行；SSL四控制计划13L/7U（维持旧33/17的大致阶段比例），显式teacher/U和use_aug=False、无星地训练。warmup2epoch444steps、every16将有250scheduledcalls/model，17selected参数、ρ=.05/系数=.1保持。此为实现准备中的方案，尚无正式run登记/发布；在真实launch前固定commands、source条件、GPU/stop/evaluation及准确成本，不加载旧E50权重续训。固定E20历史基线作为外部参考保留，新家族 matched对照用于机制归因，两者架构差异分别报告。


数值证据限定：实际probe的shadow容差为1e−6×所有optimizer参数的最大绝对值，包含未参与梯度的大Hz参数，因此首seed为8.3271195。该容差过宽，不能凭PASS证明微小feature辅助增量精确落地；另seed已观察到最大shadow差约2.4e−5。一次真实AdamW计数、辅助17张量/默认W-gain梯度None、实际初末Sinc/特征变化与独立默认头测试需分别解读。保留原probe数值及判据，不热改或重跑健康任务，不新增发布阻塞。冻结后的feature-only/默认W不影响z核对作为源端结果证据，无头拟合/query。详见shadow_tolerance_limit.json。


E20跨RX特征脆弱性 `20261015-phase1-SIGTIME-center-crossRX-sharpness-e20-clean-manysig-m15-r01` 已ANALYZED/VERIFIED：实际15行/3seed，CROSS_GAP=81.0853%，相对BASE_SSL=-0.0480百分点；源决定EXPERIMENT_COMPLETE。完整默认头truth-last/配对错误/源曲线见[结果报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-center-crossRX-sharpness-e20-clean-manysig-m15-r01/report.md)。84%三seed数值核验=False；该布尔值不替代机制对照或整体目标完成判定。未改参或选择性重跑，历史正负路线保留。


本版跨RX loss-gap闭环为负结果：CROSS_GAP81.0853%，相对BASE_SSL−0.0480百分点，净−242/504000；三seed未到84%。15模型全部20轮/4440更新/默认头truth-last，270CM与300源epoch完整。source V180真实initial/final表明17selected张量均变化，但身份间隔相对SSL两降一升；feature-only/W删除接口PASS只证明解耦结构。全局shadow容差受unusedHz参数放大，原PASS不证明紧精度。保留SIG_TIME/CENTER历史正路线，关闭本版loss-gap参数扫描；下一个最多两动作：同身份跨RX分支证据重组的source原型，以及真实feature更新采集。新主训练尚未发布，84%目标active。
