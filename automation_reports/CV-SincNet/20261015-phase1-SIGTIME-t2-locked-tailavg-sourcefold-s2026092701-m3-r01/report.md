# SIG_TIME 时间共享随机失活与同轨迹平均

状态 PLANNED。用户2026-10-15新增报告指定下一批优先研究同轨迹平均和t2时间共享失活，R-Drop第三。本轮只实施前两项，保持现有SIG_TIME+CENTER25K、默认cosine头联合训练和原hidden-U任务。84%仍是正式clean三seed目标；该真实源留出训练不替代正式结果。

当前代码确认真实位置为`id_backbone.encoder.core.id_backbone.t2.drop`，原为逐元素`Dropout(0.10)`，t2输出同时进入原主路径和签名。签名取点/6维1×1投影/tanh/四窗口/112→160路径全部保持。局部算子假设：时间共享掩码不向恒定通道制造随机时间跳变；投影/tanh/其他dropout存在，不能推出最终签名无偏或DG必然提高。新意候选在CVS时序签名与训练噪声的匹配关系，通道dropout本身已有[PyTorch2.1实现](https://docs.pytorch.org/docs/2.1/generated/torch.nn.Dropout1d.html)，时间掩码在[Gal与Ghahramani](https://arxiv.org/abs/1512.05287)已有先例，不称全局首创。

固定三条件ELEMENT/NONE/LOCKED，唯一差异是t2.drop分别为原Dropout(.1)、Identity、Dropout1d(.1)。一个seed2026092701，真正heldRX1，训练RX3/4/6/8，原L/U/V先按VALIDATED_ONCE物理角色核验，再在模型/teacher构造前过滤五个active loader。L5040/U45360隐藏/V21600，RX1的1260个原可见L包仅用于无状态eval，所有RX1训练/归一化/teacher/calibration/cache拟合输入排除。该源折checkpoint不得复用于正式全5RX初始化。

每row从零20轮、13L+7U、222步/轮、4440真实更新、U观察396368，总60轮/13320更新/1189104U观察；原AdamW lr2e-4/wd1e-4/clip5、cosine min1e-6、LS0、NoExplicitCFO内部包补偿和原Gaussian强U保留，无普通L或星地增强。不增加训练IQ前向或临时优化步骤。不修改或停止任何其他健康任务。三条件同种子/同初始化200状态张量在产物中直接比较，不用外部target分数选父。

每个轨迹同步保存第17/18/19/20轮whole-model状态并维护FP64等权平均，包含188参数张量中的原默认W、融合增益、归一化参数；12个常量buffer保持原值，无BN重估。只导出FINAL/ownEMA/TAIL_AVG三个单模型，平均不回流teacher或student，也不跨seed/RAW_HZ与CENTER25K混合。此处不改LR为SWA循环，称固定尾段平均，区别于[SWA原论文](https://arxiv.org/abs/1803.05407)的常数/循环LR方案。当前三seed BASE_SSL仅保存final/smoke/Sinc切片，缺完整tail；不为平均单独重跑基线。旧native177025参数SWA30源结果未过资格保留，与本SIG_TIME863730参数配方不混算。

每row四个tail时点在同1260个held sourceL上读出，FINAL复用epoch20读出，额外评价EMA/TAIL_AVG，共7560额外packet前向+2zero-smoke；按既有fold helper额外取1404个L包（144训练anchor未使用/不前向，1260held），明确不作新增主训练样本。保存每包物理ID/pred/label和margin/CE聚合、tail预测摇摆、平均相对FINAL纠正/破坏。四时点不是四seed。

主source资格在数据前固定：三row final训练L CE均≤.5；LOCKED_FINAL在heldRX1的错误数严格少于ELEMENT_FINAL和NONE_FINAL，meanCE不高于任一控制。未通过即关闭该固定失活候选，不扫p、层位或窗口；EMA/TAIL_AVG的正结果单列，不救援失败主机制或用于target挑选。通过只允许随后独立预登记正式三seed训练，不能继承源折权重。比较LOCKED_FINAL−ELEMENT_FINAL与ELEMENT_AVG−ELEMENT_FINAL，先分辨机制与平均导出的贡献。

代码/config由本次Git提交固定；owner `codex/root/SIGTIME-t2-locked-20261015`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_t2_locked_tailavg_sourcefold_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u experiments/cvs_sigtime_locked20/runner.py experiments/cvs_sigtime_locked20/configs/matrix.json`。唯一不可覆盖输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-t2-locked-tailavg-sourcefold-s2026092701-m3-r01`，ordinary N607/CVS-RFFI Python。一GPU上一个worker顺序完成三row，预登记按已有最少进程/最大空闲选GPU，总所有owner进程≤3；一次preflight/单archive本远SHA比较/remotecompile。每row own-scratch真实checkpoint no-query smoke首步PASS后立即训练，首row及持续执行均读回PID/CWD/cmd/GPU/log。停止规则沿用experiment.json：协议/数据角色/泄漏/非有限值/预算/输出冲突等技术错误保留产物，不自动重试，不按性能停止。source-only矩阵不构造target；所有完整source结果和负项均保留、同row报告/登记/Git交付。

已完成工程检查：9项掩码/完整头增益平均/跨轨迹和buffer负例PASS；原188参数模型CPU/CUDA的实际t2类型、mask行为、推理逐位同值、原头/t2/签名梯度及200state/平均完整性PASS，零真实IQ/CP读取。它们不证明训练收益。本候选唯一独立下一run P0/P1审查PASS；当前真实代码提交后按预登记直接发布，不增加独立设计审批。

四tail callback/四次真实合成优化步的CUDA完整导出集成PASS；独立FP64 stack与所有188平均参数逐张量相等，12buffer保持，epoch20/native final权重一致检查和默认头FINAL/EMA/AVG接续读出通过。fixture使用合成IQ与自建合成checkpoint，非真实source训练，预算counter/健康值为显式fixture，不作科学结果。
