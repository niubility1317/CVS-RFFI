# SIG_TIME 时间共享随机失活与同轨迹平均

状态 **RUNNING / 启动 VERIFIED**。下文保留预登记。用户2026-10-15新增报告指定下一批优先研究同轨迹平均和t2时间共享失活，R-Drop第三。本轮只实施前两项，保持现有SIG_TIME+CENTER25K、默认cosine头联合训练和原hidden-U任务。84%仍是正式clean三seed目标；该真实源留出训练不替代正式结果。

当前代码确认真实位置为`id_backbone.encoder.core.id_backbone.t2.drop`，原为逐元素`Dropout(0.10)`，t2输出同时进入原主路径和签名。签名取点/6维1×1投影/tanh/四窗口/112→160路径全部保持。局部算子假设：时间共享掩码不向恒定通道制造随机时间跳变；投影/tanh/其他dropout存在，不能推出最终签名无偏或DG必然提高。新意候选在CVS时序签名与训练噪声的匹配关系，通道dropout本身已有[PyTorch2.1实现](https://docs.pytorch.org/docs/2.1/generated/torch.nn.Dropout1d.html)，时间掩码在[Gal与Ghahramani](https://arxiv.org/abs/1512.05287)已有先例，不称全局首创。

固定三条件ELEMENT/NONE/LOCKED，唯一差异是t2.drop分别为原Dropout(.1)、Identity、Dropout1d(.1)。一个seed2026092701，真正heldRX1，训练RX3/4/6/8，原L/U/V先按VALIDATED_ONCE物理角色核验，再在模型/teacher构造前过滤五个active loader。L5040/U45360隐藏/V21600，RX1的1260个原可见L包仅用于无状态eval，所有RX1训练/归一化/teacher/calibration/cache拟合输入排除。该源折checkpoint不得复用于正式全5RX初始化。

每row从零20轮、13L+7U、222步/轮、4440真实更新、U观察396368，总60轮/13320更新/1189104U观察；原AdamW lr2e-4/wd1e-4/clip5、cosine min1e-6、LS0、NoExplicitCFO内部包补偿和原Gaussian强U保留，无普通L或星地增强。不增加训练IQ前向或临时优化步骤。不修改或停止任何其他健康任务。三条件同种子/同初始化200状态张量在产物中直接比较，不用外部target分数选父。

每个轨迹同步保存第17/18/19/20轮whole-model状态并维护FP64等权平均，包含188参数张量中的原默认W、融合增益、归一化参数；12个常量buffer保持原值，无BN重估。只导出FINAL/ownEMA/TAIL_AVG三个单模型，平均不回流teacher或student，也不跨seed/RAW_HZ与CENTER25K混合。此处不改LR为SWA循环，称固定尾段平均，区别于[SWA原论文](https://arxiv.org/abs/1803.05407)的常数/循环LR方案。当前三seed BASE_SSL仅保存final/smoke/Sinc切片，缺完整tail；不为平均单独重跑基线。旧native177025参数SWA30源结果未过资格保留，与本SIG_TIME863730参数配方不混算。

每row四个tail时点在同1260个held sourceL上读出，FINAL复用epoch20读出，额外评价EMA/TAIL_AVG，共7560额外packet前向+2zero-smoke；按既有fold helper额外取1404个L包（144训练anchor未使用/不前向，1260held），明确不作新增主训练样本。保存每包物理ID/pred/label和margin/CE聚合、tail预测摇摆、平均相对FINAL纠正/破坏。四时点不是四seed。

主source资格在数据前固定：三row final训练L CE均≤.5；LOCKED_FINAL在heldRX1的错误数严格少于ELEMENT_FINAL和NONE_FINAL，meanCE不高于任一控制。未通过即关闭该固定失活候选，不扫p、层位或窗口；EMA/TAIL_AVG的正结果单列，不救援失败主机制或用于target挑选。通过只允许随后独立预登记正式三seed训练，不能继承源折权重。比较LOCKED_FINAL−ELEMENT_FINAL与ELEMENT_AVG−ELEMENT_FINAL，先分辨机制与平均导出的贡献。

代码/config由本次Git提交固定；owner `codex/root/SIGTIME-t2-locked-20261015`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_t2_locked_tailavg_sourcefold_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u experiments/cvs_sigtime_locked20/runner.py experiments/cvs_sigtime_locked20/configs/matrix.json`。唯一不可覆盖输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-t2-locked-tailavg-sourcefold-s2026092701-m3-r01`，ordinary N607/CVS-RFFI Python。一GPU上一个worker顺序完成三row，预登记按已有最少进程/最大空闲选GPU，总所有owner进程≤3；一次preflight/单archive本远SHA比较/remotecompile。每row own-scratch真实checkpoint no-query smoke首步PASS后立即训练，首row及持续执行均读回PID/CWD/cmd/GPU/log。停止规则沿用experiment.json：协议/数据角色/泄漏/非有限值/预算/输出冲突等技术错误保留产物，不自动重试，不按性能停止。source-only矩阵不构造target；所有完整source结果和负项均保留、同row报告/登记/Git交付。

已完成工程检查：9项掩码/完整头增益平均/跨轨迹和buffer负例PASS；原188参数模型CPU/CUDA的实际t2类型、mask行为、推理逐位同值、原头/t2/签名梯度及200state/平均完整性PASS，零真实IQ/CP读取。它们不证明训练收益。本候选唯一独立下一run P0/P1审查PASS；当前真实代码提交后按预登记直接发布，不增加独立设计审批。

四tail callback/四次真实合成优化步的CUDA完整导出集成PASS；独立FP64 stack与所有188平均参数逐张量相等，12buffer保持，epoch20/native final权重一致检查和默认头FINAL/EMA/AVG接续读出通过。fixture使用合成IQ与自建合成checkpoint，非真实source训练，预算counter/健康值为显式fixture，不作科学结果。

启动独立确认：执行 `a4a6dd6fc79a8073113ac9a5385e75d858d325fc`，ordinary GPU0所属PID/CWD/命令/GPU成员一致，所有owner总进程1/3，真实own-scratch checkpoint no-query smoke PASS、日志增长。首row ELEMENT的真实t2元件p=.1，LS0/13L+7U/ownEMA/无L或星地增强/四RX角色及loader排除条件一致。该进程顺序完成三row，尚未产出完整机制收益或84%结果。

<!-- SOURCE_RESULTS_START -->


已完成 2/3 条件的完整源训练与导出，解析 8880 条连续step记录及 40 条epoch记录。其余条件按原计划继续，完整矩阵资格待定，不将部分结果当完整比较。


| 条件 | 导出 | heldRX1准确率 | Macro-F1 | 错误数 | meanCE | 最差day |

|---|---|---|---|---|---|---|

| ELEMENT | FINAL | 59.126984% | 55.373624% | 515 | 1.80715595 | 54.761905% |

| ELEMENT | EMA | 59.841270% | 56.082046% | 506 | 1.69754573 | 55.476190% |

| ELEMENT | TAIL_AVG | 59.047619% | 55.338053% | 516 | 1.80428535 | 54.761905% |

| NONE | FINAL | 70.476190% | 69.168923% | 372 | 1.24375390 | 62.142857% |

| NONE | EMA | 71.428571% | 70.016417% | 360 | 1.16485090 | 63.095238% |

| NONE | TAIL_AVG | 70.555556% | 69.190096% | 371 | 1.23032372 | 62.619048% |


| 对照差（候选减参考） | ΔCE | Δ错误 | 修正错误 | 新增错误 |

|---|---|---|---|---|

| ELEMENT_EMA minus ELEMENT_FINAL | -0.10961022 | -9 | 13 | 4 |

| ELEMENT_TAIL_AVG minus ELEMENT_FINAL | -0.00287061 | +1 | 2 | 3 |

| NONE_EMA minus NONE_FINAL | -0.07890300 | -12 | 18 | 6 |

| NONE_TAIL_AVG minus NONE_FINAL | -0.01343018 | -1 | 4 | 3 |


以上为原始1260个固定sourceL包的默认头读出，heldRX1全程未参与训练。每包物理ID/label/pred保存在远端source_readouts；独立从全部预测重算错误/混淆/F1/day和paired修正破坏，全部吻合。四tail完整200state逐项FP64重算与平均导出逐位一致；epoch20与native final相同、EMA导出与本轨迹原EMA相同。没有新增IQ读取、推理或训练。

ELEMENT：final训练LCE=0.02306368；tail各epoch与FINAL预测差异数[17:33, 18:23, 19:8, 20:0]。F71相对初值参数位移11.026524，AVG相对FINAL位移0.013463194；W也正常联合更新。参数变化只证明实际学习接线，不代替泛化收益。

ELEMENT四个已观察tail状态：始终正确726包、始终错误501包、预测发生变化52包、最终错但之前对14包；EMA/AVG分别修正其中始终错误包6/0个。仅描述此固定四点，不概括全部训练状态。

NONE：final训练LCE=0.01441933；tail各epoch与FINAL预测差异数[17:24, 18:35, 19:8, 20:0]。F71相对初值参数位移10.749975，AVG相对FINAL位移0.012721572；W也正常联合更新。参数变化只证明实际学习接线，不代替泛化收益。

NONE四个已观察tail状态：始终正确868包、始终错误345包、预测发生变化54包、最终错但之前对27包；EMA/AVG分别修正其中始终错误包5/0个。仅描述此固定四点，不概括全部训练状态。

这是部分row结果；LOCKED与两个控制尚未齐全，未作候选晋级、窗口选择或正式target判断。

完整逐epoch源曲线见source_complete_curves.csv，逐导出指标和配对错误见同名CSV，聚合机器证据见source_results_analysis.json。仅1seed/1heldRX，多时点复用同1260包；未读target/U真值，未另训分类头，84%目标仍active。
