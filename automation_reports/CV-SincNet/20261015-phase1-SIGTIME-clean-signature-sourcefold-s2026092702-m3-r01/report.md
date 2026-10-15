# SIG_TIME 签名分支的训练噪声位置

状态 **RUNNING / 启动 VERIFIED**。下文保留预登记。发现轮source01完整结果是ELEMENT59.126984%、NONE70.476190%、LOCKED67.222222%，原LOCKED主假设未通过。NONE相对ELEMENT修正163、新增20，CE下降.56340205；5/6类、3/3day提升，但只有一个seed/一个heldRX，不能当稳定泛化。固定17至20尾平均也无稳定错误收益，ownEMA全源V资格失败；这些负结果保留，不通过改窗口/decay救援。

既有R-Drop首源检验 `20261015-phase1-source-dropout-consistency-manysig-s2026092701-r01` 已ANALYZED：t1/t2/SIG的KL对pairedCE聚合导数cosine为.200427/.189996/.488580，5/5批非零，0训练更新。它使用旧final50/LS.01父，仅为局部兼容性；本轮不重跑或组合，保留为第三候选，不把导数同向写成已验证的泛化正收益。当前主线只执行本报告的三条件。

本候选把问题缩为“签名支路是否应直接接收逐点失活掩码”。CLEAN_SIG让原主干t2.drop继续执行p=.1，只把SIG_TIME读出从t2输出改成t2.drop输入（原pool之后）；不detach梯度，主干和签名共享同一次t2计算，保留原主干失活和随机数顺序。上游失活仍存在，不能称整条签名完全确定。NONE只关闭该处失活，ELEMENT保持原行为。只改这一处训练信号路径，0新loss/参数/IQ前向/头拟合；原默认头联合训练、6维tanh投影/4窗口/112至160/时间坐标不变。eval时失活恒等，因此同权重部署joint160和默认logits应逐位一致。

新seed固定2026092702；同RX1严格留出，ELEMENT/NONE/CLEAN_SIG各scratch20/13L+7U/4440更新，总60epoch/13320更新。新NONE是独立确认，旧seed01仅发现依据，不复用控制或权重。CLEAN_SIG保留ELEMENT随机抽样次数；NONE取消一次失活会改变之后部分GPU随机实现，不能将其全部收益归因到签名几何。

预定资格：CLEAN_SIG_FINAL错误严格少于ELEMENT_FINAL和NONE_FINAL、CE不高于任一；另一个独立判断是NONE_FINAL错误严格少于ELEMENT_FINAL且CE不升，三row final训练LCE≤.5。两结论单独报告，不以控制收益救援失败主假设，不扫参数。若通过，下一步单独预登记全5RX、三seed正式scratch20及clean自动truth-last；本轮禁止target，四RX权重不得流入正式全5RX训练。

原VALIDATED_ONCE carrier源L6300/U56700hidden/V27000及physicalIDs复用，训练前按RX1过滤所有train/probe/U/V/calibration loader；有效L5040/U45360hidden/V21600、训练RX3/4/6/8，原15静态domain映射保留。原source RX1可见L1260只在冻结final20后读一次。旧fold helper还读取144个训练L锚点后丢弃，无前向/梯度，实际额外L读取1404/row；held1260前向+真实initialcheckpoint2zero smoke，0额外训练前向，U每row396368重复观察不冒充不同包覆盖。LS0/NoExplicitCFO及内部补偿、无普通L/星地增强、原Gaussian U/oldgate/ownEMA/AdamW/lr2e-4/wd1e-4/clip5/cosmin1e-6保持。无新tail/EMA导出比较或R-Drop组合。

签名变换作为可训练层已有[Deep Signature Transforms（NeurIPS2019）](https://papers.neurips.cc/paper_files/paper/2019/hash/d2cdf047a6674cef251d56544a3cf029-Abstract.html)。当前创新候选是CVS实际训练中按支路分配随机失活并验证时序证据保留；普通关闭失活、通道失活、签名池化均不作全球新颖性声明，尚未证明该候选有效。

owner codex/root/SIGTIME-clean-signature-20261015。代码/config以实际Git commit固定；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_clean_signature_sourcefold_20261015_r01`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u experiments/cvs_sigtime_cleanpath20/runner.py experiments/cvs_sigtime_cleanpath20/configs/matrix.json`；不可覆盖root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-clean-signature-sourcefold-s2026092702-m3-r01`。只读复用原R02/先前source-fold代码，不加载其checkpoint。普通N607/CVS-RFFI Python，一GPU顺序worker，所有owner每GPU≤3，最少进程后最大空闲且>3072MiB，不干预健康旧任务。一次preflight、单archive SHA比较/remotecompile、每row真实scratch checkpoint no-query smoke后直接训练；唯一独立P0/P1审查、启动后PID/CWD/argv/GPU/log核验；非有限/协议或技术错误仅停止所属pipeline并保留产物，无科学earlystop或自动重跑。期望产物/完整矩阵见experiment.json。84%正式目标仍active。

本地7项聚焦机制/负例检查及compile PASS；实际CPU/CUDA三模型200state同初值、863730参数、同权重eval joint160/默认logits逐位相等，CLEAN_SIG与ELEMENT主干t2掩码及CPU/CUDA RNG状态一致，签名只读失活前输入且signature/t2/defaultW梯度有限非零。仅合成IQ、无数据/外部checkpoint访问，不作性能证据。唯一独立下一run P0/P1审查PASS；提交后直接按预登记发布。

启动独立核实：commit `f695b43bb8ff2c8584871cf439315557ced718cf`，ordinary GPU0所属PID/CWD/argv/GPU一致，总所有owner进程1/3，日志增长、真实own scratch checkpoint no-query smoke PASS。首row ELEMENT实际seed2026092702/LS0/13L+7U/原p.1，四RX角色及所有loader排除RX1一致。剩余两条件按固定顺序继续，未读target、未给出性能结果。

<!-- SOURCE_RESULTS_START -->


完整结束1/3条件，独立解析4440条连续step及20条epoch记录。其他健康条件继续；尚不作完整资格结论。


| 新seed02条件 | heldRX1 accuracy | Macro-F1 | 错误数 | meanCE | 最差day |

|---|---|---|---|---|---|

| ELEMENT | 65.555556% | 62.901100% | 434 | 1.67542983 | 57.857143% |

ELEMENT：final训练LCE=0.02413216；真实4440更新，U观察396368、被选观察8051.0（重复观察，不是不同U包覆盖）；F71初末位移10.805039、默认W位移0.67861619，均为联合训练。完整源V/最差RX和分阶段数值见JSON，全部epoch见CSV。

原ELEMENT在发现seed01为59.126984%，新seed02为65.555556%，跨seed变化+6.428571个百分点。只以各seed内匹配差判断机制，不把单seed变化当机制收益。

CLEAN_SIG与两个控制尚未齐全，固定资格仍待定；不以已出现部分结果改变后续健康训练。

全部held数据来自原可见source L，默认头联合训练，无另训头、无target或U真值；本独立reader只读取已完成产物，不追加IQ/模型前向或训练。84%正式三seed目标仍active。
