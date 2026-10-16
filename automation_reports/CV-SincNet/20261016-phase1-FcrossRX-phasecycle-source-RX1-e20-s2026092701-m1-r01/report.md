# C4精确相位轮转：源域预登记

REGISTERED_NOT_RUN。实际矩阵／commit／command／environment／CWD／paths／GPU／owner／stop／artifacts见experiment.json。

# 固定 F＋跨 RX：精确相位轮转的独立源检验

当前唯一假设：整包公共相位是接收观察的可逆自由度，原普通增强对它的有限随机覆盖不足；在同一物理包的两个有标签视图之间，确定覆盖非恒等 C4 相位，可使固定原网络和默认头少依赖绝对 I/Q 方向。它只是待验证假设，输入可逆或冻结模型敏感均不保证训练提高识别。

代码依据：实际 DataAugmentation normal 路径先 anti-shortcut、general channel，再可选 PA；整包 phase_rot 按 packet 取[0,2π)，触发概率却是整 batch 的一个 Bernoulli。实际 base p_phase_rot=.45，再乘 scale .1→.35；E20传入min(epoch,13)，scale由warm3/ramp15/curve1.25得到，所以实际概率从.045升至约.1128后保持，而非.45。其他旧普通扰动不改；原概率、幅度和课程不是本候选扫描参数。旧源相位曲率跨TX/day缺乏重复性，故不拟合或重开频率相关相位bank；本候选只有整包常数相位。

新增算子为C4非恒等旋转：k1=(-Q,I)、k2=(-I,-Q)、k3=(Q,-I)。原每个physical ID用独立种子20261016811产生一个初始偏移，再按访问次数循环1/2/3；不读TX、质量、loss或模型预测。只有符号和I/Q交换，精确保留逐点功率、全复数信息、时序、幅度谱、观察区域、25MHz和256长度，逆变换精确恢复。不做conjugation、CFO、相位噪声、时移、IQ失衡、PA/DAC、加噪、多径或新归一化；不将数字变换描述为更换RX硬件或反演未知TX硬件。

第一对是共同旧ordinary，第二对是对同一个已生成ordinary IQ取C4旋转；0.5 RD(primary)+0.5 RD(C4(primary))，每RD内同一IQ两次dropout，LS=.01／RDrop alpha1不改。与既有C0 ordinary/同IQ四L前向、一次ordinary生成、全部primary L/U物理顺序、模型初始化与旧U规则匹配。SupCon仍只读第一对的time_down/t2.pool，温度.2、最大.1、warm5/ramp10；不新增跨视图一致性、部署模块、特征头或分类头。保留三路SIG_TIME／CENTER_25K／CENTER_UNITS、原joint160／default cosine30联合训练、NoExplicitCFO、ownEMA/旧门控/cos。

最小实验仅一个新C4模型，scratchE20=13L+7U、4440更新。held source RX1；fit3/4/6/8 L5040/U45360/seenV21600，held L/U不训练、held V5400只FINAL。原C0同合同最终报告和trace只读复用，不读其权重或继承optimizer/EMA/统计；0E50/E200父状态，U TX始终隐藏。可见L物理ID仍5040，重复相位不是新增有标签样本。

候选必须C4−C0 held clean≥+1pp、macroF1不降、seen V损失≤.5pp、完整有限4440更新、全部5994 L/U批次相同、5040包各三旋转和TX×RX×day×k覆盖，才通过源首折。否则关闭固定C4，不调角度、概率、比例、源RX/seed或追加epoch救援。PASS只支持另登记剩余源确认，不自动进入target/As混合；不以低分停健康行。已有paired-As五源clean保护失败和realpair失败保留，不用它们救援或把不同增益相加。

训练不使用As；新row最终自动source clean＋全部六种原practical，无target。正式IQ、信道、SNR、困难样本和原residual/post_sync/EQoff G1/reference30/reflect接收动作均保持。当前研究不是正式truth-last或跨seed验证，也未证明Ag有效、low>60%或suburban接近clean。

新增20模型轮／4440更新，L2273280包；原U/ownEMA/source eval/heldclean/六practical/smoke实测单列。新增568320次精确旋转，含原IQ核对／逆变换／trace的构造墙钟单列，无额外ordinary draw、模型前向或新标签。C0既有20轮成本单列，父权重成本0。E20耗尽不称收敛。

精确矩阵/commit/command/environment/CWD/paths/GPU/owner/stop/artifacts在experiment.json。一个不可变run，最多一个worker；所有owner/preCUDA每GPU≤3且free≥12000MiB。首步自身scratchCP noquery smoke PASS后立即继续。技术失败只保留自身产物、不自动重试、不中止健康外部任务；最小工作流的一次限定P0/P1、一个archive传输比较、remotecompile和PID/log-growth核对足够，不增加审批或hash链。

发布前实际验证：LOCAL_VERIFIED_NOT_RUN。1024合成网络包／2合成optimizer／15协议负例、206初态逐键匹配、原primary IQ逐值匹配、默认头／三SIG／Sinc联合梯度、F clip、ownEMA、四前向及原首对SupCon均PASS；C4代数／精确逆／逐点功率／每包轮转覆盖与全局RNG隔离PASS。两次测试脚本顺序错误已修正（native作用域创建optimizer、native日志清理），失败合成尝试另1152网络包／1更新，无真实IQ/CP；累计2176合成网络包／3更新。隔离发布导入、实际E20=13+7参数和当前C0真实元数据核对／污染拒绝PASS，0网络/IQ/CP。一次限定P0/P1只读审查PASS无直接问题，0额外网络/IQ/CP。N607普通账户preflight于2026-10-16 16:02:23 CST核实一次。尚未发布／无新实际训练分数。
