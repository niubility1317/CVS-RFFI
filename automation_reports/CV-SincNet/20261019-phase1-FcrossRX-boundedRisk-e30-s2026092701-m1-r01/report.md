# P71：固定 As 总权重下的有界组风险

唯一假设：原 full practical 的 As CE 内部风险更多分配给 3–15 dB 中间质量组，是否比同池同顺序的 nominal M2 带来足够源 mid/low urban 改善，并控制 clean 代价。质量 bin 不是真实可恢复性标签，不预告 low urban 超过60%。

上一轮P70已ANALYZED：分层降低批次数方差，却未通过源推进条件。因此不再扫描分层seed、配额或buffer，也不扩P70候选E50。本次改变的是优化风险目标，不冒称只是降采样噪声。另一对话“审计并设计 CVS 联合增强”当前正在IID/nominal分层匹配，本侧不重复或干预；未发现相同固定AsCE内部pi实训。旧recoverable改变提案分布并增加模拟；旧Asweight同时提高As CE和KL总份额，与本次不同。

逐row配置/seed/来源以experiment.json为唯一矩阵。只新增M3-s2026092701一行：严格原P43 As20学生/自身EMA，独立重置AdamW＋原cosine10/CENTER_UNITS，追加E10/2220更新，总E30/6660。P70 M2累积E30作为精确匹配控制，原P43 NoAs30/As30作为参照；它们的冻结预测和独立评分全部复用。禁止加载P70 final作为新父。

固定F＋跨RX同TX SupCon、SIG_TIME_X3、joint160、默认6×160 cosine头/scale30，骨干和默认头正常联合更新，900722参数/206state。原L6300/U56700（TX隐藏）/V27000、五源RX1/3/4/6/8、源day1/2/3、split392005，无heldRX/Ldev。没有新增头、分支、诊断分类器、对齐或U规则。

原P70 nominal_pool在训练前从合法rawL及原P43尾L物理序列/命名流生成，142080次观测出现、15个RX/day/eq硬件session。新行只读加载并重置Reader，使用与M2完全相同的one-use indices/配额。IQ、真实标签、compact domain、物理ID/接收元数据保持同包。每包真标签都参与AsCE；不按urban学生置信度筛选。不重新模拟、不追加proposal、不丢样、不逆推未知TX硬件。

四个signal_to_added_noise_db组（不是总SNR）为<3、3至<8、8至<15、≥15dB。固定gamma=.25、v=[.15,.40,.25,.20]，pi=.75p+.25v。p由现成源nominal池N_g/N决定，不读V/target组占比。已知p=[0.15361064189189189,0.15182995495495497,0.08809825450450451,0.6064611486486486], pi=[0.15270798141891892,0.21387246621621622,0.1285736908783784,0.5048458614864865]。3–15dB内部风险由23.992821%提高到34.244616%，最噪组不额外提高，最大相对放大约1.46×。全部组pi>0，保留全部原场景/状态支持；若正pi落入空组，首次更新前技术停止，不能暗改v或重模拟。

AsCE总系数仍.25，仅内部采用sum pi_g meanCE_g；原sameIQ RDrop KL仍等权.25，不随pi重加权。ordinary/raw、ORD128 SupCon、原U教师/学生规则及6次L前向保持。L每步768样本前向；CE总系数仍1.25。额外迁移项、伪标签门控、部署参数均为0。复用的六前向hook在同logits下相对M2的变化严格为.25×sum(pi_g−p_g)meanCE_g；反事实用于公式核对，不当作跨模型性能结果。记录真正的p/v/pi、各组CE、M2同logits反事实和实际风险增量。

实际物理路径沿用已确认pinned20261015_r02的experiments/adv3b02_xuc/code/leo_practical。fs25MHz、IQ2×256、逐包RMS center=false后原practical。训练fc2.462GHz、正式固定fc2.45GHz不变。full residual/post_sync/EQ=false（G=1）、reference_snr_db30、reflect；质量适配/相位跟踪/IQ补偿/AGC沿原实现。p71不改生成器，也不改变正式receivedIQ/测试SNR或样本。

源推进只两项：全V27000 clean M3−NoAs30≥−.1个百分点；两次固定源draw的pooledmidlow M3−M2≥+.5个百分点。任一FAIL关闭固定配方，不扫描gamma/v/seed，不扩候选E50。源分组、容易组、margin÷30、固定P43As20 clean正确集合、梯度/覆盖为解释，不新增替代门槛。预算耗尽不等于收敛。

无论源PASS/FAIL，新行固定FINAL30冻结后自动clean＋六practical预测，全部7预测固定后独立CPU scorer接真值。正式成绩不反向决定参数、seed、晋级或选择性重跑。全row/seed/RX/day/TX、accuracy、macro-F1、最差RX、错误修复/新错、urban均值/全部六场景和成本均报告。仅一个模型seed，微涨不能称可重复双提升。

新增预算：一个E10尾段/2220更新，native3162962NN＋source12600＋formal1176002＋真实parent student/ownEMA smoke4=4351568新远端样本前向。父As20 native4797722NN已训练并复用，控制不重训/重测。共享现成池本轮0新物理模拟，历史142080次生成/287.656秒单列，不当成本轮新成本。本地真实CVS NN/IQ/CP均0。

运行：N607既有普通非特权账户/既有CVS-RFFI Python环境，cwd=PROJECT/releases/cvs_FcrossRX_bounded_risk_E30_20261019_r01，入口python -u experiments/cvs_bounded_risk71/runner.py；唯一owner codex/root/bounded-risk71；每GPU最多2总进程（含其他owner），可用显存≥12000MB。输出PROJECT/runs/20261019-phase1-FcrossRX-boundedRisk-e30-s2026092701-m1-r01与release不可覆盖。首步真实checkpoint无query smoke PASS后立即继续，不增审批。

技术停止：数据角色/父来源/标签domain/数值/readonlypool/π支持/预算或输出冲突错误，仅停止自己失败行并保留产物，不修改其他健康实验，不自动重试。源低性能不是技术故障。代码/config只由本地Git版本发布，一归档一次SHA传输比较/一次远端compile/一次PID-CWD-argv-GPU-log增长核对。

状态：已登记、实现和聚焦检查中；尚无P71训练或性能结果。
