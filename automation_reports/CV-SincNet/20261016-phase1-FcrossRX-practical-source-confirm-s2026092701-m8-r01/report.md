# F＋跨 RX 配对星地：其余四源折 E20 确认

状态 REGISTERED_NOT_RUN。当前唯一假设：首源 RX1 的 clean/practical 共同收益能在其余固定源 RX3/4/6/8 留出折保持。首折真实 defaulthead：clean83.4444→86.1667%、urban均值+17.7469pp、low25.6852→38.4074%；TX2 low下降20.5556pp、BLOCKED/unlocked及suburban差距仍明显，保留全部负结果。首折通过不代表完整DG、跨seed或最终60%目标。

八行新训练：held RX3/4/6/8各C0/C1，均scratchE20=13L＋7U、4440更新、FINAL20。RX1只复用已完成的两模型结果／trace，不读其权重或继承EMA/optimizer/statistics。模型seed2026092701与split392005固定；五折不是五个model seed。每折fit恰为其余四原sourceRX：L5040、hiddenU45360、seenV21600；heldRX的L/U完全不训练，heldV5400只最终只读。全L6300/U56700/V27000与5580/720合同分别说明，禁止混称。

保持已验证F三路SIG_TIME/CENTER_25K/CENTER_UNITS、25MHz/IQ256/equalized1/nativeRMS、NoExplicitCFO、原joint160/defaultcosine30联合训练。SupCon沿用time_down/t2.pool、同TX不同物理RX正对、异TX负对，温度.2/最大.1/warm5/ramp10，只读共同第一对原两次前向，0新部署参数／头／前向。旧U/ownEMA/门控/cos/strong_noise_std=.015不变；U的TX始终隐藏。

C0=.5RD(同一旧ordinary)+.5RD(同IQ)，C1=.5RD(共同旧ordinary)+.5RD(practical(raw))；四L前向匹配，每RD同生成IQ两次dropout，CE LS.01＋对称KL alpha1。无新增Ag、对齐／课程／U增强。As直接原IQ，避免ordinary的PA/AWGN/多径先叠加；原ordinary是共同研究条件，不预称每个算子身份安全。

完整原practical residual/post_sync、EQ关闭即G=1、reference30/reflect、质量自适应／相位跟踪／RX IQ补偿／AGC及物理依赖不变；训练uniform low_suburban/mid_urban/low_urban，无拒绝或学生置信门槛。RX/day/eq硬件session稳定且不按TX定义，训练epoch/step/view刷新，train/validation命名流分离且各C0/C1同流。源heldclean和六practical每row冻结后自动评估，每视图5400包，0更新／0target，保存defaulthead/30间隔及完整物理元数据；新增noise SNR不称总SNR。不修改正式测试IQ、信道、SNR或困难样本。

唯一确认规则逐折完全沿用首折：urban三场景均值≥+3pp、low≥+5pp、urban宏F1不降，heldclean与seenV损失均≤.5pp，完整有限更新、全部5994个实际L/U批次顺序相同及TX×fitRX×scene/state覆盖。五折全部PASS才称这项源确认通过；任一失败即不称稳定跨RX，不换失败源RX／seed／参数救援。所有健康row仍按预登记完成，不按低分停止；PASS不自动扩target／modelseed或宣称Ag有效，后续研究另登记。Suburban距自身clean≤2pp仅探索目标，本次如实报告差距。

最多4个worker、每GPU计入其他owner/preCUDA最多3进程且free≥12000MiB，CPU2/workers0，共享容量锁。唯一owner和不可变release/root，PID/CWD/argv/GPU/log增长独立核对。worker首步own scratchCP无query smoke，PASS立即训练。系统技术失败仅保留本run产物、无自动重试，不停其他健康任务。精确command/environment/CWD/paths/GPU/stop及八row配置在experiment.json。

新成本8×E20=160模型轮／35520更新／18186240个L网络包，另计原U/EMA/nativeV/heldclean/259200六practical及smoke；按原实际预算期望总28632144真实networkpackets，最终实测。RX1复用成本40模型轮／8880更新／7158036包单列，0父权重成本。不将总矩阵模型轮写成单模型E200，不声称E20收敛。预计新302400个held-source冻结预测，另复用RX1的75600预测；source分析不等于正式truth-last target。

聚焦验证覆盖动态源角色／37负例／原RX1波形和元数据等价／206初态／原默认头和F联合梯度／4L与原前两次SupCon；当前0真实IQ/CP。发布采用一个代码配置commit、一次限定P0/P1、一次N607preflight、一个archive本远比较、一次remotecompile和首次真实ownCP smoke。无额外签名／hash层／审批／数据重验。原RX1产物保持，后续实际状态和结果在此追加。

## 发布前实际验证

LOCAL_VERIFIED，未启动。4096合成网络包、8合成optimizer、800合成practical包、37负例、五源metadata-only角色划分、八行实际E20参数、206原F初态、共同第一IQ及原两次SupCon、默认头／三SIG／Sinc联合梯度和硬件session稳定均PASS；原RX1新旧波形与物理元数据逐包完全相同。0真实IQ/CP/target。仅实际6条发布路径的隔离目录导入及实际8row参数20/13/7成功，0模型前向／checkpoint读取。

唯一限定P0/P1审查PASS，无修复或重复审查，额外0IQ/CP/网络/改动。compile、登记VALID及2026-10-16 14:22:31 CST N607普通账户preflight通过；其他健康任务保持。首步真实own scratchCP无query smoke由worker执行，PASS立即继续；不增加smoke许可。

## 实际发布与首轮核实

VERIFIED/RUNNING，训练代码 b3fe1c7b2e04d3abada07e79335508be2af01882，归档一次本远对比及remotecompile PASS。RX3／RX4的C0/C1已在GPU2/3/4/5运行，四worker均通过own scratchCP noquery smoke；PID/CWD/argv/GPU/log增长、真实held／fit角色和5040/45360/21600/5400计数、E20=13+7、四L前向、每步F physicalclip及默认头／三SIG／Sinc更新核实通过。RX6／RX8四行由同dispatcher按四worker上限后续启动，尚无FINAL源成绩。

运行时asdict审计确认所有六场景fs25MHz／residual/post_sync／EQoff=G1／reference30／quality_adaptation=true及quality_reference20／phase_tracking=true／IQcomp=true／reflect，原模块实际导入adv3b02_xuc/code。C1一次practical生成每L step，C0为0，SupCon warm5/ramp10和0额外前向保持；原U／ownEMA、NoExplicitCFO及默认头联合训练保持。0target，未干预其他健康任务或热改release。
