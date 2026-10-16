# 真正混合 IQ 增强：失败复盘与当前检验

当前没有证明 clean 与 practical 的共同突破。用户最新将增强前基线改为 **F＋跨 RX 同 TX 对比学习**，从零 E20、原 joint160/default cosine 联合训练、NoExplicitCFO 和原 U/EMA/旧门控保持。以下已完成 OJ24 等实验使用旧 F 对照，不能充当新基线的匹配证据。没有把旧 E50 权重或性能增量移植为新 E20 结果。

## 已完成探索留下了什么

|探索|可核实结果|对下一步的约束|
|---|---|---|
|旧基座普通混合增强|B01 对 B00：low urban 约 +20.17 个百分点，clean 约 −2.85 个百分点；B11 对 B01：low 约 +0.04、clean 约 +0.01 个百分点|A_s 的适应收益存在，但当时的 A_g 几乎未带来独立增量，不能称双收益；不是当前 F 的结果|
|E50 课程、间隔、SSL|相对对应基准的 low urban 增量约 1 个百分点，clean 有代价；未到 60%|停止把额外一致性、筛选和课程直接当作主解；不同 6300 与 5580/720 合同不作配对|
|分支、joint 和语义对齐|对应源规则失败；默认头正式增量小，部分 clean 下降|分支靠近、门控变化和辅助目标下降不能替代身份识别；不继续叠加这些损失|
|独立 HQ 冻结诊断及后续训练|源 L_dev720 冻结 EMA 的 low urban 为 64.8611%→70.9722%，0 更新；真正 BRIDGE_SIG 对匹配 BRIDGE_CE 的正式 low 仅 +0.3012 个百分点，clean +0.0244 个百分点|输入变好不等于模型学会恢复；冻结结果不是目标测试提升，不能用来宣称训练成功|
|跨 RX 教师|RX1 clean 纠正 0、新错 42；low 纠正 9、新错 33，源规则失败|不扩教师库、学生或 seed，不把不同 U 当作负样本|
|旧 RAW_HZ 普通增强的五源 RX 确认|平均准确率 +2.0296 个百分点，但仅 3/5 正，最差 RX 约 −2.59 个百分点，源规则失败|单折阳性不足以证明 A_g；五折也不是五个模型 seed|
|去 PA|冻结输入诊断五 RX 小幅正；真正源 RX1 E20 对原普通增强 −3.3519 个百分点|不能由“少破坏输入”直接推出“更好的增强训练”|
|当前 F 的 raw 锚定|A0 raw/raw 81.1481%、A1 ordinary/ordinary 85.8148%、A2 raw/ordinary 85.1852%；A2 对 A1 准确率 −0.6296、F1 −0.5317 个百分点|原始监督份额必须显式记录；该候选按原保持条件失败，不换主对照救援|
|当前 F 的 WHITE30|81.1481%→81.1667%，仅净增 1/5400，F1 −0.1736 个百分点，修复 80／新错 79|执行正确且梯度健康仍不足以证明泛化；不扫噪声强度、概率或更换 RX/seed|

上述行不能跨合同、基座、预算相加。具体逐行、逐 TX/day、费用和全日志见[历史证据汇总](cvs_phase1_urban_evidence_20261015.md)、[F raw 锚定报告](../automation_reports/CV-SincNet/20261016-phase1-F-raw-anchor-ordinary-RX1-e20-s2026092701-m3-r01/report.md)、[WHITE30 报告](../automation_reports/CV-SincNet/20261016-phase1-F-white-observation-RX1-e20-s2026092701-m1-r01/report.md)。这些负结果继续保留，不据正式目标分数更改源晋级。

## 当前 F 的新定位证据

固定 A0/defaulthead、源 RX1 合法 L1260，clean 为 81.1905%；原 full low urban 为 29.4444%。固定接收动作后移除新增噪声仅 +0.9524 个百分点，拉平 H 仅 +0.7937 个百分点；重新计算动作的总效应分别 +1.5079、+0.8730 个百分点。未支持“只增加去噪／抗多径约束即可恢复”的判断。EQ 实际关闭，G=1；保持 reference30、post_sync、原接收质量、相位跟踪、IQ 补偿与 reflect 边界，不改变正式测试。

随后独立固定相位×观察原点分解：A0 仅 24 点 causal-reflect 延迟为 48.6508%，仅 low 场景核公共相位为 47.6984%，二者组合 36.9048%。raw 正确的 1023 包中，延迟后保留 468；延迟修复 145、破坏 555，TX 之间错误重分配很强，三个 day 均受损。两模型 raw 预测与原归因 clean 逐包一致，排除了这次 raw 归一化差异造成结果的解释。

这支持固定识别函数的观察原点／公共相位脆弱性，尚未证明任何增强保持身份或改善泛化。各因素含交互，不能把准确率差加成独立物理误差份额。低 SIG 激活相似度也不能证明某个分支是唯一原因。详细见[完整 practical 源归因](../automation_reports/CV-SincNet/20261016-diagnostic-F-practical-source-RX1-s2026092701-m2-r01/report.md)及[相位／原点分解](../automation_reports/CV-SincNet/20261016-diagnostic-F-phase-origin-source-RX1-s2026092701-m2-r02/report.md)。

## 已结束的 OJ24 训练检验与放弃条件

OJ24 只检验观察原点。每个 L 包独立取 d∼Uniform{0,…,24}，G(x)[n]=x[abs(n−d)]，I/Q 共用索引。它精确复用原 FIR 的 delta[d] causal-reflect 数学，不做循环拼接、插值、公共相位、噪声、PA、DAC、CFO 或 A_s。前 d 点是人工反射过去，末 d 点被截去；不冒充长记录或真实未来补齐。保留至少 232/256 个原观察位置，记录实际能量变化，不追加 RMS/AGC。

新模型从零 E20=13L＋7U、4440 更新。每个物理包使用 0.5 RD(raw)＋0.5 RD(G)，每个 RD 复用同一 IQ 两次 dropout 前向，与已完成 F A0 raw/raw 的四前向预算匹配。只增强 L，原 U/EMA/门控和 strong_noise_std 不变；无 raw/G 跨视图损失，没有新增头或部署参数。随机流独立于 TX/RX/day，逐 step/view 刷新。

首轮固定源 RX1 留出：训练 RX3/4/6/8 的 L5040/U45360，seen V21600，held V5400 仅 FINAL。OJ24−A0 必须达到 held 准确率至少 +1 个百分点、宏平均 F1 不下降、seen V 损失不超过 0.5 个百分点，并满足完整有限 4440 更新、全 TX×RX×day 覆盖及全部实际 L/U 物理序列相同。失败立即关闭 OJ24，不调整范围／概率、不换 RX 或 seed 救援；通过只支持另登记其余四个源 RX 确认，不直接展开 target 或联合 A_s。

OJ24 已在实际代码 `f601fc65f6eaafbfd98080882108f4d50e07db69` 启动，运行根和命令见[该 run 报告](../automation_reports/CV-SincNet/20261016-phase1-F-origin-jitter-RX1-e20-s2026092701-m1-r01/report.md)。真实 own scratch checkpoint 无查询 smoke 通过，实际 E20/原头联合更新、25 种偏移、每步 physicalclip、四前向及原 U 参数已核实。以上为发布时的固定设计；实际 FINAL 结果见末节。

## 如何进入真正混合验证

只有独立 A_g 的源证据成立后，才固定同 F/E20、同数据／初始化／U／默认头／前向成本的 B00、B10、B01、B11。分别检验 A_g 的跨 RX clean 收益、A_s 的 practical 适应和联合净收益及交互。初轮不同时叠 A_s(A_g(x)) 和多种新损失；全部合法有标签 satellite 视图直接参与 CE。

正式测试仍使用原 clean＋六 practical，冻结逐 row prediction 后独立 truth-last，不因目标分数调整幅度、选模、晋级或重跑。目标仍是 clean 净正、low urban >60%、suburban 接近自身 clean 以及跨 seed 复现；当前均未由新 F 联合模型证明。

## 用户最新基线覆盖

后续控制和增强变体共同使用 F＋CROSS_RX_SUPCON，不再用纯 F 对照计算新增增强收益。现有真实配方位于提交 `4df7b94d0fd8ce29056bd8236364e45afa92e07f` 的 `experiments/cvs_f_supcon50/mechanism.py`：在 `time_down` 与 `t2.pool` 上，真实 L 的同 TX／不同物理 RX 为正对，不同 TX 为负对，同 TX 同 RX 及自身不进分母；温度 0.2、最大权重 0.1，前 5 轮为 0、随后 10 轮线性增加。复用两次原 RDrop 前向，不新增投影头、参数或网络前向，不给 U 增加真值或对比损失。默认头继续通过原 CE/RDrop 联合更新。

新 E20 保持该绝对 warm/ramp 日程，即 E6 起启用、E15 到 0.1；总计 13L＋7U、4440 更新，FINAL20。保留原有普通增强作为共同条件，发布前记录真实工厂／开关／概率，不能把“无新增 A_g”写成完全无增强。完整源 L6300/U56700/V27000 与留出源折严格区分，不用 E50 权重或把 E50 成绩当作 E20。

当前机器可读契约见 [F＋跨 RX 对比 E20 基线](../configs/phase1/F_crossrx_e20_baseline.json)。这是当前基线覆盖和接入设计；新 E20 的实际状态见末节，现有 E50 的数据／权重来源与本轮契约分别核对，不能直接续训或冒称已验证新预算。新机制与 A_g／A_s 的收益均须相对这个相同配方、数据、预算和前向量的对照计算。OJ24 的失败继续保留，不因更换基线重跑救援。

## OJ24 首轮收尾

完整 E20 已完成，源规则 FAIL。OJ24−A0 accuracy -1.9074074074074021、macroF1 -1.1833066289153482 个百分点，修复 247／新增错误 350。一个源折／一个 seed，0 target；不据此宣称正式双突破。候选关闭，不调整范围或更换 RX/seed 救援。


## 新基线下的首项实际接入

当前最值得检验的是新F＋CROSS_RX下的原practical配对训练能否保住clean并提高urban，暂不再追加新Ag/一致性/课程。最小实验为C0/C1，各自scratchE20=13L＋7U、4440更新、四L前向：第一对共同旧ordinary，第二对分别相同ordinary或practical(raw)。SupCon只读第一对两次已有前向，native旧U不变，所有有标签satellite直接参与原默认头CE/RDrop。旧ordinary仅是共同条件，不冒称有效新Ag，As前不叠它的PA/AWGN/多径。

训练均匀三场景low_suburban/mid_urban/low_urban，源heldRX1原V5400在FINAL固定clean＋六原practical流，无target访问。判据C1−C0：sourceurban均值≥+3pp、low≥+5pp、urbanF1不降、clean及seenV损失≤0.5pp，并完整预算/全物理顺序和覆盖；FAIL关闭固定paired-As配方，PASS只支持另登记source确认，不自动扩target或Ag。suburban2pp差距如实报告为探索目标。完整矩阵/command/paths/stop见[该run报告](../automation_reports/CV-SincNet/20261016-phase1-FcrossRX-practical-source-RX1-e20-s2026092701-m2-r01/report.md)。

实现、1024合成网络包/188信道输入/2合成optimizer/10负例、206初态匹配、隔离导入、实际E20args和唯一限定P0/P1（锁父目录P1已修复）均通过，0真实IQ/CP；这是发布前的REGISTERED_NOT_RUN记录，当前状态见后续实际记录。不能把新基线接入写成已有正式性能收益。

实际代码8235c71cc已发布，C0/C1在N607 GPU2/3 verified running；own scratchCP smoke、PID/CWD/argv/log增长、真实F/默认头更新及As调用核对通过。当前仅源域RX1检验，完整source基线、正式clean+6practical及跨seed尚未运行。离线读取工具增加SupCon贡献核对与完整状态分层，未修改运行release。

C0已完成E20及七源视图、默认头独立复算：held RX1 clean83.444444%、low urban25.685185%；完整4440日志／3330活跃SupCon、0U真值／跳步／错误。真实网络包3579018。C1仍运行，尚无配对收益或正式target结果；这是新F+CROSS_RX的实际单源折基线，不和不同信道流的旧F直接配对。

## F＋跨RX首轮paired-As实际结果

20261016-phase1-FcrossRX-practical-source-RX1-e20-s2026092701-m2-r01 已完成40模型轮／8880更新和全部七源视图，源规则 PASS；urban均值 17.746913580246915、low 12.722222222222223、urban宏F1 21.616060692881106 个百分点，clean损失 -2.722222222222226 个百分点。单源RX1／单seed／0target；不等于正式或跨seed收益。完整日志、全部5994L/U顺序、实际F clip／默认头、训练覆盖和同流prediction已独立核对。[实际报告](../automation_reports/CV-SincNet/20261016-phase1-FcrossRX-practical-source-RX1-e20-s2026092701-m2-r01/report.md)。

只支持另登记源确认，不自动扩target或宣称Ag有效。 真正Ag/As四格及其交互尚未执行，联合目标保持active。

下一项源确认的固定问题：保持当前F+CROSS_RX／E20／pairedAs配方不变，在held RX3/4/6/8各新C0/C1确认，并复用RX1结果。逐折沿用同一通过条件，任一源折失败不声称稳定跨RX，不换失败折或seed救援。该八row矩阵目前尚未发布；后续独立Ag和四格交互仍未证明。当前主要收益在LOS/PARTIAL/locked，BLOCKED/unlocked改善较小，lowurban TX2下降20.5556pp必须保留。
