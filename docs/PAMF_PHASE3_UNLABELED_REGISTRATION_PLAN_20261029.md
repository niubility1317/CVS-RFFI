# 冻结身份的无标签适应与匿名注册计划

run为`20261029-phase3-unlabeled-registration-s2026092701-r01`。上一轮完整身份网络增量使新RX的closed准确率已低于允许的open下限，因此本轮冻结原F0/E50及其默认六类头，把身份判断与辅助拒识分支分开。辅助分支使用既有E51、原clean-L聚合几何和上轮四视图augmented读出，均不重新训练。DURA固定R69/v1、FP32/TF32off、reuse/graph true/native false；两F加G的实际参数、计算代价另行报告，不把它称单F或原部署时延。

本轮将真正完成四态：DA0_REG0、DA1_REG0、DA0_REG1、DA1_REG1，主状态固定DA1_REG1。目标仍是跨RX clean未知拒识≥95%、星地≥90%，每主景保护；registered宏FRR≤5%、旧类open较同输入原F0closed最多损失2pp。适应须旧类open正收益并保住拒识；匿名注册须未来新类macro≥80%、旧类较注册前最多损失2pp，后续未注册设备拒识达到相应条件目标。失败如实报告，不能以anonymous输出存在替代性能成功。

## 源统计与冻结

复用原source五RX/三日L6300及已验证G缓存，仅source_L进入新统计。C0每条使用clean；P1每条用seed2026102935从三个既有source practical/G计算view中iid取一个，与y、TX或群配额无关。每条件6300原物理记录，不将四view当独立邻居。L已知标签只用于源阈值/聚合统计；新神经优化0，原U/OE21/novel/target不参与新source统计。E51与既有辅助读出自身的源监督祖先仍完整披露，不能称全系统无源标签。

复用源known V27000四view的compact与原F0同输入closed。拒识分数来自不变E51aug读出；身份closed与baseline均为F0。以同一个global τ/方法保护overall、5RX、4view、20RX×view共30组FRR5%/原F0正确预测的误拒损失2pp，整组ties；不用unknown分数校准。C0/P1 score尺度σ分别由known V的clean27000／三star81000计算曝光估计，floor1e-6，不能把重复view作更多独立验证。

密度ε由source同knownTX/RX/day组内，每点第7个其他物理邻居距离的95%分位数得到；每组70条，DBSCAN min_samples8含自身，与7others对应。embedding为fused/time/frequency/PA/PA_moments五branch按固定source几何白化、unit后等权拼接688维。新类接受半径r为source F0_unit到同组leave-one-out中心的cos距95%分位。部署只保留90组unit F0均值/knownness均值，每组70物理记录，外加ε、r、σ、阈值与既有几何/读出；不回读source IQ或逐行特征bank。

## 无标签适应与发现

源状态先冻结再读目标U。C0/P1独立使用RX24-1 D1的U384，真实known/unknown角色、未知类数/配额/TX均不进入算法。可靠旧core要求F0与E51的MSP均≥0.95且argmax一致、fused源原型近邻同类且cos≥0.5、time/frequency/PA至少两票；每旧伪类至少8条且支持至少3类。每branch用合格伪类等权的source unit原型与U unit均值差，乘0.5再限制bias范数≤0.1；reference坐标27保持0。只有该小型判决特征状态适应；身份F0、辅助E51、读出及τ均不更新。core不足则identity并记科学失败，不放松条件救分。

DA0和DA1分别在当前primary MLP拒绝的U观测上做DBSCAN，ε只来自source、min_samples8；每簇至少16个独立opaque ID才注册，簇数自由推断，0簇合法。anonymous ID按opaque ID稳定排序命名，不分配真实TX名称、不强行得到真实类数。每簇保存原F0单位特征均值，source半径r固定；利用source聚合锚点给newmargin加惩罚，防止任意新原型直接吞并旧锚点。只使用聚合源状态和U，不用未来Q校准。

注册后所有old6和anonymous候选共同竞争。old margin为`(F0logits-max)/30+(knownness-τ)/σ`，new margin为`1-cosdist(F0unit,proto)/r-penalty`，最大margin<0拒绝，等于0接受，ties确定性旧类优先。REG0无新原型时严格退化为原F0身份＋knownness≥τ；REG1不按真值或隐含旧新角色限制候选集。这里是预定统计判决，不是声称源半径能泛化或新设备一定可识别。

## 独立future Q与评分

同条件U适应/聚类/注册状态全部冻结后，逐样本预测D2 Q864，U/Q物理互斥；Q只计算四态输出，不计算跨Q统计。独立scorer在四态Q预测全部固定后，才打开私有U和Q真值。anonymous ID→TX置换仅由U的聚类成员真值匹配，不在Q上拟合映射，不回流预测。新类macro分母包括U中所有真实新设备，未发现/未匹配者仍计错；未匹配anonymous输出是错误接受，不能冒充未知拒绝。后续未注册设备只以预测-1为正确拒识。

逐条件报告旧六类接受/误拒/正确open/原F0损伤、新类未来macro、全部预期注册类macro FRR、后续未知拒识、簇数/噪声/纯度/合并拆分/旧类误注册、逐scene/TX及四态差分。REG0新类注册准确率为N/A，不能填0宣称REG1优越；另保留七种拒识分数的REG0诊断，G2只用事前固定MLP。仅单模型seed、一个开发RX和三个star主景，不扩种子或正式池，除非后续预登记门槛通过。

本轮复用1248个U/Q原物理记录；clean与star配对不计新增记录。source与开发跨RX，日期非严格未见，novel设备此前source评过；不是独立新TX盲测。正式RX24-18 U768/Q1920保持未访问。本轮首次允许传递U truth到独立score侧，但绝不传入predictor的公开配置/目录；这是评估匿名ID所需，不是目标监督授权。

## 执行

新不可覆盖run/release、唯一owner，精确commit、Python/CWD、命令、路径、GPU、stop与artifact在experiment.json/runtime登记。source→C0 Ufit/四态Qpred→独立U/Qscore→P1 Ufit/四态Qpred→独立score共5jobs，GPU一次1worker、每卡最多新增2且free≥4096MB，不停止其他owner。真实checkpoint合成smoke是source/predict首步，PASS立即继续；新source统计不要求梯度smoke。执行既有八项最小流程，不新增签名/hash链或审批。

技术停止仅输入/继承/角色错误、非有限值、输出碰撞或无法产出合法完整预测；保留partial。0簇、core不足或性能不达标不是重试理由。输出包括源小聚合bundle、U bias与anonymous prototypes、匿名U assignment诊断、四态Q predictions与margins、独立score、完整log和post-state；不提交IQ/权重/预测/原始日志或私有runtime。总Phase3目标不因此缩减或预先标完成。
