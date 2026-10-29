# 默认头开放暴露与相对关系保持

Run `20261030-phase3-star-relation-oe-s2026092701-r01`。唯一主 `relation_oe::entropy_global`，匹配控制 `relation_control::entropy_global`。本轮检验：用相对关系保持代替逐向量锚定后，默认头上的开放暴露能否改善源开发及小规模跨接收机星地拒识，并保住原六类身份。

uniform OE 本身已经做过，本轮不把它写成新机制。此前 jointOVA 更新了骨干和默认头，不能说此前全部冻结 F；其逐向量权重5是否限制开放几何重塑仍是假设，不能凭两个损失值大小证明。现在的两臂仅检验“关系保持条件下默认头 OE 暴露是否有增益”，并不能单独归因于取消旧锚定。

每臂从原 Phase1 F0 E50/11100 独立开始，不继承任何按开发成绩选出的 Phase3 增量。保留三视图、固定局部排列和原六类默认头，共232993 active参数；头与骨干一起训练，无新增身份头/OVA层/MLP。固定 DURA v1.0 R69/266951，训练只读既有 G10 P1缓存，不训练恢复器。

## 两臂固定差异和预算

两臂均 CE_L＋T2 logits KD（KL×4、权重1）＋OE关系MSE（权重1）；主额外加默认头 `KL(uniform6 || softmax(logits_OE))` 权重1，控制关闭这一个损失。OE关系为单位 fused160 的批内非对角 Gram 与同记录原F0单位特征 Gram 的均方差，原教师detach；不做逐向量锚定，不用OE真实TX/K。该损失允许共同正交变换，不保证跨设备可分或未来注册，需独立评分观察。

原生可微 Sinc/CENTER_UNITS AdamW冷启动、lr1e-4恒定、weight_decay继承原F0、clip5；eval BN/dropout但梯度开启，持久buffer冻结。每步L128 IID＋OE128均匀RX/day域采样，concat256一次完整F前向。L seed2026092701、OE seed2026102907，两臂采样流相同。无普通增强/新信道/As/PL/EMA/U。各2220次固定更新，累计13320次；不是十遍自然epoch。无早停、末轮选模或权重扫描。

各臂L/OE采样曝光各284160次，F真实训练前向各568320条；两臂1136640条，共4440次optimizer step。原F0→教师OE特征只复用旧r02/code `5aa041122396d48c49b8cb3d5912649bdc3217e1` 的source-only teacher，非其训练后的joint模型，不重导教师。L teacher logits复用同P1缓存。原父训练、缓存、教师导出与重复曝光成本分别保留，不当新独立数据。

## 数据和标签权限

源仍是L6300/OE6134/known V27000/source novel3150，五已见源RX1/3/4/6/8＝1-19/18-2/19-2/2-19/3-19，2021-03-08/15/23。L旧六类14-10/14-7/20-15/20-19/6-15/8-20，训练可见合法旧TX；OE仅“当前六类之外”的成员监督、无OE TX/K，不能写整个系统零源标签。原Us56700不新增读取，Phase1父split392005与10%有标签契约不变。V每旧TX4500、source novel三个历史设备各1050，仅冻结后评分，不是新TX盲测。

源真实IQ、teacher及新评价均是原 G10 MIX01 同一物理观测的P1：600km/25MHz/2.45GHz、receiver2027、scene/channel2026103001/2026104001、六景面积概率、residual/post_sync/noEQ；输入一次RMS，输出不重归一化，P0始终是星地received。合成checkpoint smoke另计，不当物理样本。

每臂直接执行同一已暴露开发RX24-1的G10 future Q864（2021-03-15），接收机与源五RX互斥。旧六类576、三个未知设备288的真实角色仅独立scorer可见，producer配置无未知名单、K或配额；公共数据只用匹配 `phase3_min_v1`/`VALIDATED_ONCE`/capsule/split。Q仅逐批前向，不计算跨Q统计、不拟合模型/阈值/聚类。原合法U384保持未读，future-Q不回收训练，正式RX24-18保留。此处仅DA0_REG0的跨RX证伪，无新UDA/发现/注册结果。

## 冻结阈值和独立评分

唯一分数是原默认头 `1−H(p)/log6`；身份仍其六类argmax，拒识不换身份。每臂只在known V上选择一个最大合法全局观测值阈值，接受 `score>=tau`，完整ties。硬约束为overall及六主场景共7组的逐真TX宏CK≥90%、原F0同P1 closed−新AK≤2pp。RX和RX×scene等全部42源切片完整报告，但余下35组仅诊断；没有六个预测类共同尾率等额外约束。

若候选闭集本身已损伤超限，有限最小score阈值接受全部并明确known失败，仍完成固定novel及目标Q，不技术重训。known V无神经拟合、无早停。模型、阈值和known预测先冻结，再source novel，最后target Q；全部本row源/目标预测完整后，独立CPU scorer先重建defaultargmax/entropy/margin/known7校准，才连接source novel和target私有真值。

本轮**不以source RU先达90%作为开发target前置**。两臂均按同一预登记顺序只跑一次，小跨RX结果才是本机制的主要证伪。source分数低、结构诊断差、控制较高不改变另一个健康行或主方法；不按历史未知类设置专属规则，不修改旧实验阈值/判定。

主报目标逐TX宏AK/RU/H、CK、原F0 closed损失、候选闭集、AUROC/OSCR、逐场景和逐类计数/CI。每个主场景覆盖不足明确N/A，不能认证通过。source同时报42组和novel heldRX1NN，仅结构诊断而非AN。只有固定主在target overall+六场景同时满足CK90/RU90/原F0损≤2pp，才取得下一独立UDA证伪资格；不据控制替主、不扩seed/正式矩阵、不宣布Phase3完成，AN/HOW仍N/A。

两臂同Q/同observedIQ/同原F0基线，最终报告正确接收和未知拒对的得失配对、已知身份变化及结构保真。当前分数只是单默认头读出，不额外换分数以选择最高。目标阈值不拟合，source统计不作为目标逐样本bank。

## 执行、检查和停止

N607、唯一root launch owner，最多2行分别占不同实际可用卡，每GPU至多新增2本任务train/predict，启动前实查≥4096MB；其他owner/用户进程保持。独占run/release/output，训练冻结后自动源和target预测、独立评分、同row分析与登记。失败行保留partial，健康行继续；科学失败不触发重启/重采样/调参，UNKNOWN先核实原PID/产物，不重复launch。

只执行现有最小流程：单实际代码配置commit、聚焦协议负测、首步实际checkpoint合成星地梯度/无query smoke、一次直接P0/P1审查、最小登记、一次N607 preflight/单archiveSHA比较/compile、启动后PID/CWD/argv/GPU/日志增长、完整预测后独立truth-last。代码和公开汇总Git交付，IQ/权重/预测/原始日志保留private。原F0/DURA/teacher与数据句柄核对属于输入权限，不加hash链、封印或额外审批。

相关方法依据：OE对辅助离群样本使用均匀目标，见[原论文](https://arxiv.org/abs/1812.04606)。本轮加入源关系保持和项目默认头/数据约束，属于独立项目变体，不声称原论文复现或其图像基准效果适用于RFFI。
