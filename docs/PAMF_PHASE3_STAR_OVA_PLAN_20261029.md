# G10 类条件开放边界与难负类学习

Run `20261029-phase3-star-ova-evidence-s2026092701-r01`；唯一主方法 `ova_max`，固定四决策 `membership_routed/membership_max/ova_routed/ova_max`，只训练两个共同初始化的读出模型。该候选在本轮任何新 IQ 或 novel 评分前确定，所有控制均不能在结果后替代主方法。

上一联合管线已经保留负结果。本轮回到源端检验开放边界的表示与训练目标，不继续在开发 Q 上调整邻域或阈值。完整目标仍是跨 RX 星地拒识、无标签适应、发现及未来匿名注册；源端试验本身不能证明上述目标端性能。

## 方法依据与实现范围

[OVANet 原论文](https://openaccess.thecvf.com/content/ICCV2021/html/Saito_OVANet_One-vs-All_Network_for_Universal_Domain_Adaptation_ICCV_2021_paper.html)用每类开放分类器及难负类训练学习已知边界，并研究目标无标签开放熵适应。本轮仅借用“源真类正项＋最难错误旧类负项”的思想；冻结 PAMF 特征，使用共享非线性条件网络，加入合法 source OE，并保留已知 V 校准，不执行其目标 OEM。因此它是项目自定义 OVA 变体，不能当作原版 OVANet 复现或继承论文性能。

每条记录在 P1、P0 各有六个对应旧类槽。每 path 的八分支先用该 path 源 L 几何标准化、单位化；对每 class 输出 cosine_c、其他类最大 cosine、两者差，以及 negative squared MD_c、其他类最大 negative MD、两者差，共48维。再加 prob_c/logit_c/其他类最大logit/差/energy共5维，得53维；P1在前、P0在后，形成[N,6,106]。同一 class 槽在两 path 表示同一原旧 TX，类别轴随 head/prototype 一起置换；部署不用真实 TX 路由。

每个读出都是一个共享的106→32ReLU→1网络（3457参数），六 class 条件输入复用同一权重，不是六个独立 MLP。两个 arm 每个从相同 seed2026092701权重开始，CPU FP32 AdamW(lr.001,wd.0001,betas.9/.999,eps1e-8,foreachFalse)，各1000全batch更新，共2000实际更新，完整loss保存。无F/defaulthead/G更新。

normalizer仅由L/OE形成共同106维FP64 population统计，roleprior .6/.4、每role15个RX/day域等权、每记录六条件槽等权；scale≤1e-8回退1。loss role .5/.5，同样每role域等权。membership的known项为原P1预测槽的softplus(−g)；OVA的known项为.5×[真源y槽softplus(−g)＋最难错误旧类槽softplus(g)]；两arm的OE项均为六槽最大softplus(g)。OE只提供membership及域，不读其TX身份或类别数。

每个模型都预登记两种读法：routed取原P1 argmax槽；max取六个旧类存在性分数最大值。后者检验当原头错分已知身份时，另一旧类的存在性证据能否保住已知接收。两种读法均保留原默认P1身份输出，不以OVA获胜类别重分类。因此C_K包含错分已知，A_K仍只认身份正确且接收。

## 数据与冻结顺序

原 F0 为三视图 Tri-SIG-TIME SSL Main split E50/11100/232993参数；固定 DURA v1.0/R69/E20/4940/266951参数的 P1 cache 与原 received P0复用。所有新消费IQ均为G10 MIX01：600km/25MHz/2.45GHz，receiver2027，scene/channel2026103001/2026104001，六景面积概率，residual/post_sync/noEQ，一次输入RMS、输出不重归一化。每原记录只有一个固定received，两path不是独立物理样本。

新source consumption为L6300、OE6134、V27000、novel3150，共42584原物理记录的既有G10cache，每条P0/P1；不新读原U_s56700、rawManySig或真实记录信道生成。F0祖先仍为L6300/U56700/V27000，训练L/(L+U)=10%，V标签额外用于验证；本轮L的六类真实标签用于OVA正负项，不能称全系统零源标签。OE为21源辅助设备、无TX标签，novel为保留的19-15/19-16/7-1三设备各1050，真值只由独立scorer连接。后两项不把设备数传给学习器。

源五RX为1-19/18-2/19-2/2-19/3-19（indices1/3/4/6/8），日期2021-03-08/15/23；L每旧TX/RX/day70。novel与L/OE/V物理互斥，但这三个novel设备及source RX已经有历史开发曝光，不能声称新TX盲测或目标跨RX泛化。正式RX24-18、开发目标U/Q均不进入配置，不访问其IQ/真值/统计；不回收历史query。

源真实F0合成星地IQ no-query smoke先于实际cache输入，PASS即继续，G本轮只复用已验证cache身份。L只拟合两path几何，L/OE只拟合共同normalizer及两reader；这些全部冻结后才开V。每四scalar用V27000只运行既有predicted-class risk calibration，按原P1预测类保留六τ，globalτ只诊断；42个true-TX宏known组（overall、5RX、6scene、30RXscene）同时满足C_K≥90%、原F0同P1 closed−open≤2pp，whole ties，缺类不能认证。

24个conditionalτ全部写完calibration complete后才开novel；逐条输出[N,2,6]实际class_scores，以及由它确定的四个margin：raw scalar−τ[method,原P1预测类]。默认margin≥0接收，identity保持原头。所有四决策预测完成后，独立CPU scorer先核对raw4算子与24τ路由，再连接private novel truth；没有以真实TX/scene路由部署τ，V不拟合几何/normalizer/reader。

## 预登记判定与执行

主报true-TX宏A_K、R_U、H=2A_KR_U/(A_K+R_U)，并列C_K、原F0同P1闭集、closed−open差及AUROC/OSCR、逐TX/RX/scene、同ID净正确/接收/拒识。源known42保护和主ova_max的novel overall＋5RX＋6scene宏拒识≥90%且必要TX覆盖全部成立，才记SOURCE_OVA_READY_FOR_TARGET_FALSIFICATION；任何控制通过不能替代主方法。该source通过也不是跨目标RX成功，目标端需另行预登记一次小矩阵。失败记NO_PROMOTION，保留全部结果，不以novel分数改变网络、目标、τ、排序或选择性重跑。未执行DA/发现/注册，A_N/H_OW为N/A，完整目标保持ACTIVE。

当前数值要求来自v3：星地跨RX的C_K/R_U≥90%、同输入原闭集损失≤2pp；未来匿名A_N≥80%、注册前后旧损≤2pp/后续未知≥90%。本源试验只处理完整链中的拒识学习环节，未缩减全目标，也不以source达标替代target验收。

N607唯一owner当前root，每次最多1个ownGPUworker，actualfree≥4096MB、每卡至多新增2，不改其他owner；F0前向使用GPU，两个新reader在CPU训练，独立score为CPU。不可覆盖run/release；一次code/config commit、聚焦协议负测、唯一独立P0/P1审查、最小prereg、一次preflight/archiveSHA比较/compile、真实checkpoint first-step smoke、独立PID/CWD/cmd/GPU/log增长与truth-last同row报告。方法说明和判定就在本报告/配置，不新增审批或封印。

技术停止限祖先/角色越界、输入或输出碰撞、非有限、无法产生合法完成prediction；保留partial不原地重启。科学低分/缺覆盖不重试、不触目标。实际F0前向2×42584+2smoke=85170，真实G新前向0，新readout更新2000；源输入、真实物理与合成smoke分别计数。精确commit、命令、Python/CWD/paths、GPU及artifact保存在本run预登记。
