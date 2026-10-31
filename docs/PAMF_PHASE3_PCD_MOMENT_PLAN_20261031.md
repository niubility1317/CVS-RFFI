# 冻结 PCD 的目标类内矩校正

Run `20261031-phase3-pcd-moment-s2026092701-r01`，唯一owner `codex-phase3-pcd-moment-20261031`。本轮在当前PCD研究基线上检验新的无标签统计适应：神经网络、源密度和阈值均不改，用合法目标U估计共同偏移与完整协方差校正。两项固定适配为SHIFT平移控制和CORAL主候选，旧E0/PCD作为冻结控制；不替换默认模型或追改历史失败。完整目标ACTIVE。

参考[Sun等CORAL原论文](https://arxiv.org/abs/1511.05547)的二阶统计对齐思想。本行使用原模型筛选的伪已知core、类内残差、目标到源变换与固定分类器，是自选RFFI变体，不是原论文的source→target后重训分类器复现。既有正对角MAP-EM、混合先验UOT、低秩平移及源PCD-RMD均不等于此完整协方差目标矩校正；不重复其旧RX24-1实验。

## 固定输入与标签边界

原F0为Tri-SIG-TIME SSL E50/11100、232993参数；最终RX34 PCD为fresh-development f4b5的独立r03初始化＋U576/444步模型，334721参数；DURA v1.0 R69 E20/4940、266951参数，reuse/graph开/native关、FP32/TF32off保持。r03和全部原参数/buffers不更新。PCD与其源密度有既有U576神经祖先，本轮适配器另有同一U的统计祖先，不能称整体纯源。

U使用`20261031-phase3-pcd-cluster-registry-s2026092701-r01`/`16e6f04e7475827e7c8e24b398fb029a39a8138c`已经导出的最终PCD特征，读`discovery/U.npz`的ids/features/default_logits/closed/base_scores五个member，**不读assignment、注册状态、true TX/K、配额或真值**。原fresh训练cache中的parent_z是r03而非最终PCD，禁止混用。对应模型身份、原F0/G、U的VALIDATED_ONCE capsule/split/count必须与fresh最终PCD一致；新Ufeature/ID各读576，不重新生成IQ或神经前向。

源原合同392005保留L6300/Us56700及OE6134祖先，源RX1/3/4/6/8（1-19/18-2/19-2/2-19/3-19）、03-08/15/23、eq1、旧六TX14-10/14-7/20-15/20-19/6-15/8-20。新L/OE/Us IQ读取0；源参考矩只由冻结known90高斯聚合推导，不用V或NV拟合。

目标为RX3-2/index34早日03-01的星地U576，与future03-15/23 Q物理互斥。Q为5525＋4713＝10238条，已知4508、未知5730；四未知5-5/10-7/2-19/7-11仅builder/scorer，后来7-11整类不在早日U。03-23缺20-19/8-20，完整名单AK/CK/H/闭集保持N/A。源V27000已有阈值校准用途，不是独立已知测试；NV3150的19-15/19-16/7-1各1050只评分，不能当作目标四未知。

三类主输入均经过既有DURA。source/clean/star预测只消费已冻结fresh的同记录features/logits；clean只测试，不能参与core或矩估计。G10保持600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、input RMS一次/output RMS无，源scene3001/channel4001/future3133/3134/receiver2027不重抽。NV/Q历史曝光、单RX/seed、global physical history UNKNOWN及正式RX24-18未读保持。

## 确切统计方法

令w＝(unit(z_PCD)−origin)W，沿用原源密度坐标。known90按六类×十五RX/day模式排列，m_c为该类十五均值的平均。源model-implied类内协方差为：Cs＝mean(known_variances)I＋mean_{c,g}[(mu_cg−m_c)^T(mu_cg−m_c)]。它来自既有等权球形高斯模型，既不是经验全局总协方差，也不能因已有残差白化器就设为I。

Cs的对称特征分解给P＝Cs^(−1/2)、R＝Cs^(1/2)。只用适配前冻结F0 softmax置信度≥0.95且原PCD predicted margin≥0的U记录作为core。每个非空预测旧类等权，类内core记录等权，权重和为1；这是预测筛选，不是已知真值或真实类别配额。空预测类不补样本，算法不接收真实未知K或U类别构成。

对core计算v_i＝(w_i−m_closed_i)P，nu＝sum a_i v_i，Ct＝sum a_i(v_i−nu)^T(v_i−nu)（population covariance）。使用Ct_reg＝0.9Ct＋0.1I；其逆平方根响应按特征值截到[0.5,2]，得到对称B。A＝PBR，delta＝nu R。SHIFT使用h＝w−delta；CORAL使用h＝m_c＋(w−m_c−delta)A，c始终为原F0在该记录上的预测，校正后不再次单位化。

两适配共享同一core、权重和delta，差异为是否增加完整矩阵校正。若core少于2条或有效样本数≤1，显式记录NO_CORE、两态恒等和fit_succeeded=False，不伪报适应成功。保存core数量、预测类直方图、有效样本数、平移范数、Cs/Ct谱、裁剪数量与矩匹配诊断；它们不是伪标签准确率，也不能把全部矩差解释为纯接收机变化。

部署用h＝wA＋offset[c]，CORAL offset_c＝m_c−(m_c＋delta)A；SHIFT为单位矩阵和重复−delta。适配器平坦状态含schema/kind/matrix160×160/offset6×160，主数值26560个FP64／212480B。原source统计的Cs/P/R只在拟合中使用，不额外部署逐条source/U bank。目标估计量为160个均值与12880个对称协方差自由数；不称神经训练参数。

新score仍为原known90/OE32在h上的pooled或predicted density ratio，原F0预测选择同类15模式，已知与背景的同一线性Jacobian在比值中抵消。pooled指密度边缘化，CORAL变换本身仍依赖预测c；不得称该读出完全不依赖身份。source RX不应用目标适配器，新SHIFT/CORAL源读出严格等于PCD；target clean/star共享唯一星地U状态，不按场景拟合或使用query批统计。

## 预登记评估与执行

固定8方法×3DURA：E0、PCD、SHIFT、CORAL各pooled/predicted，唯一主CORAL_predicted_ratio。原E0/PCD四读出精确复用；新48个FP64阈值384B仅由既有四组六τ24个唯一值192B复制，不新增校准、normalizer、GMM或神经更新。统计适配器先于任何新source NV/Q特征消费冻结；原F0身份及同G闭集全部保持，AK变化只来自拒识变化。

全部三条件预测固定后，独立CPU scorer读取NV/clean/star三truth各一次；Utruth0，scorer只用存下的U IDs核实未来Q互斥，不重新估计U矩。完整报告逐TX宏AK/CK/RU/H、闭集损伤、AUROC/OSCR、TX/RX/day/scene/day×scene/覆盖/CI与主对E0/PCD/SHIFT配对。source known12与42诊断分报；cleanRU≥95%、starRU≥90%、CK≥90%、同G原F0closed−AK≤2pp和各必需主景完整，且starAK高于PCD/SHIFT、RU不低于E0/PCD/SHIFT，才adaptation_success=True，否则NO_PROMOTION、不扩seed或换主。即使通过当前开发单元，也不意味着未知发现、匿名注册或整个目标完成，AN/H_OW仍N/A。

首步骤实际F0/r03/PCD/G无query synthetic smoke PASS立即继续；真实新F0/open/G前向均0，合成F0/r03/PCD/G各2行另列。新读source缓存预测30150行、target Q缓存预测20476行、U最终PCD特征576行；真实记录数不因反复读特征增加。继承原F0/r03/PCD/G训练、source fit/τ及缓存生成成本；主神经部署仍834665参数，另有212480B适配数值状态，阶段墙钟不是B1推理延迟。

实际commit/命令/环境CWD/路径/GPU见本run登记。默认N607，卡0至7free≥4096MiB、每卡新增最多2本任务worker，不干预其他owner。单不可覆盖run/release/output，单包一次SHA比较、remote compile及PID/CWD/argv/GPU/log增长核实。停止仅协议/祖先/输出冲突、非有限、无法执行或无法生成合法prediction；不按低分中止，不盲重启。预期smoke、源state/τ、两适配器及U IDs、完整三条件prediction、独立score、pipeline/logs、同row报告与登记Git交付。
