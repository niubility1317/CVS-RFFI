# Phase3源侧物理分支证据开发

2026-10-29。下一独立候选拟为`20261029-phase3-source-branch-s2026092701-r01`。本候选不重训模型，不展开失败源分数候选的三fold，不访问正式target；研究不同物理证据是否保留了最终分类表征未体现的新设备差异。完整Phase3目标保持95%拒识与旧类保护，源代理结果不替代正式验证。

## 已有证据与假设

固定源留出模型训练20轮/540步已完成，已见RX的V已知98.6528%，RX8为66.4722%；8个最终160维分数未达源推进。该结果只证明这组四类CE模型/分数不足，不能证明完整PAMF或联合无标签适应不可能达标。

CPU合成结构核对表明，完整特征为核心时频PA融合加参考响应投影及SIG-TIME残差。原始参考响应29维、3签名拼接336维以及更早频谱/PA描述可在同一完整前向中导出。它们并非统计独立；学习分支仍受4类CE影响，不能称无标签训练特征。

## 固定候选矩阵

只复用来源核实的`20261029-phase3-source-holdout-s2026092701-r01/train/source_model.pt`，新source角色与其训练契约一致；strict load四类state_dict，不加载原六类checkpoint、teacher或恢复器。训练选择仍labels2/3/4/5、RX1/3/4/6的3360条L，V27000完整read-only。

|表示|维度|导出位置|标准化|
|---|---:|---|---|
|fused160|160|reader.features|与前轮相同L2单位向量，作为对照|
|time160|160|core aux t_emb|仅L的全局location/scale，然后L2|
|frequency160|160|core aux f_emb|同上|
|pa160|160|core aux feat_pa|同上|
|reference29|29|reader.response(x)|同上；坐标27必须保持零|
|signature336|336|三个sequence_pool.views输出拼接|同上|
|spectrum128|128|freq_gate输入的raw_fft4×32|同上，不使用gate后的表示冒充raw|
|pa_moments48|48|pa_lift24通道沿时间的mean/std|同上，固定数学汇总|

每个表示比较cosine中心距离和全局类内收缩Mahalanobis（固定shrinkage0.2），共16种明确source规则。只在训练L拟合location/scale、类中心、协方差；常数坐标scale置1，不以V估计统计。V known固定99%保留阈值，V proxy unknown只评价源推进。最终类别始终来自完整reader.classify_features(reader.features(x))，禁止使用缺少reference/signature的core aux logits。

第一轮只做当前同一个TX-pair/RX fold；全部16种事前固定，不从正式target挑分支/阈值。源95%/5%/2pp共同条件保持，通过的分支才进入后续独立source一致性检验；未通过保留负结果，不扩正式矩阵。两种distance采用同一公式/超参数处理全部类，不为具体TX或RX单设规则。

## 前向与状态验证

call-local hooks必须finally移除，单次完整前向同时捕获分支与最终logits，不改checkpoint。保持eval/no_grad；原模型实际含0 BatchNorm、10 GroupNorm、4 LayerNorm，按样本计算，不用批量或query统计。合成检查特征重建与同行替换独立性，真实own-checkpoint无query smoke从合法训练L取两条，PASS立即继续源导出。原response.components中的未屏蔽CFO不得进入本候选；使用response(x)保持既定NoExplicitCFO策略。

## 后续无标签代理设计（未实现/未启动）

还可在F0原source L_s中用RX8且source labels0/2/3/4/5的1050条记录建立标签隐藏的源代理池，留出label1的210条L不开放。该池与四类模型train3360和V27000在ID及canonical物理记录上均互斥；初始已注册4类，算法不得获得真实未知类数、配额或old/new标记。V继续只评分，不能适应/注册/BN或原型更新。RX8 V中旧4类3600、新发现类900、后续未知900。

该代理只有一个可发现类，不能证明多个未知设备之间的分类。它保持F0原source角色，不从F0/R01目标query artifact转训练；历史独立F1/F2划分曾将RX8作为目标，应披露跨划分历史曝光，不能声称全历史从未query或新的盲测。不同独立契约的source/target集合交叉不自动造成当前scratch模型继承污染，但不能继承别划分权重或读取其query反馈。

正式Phase3无标签适应/多新类注册仍优先使用待实际Full eq/packet身份核实的独立新RX24-18池（2发现类＋2后续未知），不因源代理可运行而缩减正式目标或数据隔离。

## 分工与当前状态

模型代理已完成只读接口/尺寸/归一化/状态核对；主Agent负责实现与集成，数据代理负责相同源角色复用。候选实施后按新候选执行一次P0/P1审查和既有最小发布流程。

DESIGNED_ONLY。未实现分支导出器或16规则runner，未建立正式无标签池，未启动本候选。源码、测试、run登记和实际结果不能由这份设计替代。
