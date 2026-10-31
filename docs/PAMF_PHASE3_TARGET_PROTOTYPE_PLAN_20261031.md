# Phase3 目标原型投影适配（2026-10-31）

Run `20261031-phase3-target-prototype-s2026092701-r01`。原 `predicted_ratio` 主基线与 `pooled_ratio` 对照继续保持。本轮检验独立开放投影的无标签目标结构学习，不复用先前 target-adapted 权重，不启动延后的 native-energy 草稿；当前无新性能，完整目标 ACTIVE。

## 固定方法与对照

先前全信号实例对比 TARGET_SSL 出现源损失升高、跨RX正确接收下降；本轮冻结 r03 headless 信号编码器232033参数，只训练既有640→128→160开放投影102688参数。原F0/default6、DURA及原r03均保持。两学生均独立严格从同一source-only r03初始化，SOURCE和TARGET_PCD各fresh AdamW444更新、lr1e−4/wd1e−4/eps1e−8、普通参数norm clip5；冻结Sinc不进入优化器，不称CENTER_UNITS或物理Sinc更新。

SOURCE仅源训练；TARGET_PCD增加目标原型分配学习。两臂source每步192L/64OE、种子2026103017/18、逐步索引完全匹配；源目标为原crossset_metric＋原OE软邻域结构KL，再加10×源父z160点对点平方距离（L/OE两组等权），保护既有源几何。只用合法L旧TX与OE outside-six成员/公共RX/day，不读OE TX、原U_s、V/NV或Q训练。SOURCE完成并冻结后才打开目标U。

目标U384先经同一冻结DURA恢复一次，再由原r03冻结编码器提取弱view及两个固定独立phase/gain派生view的u160/c640。seed2026103114；每条仍只有一个原received，派生view不是新信道或物理数据。source L/OE也只由父模型一次缓存u/c/parent_z供投影训练；缓存不含V/NV/Q，原F0的OE teacher u沿既有合法特征缓存复用。

TARGET_PCD以父弱z的spherical kmeans++初始化16个单位原型（seed2026103113，固定25轮Lloyd），原型16×160＝2560参数仅辅助训练，16为事前容量而非真实设备数。每步对当前学生全部弱U384与原型做epsilon0.05、固定20轮平衡log Sinkhorn；逐样本q归一，原型参考质量均匀。q停止梯度，以温度0.1的强view logits预测q；global为两强view对自身q的CE平均，local为两强view对父弱z最近其他记录的q的CE平均。邻居固定、排除自身，不读真实TX/known角色。目标批64、seed2026103112，训练总损失=source＋0.1×(global＋local)；每步后原型单位化，不扫描权重/温度/容量。

上述借鉴[SwAV的跨视图原型分配](https://arxiv.org/abs/2006.09882)和[UniOT的目标全局/局部结构学习](https://papers.neurips.cc/paper_files/paper/2022/hash/bda6843dbbca0b09b8769122e0928fad-Abstract-Conference.html)，是PAMF固定编码器变体，不是两论文完整复现，不继承论文性能、私有类发现或注册声明。原型不替代六类默认头，也不作为最终匿名设备数；本轮部署仅学生SignalOpen与源拟合密度，aux原型不部署。

## 冻结评估与主方法

实际原F0/r03/G checkpoint各2条无query smoke先执行，PASS立即继续；投影/原型合成梯度probe验证可训练参数，冻结编码器梯度为空、原参数及buffers不变。两个训练臂均完成并冻结后自动进入测试，不以训练完成作为最终结果。

六列顺序固定：E0_pooled_ratio、E0_predicted_ratio、SOURCE_pooled_ratio、SOURCE_predicted_ratio、PCD_pooled_ratio、PCD_predicted_ratio；唯一主最后。E0两列及其两组六τ精确复用4af。两新模型分别以源L/OE拟合自己的known90/OE32及normalizer，全部密度冻结后才以已知V27000各校准两组六τ，共24新阈值；全36阈值冻结后才预测NV3150与clean/star Q各864。全部6×3预测固定后独立CUDAempty CPU scorer才连接源NV、cleanQ、starQ真值各一次，Utruth读取0。源科学失败仍测全部Q，不换主或取消。

身份始终来自原F0默认头；同DURA输出同时供原F0和两个新学生，目标原F0 logits/closed核对原4af参考，不能以新模型自身或另一输入路径闭集替代保护基线。两新模型全三条件均走DURA；clean仅冻结测试，不参与任何训练/校准/发现。

## 数据与科学边界

原源训练L6300/隐藏U_s56700（训练池L占10%）、OE6134；V27000是额外已知校准监督，非独立known盲测。L每类×五RX×三日70，源RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23；NV3150，19-15/19-16/7-1各1050。旧六TX14-10/14-7/20-15/20-19/6-15/8-20。

目标U为RX24-1/index26、2021-03-08的固定星地384条，与全部源及future Q物理互斥；U真TX/K/比例/配额均隐藏。capsule/split为p3dev-star-g10-6e1292b62ab09286ebf1a0df / 83c5cc1fe2f2a1512dbc1287。未来clean/star Q为同RX、03-15的同864物理记录（旧六类各96＝576，三非旧类各96＝288）；重复处理不增加独立物理数。U中无标签接触的设备不能宣称对TARGET_PCD严格训练未见，使用“未注册未知”口径；SOURCE仍无U祖先。

全部星地沿既有G10 MIX01：600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、原六景权重、郊区城区各0.5、receiver2027/scene2026103001/channel2026104001；信道前一次unit RMS，received不再逐样本RMS，不重抽信道。历史NV/Q曝光、单seed2026092701/单开发目标RX、源日期重合保持；high未知0/4/2缺19-15，宏RU/H仍N/A，正式RX24-18未读。

## 门槛、预算和执行

主报逐真实TX宏A_K/R_U/H，另报C_K、原F0同G闭集−A_K、AUROC/OSCR、全部TX/RX/day/scene/coverage、源known12及30交叉诊断、同ID配对gain/loss/net。cleanRU≥95%、starRU≥90%、CK≥90%、同路径原F0闭集损失≤2个百分点及必需主景完整覆盖保持；还要求主starAK严格高于E0_predicted和SOURCE_predicted，RU不低于两control。H不替代单项约束，不按Q调参数/重排/选择性重跑或扩seed。当前不做匿名REG，AN/HOW为N/A，完整目标ACTIVE。

新增神经更新总888，仅projection各444，TARGET另2560个aux原型参数；编码器累计特征更新仍13320，projection从2220到2664，原F0/head/G/Sinc更新0。source frozen-parent缓存12434行、TARGET三view父forward1152行、G恢复384行；每臂source投影前向113664，TARGET另weak全U/两strong每步512，共227328行，合计学生投影454656行，实际计数分列。上述均为已有物理记录重复计算，不是更多独立样本。新增两源密度及四组六τ不得称零统计拟合，target学生及其源聚合有U祖先，不能标source-only。

部署单方法F0 232993＋新open334721＋DURA266951＝834665参数，aux原型和逐行source/U训练bank不部署。两新GMM各known90/OE32、normalizer与源校准成本另列；原F0 E50/11100、r03原2220、DURA E20/4940及原数据缓存成本继承。实际训练/预测/scorer时间均非B1包时延。

一个实际代码/配置Git commit、最小登记（精确命令/环境/CWD/路径/六类seed）、聚焦负测/真实CP无query smoke、一次直接P0/P1、一次N607 preflight/单archive SHA比较/远端compile。入口star_target_prototype_launch start --runtime <release>/runtime.json，单row内两臂训练完成即自动predict→CPU score→同row报告；默认N607/GPU0至7实际剩余≥4096MiB、每卡新增本任务worker≤2，本轮一个GPU worker。不干预其他owner或覆盖旧模型/输出。

非法角色/祖先、非有限loss/gradient、冻结编码器或原状态改变、输出碰撞/无法形成合法prediction时保留全部产物；partial不得接truth，不覆盖/自动重启，UNKNOWN先核原PID/root。科学低分不停健康任务。保存完整444×2日志、索引匹配/模块冻结/实际曝光/aux计数，评分与报告不回流训练。原选定基线及延后native-energy状态保持。
