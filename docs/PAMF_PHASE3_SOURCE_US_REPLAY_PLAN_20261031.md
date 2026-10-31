# Phase3 原源域 U_s 星地重放与置信度伪标签（2026-10-31）

Run `20261031-phase3-source-us-replay-s2026092701-r01`。当前源原型判决已完成且 NO_PROMOTION，本行回到原 F0 身份头，检验在开放分支增加合法源 U_s 的星地无标签训练，是否改善源 L 过拟合及跨 RX 拒识。不是重新调当前目标阈值，也不直接升级失败的聚类/原型身份规则。

## 新源数据角色

沿用实际 F0 的原 source contract392005：L_s6300、U_s56700、V27000，源 RX1/3/4/6/8、day1/2/3、eq1、六旧TX。U_s 与 L/V 原角色ID互斥，不重新划分、不增加人工标签。Phase1原F0已经有clean U_s半监督祖先；“新使用”仅指本次 Phase3 G10/P1 增量。

新 U_s 数据按原 contract.role_ids 定位 ManySig 原始256 IQ。可信离线 builder 可以用原 TX/RX/day/eq/packet 坐标计算既有24位sample_id，仅用于物理定位；不读取 U_s 的 y，不把 TX/eq/sig/base_index/原始路径或定位表传给训练端。禁止把已经 unit-RMS 的 clean 缓存当原始 IQ 再归一化。原始输入经既定 G10 observe 一次 input RMS，输出不再RMS/EQ，再经冻结 DURA v1.0 R69 一次恢复。

沿用 source MIX01：scene2026103001/channel2026104001/receiver2027、600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect；physical_id直接使用原Us ID，RX/day使用原名称。每物理记录一份固定received。新独立cache/U_s公共目录仅 manifest.json、ids.npy、received.npy、restored.npy、metadata.npz；metadata恰为rx_i/day_i/domain（十五域0..14），无y/TX。建立新的VALIDATED_ONCE句柄，不改旧四角色缓存。

公共capsule `pamf-source-Us392005-G10-MIX01-20261031-r01`，split `392005-Us-source5rx-3day-eq1`。完整 builder 审计置于public之外。训练配置只接收公共Us句柄，不接收raw路径或builder配置；不创建U_s私有真值文件。

## 固定训练对照

精确复用 body-prototype r01/a33612e 的纯源 SOURCE444 及其完整L192/OE64采样记录。新 SOURCE_US 从原 source-only r03独立初始化，固定444步、lr1e−4/wd1e−4/AdamW/clip5，冻结Sinc48与buffers；更新body231985＋projection102688。原F0/default6、DURA、r03及旧学生保持。

L/OE损失和索引流完全沿SOURCE：crossset metric＋OE structure＋10×父z锚点，独立rng2026103017/18。Us批128用独立rng2026103150的无标签全池排列，跨末批循环，444步56832曝光、56700各至少一次；增强另用独立2026103151生成同观测全局相位/0.95至1.05增益，不重抽信道/不二次RMS，不扰动L/OE流。

固定原 F0 对全部 Us 的同P1弱输入给出六类概率，置信度≥0.95的argmax作为伪标签。伪标签来自冻结的源教师，不是Us真值或本轮新人工标注。新学生强view的z与原r03合法L六个固定、单位化类均值相似度/温度0.1构成六类训练分数。PL为逐条硬CE乘置信mask后按完整批128平均，权重1；不新增神经分类头、EMA、真实Us标签、类别配额或目标数据。固定六原型仅为训练锚点，不部署；新密度拒识仍由原 F0/default6 提供身份路由。

保存完整444步损失、梯度/裁剪、L/OE索引匹配、Us实际唯一数/曝光、伪标签接收率和预测类直方图；不能用Us真值计算伪标签准确率。新学生与匹配SOURCE都仅源数据，旧PCD控制具有目标U576祖先，两种数据权限和成本分别报告。

## 自动完整评估

固定六读出：SOURCE_pooled/predicted_ratio、OLD_PCD_pooled/predicted_ratio、SOURCE_US_pooled/predicted_ratio，唯一主最后一项。前四列从fresh f4b5的 SOURCE/PCD 完整预测、密度和四组六τ精确复用。新学生冻结后只在原L6300/OE6134拟合known90/OE32/normalizer，V27000只校准两新六τ；旧GMM/旧τ不更新。

新τ全冻后预测 source NV3150及同开发RX3-2/index34的03-15/23 Q10238（旧4508/四未知5730），包括clean与固定星地；两个Q均经同DURA，并使用原F0默认头，同路径原F0闭集基准重核。源V已是校准角色，非独立已知测试。clean仅冻结测试，无clean Us或target学习。

全6×3条件预测固定后独立CPU scorer连接source NV/clean Q/star Q三truth各一次；Us标签/private truth0、新target U访问0。主报逐TX宏A_K/C_K/R_U/H、原F0闭集损失、完整TX/RX/day/六景、与SOURCE及OLD_PCD的配对与缺类。source known12、cleanRU95/starRU90/CK90/原F0损≤2pp及必需景覆盖须同时满足；主starAK须高于两predicted控制且RU不下降，否则NO_PROMOTION、不扩seed。03-23缺两个旧TX时完整宏指标N/A。无新target UDA/匿名REG，A_N/H_OW=N/A，完整目标ACTIVE。

## 生命周期与成本

单owner root、单模型seed2026092701；新不可覆盖root，cache→train→predict→独立CPUscore自动串联，不因源科学分数低取消已登记测试。每GPU新增≤2、实际free≥4096MiB，不干预他人；SOURCE444只复用，不重复训练。

新增cache生成56700固定received＋G56700恢复；训练额外F0教师56700、父r03 L/OE12434及学生L/OE/Us170496行（具体实际调用全记），新优化444，旧训练成本继承。评估新学生L/OE/V/NV42584＋两Q20476，Q F0/G各20476；新增统计1个源GMM/normalizer与12τ，旧四控制复用。合成烟测、模型加载和缓存成本另列，不以缓存时间冒充部署B1。

逐步按实际记录预算，统计曝光不算新增独立物理样本。非法角色/祖先、Us标签泄漏、冻结状态改变、非有限、输出碰撞或无法合法闭合预测即保留停止；低科学分数完成完整矩阵。代码commit、聚焦负测、一次直接P0P1、单preflight/单archiveSHA/compile、实际PID/CWD/argv/GPU/log增长和独立truth-last收尾。历史Q曝光、单开发RX/seed、global packet history UNKNOWN及正式RX24-18未访问保留。
