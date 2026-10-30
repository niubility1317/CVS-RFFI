# PAMF 固定密度拒识边界训练：单模型配对研究

Run `20261031-phase3-density-boundary-s2026092701-r01`。唯一主BOUNDARY_predicted。REPLAY与BOUNDARY两独立r03副本各2220更新、同L/OE索引；原E0两种入选ratio固定。当前没有新性能，完整Phase3目标ACTIVE。

## 科学改变

最近局部协方差和联合特征都未满足跨RX联合目标。本轮检验：直接优化已部署的固定条件密度比，是否比只维持身份/拓扑的源续训更好。它不是此前U实例NT-Xent、普通uniform-OE、另训身份头或新统计复杂化。

两arm共同使用原crossRX且跨day的L监督cross-set KL＋原F0 OE软拓扑KL，各权1，并增加权1的r03点特征锚：0.5×mean_L(||z−z_ref||²/2)+0.5×mean_OE(||z−z_ref||²/2)。z_ref由原冻结r03对同P1源记录一次导出，维持原GMM坐标。该锚是两arm共同条件，不把它的作用单独归因于BOUNDARY。

仅BOUNDARY另加权1的role-balanced拒识BCE。令r(z,c)=log p_L(z|c)−log p_OE(z)，p_L为原预测类15模式各1/15、p_OE原32模式；共享白化Jacobian抵消。c是原F0余弦头作用于缓存u160的预测，不是L真实y或新头。L_B=0.5 mean_L softplus(−r/160)+0.5 mean_OE softplus(r/160)。按维数160缩放固定，不扫描温度/权重/阈值。原11数组通过Torch FP64常量保持梯度到z及完整Sinc/浅中层/PA/ref/signature/投影；GMM无可训练参数，原F0和G均不更新。

原GMM在训练和三模型评估中共用且全程不重fit；原F0/default6提供身份，主只拒识。方法参考[Outlier Exposure](https://arxiv.org/abs/1812.04606)的源辅助异常监督思想，以及[能量OOD论文](https://arxiv.org/abs/2010.03759)直接塑造拒识分数的研究背景；本轮采用上述固定GMM-BCE，并非复现其原损失，也不继承视觉任务性能保证。

## 数据、初始化与预算

两个学生严格从纯源P1 signal r03初始化（code dc01a402582f93e452dbaa3ef59a9fbb44151818，334721=headless232033＋MLP102688），原F0为Tri-SIG-TIME SSL E50/11100/default6/232993，sourcecontract392005。DURA v1.0 R69/266951/E20/4940/reuseTrue/graphTrue/nativeFalse/FP32/TF32off。固定source GMM与原两tau来自已完成class-density code4af0668331682636ee212077b38cd912857a1b31；训练只读其完成身份与models，禁止加载旧tau/calibration或V/NV/Q数组。新评估在全部神经冻结后才读取旧V校准。旧U适配学生、P0或local/joint统计不加载。

训练只使用原cached G10/P1 L6300/OE6134。L六旧设备各1050、每TX×五RX×三day70，有旧TX标签；OE只见outside-six成员和公共RX/day，真实TX/K隐藏。源RX1/3/4/6/8为1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。原Us56700不读，L占原训练L+Us的10%；V27000是额外已知校准标签。全source角色物理互斥，VALIDATED_ONCE句柄与原F0/G/teacher祖先直接复用。

原r03一次对L/OE导出z_ref共12434真实parent-signal行；原F0只对合法缓存u做head-only分类12434行，不运行其source backbone/G。源缓存IDs/metadata/角色严格对齐，私有reference_signal.npz不是部署bank或checkpoint内容。实际CP首步synthetic F0/G/r03各2；随后source与BCE两次no-step合成梯度probe，学生16行、optimizer0，PASS继续真实source。

两arm各fresh AdamW/CENTER_UNITS lr1e−4/wd1e−4/eps1e−8，Sinc坐标倍率1000/eps1e−11/decay0及原physicalclip5；全部BN/dropout eval、buffers固定。L192/OE64，原Sampler seeds2026103017/18，各2220步；REPLAY全完成后BOUNDARY从同原r03重新复制，逐step核L/OE索引配对。每arm L426240/OE142080曝光，双arm学生真实1136640行，共4440新更新；无aug/U/EMA/早停/V选模。新信号特征祖先11100＋2220＋2220=15540、MLP4440，原F0仍11100。原模型、原OE teacher、r03点teacher与GMM保持不变；模块梯度前2步记录，全程finite/physicalnorm/Sincstep/clip记录，不以少数步推广全部非零。

## 冻结后的完整测试

V27000旧六设备各4500：两神经全冻结后，仅为REPLAY/BOUNDARY的pooled/predicted四读出各校准6tau；E0原两tau/cal精确复用。原known12规则为overall＋5RX＋6scene，30cross完整诊断。全部六套tau先冻结，再读sourceNV3150（19-15/19-16/7-1各1050）和两个Q。无新密度/normalizer拟合，不能将新NN学习误写成只改读出或将已发生V阈值校准写为0。

三条件为source_DURA、clean_DURA、star_DURA。sourceV/NV复用原F0身份/E0 z，并新增两学生P1前向共60300；目标Q均为RX24-1(index26)03-15同864物理记录、六旧/三未知各96（576known/288unknown），新同DURA输出共享F0与两学生、E0 z按父缓存逐值复制，核新F0logits/closed。目标新G/F01728、两学生signal3456，eval signal合计63756；eval synthetic G/F02各、三open共6。source/目标同DURA闭集保护基线不变，clean也G。U384/03-08、正式RX24-18和未来query不访问；source与Q日期重合，非严格新day。重复处理不增加物理样本或盲测次数。

G10沿既有MIX01：600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、郊区城区各.5及原六面积概率，receiver2027/scene2026103001/channel2026104001，一次输入unitRMS、输出不再RMS。所有学习IQ为星地P1；clean仅测试，不新增信道或clean参考。

六列×三DURA预测完整后，独立CUDAempty CPU scorer核完整配对training traces、模型/固定state身份、原E0精确列、全部raw/margin后才sourceNV/cleanQ/starQ truth各一次。score不读CP、IQ、teacher缓存值或Utruth，不fit/重tau。主报宏AK/RU/H、CK/同G原F0closed−AK/AUROC/OSCR、全TX/RX/day/scene/42组及主对E0_pred/REPLAY_pred/BOUNDARY_pooled配对。gate为sourceknown12、cleanRU95/starRU90/CK90/损≤2pp/必需主景覆盖，starAK/RU还须同时不低于E0与REPLAY。sourceRU仅诊断，不因源未过取消Q，不改主/扫描/重跑。

sourceNV/Q历史开发曝光、单RX/模型与high未知0/4/2缺19-15保持N/A，不认证六景。无目标适应/发现/REG，AN/HOW为N/A，完整目标不缩小。单主活跃NN834665，三open评估合计活跃1504107；固定state364340B共享，全部tau288B（新192B/旧96B），teacher行cache不部署。实报源/总/增量成本与批处理时长，不冒充B1或多seed确认。

N607唯一owner/不可覆盖run root，单train/eval GPU、每卡新增本任务≤2且实际free≥4096MiB，独立CPUscore；不停止/修改/重启他人任务。精确commit/命令/环境CWD/路径/GPU/产物写experiment.json/report；一次聚焦验证/直接P0P1、单preflight/单archive SHA比较/remotecompile、PID-CWD-argv-GPU-log核实后自动完整测试/报告/Git。非法来源/角色、非有限、输出碰撞或不能合法预测时保留，不partial评分/自动重启；UNKNOWN先核原PID/root，科学负结果不停止健康任务。
