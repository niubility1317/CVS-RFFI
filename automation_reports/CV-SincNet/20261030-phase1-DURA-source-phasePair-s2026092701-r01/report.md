# DURA 精确相位配对单阶段恢复对照

- run_id：`20261030-phase1-DURA-source-phasePair-s2026092701-r01`
- group_id：`phase1-dura-source-phase-covariance`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

源clean/received同旋转，真实宽线性action共轭变换；单前向从零E20/4940，F-free，自动全七景truth-last。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 方法差异、证据与固定验收

本版只在固定 noVar E20 对照上加入源配对的全局相位坐标变换。每次访问独立抽 theta∼Uniform[-π,π]，clean 与 received 同乘 R=exp(i theta)；真实源 action 的 FIR 核和 main 保持，image 乘 R²。它满足 A₂=R A R⁻¹，残差同旋转、功率和方差保持。source 原缓存本就含 15% clean，数量与角色均不变；曾考虑补充 clean 混合的思路在实际缓存核实后取消，没有实现或启动。

仍从零单阶段 20 轮/4940 更新、每批一次 G 前向；没有 F 特征、直接 TX 标签、学习教师、V 或目标训练，不继承任何恢复器状态。F50/defaulthead 只用于真实无 query smoke 和冻结评测。固定最终 checkpoint 自动完成 source1800×7 与 target168000×7，各角色全部七景预测后才由独立 CPU scorer 读取真值。正式结果不得回流角度、系数、轮次或候选选择。

数学恒等式与实际 smoke 只证明配对实现，不能保证身份保真或跨 RX 泛化。与已有 [Wireless Fingerprinting via Deep Learning: The Impact of Confounding Factors](https://arxiv.org/abs/2002.10791) 的关系限于检验干扰变化这一研究动机，不挪用该论文的精度或方法权限。逐景距同一 raw clean≤3pp、clean 保持与 R17逐景1pp容忍分别报告；默认DURA保持。记录实际旋转曝光/随机数复算、完整20轮曲线、126目标分组以及G训练/全流程成本，不能把单阶段或源分数当作同质量加速。


本地验证：8项针对性检查通过，覆盖非交换共轭分量、真实伴随、噪声/功率、旋转复合与零旋转、缓存不变、随机数隔离、单前向和原合同拒绝。唯一独立P0/P1审查PASS，无阻断缺陷；N607只读preflight已核实身份/路径与实际显存。远端真实checkpoint/source2Ls smoke由launcher首先执行，PASS后立即继续。


## 已核实启动

VERIFIED / RUNNING。执行commit `028be06be1b7009867790b82ea7522de8b521dc2`，8项聚焦检查与唯一独立P0/P1审查PASS；一次N607 preflight、单归档两端SHA和远端compile通过。控制器PID797107、GPU7 worker PID797202实际存活，独立核实release CWD、完整命令、GPU绑定与训练日志增长。source2Ls实际旋转/单G目标/有限非零梯度/严格checkpoint重载/同F默认头no-query smoke PASS。残差旋转最大误差4.8054801e-07，相对样本功率最大误差2.82035245e-07；GPU训练已重建独立phase RNG。

当前E1/20、247/4940更新，累计60.167秒。F、TX标签、学习教师和初始化checkpoint训练读取均0。固定E20最终轮次后自动全七景；尚无正式成绩，其他owner任务保持。


## 完整训练与正式七景结果

VERIFIED / ANALYZED。完整20条epoch/4940更新，L126000/U1134000曝光，从零单阶段、0恢复器祖先/F指导/神经教师/直接TX/V/target训练。独立源端CPU重放全部4940次角度调用，1260000条旋转的8箱分布、角度总和及平方和一致；全部L/U分别126000/1134000，原缓存clean实际占比14.974365%，未改写为20%。真实source2Ls相位/残差/功率、完整梯度、G重载与原F默认头no-query smoke通过，训练重建phase RNG。执行commit028be06be1b7009867790b82ea7522de8b521dc2。

source1800×7和target168000×7全部固定后各自独立CPU truth-last；再次独立复算126个总体/TX/RX/day、MacroF1及修复/损伤计数一致。完整日志无Traceback/OOM/数值失败，控制器与worker自然退出。基准此前暴露，不声称新的独立盲测或多seed复现。

|场景|准确率%|相对noVar E20 pp|相对默认DURA pp|距raw clean pp|净纠错（相对raw同景）|
|---|---:|---:|---:|---:|---:|
|clean|83.858929|-0.883333|-0.819048|0.817262|-1373|
|practical_high|78.670833|-2.233333|-4.154167|6.005357|+63736|
|practical_mid|75.448810|-3.957738|-6.111905|9.227381|+59694|
|practical_low_suburban|72.675595|-5.193452|-7.382738|12.000595|+56525|
|practical_high_urban|73.790476|-6.208333|-7.955357|10.885714|+57275|
|practical_mid_urban|55.311310|-19.081548|-20.175000|29.364881|+34991|
|practical_low_urban|43.226786|-25.244048|-25.663095|41.449405|+21718|

六景3pp验收=False，clean不低于raw=False，原R17逐景最多下降1pp容忍=False。默认DURA保持，结果不回流角度分布、轮次、父权重或目标分组权重；没有选择性重跑。

G训练循环及保存1018.048秒，完整G训练流程1019.345秒/16.989分钟；控制器启动至完整分析1329.381秒/22.156分钟，目标七景预测279.967秒。原模拟缓存初次构建成本另计，未把共享GPU跨时段耗时差归因于单一方法。仍266951参数/2prior/4CG/27FFT，推理不旋转/平均/读取RX元数据，无额外推理分支；同卡交替G/G+F测量见matched_profile。

源训练误差对应旋转后的clean/received参考，源V与目标测试均用原固定未旋转输入。预定1800条源V面板结果见source_V_comparison，不扩展为整个V或TX细节保真证明。数学配对一致与训练收敛不是正式性能保证。


## 同行解释与边界

精确相位配对在六个星地场景中有0/6景高于固定noVar E20对照，完整差异逐景保留。预定1800条源V面板的low urban从83.000000%变为51.555556%。六景3pp验收=False、clean保持=False；源拟合、数学配对和随机数重放与正式性能分别判断。

这项实验只检验独立均匀全局相位配对。它不证明某个RX偏差是因果，也不外推所有复数等变架构或其他相位方法的结果。固定最后20轮、原物理数据/输入/分类头/七景与独立评分保持，没有用正式结果调整角度、采样或轮次。默认DURA保持，尚无同质量加速结论时不能仅凭单阶段声称达成效率目标。

另附source_phase_semantics.md/json：同源完整缓存元数据中的未锁定访问为129109/1260000（10.246746%）。简化随机相位模型给出了“配对正确但单条观测逆问题仍有歧义”的反例，作为解释假设；未将简化误差或未锁定比例外推为实际source/target下界。该说明由源端退化和原生代码形成，不按目标分组设计下一损失，未启动新训练。
