# DURA 精确相位配对单阶段恢复对照

- run_id：`20261030-phase1-DURA-source-phasePair-s2026092701-r01`
- group_id：`phase1-dura-source-phase-covariance`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

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
