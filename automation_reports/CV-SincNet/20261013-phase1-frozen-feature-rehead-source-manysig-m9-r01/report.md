# joint160冻结特征与普通cosine换头源域诊断

- run_id：`20261013-phase1-frozen-feature-rehead-source-manysig-m9-r01`
- group_id：`transferable-feature-source-rehead-20261013`；类别：`diagnostic`；阶段：`phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

逐行丢弃旧类别W，完整保留norm/gain/reference投影；相同源L拟合普通cosine W、V评价；独立固定少样本跨RX头拟合诊断。骨干0更新，无U/target/Phase2读取

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

最小预登记

候选：全部原9个终轮特征，无目标选择。代码提交由本报告同次Git commit确定，release_commit.txt记录；父执行43bd20494aa470568a68553ce664dc76811b9707，评估恢复9ea84c4d22076532dd613a28d1b2da3988591feb。

命令 python -u -m experiments.cvs_source_transfer_probe.dispatch；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_source_transfer_probe_20261013_r01，N607普通账号/CVS-RFFI Python。输出 /home/szu2070436088/2510044040/CV-SincNet/runs/20261013-phase1-frozen-feature-rehead-source-manysig-m9-r01，唯一owner codex/root/source-transfer-probe-20261013，共享GPU容量锁总进程≤3，free≥12000MB。

输入权限：原source config/实际角色和终轮freeze逐行复用。载入nocfo preparation仅取同契约L/V缓存，闭合U、不继承其模型；V从不参与头拟合/骨干更新/源选模。真实checkpoint无query烟雾为preflight首步，PASS后立即启动。

矩阵：每row原头、全L随机初始化普通cosine头、全L原型；5RX×3support draw×K5/10的普通cosine头。W6x160，无bias、scale30、eps1e-4、Adam .01/WD0、20轮；全L每轮25batch，共500更新；少样本每轮1batch共20更新。骨干50轮固定后0增量更新。旧头权重在导出副本中实际删除，norm/gain/reference完整继承。

停止规则：checkpoint或数据角色不符、旧W残留、真实source IQ特征不完全相等、非有限loss/gradient、冻结参数变化、输出碰撞或闭环缺项停止新调度，健康其他行完成，保留产物。无性能stop/promotion；不读取目标。

预期：297项readout（33/row）、279个独立新头、实际source feature exact equality、source物理support ID、每头20轮完整loss序列、seed与support波动、混淆矩阵和同row分析。该诊断未进行新类或Phase2测试，不能据此声明旧新统一识别。

独立P0/P1审查：纯特征接口一次PASS；新源域诊断输入/拟合/闭环一次PASS，无重复审查。focused refit checks14 negatives PASS、三臂feature synthetic精确一致/旧W删除/默认分类/每包独立/梯度与10 negatives PASS。

兼容性直接P1修复：IdentityOnlyCVS仅在ResidualPhysicalHead下走feature_only，native多输出头保留旧return_aux特征。focused真实构造native/residual的特征精确一致PASS，一次限定复审PASS。
