# 恢复误差的周期与周期间残差：固定源端等能量检验

- run_id：`20261031-diagnostic-DURA-source-period20-utility-s2026092701-r01`
- group_id：`phase1-dura-source-period20-error-utility`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用defaultDURA/source512x01/20的nativeIQ/F缓存，固定[80,160)四个20点周期的误差正交分解；主残差oracle与同修正能量全窗控制，0G/优化/TX-V-target。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整源诊断结果与边界见analysis.md/results.json；无正式target预测。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 固定源误差分量检验

本轮只读defaultDURA既有源512条（128L_s/384U_s）×原缓存01/20的nativeIQ及同F50/source4df7的clean/native特征和logits，不加载G或使用V/target/TX真值。固定坐标沿项目已有公开响应路径的[80,160)、四个20点周期；不拟合period、不补偿CFO，也不声称整个256点都严格周期化或残差属于纯TX/RX噪声。

设E=sourceclean−native，只在80点内对四周期取均值形成正交投影P，R=window(E)−P。分别理想修正native+P与native+R。主假设仅R；对照native+||R||/||E||×E，在全窗沿最近clean方向作完全相同L2半径修正，零误差保持。P/R窗口外逐位不变，无额外RMS归一化；P/R能量和仅等于窗口误差能量，不是全窗误差。实际FP32修正范数差容忍为2e−6×max(cleanIQ范数,1)，完整数值保留。

双视图R修正与clean F选择一致数须各较native至少+4/512、较同能量全窗至少+2，五RX均不负、clean场景不负、其他各景最多−1、总体F余弦误差不高于native。periodic均值、day/role和能量仅描述，不改主分支、窗口、period或阈值。pairedclean是部署不可得的理想参照；Fclean本身可能判断错TX，源训练池也不是独立V，因此不把PASS当目标身份收益、可恢复性、硬件因果或物理上限。PASS最多支持另登记学习假设，不自动训练G。

与旧20261018 repeatShrink检验分开：旧版对不同E20识别器/sourceV做.1幅度带窗IQ收缩保护，本版问固定恢复误差的成分效用，并非重跑旧增强。旧尾部、Q频段、相位/RX投影等FAIL保持。新设置在读取本轮SIGfastTrain正式结果前已登记，不据该结果改设置。

3项聚焦检查（dense80x80正交投影/支撑域、配对P/R与等能量全窗/零误差、双view与RX保护判定）及actual entry/help/CPU-path负测先于source通过，唯一P0/P1 PASS。首步真实2非clean L_s的clean/native F缓存logits与实际F对齐（rtol/atol3e−5），F4/0G，PASS直接跑完两视图三新case；主F3072，0优化/TX/V/target/新F缓存，F全state和grad保持。F50/50轮11100步知识与旧defaultG/缓存历史成本披露，属于F-informed源诊断。

CPU2/空CUDA，独立不可覆盖输出；18组/视图×4case×2=144组case统计、全部1024×3新IQ与dense投影/等能量控制、Fchoice/余弦/NMSE和判定由独立NumPy复算，0新增NN。完整IQ/F向量仅留私有NPZ，Git只保存汇总。一次N607 preflight/单归档SHA/compile/实际PID-CWD-argv-空CUDA-loggrowth与完成产物读回。当前SIGfastTrain按其原固定30轮/七景结束，其他owner/defaultDURA保持，完整目标ACTIVE。


## 完整源结果

VERIFIED / ANALYZED_SOURCE_ONLY，主假设FAIL。固定defaultDURA/source512（128L/384U）×缓存01/20、1024次观测/512物理记录，0G加载/前向、0优化/TX/V/target。固定窗口80:160/四个20点周期，将pairedclean−native误差分为周期均值P和窗口残差R；主R与同修正L2的全窗最近clean控制比较。R的clean F选择一致数485/498，相对native486/498净−1/0、相对同能量全窗496/504净−11/−6；两view额外效用失败，第一view源RX保护也失败，其他clean/scene/Fcos条件通过。periodic修正492/501净+6/+3仅描述，修正能量不同且没有其单独同能量控制，不换主/窗口/period/阈值或据此加训练loss。clean窗口周期间残差能量约6.03%，窗口恢复误差中残差占44.14%/56.69%，高相对误差不自动意味着更高身份效用。CPU2主循环3.079785秒、F完整输入3072+真实2非cleanLs clean/native缓存logits smoke4，F全state保持、grad为空。独立dense80×80投影/全部1024×3IQ与同能量/窗口外实际bits/原cache配对、36组144case统计和判定复算，最大IQ公式差1.338257e−7、指标差0、0新增NN。复核中FP32先减再升精度与FP64先升再减的标量分别重算并保留原2e−6约束，无模型重跑或判据变化。F50/source4df7/50轮11100步知识及旧defaultG/缓存历史成本披露；这是源训练池Fclean选择代理，非真TX、RX噪声/硬件因果、可部署修复或物理上限。默认DURA及完整目标ACTIVE保持。

执行58da3da07023b93e55667f924e07af5db9ff745c，PID1915074的实际CWD/argv/空CUDA/线程及日志增长已独立核实，完整产物/自然退出读回；不重复执行。详见[完整分析](analysis.md)、[源图](source_utility.png)和independent_check.json。
