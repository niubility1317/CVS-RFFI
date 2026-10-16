# 重复差残差相关性：零更新源域检验

当前训练底座保持 F＋跨 RX 同 TX 对比学习、E20、默认 cosine 头联合训练。本轮 G0 不建立或加载模型，0 个 checkpoint、网络前向、优化更新和目标访问，不改变任何 IQ 或正式 practical 条件。

上一轮课程已完整结束，源 RX6 的 C2−C1 clean −0.277778 pp、urban +0.462963 pp、low +0.055556 pp；相对 C0 clean −3.925926 pp，固定候选关闭。新的检验不调课程参数，也不重新训练已关闭的 WHITE30、RXMIX、相位或包络候选。

唯一假设：合法源 IQ 的固定重复差中存在跨 TX/day 可重复的 RX 相关短时结构，可为有色观测扰动提供先验。残差可以保留 TX 瞬态、传播和已有处理，不能先称为纯接收噪声。统计阳性也不证明身份保持或泛化。

只使用现有合法缓存中、与当前 C0 RX1 留出合同完全一致的 fit RX3/4/6/8 L5040；6TX×4RX×3day，每格70包。RX1 IQ 在取值前排除；U/V 不参与统计，缓存反序列化会带入未使用的其他源角色对象，如实披露。复用已有数据，不另建 hash 或验证链。

固定原256点中 `[80,160)` 四个20点周期。由三个相邻周期的 pooled 复相关得到一个单位相位 q，不拟合幅度。构造两组不重叠差 `(cycle1-q cycle0)/sqrt2` 和 `(cycle3-q cycle2)/sqrt2`，只在各20点差内计算复 lag1 归一化相关，不跨两组边界连接，不去均值。零残差等使 rho 未定义时保留包、存零哨兵和 invalid 标志，任何 invalid 都使候选失败。

主描述子只有复 lag1 相关的实、虚两维。每 TX/RX/day 取均值，减去同 TX/day 的四 RX 均值，得到 RX 关联对比。18 折同时留一个 TX 和一个 day，用其他5TX×2day估计四 RX 对比。零对比为主参照，并保留3个循环错配 RX 对照；无拟合分类器或身份头。

以下条件全部满足才通过：全部5040包 rho 有定义；重复相干度 q10≥0.8；四 RX 的残差/周期功率比中位数均≤0.1；RX对比风险聚合下降≥10%、至少14/18折正、四 RX 风险增益均非负；超过最佳错配 RX 至少2个百分点；两组 TX 的中心 RX profile 余弦≥0.8；RX profile 绝对 rho 中位数≥0.05、RX对间 profile 距离 p95≥0.05。风险下降不是识别准确率。

FAIL 即关闭固定重复差 lag1 着色依据，不改周期、窗口、lag、统计量、TX/day/RX 或阈值补救。PASS 只支持另行预登记固定功率、匹配白噪声控制的 IQ 训练；不自动启动，不扩目标或宣称 Ag 成立。未来仍需默认头实际 E20 身份收益和 B00/B10/B01/B11 交互证据。

本轮与已关闭的周期相对参考/正交补预训练不同：不预测任何 residual 或拟合 decoder，不强迫骨干学习残差，不以该残差替代部署特征。此前残差身份关系弱的负结果继续保留，正是不能将它当作 TX 目标的原因。

缓变增益斜坡曾在源数据得到 RX 关联风险下降，但线性 R² 仅0.303317，未达0.8，已关闭；本轮不以换窗口重开该路线。公共相位、观察原点、分数延迟、频偏、常数增益、DC 和谱响应等已登记结果按原判据保留。

运行计划为普通 N607 账户、CPU2线程、CUDA隐藏、唯一不可覆盖 run/output root、一次发布和独立回读。实际命令、路径、owner、stop 和预期 artifacts 由同 run experiment.json 维护。无模型的 G0 不加载无关 checkpoint 来凑 smoke。科学失败不做技术重试。

实际结果：**VERIFIED／ANALYZED，科学判据 FAIL**。执行代码为 `3812d3825cdf08546b4e37e8607559d34d2b6928`，唯一 CPU row 自然完成，0 技术 failure／训练重试／额外模型。

5040 个合法 fitL 包全部保留，72 个 TX×RX×day 单元各 70 包。独立 NumPy 读取保存的描述子、重新聚合全部单元并逐折重算 18 个 leave-TX-AND-day 结果，全部与原报告相符；没有重新访问 IQ 或 checkpoint。

|指标|实际值|预定条件|
|---|---:|---:|
|有效 rho 包|5040/5040|全部有效|
|重复相干度 q10|0.803834655|≥0.8|
|RX3 残差功率比中位数|10.275181%|≤10%：失败|
|RX4 残差功率比中位数|0.181328%|≤10%|
|RX6 残差功率比中位数|0.369949%|≤10%|
|RX8 残差功率比中位数|9.320242%|≤10%|
|RX 对比平方风险聚合下降|62.523108%|≥10%|
|正增益折|15/18|≥14/18|
|TX 两半 profile 余弦|0.953371987|≥0.8|

风险下降和 profile 余弦描述 RX 关联统计的可重复性，不是默认头识别准确率。四 RX 风险增益分别为 69.759840／49.379409／56.556633／81.336763%；但 TX4 的三个留出日期均为负，不能用总体风险下降掩盖这三折。残差可能混入 TX 瞬态、传播或既有处理。失败不证明 RX 无有色噪声，也不证明所有有色扰动必然无效；它关闭的是本次固定重复差估计可直接支撑轻度噪声注入的依据。按预登记不改窗口、period、lag、统计量、TX/day/RX 或阈值救援，不自动训练 colored-versus-white。

源统计 wall=1.689139605 s，CPU2 线程、CUDA 隐藏；新模型、E20 预算、网络前向、优化更新、checkpoint 和目标访问均为 0。离线复算与绘图另行完成，未统一计时；不存在本轮训练曲线、默认头性能增量或跨模型 seed 结果。

发布恢复：第一次系统 SCP 客户端 exit255，尚未调用远端发布步骤。独立检查确认同名远端归档、release 和 run 均不存在后，改用已有可用的 Git SSH/SCP 原生程序，以 shell=false 恢复同一归档；未重建归档、未改 SSH 配置、未使用管理员或 WSL。远端单次 archive SHA 比较、compile、普通账户与主机预检均 VERIFIED。唯一 dispatcher 在首次回读前已自然结束（1.689 s）；新 dispatcher.log 为 527 字节、completion 齐全，无 PID 存活；不为补采运行态而重跑。execution.json 的 RUNNING 是启动时快照，terminal 以 completion 和独立进程回读为准。

本地测试：17 协议负例、5433 次有效合成描述子求值（含完整 5040 个合成波形）、disjoint 边界／RMS 不变／undefined 保留与失败／白残差和 TX 混杂及坏质量阴性通过；一次限定独立 P0/P1 静态审查 PASS。真实 checkpoint smoke 不适用：此 G0 无模型，也不为形式要求加载无关权重。

已生成并查看[四面板源统计图](E:/type10-7/local_artifacts/cvs_source_repeatcorr_20261016/repeatcorr_source_results.png)，[PDF](E:/type10-7/local_artifacts/cvs_source_repeatcorr_20261016/repeatcorr_source_results.pdf)、[18 折表](E:/type10-7/local_artifacts/cvs_source_repeatcorr_20261016/folds.csv)、[四 RX 表](E:/type10-7/local_artifacts/cvs_source_repeatcorr_20261016/RX_statistics.csv)与[独立复算](E:/type10-7/local_artifacts/cvs_source_repeatcorr_20261016/independent_recount.json)均为实际文件。只读保存的源派生统计，不含新 IQ／模型访问；原 NPZ、日志和图表保持私有产物，不提交 Git。

这轮关闭一个噪声先验候选，没有缓解任何已实证的 urban 退化，也没有提高默认分类头或产生 clean 代价数值。当前最可靠的训练证据仍是完整 practical 配对 CE 对 LOS/PARTIAL/locked 的适应收益；BLOCKED/unlocked、clean 保护、low>60% 和跨 seed 联合收益仍未解决。目标 active，未把预算耗尽或统计检验完成写成收敛。
