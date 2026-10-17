# 固定IQ重排：原生星地传播前后顺序对照

状态：ANALYZED，源域顺序对照完成，未训练、未访问 target。上一轮后置Ag(As(x))在源urban显著退化，保持FAIL。新实验不改变算子/窗口/周期/排列群/phasefit/概率/seed，只检验原目标明确要求的不可交换问题：As(Ag(x))是否仍能保留原As(x)。当前并未找到有效Ag或联合提升，目标ACTIVE。

## 固定方法与判断

直接复用不可变P30发布commit7e8dc09的四周期算子，仍是[80:160]四个20点周期及原pooled lag20线性相位运输；24排列均匀私有流202610173001。对每个物理ID/draw/scene使用与原post-order完全相同的排列，只因输入位置不同而由同一算法计算q；不换窗口/相位估计或调小幅度。原F＋跨RX同TX、三SIG_TIME/统一Sinc/joint160/cos30/NoExplicitCFO与内部补偿固定，两个fullsource scratchFINAL30/6660学生默认头冻结。无新loss、头、部署结构、训练或EMA更新。

主矩阵V1800，全部5源RX1/3/4/6/8×3day×6TX各20包；两个原source信道draw、low_suburban/mid_urban/low_urban。原L6300/U56700隐藏TX/V27000合同不变，不留训练RX、不切L_dev。先strict own_model来源/角色核验和两真实CP共4合成包no-query smoke，再打开源缓存；可能反序列化未用L/旧特征，实际只消费V1800，U IQ/真值和target不打开。

每个物理源观测先按原P24 normalize_input及原nativeResidualChannel重放As(x)，必须对旧full缓存在2e−6内一致；增强输入按同一既有As输入归一化步骤处理后，用相同cfg/seed/receiver运行原nativeAs(Ag(x))。不新增As输出RMS。固定原sourcechannel种子202610172402和sourceV/residual_diag24/draw*/scene/ID命名流，receiver种子2026092701按RX/day/eq且跨TX稳定。原geometry/真实hardware/遮挡state/传播kernel/新增噪声方差必须同一，EQoff G=1，不反演未知TX。quality/补偿/AGC自然按原native依赖重算，记录而不人为强制等于原输入；新增噪声比不等于总SNR。formalIQ/SNR/boundary完全不变。

原模型和post-order已有logits只读复用，不重复其NN或正式测试；每个新模型只预测前置10800个源卫星观察。源clean保护沿用同一算子的P30固定结果，两模型clean/RX/TX/保留率/相干度/非平凡变化均原已通过，后置urbanFAIL不改写。B01前置两个urban×两个draw必须各≥原As−.5pp（1e−10pp数值容差），否则放弃这条组合路线，不换RX/seed/phasefit/窗口/概率或预算救跑。B00仅为未训As的读出控制，不能用它的分数代替B01条件。源PASS也仅说明前置兼容性，不证明Ag跨RX效果、正式提升或跨seed。

## 成本、路径与保护

新增NN=2×(2smoke＋10800sat)=21604；新增IQ增强10800；nativeprocess21600（10800原输入replay＋10800前置输入），unique潜在draw10800沿用旧seed，不称物理计算免费。0新训练/头拟合/teacher/optimizer/EMA更新。两父各E30/6660/5407922NN及P30源25204NN成本另列，不计作本轮新增或忽略。共享physical观察只生成一次供两个模型使用。

每包保存原/新quality、lock/state、频率补偿、signal_to_added_noise、AGC/限幅、kernel长度/能量/数值延迟、同noisevariance及preAGC noise差、重放误差和原排列，保存全部新IQ与默认头logits供独立源CM/配对复算。训练曲线不新增，不能把诊断完成称训练收敛。

普通N607账号/CPU2/空CUDA，唯一owner/root，独占run/outputroot，精确CP/配置、命令/CWD/GPU/paths由experiment.json维护。一次archive传输SHA比较和remotecompile，首步真实CP no-query smoke PASS即继续；一次独立PID/start/CWD/argv/CPU/loggrowth或自然终态读回。role/provenance/replay/kernel/G1/hardware/noisevariance/非有限/模型状态/NN/输出冲突等技术失败保留本run partial/log、不自动重试，不修改健康旧任务；科学FAIL也完成两模型全部矩阵。

本地3项focusedtests及compile PASS：原生重放差/G1/形状/非有限/错误kernel反例，恰好−.5pp浮点边界及越界/坏矩阵反例，priorclean guard独立于已失败urban且不可缺项。0真实IQ/CP/NN。唯一限定新增P31接线P0/P1审查PASS，不重审不可变旧算子。普通账户/CPU2/输出无冲突/实际engine句柄preflight VERIFIED，不新增签名、hash链、报告审批或重验数据。

整体目标仍未完成：Ag跨RX clean净收益、正式low>60%、suburban接近clean及跨seed联合收益均未证明。本轮只源反证，不自动target或扩seed，后续训练须另行固定匹配矩阵和实际输入路径。

## 已核实结果

实际代码 commit 5692278a88c025f18d21932087ede41a525be9a6。两真实 FINAL30 默认头共 4 个合成包 smoke PASS 后继续。一次发布 archive 比较、远端编译、PID570422/start_ticks148293505/CWD/argv/CPU2/空CUDA/log增长及自然终态 VERIFIED，stderr为空。

B01 同一源V1800、两个信道draw池化：原As(x) / 后置Ag(As(x)) / 前置As(Ag(x))准确率分别为 low_suburban 90.444444% / 85.972222% / 89.972222%，mid_urban 76.305556% / 72.666667% / 76.027778%，low_urban 62.555556% / 59.916667% / 62.305556%。前置相对原As分别−0.472222、−0.277778、−0.250000个百分点。四urban单元为[[-.5,-.277778],[-.055556,-.222222]]pp，按原含边界判据4/4通过；后置仍FAIL，不改写为成功。clean仅复用前轮同算子结果，两模型均−.111111pp、clean正确集合保持均>99%；并非新增训练clean结果。

10800个原输入原生重放最大绝对差为0；相同物理状态、传播核、RX session、新增噪声方差和G=1检查通过。15个RX/day session；lock变化0；preAGC噪声最大相对差3.091536e−16。接收质量、补偿和AGC按原流程重算；本轮保存AGC增益，未保存逐包agc_limited布尔项，该项为明确缺项，不据此宣称无AGC限幅。不能将新增噪声比等同总SNR，也不能从顺序效应唯一归因于某一内部机制。

新增native calls21600、IQ增强10800、NN21604、0训练/头拟合/教师更新，墙钟48.543259秒；独立复算342个CM、96个分层、10800个private排列和10800条物理记录，NN/IQ/CP全为0、墙钟1.659808秒。实际图E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_placement_E30_20261017/analysis_r01/cycle_order_placement_source_comparison.png及同名PDF，已查看PNG。CSV/summary/completion保存同目录。

结论只限于固定算子的前置源域兼容性，Ag泛化价值、联合训练净收益、正式low>60%及跨seed均未证明。原正式E30 ordinary拼接参考clean82.797024%/low53.137500%不变，source low62.305556%不能替代正式目标。无新训练曲线。下一步才是匹配原始secondary输入的scratchE30四因子训练，必须清楚区分旧ordinary-secondary正式参考，不能把raw输入变化冒称Ag收益。
