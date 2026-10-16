# F＋跨 RX 同 TX：有界平滑频谱 IQ 实现与身份保护

状态：ANALYZED_SOURCE_ONLY／VERIFIED。25视图及独立完整复算已完成；固定滤波候选因实际IQ频谱保真失败关闭。唯一逐行登记见 [experiment.json](experiment.json)。代码身份由实际Git提交和release_commit.txt/resolved_config记录。

## 唯一假设与固定矩阵

固定公共相位训练已源证伪并关闭，不调角度或换seed救跑。历史 `20261011-phase1-rx-spectral-crossfit-manysig-m1-r01` 的预指定rectangular/DCT1to8报告部分RX相关频谱变化可跨TX/day复现（完整L6300主观测MSE比0.6141、18/18折改善）。这不是纯RX硬件辨识或Ag识别收益；该历史分析包含RX1，不能把其系数带入当前四RX留出契约。旧E200＋40轮建议不沿用，当前真实训练预算仍E20。

唯一新问题是：仅用当前fit四RX重估的平滑变化，能否实现为轻度有限长IQ滤波，且不明显损害冻结默认头的fit源身份识别。通过只支持另行受控训练，不能证明Ag跨RX收益。

| 对象 | 数据 | 固定视图 | 训练／头拟合更新 | 新网络包前向 |
|---|---|---|---:|---:|
| 自有F＋CROSS_RX B00／seed2026092701／fixedfinal20，完整冻结 | fit RX3/4/6/8 的L5040，6TX×3day×4RX×70 | raw＋6RX对×2方向×structured/control＝25 | 0／0 | 126000＋合成smoke2 |

已有合法carrier和缓存复用。按物理ID先排除heldRX1，再做IQ统计、变换或推理；fitID必须完全等于冻结模型训练L5040。U45360、V21600、held RX1原L1260均不用于本次计算，U真值隐藏，target不访问。原缓存可反序列化其他源角色/旧特征但不使用；不复用旧完整L6300频谱系数。

冻结CP来自 `20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01` 的B00，自身scratchE20/4440更新、无祖先、无目标选模，四RX实际角色一致；已有来源结论复用，严格载入后第一步只跑2合成包no-query smoke，PASS直接继续。不初始化新训练、不加载E50/E200或不匹配权重。SIG_TIME三路、统一Sinc、NoExplicitCFO与必要内部补偿、joint160、默认cosine头保持，0新分类器/部署分支/教师。

## 观测、算子和对照

仅用固定矩形窗：FFT256移中，8bin平均成PSD32，按包平均功率归一化，relative floor1e−6取log，再减32维均值，投影正交DCT1to8。18折分别排除一个TX及一个day，只用其余5TX、2day、4RX的40单元拟合RX对比，对排除TX/day的4单元预测。不能事后换到历史效果更好的Hann观察。

全fit4RX平衡单元均值生成一个共同bank，不按每包TX/RX/day选kernel。L标签仅用于统计分组与交叉排除；kernel应用本身没有标签或域参数。6个无序RX对都取正负方向。structured为DCT差，control用预定符号mask[-1,+1,-1,+1,+1,-1,+1,-1]打乱结构，保持每个DCT模能量。先仅衰减、不放大源幅度差到0.5dB上界；生成33点Hann有限核后，匹配control与structured的FIR扰动L2能量，再共同衰减，使4096频率网格上的实际幅度响应在±0.5dB内。所有方向和对照全部报告，不挑最好者。

算子是complex-linear FIR，不含conjugate/image分支、TXIQ失衡或PA非线性。reflect16填充、中心卷积、抵消数值16点延迟，输出仍256点；不roll、剪裁或移动固定参考窗口。无新增CFO、噪声、多径、DAC、剪裁或星地信道。FIR后显式恢复原包RMS，记录实际恢复gain和频谱变化；纯全局gain被去除，频谱形状保留。±0.5dB是RMS恢复前的网格幅度界；恢复后的频谱峰峰形状界1dB不变，不把恢复后绝对gain伪称仍±0.5dB。数字滤波不等于更换真实RX。

同一FIR的循环卷积只作为DSP参考，检查有限边界和频谱实现，不送模型。有限输出与循环参考在内部16:240应一致；固定参考80:160在其中。按完整L累计实际PSD差与参考PSD差的SSE、边界差能量、滤波总变化能量、内部误差、RMS误差及restore gain，所有包参与，不删困难包或低能量差视图。

## 固定判断、成本与停止

全部同时满足才记可实现性/fit身份保护PASS：18折相对MSE<1、至少15折优于零对比，且结构预测SSE优于固定错符号对照；每核相对响应RMSE≤0.35、4096网格幅度≤0.5dB；structured/control核扰动能量相对误差≤1e−8；每视图实际PSD相对RMSE≤0.35、边界差能量/循环滤波变化能量≤0.25、内部相对MSE≤1e−10、包RMS相对误差≤1e−6。12个structured视图分别要求相对raw的fit准确率下降≤0.5pp，且各RX、TX、day下降≤1pp。control全部评分，但不以最好control替代候选。任一失败关闭固定滤波候选，不调上界、核、control、seed或RX救跑。

统计条件无论正负都继续完成所有固定视图，用完整证据解释失败；不自动训练。0训练/0头拟合/0teacher/0As；实际新增126002网络包前向，另120960个有限FIR和120960个循环DSP参考。原CP父成本E20/4440更新/4686678网络包前向单列，不继承新训练。记录CPU墙钟，不用辅助loss或模型健康代替身份准确率。PASS仍不证明Ag、As或联合目标，不启动target。

普通N607账号、CPU两线程、CUDA_VISIBLE_DEVICES为空，不占GPU槽，不修改健康旧任务。owner `codex/root/F-CROSS_RX-spectralIQ-20261016`；CWD `PROJECT/releases/cvs_F_crossRX_spectralIQ_fitL_20261016_r01`；命令 `CVS-RFFI/bin/python -u experiments/cvs_spectral_iq16/runner.py`。唯一不可覆盖run root、输入/输出和log详见experiment.json。一次N607 preflight、一个归档SHA比对、一次远端编译；启动后独立PID/CWD/argv/CPU环境/log增长核实。角色/CP/非有限/输出冲突/状态或成本错误为技术失败，保留产物，无自动重试；科学低性能不停止或改配方。

保存fit观测、18折、24核及期望频响、25视图完整默认头预测/logits、每batch充分统计、完整stdout、resolved/CPsmoke/progress/result/completion，或failure。真实IQ、CP、物理ID、核bank、逐包预测/日志均私有；只代码、配置、登记与汇总报告入Git。全部预测后独立复算accuracy/F1/CM/CE、RX/TX/day、18折、核频响与能量、DSP比例及固定判断；不另拟合诊断头。

本地9个协议/算子负例和5个科学阈值负例PASS；合成可加效应精确恢复、排除TX/day不进入拟合、交互反例拒绝、中心卷积内部一致、控制能量匹配、RMS恢复和零核identity通过。合成本地400个有限滤波/400个循环DSP参考，0真实IQ、0CP、0网络前向、0更新。一次限定P0/P1独立审查PASS，无发现。真实源与默认头结果仍待执行，不能由合成检查替代。


## 实际完整结果

执行commit `132f344af1e22a53beb516b1881689ba42c574a2`，本地与远端分支OID已独立一致。归档单次比对、远端编译、真实CP smoke、运行PID/CWD/argv/CPU环境与日志增长VERIFIED。进程自然退出、completion齐全、failure不存在；无重启或热改。

预登记判定：{"stable_RX": true, "kernel_realization": true, "matched_control": true, "finite_IQ_realization": false, "source_identity": true}。四RX18折MSE比0.577122005，18/18折优于零对比，预测SSE=122.306262798、零对比SSE=211.924448696、错符号SSE=477.064528655。这支持部分共同观测变化，不是纯RX硬件辨识。

| 固定源视图 | 准确率 | 相对raw下降 | 有限PSD相对RMSE | 边界误差能量比 |
|---|---:|---:|---:|---:|
| identity | 99.841270% | 0 pp | — | — |
| structured_RX3_4_minus | 99.841270% | +0.000000 pp | 0.271952 | 0.008137 |
| control_RX3_4_minus | 99.841270% | +0.000000 pp | 0.406231 | 0.004984 |
| structured_RX3_4_plus | 99.861111% | -0.019841 pp | 0.282345 | 0.008321 |
| control_RX3_4_plus | 99.821429% | +0.019841 pp | 0.422909 | 0.004897 |
| structured_RX3_6_minus | 99.841270% | +0.000000 pp | 0.284589 | 0.006901 |
| control_RX3_6_minus | 99.841270% | +0.000000 pp | 0.374422 | 0.004670 |
| structured_RX3_6_plus | 99.880952% | -0.039683 pp | 0.293072 | 0.007124 |
| control_RX3_6_plus | 99.821429% | +0.019841 pp | 0.388058 | 0.004593 |
| structured_RX3_8_minus | 99.821429% | +0.019841 pp | 0.584366 | 0.007474 |
| control_RX3_8_minus | 99.841270% | +0.000000 pp | 0.364259 | 0.006036 |
| structured_RX3_8_plus | 99.861111% | -0.019841 pp | 0.601925 | 0.007773 |
| control_RX3_8_plus | 99.821429% | +0.019841 pp | 0.373926 | 0.006059 |
| structured_RX4_6_minus | 99.841270% | +0.000000 pp | 0.498750 | 0.006803 |
| control_RX4_6_minus | 99.841270% | +0.000000 pp | 0.351647 | 0.005853 |
| structured_RX4_6_plus | 99.821429% | +0.019841 pp | 0.477097 | 0.007124 |
| control_RX4_6_plus | 99.801587% | +0.039683 pp | 0.343489 | 0.006111 |
| structured_RX4_8_minus | 99.861111% | -0.019841 pp | 0.348736 | 0.015304 |
| control_RX4_8_minus | 99.821429% | +0.019841 pp | 0.482577 | 0.008062 |
| structured_RX4_8_plus | 99.821429% | +0.019841 pp | 0.345876 | 0.015553 |
| control_RX4_8_plus | 99.841270% | +0.000000 pp | 0.463763 | 0.008231 |
| structured_RX6_8_minus | 99.861111% | -0.019841 pp | 0.325410 | 0.013414 |
| control_RX6_8_minus | 99.821429% | +0.019841 pp | 0.457971 | 0.007402 |
| structured_RX6_8_plus | 99.821429% | +0.019841 pp | 0.319452 | 0.013441 |
| control_RX6_8_plus | 99.841270% | +0.000000 pp | 0.441629 | 0.007495 |

全部126000默认头决策及logits/CE/CM/F1、RX/TX/day和1800个视图单元覆盖已独立重算；原始fit准确率99.841270%，structured最大下降0.019841pp，原身份保护阈值通过。全部18折和24个核的FFT频响/扰动能量、batch充分统计和科学判断亦独立复算。

核响应相对RMSE最大0.185859803，能量配对最大相对误差6.27733912e-16。实际有限PSD相对RMSE范围0.271952222至0.601924837，部分超过原0.35；不得丢弃失败方向或只保留control较好的核。边界差能量比范围0.004593413至0.015552916，低于0.25；不是边界能量主导。内部相对MSE最大1.30801702e-31，RMS恢复误差最大1.60675746e-08。

已证实：当前四RX存在可跨TX/day预测的部分频谱差异；33tap核可满足幅度和控制能量约束，源fit默认头身份基本保持。被否定：这组固定有限IQ输出可同时满足原定实际PSD保真条件。不能由source身份不降推导Ag有效。固定候选关闭，不改核/幅度/control/阈值/seed/RX救跑，不启动该滤波的E20训练。

实际新增126002个NN包前向，120960有限FIR和120960循环DSP参考，0优化更新/头拟合/教师/As，模型state_dict不变，CPU墙钟74.685727秒，完整stdout28行。parentE20/4440更新/4686678NN与本地400有限FIR/400循环参考另列；0heldRX1/U/V/target计算。分析曾仅在JSON导出NumPy布尔类型时报错，已修导出类型后用同一批已完成产物复算，没有重跑实验或改变判据。

下一唯一拟检验回到已有普通IQ管线：实际启用的PA-normal是否在触发样本上引入可避免的身份错误。最小源实验用同一原生增强抽样，对比raw、完整ordinary、只旁路PA的ordinary，保存实际输入差异和触发标志，用冻结默认头配对读出；不解封U、不读held/V/target、不拟合诊断头。先读取实际调用并固定抽样/阈值/成本，再登记运行。若旁路PA在真正受影响包上不恢复身份或随机流不匹配，就不以该证据启动训练。当前仅提出，未实现或启动；即使源配对改善，也还须匹配E20留出clean检验，不能称Ag完成。
