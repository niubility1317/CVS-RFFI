# F＋跨 RX 同 TX：有界平滑频谱 IQ 实现与身份保护

状态：PLANNED；尚未运行或取得源身份结果。唯一逐行登记见 [experiment.json](experiment.json)。代码身份由实际Git提交和release_commit.txt/resolved_config记录。

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
