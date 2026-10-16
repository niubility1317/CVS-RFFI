# 联合增强活动基线：F＋跨 RX 同 TX 对比学习

当前没有通过晋级的新增 A_g：有界公共相位E20已完整源证伪并关闭（clean −3.968254pp，六practical均负）。下一唯一拟检验有限长平滑频谱IQ滤波的可实现性和身份保护，基于已有固定观测的源统计；先在fit4RX重新估计，尚未实现/登记/启动。
用户于 2026-10-16 指定，增强前参照更换为 **F＋跨 RX 同 TX 对比学习**。该选择覆盖本对话旧的 F-only 参照；训练预算仍为 E20，没有延长到 E50 的新授权。唯一活动配方见 [配置](../configs/cvs_joint_augmentation_F_crossrx_e20.json)。两种子源 B00/B01 已完整分析；独立seed02因clean下降13.095238pp未通过，固定75/25候选关闭。正式完整源四因子尚未训练。

已核实的来源是 `20261016-phase1-F-supcon-e50-clean-manysig-m9-r02` 的 `CROSS_RX`，执行 commit `4df7b94d0fd8ce29056bd8236364e45afa92e07f`。已读取完整报告、最终分析、实际 design/model/mechanism/source/observer，确认实际原生 extra 格式为 `(domain, metadata)`；不复用 r01 的错误读取方式。

保留 F 三路 SIG_TIME、统一完整 Sinc 修正、真实 joint160、默认 cosine 分类头联合训练、RDrop alpha1、LS0.01、已有普通增强和原隐藏 U 规则。浅中层 `time_down`、`t2.pool` 的同 TX 异 RX 正对只作用于原 L 的两次既有 RDrop 前向；同 TX 同 RX 不作负对，不增加网络前向或部署参数。温度 0.2，最大权重 0.1；E1 至 E5 为零、E6 至 E15 线性启用，E16 至 E20 保持 0.1。不把对比学习或 RDrop 称为 IQ 泛化增强 A_g。

E20 是从零训练 13 轮 L＋7 轮原 U，4440 次实际更新；预期对比调用 4440 次、非零权重 3330 次、层位／前向目标 13320 项。所有匹配候选用同 seed 初始化规则和同真实物理数据契约。原 E50 权重不用于新 E20 或内部开发折初始化。

正式 B00/B10/B01/B11 均包含该基线。B00 的“无新增 A_g”仍保留四组共有的旧普通增强，不称为无增强。对比目标保持在共有原 L RDrop 对，不因卫星视图或新增 A_g 隐式扩展到其他视图或 U；所有额外视图均在发布前固定损失尺度和对应前向控制。

正式完整源矩阵使用 L6300/U56700/V27000。若先做 RX1 源留出证伪，则单独登记 L5040/U45360 训练、RX1 原 L1260 检验及已见 V21600 健康诊断；该契约不能冒充 L5580/Ldev720 或完整 L6300。旧 F-only 的 RX1 B00 不作为新 CROSS_RX 的匹配参照，各 cell 从零建立，不继承看过开发包的父权重。

历史 E50 的目标 clean 已暴露，只能说明用户指定配方的来源；不在此据其目标分数重选温度、权重、seed 或启用日程，不宣称新独立盲测。E20 clean 和六 practical 的实际收益都待冻结后的独立评分。正式预测、received IQ、测试信道和真值隔离保持原协议。

本轮包内缓变增益源检验已完成，固定平滑斜坡假设失败：RX 风险降低 63.164130%，15/18 折改善，但包内线性 R² 中位数只有 0.303317，未达到预定 0.8。源统计不依赖模型权重，结果保留；未开展增益增强训练，不调整阈值救跑。

当前唯一候选是 A_s 的监督份额组织：75% 原 L 监督、25% 完整 practical。已实现并登记 `20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01`，B00/B01 均为新的 F＋CROSS_RX/E20，旧 F-only B00 不复用。每组 raw.5＋共有旧普通增强.25＋S槽.25；S 槽为 raw(B00) 或完整 practical(raw)(B01)，共有对比只作用于旧普通增强对。固定 6 次 L 前向和真实 raw.5 份额，防止普通 AWGN/multipath 在 practical 前重复叠加。

最小实验为既定 RX1 源留出契约、两组同 seed 从零 20 轮。源阈值已固定：low urban≥+5pp、urban均值≥+3pp；clean、各 suburban、seen V 均≥−0.5pp，完整有限预算及最终 LCE≤0.5。失败关闭此比例；目标分数不参与。实现、定向检查和一次 P0/P1 审查已通过，实际远端状态以该 run 产物／报告为准。该源证伪不等于正式全源四因子结果，也不是联合目标完成。

联合目标仍完整保留：A_g 跨 RX clean 净收益、A_s practical 适应、二者联合保留各自收益，low urban >60%，suburban 各场景接近本模型 clean（建议探索差距 2 个百分点，未冒称已达到）。目前 A_g 无通过候选；A_s 固定 75/25 方案仅通过一次源配对，联合目标未完成。


源 run 已自然完成两组 E20 和 17640 个七视图默认头决策，独立完整重算 PASS。B00/B01 clean81.666667%→84.761905%，low urban28.571429%→36.111111%，urban均值+13.095238pp；所有原科学条件通过。两组训练物理序列相同，每模型实际NN前向4686678，合计含公共smoke2为9373358。没有目标接触。suburban gap仍22.222222/26.746032/28.809524pp，TX2/TX3在全部卫星场景均下降，不能宣称普遍身份保持。详见该run最终报告与source_analysis.json。

下一唯一假设是该固定组织可在独立seed复现。最小确认拟为seed2026092702同RX1、同B00/B01各从零E20、原阈值和全部数据/视图/前向控制不变；尚未登记或启动。独立确认不通过原联合源条件即不晋级，不通过换seed、RX、比例救跑。正式矩阵仍缺Ag、B10/B11及交互项，当前不启动target。


独立seed02确认已登记为 `20261016-phase1-F-crossRX-satmix25-confirm-sourceRX1-e20-s2026092702-m2-r01`，状态PLANNED，尚未启动。固定runtime profile只改模型seed/run/release/output，底层F与原生seed均为2026092702；split/channel/receiver/eval、六前向、损失/阈值/U不变。两个模型各从零E20。10个负例与种子传递检查PASS，新入口一次限定P0/P1审查PASS；不重复审旧机制。源确认失败不扩seed03、不改RX或比例救跑，联合目标仍未完成。


上述seed02确认已实际发布，执行commit `d71531d53c1d9ec46e3915963e05bff96760e80e`；两组native/config种子均为2026092702，真实smoke、archive单次比对、编译及PID/CWD/argv/GPU/log增长VERIFIED。当前RUNNING，B00 E2/20、B01 E1/20是已观察进度，尚未评分或确认收益。不重新提交同row。


## 当前结论：75/25源确认失败并关闭

seed02两行均自然完成E20/4440更新与全部七视图，独立重算VERIFIED。clean83.095238%→70.000000%（−13.095238pp），lowurban28.492063%→36.031746%（+7.539683pp），urban均值+11.825397pp。原clean保持条件失败，其余条件通过；不追加seed03、换RX/比例或延长预算。两个seed的clean差分别+3.095238/−13.095238pp，均值−5.000000pp；urban均值差+12.460317±0.897913pp，lowurban差+7.539683pp。As适应净收益被重复观察到，稳定clean保持没有成立。seed02差距缩小部分来自clean下跌，不能记为联合成功。四模型累计80模型轮/17760更新/18746716实际NN包前向，全部结果与CM、代价保留。

下一唯一源假设为当前F+CROSS_RX对有界共同相位变化仍有默认头识别缺口。最小拟用自有B00/seed01/final20冻结模型，在fitL5040上做identity/+0.37rad/−0.37rad共15120决策、0更新；不读heldRX1、U、V、target或训练新头。两个方向均需至少2pp整体下降，且各自至少3/4 fitRX下降1pp，才支持另行考虑B00/B10；否则关闭相位方向，不扫描幅度。仅为待实现/登记/启动假设，敏感性不等于Ag有效。历史CNN相位和旧E200诊断不被当作当前F成绩。


当前相位源检验已实现并登记为 `20261016-diagnostic-F-crossRX-commonphase-sourceL-s2026092701-m1-r01`。三固定视图、fit L5040、15122包前向（含合成smoke2）、0更新/0头拟合/0教师/0As；10个负例和一次P0/P1审查PASS。尚未启动或取得识别结果，原正负双向阈值与失败关闭规则不变。


公共相位冻结源检验现已完整完成并独立复算：fitL identity99.841270%，+0.37rad75.337302%（−24.503968pp），−0.37rad63.015873%（−36.825397pp），各4/4源RX超过1pp下降。0更新/0头拟合/0教师/0As，15122实际NN包前向，功率误差≤2.981e−7，全部15120决策/logits/216视图单元核对通过。真实smoke/发布/进程/日志/自然退出VERIFIED。只证实当前F+CROSS_RX的公共相位脆弱性；未证明Ag有效。下一唯一拟检验独立相位L视图的匹配B00/B10 scratchE20，仍须单独登记，尚未训练，不访问target。


独立相位训练候选现已实现/登记 `20261016-phase1-F-crossRX-commonphase-sourceRX1-e20-s2026092701-m2-r01`，匹配B00/B10各scratchE20/同六L前向/.5raw+.25ordinary+.25槽；B10槽为每包uniform[−.37,.37]rad私有随机流。CROSS_RX仅共有ordinary对，U/ownEMA不变，0As训练。11负例与实际F合成配对控制检查PASS、一次限定P0/P1审查PASS。源clean≥+1pp/F1不降/seenV最多−.5pp，失败关闭固定配方无angle/seed/RX救跑；尚未启动或取得训练结果。


上述公共相位B00/B10已实际发布并运行，执行commit `39d95f79d91ccf2a570f1c1349adacfb409c0833`；archive/compile/真实CPsmoke/两worker PID-CWD-argv-GPU及日志增长VERIFIED。实际seed2026092701、20/13/7和无训练父权重均核对；当前未完成训练/源预测/独立评分，不能称Ag有效。


相位E20源run现已ANALYZED/VERIFIED：两行完整20epoch/4440更新及七视图，全部17640决策/完整step/相位角/物理序列独立核对。clean差-3.9682539682539653pp，clean F1差-4.4209738408452885pp，seenV差-0.07407407407407085pp，原源条件FAIL，固定相位配方关闭，无角度/seed/RX救跑。仅源证据，尚未证明Ag目标泛化或联合目标；实际总NN包前向9373358。


频谱IQ可实现性候选已实现并登记 `20261016-diagnostic-F-crossRX-spectralIQ-fitL-s2026092701-m1-r01`：fit4RX L5040重估固定rectangular/DCT1to8，33tap/.5dB、6RX对双向与匹配control共25视图；126002NN包前向/0更新，不读held/U/V/target，不复用旧fullL系数。9个算子/权限负例和5个门槛负例PASS、一次独立P0/P1审查PASS。原阈值失败即关闭固定滤波，尚未启动或取得真实源身份结果。
