# F OJ24 独立源训练：预登记

状态 REGISTERED_NOT_RUN。精确command/env/CWD/paths/GPU/stop见experiment.json；实际代码以本次commit确定。

# F/E20 有界观察原点抖动的独立训练检验

状态 IMPLEMENTED_REGISTERED_NOT_RUN。源诊断表明固定F存在原点脆弱性，不代表增强训练有效。仅delay24使A0源L1260从81.1905%降至48.6508%，修复145/破坏555，三day均明显受损；相位和delay存在交互。当前唯一训练假设是：**原始监督保持一半，另一半只使用轻度有界的观察原点变化，能够减少对固定切片原点的依赖，并改善跨RX clean识别。** 首先检验原点，不同时加入相位、噪声、PA或As。

唯一算子 OJ24：每个L输入独立取整数d∼Uniform{0,…,24}，G(x)[n]=x[abs(n−d)]，I/Q共用同一索引。该式严格等价于原linear_filter的delta[d] causal-reflect处理；无循环拼接、未来/长记录假设，前d点是人工reflect过去，后d点被截去。d=0保留输入。无插值或相位改动，不反演未知TX硬件。24来自当前原practical FIR数值延迟，发布前固定，无范围/概率扫描。这个候选仍可能损伤身份信息，必须由训练证据判断，不预称身份保持成立。

原始nativeRMS输入后施加G，不另做packetRMS/AGC、幅度重标定或剪裁；记录实际输入/输出能量比、MSE、d计数及有效原观察范围。每epoch/step/view刷新私有随机流20261016321，不改变loader/dropout/U随机流；d不由TX/RX/day标签决定，属于观察变化而非硬件参数session。当前不生成As，避免普通AWGN/多径在practical前重复叠加。

模型为用户指定F：SIG_TIME_X3/CENTER_25K/CENTER_UNITS/NoExplicitCFO/原joint160和默认cos30，所有原参数按原流程联合训练，不新加分支、头或损失。新OJ24从零总E20=13L+7U，4440updates，每L四前向：0.5RD(raw)+0.5RD(G)，RD为同一IQ的两次随机子网络CE(LS.01)+原对称KL(alpha1)。两个RD对之间无一致性或对齐。旧U/ownEMA/门控/cos/strong_noise_std保持，只增强L。

只训练首源fold：seed2026092701/split392005；fitRX3/4/6/8 L5040/U45360/seenV21600，heldRX1 L/U不训练，heldV5400仅FINAL。匹配控制复用已完成F A0 raw/raw四前向的完整源结果、配置及全部实际L/U物理顺序，不加载任何A0/旧E50/失败候选权重、EMA、optimizer或统计。源L1260归因的使用不改变训练合同；不把它称独立盲测或正式目标测试。

主比较唯一OJ24−A0：heldV accuracy≥+1pp、macroF1不下降、seenV均值损失≤0.5pp，完整有限4440更新、72个TX×fitRX×day单元覆盖及全部实际L/U物理序列一致。失败关闭OJ24，不扫范围、概率、换RX/seed或主终点救援。通过只允许另登记固定其余4个源RX的同方案；五fold均值≥+1pp、至少4/5正、最差≥−1pp、meanF1≥0、每折seenV损失≤0.5pp才建立Ag源证据，不自动扩target或As。

新增成本1模型E20/4440更新/2273280 L网络包，原A0历史E20/4440单列；U/EMA/源eval/heldV/真实CPsmoke按实际记录。合法scratchCP noquery smoke为worker首步，PASS立即继续。目标指标、增强幅度和候选晋级不相连。正式混合仍需匹配四因子、原clean+六practical冻结预测后独立truth-last；本项明确source-only，lowurban>60%、clean与星地双提升、跨seed尚未证明。

实现已完成并登记，尚无真实训练成绩。一次限定P0/P1 PASS；全25种整数偏移逐值匹配原FIR、标签独立/随机流恢复、四前向/原头三SIG与Sinc联合梯度均通过；206初态张量与匹配F A0相同。累计512合成网络包、896合法jitter输入/50公式输入、1合成optimizer、13负例，0真实IQ/CP。四包隔离导入、实际E20args、compile、登记和N607普通账户preflight通过。真实scratchCP noquery smoke在worker首步PASS立即继续。

## 实际启动核实

VERIFIED/RUNNING，实际代码 f601fc65f6eaafbfd98080882108f4d50e07db69。PID/CWD/argv/GPU/log 增长、真实 scratch checkpoint 无查询 smoke、实际 E20=13+7/F 三路/统一 Sinc 每步 physicalclip/defaulthead 联合更新均通过。OJ24 已覆盖全部 25 种偏移，四次 L 前向和半 raw 权重实际生效；其他普通增强概率为 0、PA/classsignature 关闭、As 调用关闭，原 U/EMA/strong_noise_std 保持。实际 leo_practical、DataAugmentation、SSDG 导入来自依赖 release 的 experiments/adv3b02_xuc/code/。源角色5040/45360/21600，无 target。单 archive transfer 和 remotecompile 通过。遗留描述字段 updates/旧阶段轮数并非执行预算，实际 nativeargs 为20/13/7，成功更新目标4440；保留原发布配置，未热改。尚无 FINAL 源识别收益。
