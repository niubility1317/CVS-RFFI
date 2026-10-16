# 普通增强去 PA：独立 E20 训练因果检验

状态REGISTERED_NOT_RUN。旧ordinary五foldFAIL不回溯修改；已完成同IQ PA输入干预给出五RX保持率全正mean+1.126556pp/accuracymean+.600198pp，不能据此宣称训练有效。本轮只检验一个新N2-RX1 scratch总E20，复用同预算N0/N1源结果不载权重，不换RX救援。

具体矩阵、命令/commit/CWD/环境/路径、六类seed、GPU、stop规则见experiment.json。固定SIG_TIME/RAW_HZ/NoExplicitCFO/joint160/defaultcos、LS.01、AdamW/cos、RDrop、oldgate/ownEMA和U配方。唯一科学差异为actual aug_enable_pa_normal=false；原p_pa/schedule和其余ordinary保留，guard拒绝任何simulate_pa调用；不新增硬件效应、不反演TX。初始化本地198state张量与N1一致，实际args只差PA开关和run/candidate/output标识。

同原fourRX训练契约L5040/U45360、seenV21600，源RX1 L1260/U11340不使用，原V中heldRX1 V5400仅FINAL。先前诊断读过heldL不改变本次角色；U TX隐藏，不将V变训练。源过滤、原222step/epoch循环和私有augmentation/loader流复用；结束时将N2/N0/N1全部实际L/U序列逐条比较。默认头与全网按原流程训练，未另训probe，无As/target/新loss/U机制。

首轮标准：相对N0 heldaccuracy>=+1pp/macroF1>=0，seenV均值损失<=.5pp，有限完整预算/覆盖/序列；相对N1 accuracy和macroF1各损失<=.5pp。失败结束NoPA候选，不改强度/seed/RX重试；通过仅允许另登记固定其余四RX，再使用原五fold规则，不能直接进入target/joint或宣称泛化。

新成本20model-epochs/4440updates/1136640L网络前向，U/EMA/sourceeval/held5400/CPsmoke按实际计数。旧N0/N1合计40model-epoch为历史对照成本，无模型继承。没有从E50/E200加载或扩展单模型预算。

本地256合成网络前向/128增强包/8负例通过，0realIQ/CP/optimizer，默认W/gain/SIG梯度有限非零、RDrop两前向同IQ。首次argsfixture将两个双方相同的nativeNaN占位判成差异，在模型构造前0fwd失败；修正NaN对等测试后通过，没有改实际native参数或模型。一限定独立P0/P1审查正在执行；真实scratchCP无query smoke在worker首步PASS立即训练。尚无本新候选训练结果或正式测试结果。

唯一限定P0/P1增量审查PASS，无直接缺陷；检查再执行256合成前向/128增强包/8负例。合计本地512合成网络前向/256增强输入，0真实IQ/CP/optimizer。登记字段及compile通过。真实scratchCP smoke留在worker首步，尚未发布。

## 启动已核实

状态RUNNING，实际训练代码2b916b8892fce0dd1ae80ff9c8f1d649445b245b已提交推送并独立读回同OID，单归档transfer对照与remotecompile通过。worker PID3609877/GPU0的CWD/argv/GPU、真实scratchCP noquery smoke与日志增长6396→16669已独立核实。当前E2，实际5040/45360/21600角色、E20=13+7、LS.01、ordinaryON/PAOFF、NoExplicitCFO/defaulthead、ownEMA/scratch、AsOFF全部通过。尚未读取FINAL heldV，不能宣称源收益或target改善。原release/健康任务未热改/重启。
