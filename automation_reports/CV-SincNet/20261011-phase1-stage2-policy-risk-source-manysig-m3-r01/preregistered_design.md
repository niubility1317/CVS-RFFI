# L/T 策略刷新与有限风险拟合：源域预登记

RUNNING，远端 PID/CWD/argv 独立核实 VERIFIED。本轮落实用户报告第九节步骤四：先判断固定动作策略是否遗漏了更新后身份网络的合法风险，再考虑预定低频刷新。当前 R 和物理附加读出没有形成足够源增量；本轮依据训练目标不一致及报告路径设计，不消费目标测试成绩。

固定种子 2026092701/02/03 的 B2 最终 E40，作为已经完成训练的诊断对象。B2 只含确定性 L/T 有限风险与身份保持，避免 R 产生额外解释混杂。每个 B2 都来自相应纯 CE E200，运行时核对原父、B2 完成、冻结、实际 checkpoint/config 和源角色。E40 不用作未来第二阶段身份初始化。

四策略同时固定：coverage；old（原 E200 上拟合并在 B2 训练期间冻结）；fresh_raw（B2 上重建原 raw margin 下降目标）；fresh_risk（主候选，B2 上直接最大化身份训练同一定义的有限风险）。不追加策略、不选最好种子。fresh_raw−old 检验相同拟合预算刷新后暴露的风险差；fresh_risk−fresh_raw 检验优化目标差。不将其称为随机化训练干预的因果效果。

两个新策略均从同一私有初值开始，L/T 各两遍完整 L_s，batch128，AdamW lr2e-4、weight_decay1e-4；相同每遍 shuffle、radius、coverage cube、mixture mask、code、phase process 随机流。每目标200辅助更新，两目标400/seed；总1200。L_s 6300、V 27000，身份更新0，U/教师/星地增强/target访问0。

raw 损失为 -mean(u_clean−u_aug)；主损失为 -mean(0.5 relu((min(stopgrad(u_clean),log95)−u_aug)/log95)^2)。clean 是当前固定 B2 的确定性 margin，不沿用 P0 clean margin。两者沿真实 IQ→冻结身份分类器反传，只更新动作网络。有限风险在安全区梯度为0，因此必须记录全部梯度、风险与零风险占比，不能预设它更容易搜索。

物理算子、边界和 mixture 与旧 geometry 完全一致：每动作0.5覆盖＋0.5条件，L∞半径 U[0,1]。不扩大 CFO、相位、delay 或线性 taps。相同 radius 不代表相同 IQ 能量，逐策略报告实际 IQ 相对变化、RMS 比、边缘能量、参数及半径。有限风险增大也可能集中在少数包，报告 top1% 风险份额和全部 TX/RX/day/TX-RX 分层。

拟合仅 L_s，全部策略权重在第一次 V 前向前保存并冻结。V 不参与拟合、梯度、early-stop、归一化或模型缓存写回。评估全 L_s 与 V，用新独立私有种子 seed+941003+kind_index*1009+(500009 if V else0)，每种策略从相同随机流开始；每样本/因素/策略一次随机扰动，不能称穷尽最坏风险。模型 eval，ReadOnlyState 核对身份参数、buffer、模式、requires_grad、grad、forward cache 和全局 RNG。逐样本配对数据留远端。

每seed前向预算：clean33300＋拟合50400＋8视图266400=350100样本；三seed1050300。CPU双线程串行，CUDA不可见，无GPU占位。身份更新0不等于没有辅助训练。

推进规则：每seed的 fresh_risk 在 V 等权 L/T 平均有限风险均严格高于 old、fresh_raw 和 coverage。只有主候选可推进到下一项低频刷新设计；通过不是提升识别准确率、身份保持或达到84%的证明。每因素与配对错误全部报告，失败不改选 raw，不自动 E40/target。即便通过，未来正式训练仍从对应原始 E200 重启辅助策略，并在预定时点重新拟合当前身份模型；不把终点策略倒灌到起点。

|ID|报告要求|实现|核验|
|---|---|---|---|
|P01|动作过时与独立合法动作风险|policy_audit.py 四固定策略|配对 radius、独立评估流、真实 IQ|
|P02|动作网络服务有限身份要求|search_loss/fit|有限风险梯度、安全区0梯度、原目标对照|
|P03|源角色与身份状态保护|fit/run|V拒绝拟合、完整来源、状态恢复|
|P04|真实损失与成本证据|audit/summary/逐包记录|辅助1200步、身份0步、完整分层|
|P05|不可覆盖发布与全部seed|policy_publish.py|本地验证、独立P0/P1、远端读回|

7项定向检查通过；独立P0/P1审查通过，源数据/拟合隔离、身份状态保持和发布路径无阻断问题。

执行提交 `c6e43735df6d21a2742a4df4dd0284dc765a15f4`；单 CPU2 进程 `3971531`，CUDA 不可见。三个种子固定顺序执行。运行证据见 [发布读回](evidence/launch_readback.json)，尚未给出完整源结果；不启动新的身份训练或 target 测试。
