# R30：固定三步恢复器的 IQ 噪声协方差源诊断

持续目标保持：当前受损 IQ→单份恢复 IQ，原 PAMF 的 FFT/PA/时域/统计均从同份 IQ 派生；训练使用合法 L6300/U56700，0 TX/F/target 信号，原 F50/defaulthead 冻结验收。正式 low urban≥70%，其他五受损场景各距原始 clean84.67619047619047%≤5个百分点，恢复 clean 损伤单列。本轮是固定模型源机制诊断，不替代正式目标或逐细节 TX 归属证明。

上一轮 R29 的完整 Haar 子带使 source mean6 NMSE 相对匹配低通降低2.1692%，但 mean6 accuracy仅+0.0093个百分点、mid urban−0.8333、最差low urban TX−2.3891，身份验收失败，0target。R28 已验证已知作用/标量模拟器方差在三步网络上的反事实，不能重复算成新设计。本轮只检验既有报告明确未覆盖的 IQ 噪声交叉项，父模型仍固定历史 source-selected R17 unroll_unshared，不依R27正式排名选父。

源码依据：`code/leo_practical/residual.py` 中，噪声为 `alpha*w*main_rotation + beta*w.conj()*image_rotation`，随后只经过均衡滤波G；本任务原R07 teacher明确拒绝EQ开启，因此G=1，不存在额外的均衡时间滤波。载波/相位乘法与逐时IQ混合仍存在。原teacher保存包含实际AGC的main/image及白输入等效实方差，不是完整的实2×2噪声协方差。噪声采样在各时刻独立；在冻结实际AGC/旋转的变换模型下，遗漏项是各时刻I/Q的椭圆性，不应泛称未知的强时域色噪声。

设已缓存系数为a(t)、b(t)，注入圆复高斯噪声w的E|w|²=σ²。冻结这些变换后，n=σ²(|a|²+|b|²)/2，q=2ab/(|a|²+|b|²)。实协方差为

\[
C_t=n\begin{bmatrix}1+\Re q_t&\Im q_t\\\Im q_t&1-\Re q_t\end{bmatrix},
\quad C_t^{-1}(z)=\frac{z-q_tz^*}{n(1-|q_t|^2)}.
\]

特征值是n(1±|q|)，条件数(1+|q|)/(1−|q|)。n沿用原R07缓存的标量模拟器方差，原noEQ系数幅值在帧内恒定；q保留逐时旋转方向。本轮记录每个场景全部1800×256个时间点的|q|均值、95分位、最大值和最大条件数，不按TX标签估计矩阵，不拟合任何参数。

**适用边界**：AGC增益本身由当前受噪接收波形计算，本轮冻结其实际值；以上不是“在已测AGC条件下”的完整后验噪声分布，也不包括原地面录音中未知噪声、AGC诱发的联合依赖或估计作用误差。它只描述冻结变换作用于新注入白噪声的协方差。本轮已知系数是源oracle，不能部署到目标。冻结prior与原近似共适应，因此结果不是理论性能上界，也不证明重训对应网络一定无效。

|新读出|作用A|MAP噪声权重|对应原R28控制|
|---|---|---|---|
|estimated_covariance|原估计|已知逐时2×2 IQ协方差|est_action_simvar：同估计A、同n、各向同性|
|known_covariance|原已知条件作用|相同逐时2×2 IQ协方差|known_action_simvar：同已知A、同n、各向同性|

两条轨迹均固定原R17三步不共享Core9、原coarse锚、SCALE²、每步2次warm-start CG。**只改变三个MAP的噪声精度算子**；prior的标量logvar与归一化、未加权伴随innovation定义均保持对应R28路径，后续current因MAP而改变是本次干预的结果。新控制复用原R28已完成预测，同时保留其native_estimated；原R10 identity/coarse只作参考。0新训练、优化器、随机抽样、L/U曝光、V持久拟合或target访问。

原父严格完整末轮E20/4940，R02 coarse20＋R07 action20＋R17prior20=组件E60，0TX/F/target训练；F50历史监督E50/11100另列。R28原method commit `2af222f06595e8f7d355d032c0144d49802a6174` 的source-only完成状态、原F/defaulthead、同1800 IDs及预测契约核实。原R17/R07/R10既定数据来源与原7/63/30物理角色沿用，不新增数据重验证、hash链或签名。

新2×7=14组IQ/features/logits全部完整有限、ID匹配，复用源参考完成句柄核实后，独立CPU才连接V标签与配对clean评分。七读出（2新、3原R28控制、2原R10参考）×七视图×15分层=735组；物理记录仍为1800条，不把相关视图与反事实当成独立样本。输出保留clean/三郊区/三urban、TX/RX/day、accuracy/Macro-F1、repair/damage、IQ/FFT/前232后24点及配对clean特征。

主问题是known_covariance相对known_action_simvar，而不是相对较弱native：固定门槛沿用R14/R28的机制尺度，mean6 NMSE≤对应控制0.95倍、low urban≥+1个百分点、mid urban≥+.5个百分点、clean相对原输入下降≤.1个百分点、low urban最差TX变化≥−1个百分点，才支持优先研究可估计协方差。估计A路径作为次要对照。无论结果如何，本轮不晋级oracle、不自动访问target、不改变既有源资格或健康任务，单seed源点估计不冒充正式收益或显著性证明。

最小实现检查：实2×2矩阵逆、正定/伴随、固定widely-linear噪声的Monte Carlo协方差、两次CG闭式解与warm-start、q=0时三个MAP精确退化为原对应标量轨迹、源模型状态不变、错误来源/非法协方差/不完整预测/CPU truth-last负测、原F参数版本缓存与完整七视图接口。真实2条L_s/完整checkpoint/原F50 no-query smoke在入口第一步，q=0对照与非零q精度均核实，PASS后立即源预测。

普通账户N607、新不可覆盖run `20261020-diagnostic-restoration-IQ-noise-covariance-F50-s2026092701-m2-r30`；最多一条本任务GPU worker，池0/1/5/7择空，全owner cap2/free≥12GB，共享capacity4锁保持至CUDA PID可见。R29已自然退出。非法来源、输出冲突、非有限或无法合法预测时保留产物、停止本run，无自动重试；其他owner健康任务不动。精确commit/命令/环境/CWD/seed/路径/停止规则/预期artifact见登记。

REJECTED_EXTRA_GATE：额外设计/spec审批、报告单独审查、重复全测或数据hash链不在项目八项最小流程内；按持续授权、聚焦检查和一次独立P0/P1推进。当前仅证明代码与数学检查，真实源身份结果待返回；总目标仍active。

实际结果：**SOURCE_COVARIANCE_PRIORITY_NOT_SUPPORTED**。14组新预测、35组复用参考、735分层及独立复算完成，0训练/target，所有进程退出。由源缓存推导max κ=.01567078、max条件数1.03184052；已知作用下mean6 NMSE仅降低0.005215%，low/mid urban准确率增益均为0。没有支持优先增加协方差估计网络，不宣称全模型理论上界或完整后验协方差。[完整结果](../automation_reports/CV-SincNet/20261020-diagnostic-restoration-IQ-noise-covariance-F50-s2026092701-m2-r30/report.md)。
