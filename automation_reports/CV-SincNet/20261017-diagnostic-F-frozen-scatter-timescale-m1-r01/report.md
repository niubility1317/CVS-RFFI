# 256 点 practical residual 的包内冻结时标审计

状态 **ANALYZED／VERIFIED：nominal 包内遗漏项量级不足，未进入训练**。本轮只检验实际 nominal 传播模型是否遗漏了有足够量级的包内散射、阴影和遮挡状态变化。这是代码参数与解析公式审计，不是 IQ 增强或身份性能实验；0 IQ/物理 ID/标签/CP/信道 process/NN/训练更新。E30、全部 5 源 RX 和默认头等训练契约保持，本轮不启动模型。

实际训练适配器从 native 路径导入 leo_practical，显式 fs=25 MHz、fc=2.462 GHz、residual/post_sync、EQ=false。原完整 ChannelStream 有 ray_hz 的时间演化；ResidualChannel 明确 frozen within record，用 ray_coef 的和构造 H，公共轨道 Doppler、残余频偏与相位跟踪仍保留。不能将公共轨道 Doppler 再加到 terminal scatter 充当新增强。

唯一假设：在原 nominal 参数下，未建模的包内 scatter/shadow/state 动态已经有足够物理量级，值得进入新的 A_s 训练。最小实验是从实际代码 AST 读原终端速度、scatter 频率、shadow 相关时间、状态混合时间及显式采样率/载频；计算 255/fs 采样跨度、256/fs 包长、最大 ray 相位变化、状态刷新概率上界、平稳 shadow 增量 RMS 和三状态系数向量集合 RMS 误差界。无随机数、无源/目标数据、无滤波或网络。

固定否证规则：若最大 ray 旋转 <.001 rad、包内状态刷新概率 <1e−4、平稳 shadow 变化 RMS <.01 dB 同时成立，就关闭 nominal 包内遗漏项量级显著的训练理由。此工程量级筛查不是 identity 通过门槛，也不能证明任意模型的分类绝对不受小扰动影响；深衰落下不声称逐 realization 的相对 IQ 误差统一有界。规则来自原参数和短窗口，不使用测试误差、分层或候选成绩。

命令：F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_scatter_timescale17.audit --out E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_frozen_scatter_timescale_20261017/r01

本地 verified F Python 标准库、当前 worktree CWD；唯一 owner codex/root/F-scatter-timescale；不使用 GPU；独立新输出 E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_frozen_scatter_timescale_20261017/r01。实际 Config 不唯一、route/noEQ 或 frozen scatter 假设改变即失败；不切换 root/code，不读取数据或改 speed/strength。预期 result.json 与同 row 分析；无 checkpoint/remote release/query，未消费的真实 CP smoke、远端编译和发布 machinery 不适用。不得据此宣称 A_g/As/联合目标成功，正式训练 E30 预算不变。

## 实际结果与边界

执行代码 commit 86c7733116e33089e66020542415ca5286b6c727。本地解析和实际执行成功，固定新输出 result.json 存在并可重新读取；未重写旧运行。最大射线旋转、状态刷新、shadow 三项工程量级筛查均满足预登记的关闭条件。

|读数|实际值|解释|
|---|---:|---|
|采样跨度／包长|10.2／10.24 μs|分别用255/fs和256/fs|
|scatter Doppler 上限|8.212348 Hz|原terminal_speed=1 m/s及fc=2.462 GHz|
|单 ray 最大旋转|0.000526317 rad／0.030155742°|原频率和原窗口，没有加大速度|
|单位 ray 复值最大变化|0.000526317|不等同任意深衰落信号的相对误差|
|包内 state 刷新概率上界|0.001023995%|实际状态改变概率还受原状态转移分布限制|
|平稳 shadow 增量 RMS 上界|0.007838360 dB|原3s相关时间，取最大3dB状态标准差|
|LOS／PARTIAL／BLOCKED系数向量集合相对RMS误差界|0.006488809%／0.014222435%／0.052631699%|只界定冻结scatter的系数向量，不是IQ或分类误差保证|

结论 CLOSED_INSUFFICIENT_NOMINAL_PHYSICAL_MAGNITUDE：关闭“在当前原参数下遗漏的包内动态已经足够明显，可以直接开启新A_s训练”的理由。它不证明模型对小扰动零敏感，也不证明时变信道在其他时间尺度或有独立实测依据的移动场景无用。没有读取 source IQ、默认头间隔或任何 target 成绩，因此不能声称身份保持、跨RX泛化或urban识别有收益。更不能把已补偿的公共轨道Doppler当作散射展宽再施加一次。

0个物理记录、0L/U/V/target IQ或标签、0CP、0信道process、0NN、0训练更新／模型epoch，无RNG、无GPU；本轮没有留出或减少任何训练RX。正式E30和上一轮已完成PDP配方的实际指标保持各自独立。其他对话的局部衰减路线不在本次候选中，也不将其目标成绩用于本算子的参数。

登记工具最初因authorization/seeds字段类型不符未创建记录，随后因本地预建空目录返回已存在；核对空目录后由登记工具创建成功。新增代码编译与零数据实际执行通过。结果是真实完成的代码物理审计，未训练模型，不冒称正式七场景实验完成或联合目标达成。

最近完成：固定原参数的解析否证。证实：residual有意冻结包内scatter，原默认时间尺度的遗漏项很小。未证实：任何Ag/As身份收益。唯一假设已按最小实验和原放弃条件关闭；不提高speed/fd、放宽阈值或追加模型来挽救该配方。联合目标仍ACTIVE；下一候选必须先有不同的源域或实际算子依据。
