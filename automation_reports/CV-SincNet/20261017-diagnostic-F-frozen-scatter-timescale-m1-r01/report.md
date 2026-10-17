# 256 点 practical residual 的包内冻结时标审计

状态 PLANNED。本轮只检验实际 nominal 传播模型是否遗漏了有足够量级的包内散射、阴影和遮挡状态变化。这是代码参数与解析公式审计，不是 IQ 增强或身份性能实验；0 IQ/物理 ID/标签/CP/信道 process/NN/训练更新。E30、全部 5 源 RX 和默认头等训练契约保持，本轮不启动模型。

实际训练适配器从 native 路径导入 leo_practical，显式 fs=25 MHz、fc=2.462 GHz、residual/post_sync、EQ=false。原完整 ChannelStream 有 ray_hz 的时间演化；ResidualChannel 明确 frozen within record，用 ray_coef 的和构造 H，公共轨道 Doppler、残余频偏与相位跟踪仍保留。不能将公共轨道 Doppler 再加到 terminal scatter 充当新增强。

唯一假设：在原 nominal 参数下，未建模的包内 scatter/shadow/state 动态已经有足够物理量级，值得进入新的 A_s 训练。最小实验是从实际代码 AST 读原终端速度、scatter 频率、shadow 相关时间、状态混合时间及显式采样率/载频；计算 255/fs 采样跨度、256/fs 包长、最大 ray 相位变化、状态刷新概率上界、平稳 shadow 增量 RMS 和三状态系数向量集合 RMS 误差界。无随机数、无源/目标数据、无滤波或网络。

固定否证规则：若最大 ray 旋转 <.001 rad、包内状态刷新概率 <1e−4、平稳 shadow 变化 RMS <.01 dB 同时成立，就关闭 nominal 包内遗漏项量级显著的训练理由。此工程量级筛查不是 identity 通过门槛，也不能证明任意模型的分类绝对不受小扰动影响；深衰落下不声称逐 realization 的相对 IQ 误差统一有界。规则来自原参数和短窗口，不使用测试误差、分层或候选成绩。

命令：F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_scatter_timescale17.audit --out E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_frozen_scatter_timescale_20261017/r01

本地 verified F Python 标准库、当前 worktree CWD；唯一 owner codex/root/F-scatter-timescale；不使用 GPU；独立新输出 E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_frozen_scatter_timescale_20261017/r01。实际 Config 不唯一、route/noEQ 或 frozen scatter 假设改变即失败；不切换 root/code，不读取数据或改 speed/strength。预期 result.json 与同 row 分析；无 checkpoint/remote release/query，未消费的真实 CP smoke、远端编译和发布 machinery 不适用。不得据此宣称 A_g/As/联合目标成功，正式训练 E30 预算不变。
