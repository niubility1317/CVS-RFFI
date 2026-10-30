# Phase3 源 RX／日期扰动积分（2026-10-31）

Run `20261031-phase3-nuisance-marginal-s2026092701-r01`，单模型种子 2026092701；唯一主方法 `marginal_predicted_ratio`。本项检验源端估计的接收机变化分布能否改善新接收机拒识，同时保护已知正确接收。属于自主研究中的源端统计候选，不是无标签目标适应或匿名注册完成证据。

## 方法与对照

保留原 `pooled_ratio`、`predicted_ratio` 的全部身份预测、密度、阈值和运行点作为精确对照。新候选使用同一冻结 P1 signal r03 表征、原 F0 默认六类头和 DURA v1.0；不使用上一轮 BOUNDARY、TARGET_SSL 或其他目标适配权重。

令原已知 90 个模式的白化均值为 μ[c,g]，c=6 类，g=15 个源 RX×day。计算 o[g]=mean_c(μ[c,g]−mean_g μ[c,g])，再去除 o 的均值；用全部正奇异值定义 C=oᵀo/15=Bᵀdiag(λ)B，秩至多 14，不搜索秩或尺度。只有原源端 Gaussian 聚合量进入这一步，不新读 L/OE IQ。

对已知与 OE 的每个原 Gaussian 分量，将条件观测模型 N(z;μ_j+δ,v_jI) 中 δ∼N(0,C) 积分，得到 N(z;μ_j,v_jI+C)。原均值、权重、标量方差和白化器均不改写。该假设提供额外域变化容忍，也可能使未知被接收，必须通过固定矩阵证伪；不预先称作域不变表示。

采用低秩精确 Gaussian 算式，logdet=(160−r)log(v)+Σlog(v+λ)，二次型使用正交残差/v 加投影分量²/(v+λ)，避免减去大数。新 `marginal_pooled_ratio` 为全 90 模式与 OE32 的比；唯一主 `marginal_predicted_ratio` 按原 F0 预测类别取 15 个等权模式，OE32 相同。密度比中共同白化 Jacobian 抵消。r=0 严格回到原两种分数。

## 数据、拟合与权限

沿用已验证 source contract split392005，原 F0 Tri-SIG-TIME SSL E50/11100、r03 source-only 开放分支及 DURA R69 E20/4940 祖先均保持。原密度基于 L6300（旧六类各1050，每类/RX/day70）和 OE6134（仅 outside-six 成员及公共 RX/day，不解封 TX/K）；原 U_s56700 不新读。源 RX1/3/4/6/8=1-19/18-2/19-2/2-19/3-19，日期 2021-03-08/15/23。Phase3 的源 IQ 祖先均为固定 G10 MIX01 星地＋DURA P1。

实际三 checkpoint 合成两条 G10 无 query smoke 先执行，PASS 后即继续。从旧聚合量导出并冻结 C 后，才读取源 V27000（每旧类4500）拟合两组新六阈值，原两组六阈值精确复用。采用既有 known12 保护规则，另30个 RX×scene 组仅诊断；不用 NV、U 或 Q 优化阈值。原密度/normalizer 不重拟合，不更新神经模型或 BN。

全部四组阈值冻结后，计算 NV3150（三未知 19-15/19-16/7-1 各1050）的固定源接收机分数，以及 crossRX clean/star 两份既有 DURA 缓存分数。两份 Q 均是 RX24-1/index26、2021-03-15 的同864个物理记录：旧六类×96=576，未知三类×96=288。原 F0 身份、同 DURA 闭集基线与特征均逐值复用，不重跑真实 IQ 或增加物理观测。

全部 4 方法×3 DURA 条件预测完成后，独立 CUDAempty CPU scorer 才连接三份真值，验证控制、prior 源协方差、全部分数及阈值。目标 U384/2021-03-08 和正式 RX24-18 不访问；没有 clean 学习、真实未知类数/配额、Q 统计拟合、发现或 REG。源 NV/devQ 已曝光、单 seed/RX/日期重合如实披露。high 景未知计数0/4/2缺19-15，宏拒识及 H 为 N/A，不重抽、不省略、不认证完整六景。

## 验收、成本与执行

主报宏 A_K、R_U、H-score，列 C_K、同路径原 F0 闭集−A_K、AUROC、OSCR、逐类/RX/day/scene 和覆盖，保留主方法对两原控制及新 pooled 的同 ID 增/失配对。目标沿用最新指定：clean crossRX R_U≥95%，star crossRX R_U≥90%，C_K≥90%，闭集至 A_K 损失≤2个百分点，主场景覆盖完整。另要求主 star A_K/R_U 均不低于原 `predicted_ratio`，源 known12 保护通过；源 RU 仅诊断，源失败仍自动完成全部已登记 Q。不得借控制的单项高分换主或改变历史结论。

新神经更新0、新真实 F0/G/signal 前向0；实际合成 smoke 三模型各2。新增一份源低秩协方差聚合与两组六阈值（12标量），两组原阈值复用，旧 source/cache/G/神经训练成本继承。完整 NN 部署参数仍834665；新增 prior 与全部 τ 的字节数按实际数组报告，不把缓存复用写作零总成本或把进程耗时写成 B1 延迟。

N607，唯一 owner `codex-phase3-nuisance-marginal-20261031`，不可覆盖 run/release roots，见 experiment.json。一个 GPU 仅执行真实 checkpoint smoke，其余统计；free≥4096MiB、每卡新增本任务≤2，独立 CPU scorer 不接 CUDA。绝不停止/修改/重启其他 owner。启动后核 PID/CWD/argv/GPU/log 增长，缺遥测如实标明，不为补采样重跑。

一份代码 Git commit、聚焦数学/权限与实际 toy producer→scorer 检验、一次直接 P0/P1 审查、一次 preflight/archive SHA 比较/compile。非法角色/祖先、非有限、输出碰撞或无法合法闭合时保留产物；不自动覆盖重启、不 partial truth 评分。观察超时先核原 PID/root；科学负结果不作为技术停止。完整 Phase3 目标保持 ACTIVE。
