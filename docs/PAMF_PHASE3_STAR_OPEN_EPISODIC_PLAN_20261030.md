# PAMF-OW 动态注册集合首轮实验

当前执行 run `20261030-phase3-star-open-episodic-s2026092701-r02`。r01 在第一步原型槽 metadata 的 NumPy2/Torch2.1 转换处技术失败，三行均完成真实权重 smoke、尚无 optimizer 更新或目标访问；原输出保留。r02 仅修复 int64 metadata 转 Python 整数列表的兼容边界，矩阵、模型、预算、采样与目标设置不变。本轮落实[再设计](PAMF_PHASE3_OPEN_REDESIGN_20261030.md)，检验源端训练成员关系能否迁移到跨接收机星地开放判决。唯一主候选为 M，S0/S1 是固定控制，不按目标成绩换主。完整目标仍为 ACTIVE。

三个模型分别从原完整 Tri-SIG-TIME SSL `20261027-phase1-pamf-repl-shuffled-views-m16-r02/main_split-repl-shuffled_time-3view-s2026092701` 的 E50/11100 终点独立开始。严格构造原生可微 Sinc、骨干和默认六类头，再加入开放模块；不复用训练后的 Phase3 候选。当前 Phase1/2 默认模型不变。

同次前向使用 t2 时间均值与 population std 共 192 维、PA160、实际 bounded SIG160，拼接 512→64→64/LN/unit。std 使用 `sqrt(var(unbiased=False)+1e-12)`，仅单记录时间统计。身份残差 `0.1*norm(f)*tanh(g)*unit(H z)` 中 g 零初始化，原 raw fused160 不整体归一化。共享关系 MLP 为 `concat(abs(z-p),z*p):128→32→1`；原默认头始终决定旧类身份。新增 51841、识别器 284834；训练容器 952563，含未使用原生域参数 667729。真实构造和首步烟测核对参数范围。

| 行 | 骨干和原头 | 关系任务 |
| --- | --- | --- |
| S0 | 完整 L 目标＋同 P1 原 F0 KD | 开放特征 detach，固定六类注册集合 |
| S1 | S0＋成员梯度联合更新 | 固定六类注册集合 |
| M | 同 S1 | 轮换五类注册集合，留一源旧类为临时非成员 |

每行 fresh CENTER_UNITS AdamW，lr1e−4、继承原 weight decay、物理 clip5，固定 2220 更新，累计13320；不是 IID 数据的自然 epoch。新模块同 seed2026092701，源批/episode/OE 采样 seed2026103017、槽排列2026103018，均按 step 派生，三行轨迹一致。旧构造器 L/OE seed 仅用于其验证接口，不冒充本轮采样 seed。

每步原 L128 完整六次监督前向，CRIC0.1、ordinary/raw KL、As CE 且 As KL0、LS0.01，U/PL/EMA 关闭。输入全部为既定星地 P1；ordinary 仅施加原普通变换，As64 槽重放同一 cached P1，不再次生成信道。训练容器保持 train 以执行原 L hook，encoder、Dropout/BN 子模块保持 eval，梯度保留、运行统计不更新。

额外同 L128 原 P1 默认头 logits KD，T2、标准 teacher||student KL×T²、权重1。每步另取 support 每类16共96、query 每类16共96、OE64，一次256行前向。query 固定在当前 held RX/day；support 同时排除该 RX 和 day，物理 ID 互斥。held class/RX/day 按 step 遍历，原型槽随机排列。原 F0 已见全部六类和源 RX，这些是训练代理，不能宣称严格未知类别／接收机实验。OE 只按可见15个 RX/day 非空域等权再记录等权抽取，不读 TX/K、不按未知身份配对或补空 cell。

成员损失为 `0.1*(0.5*balanced pair BCE+0.5*balanced set BCE)`，正负分别平均；set score 为当前原型关系 logits 最大值。共享梯度先消除与完整 L＋KD 冲突的分量，再将成员梯度范数限到锚梯度范数；`open_world.relation_scorer.*` 独立接收成员梯度。不加入 uniform-six OE、VICReg、Gram 或目标伪标签。

每行训练原 L 前向 1704960 行、KD284160 行、episode568320 行，合计2557440行；源 L native 抽取284160、support/query各213120、OE142080次曝光。三行训练 F 共7672320行，独立物理记录数不随重复抽样增加。冻结后 F/G 前向和实际耗时另计，不预报时延或速度收益。

## 数据与冻结评估

| 角色 | 数量 | 权限与位置 |
| --- | ---: | --- |
| source L | 6300 | 六类真标签、源五RX/三日；训练和冻结六个 open64 原型 |
| source OE | 6134 | outside-six 成员监督，可见 RX/day；真实 TX/K 隐藏 |
| source V | 27000 | 只在模型和 L 原型冻结后校准六个预测类阈值 |
| source novel | 3150 | 三个历史暴露设备，仅冻结预测与独立评分 |
| target U | 384 | 本轮不读；后续适应和匿名注册单独实施 |
| target future Q | 864 | RX24-1、2021-03-15；逐样本预测、不得拟合 |

源 RX1/3/4/6/8 与目标 RX24-1 互斥，源日期2021-03-08/15/23。所有新消费记录均为原 G10 MIX01 received 或同观测经冻结 DURA v1.0 的 P1。源缓存复用 `20261029-phase3-star-representation-s2026092701-r01/cache`；不打开旧 clean SourceDataset 或 Us56700。固定 G10 的 receiver2027、scene2026103001/channel2026104001 及完整信道定义见[统一输入](PAMF_PHASE3_STAR_G10_INPUT_20261029.md)。

原型为 source L 各真实旧类开放特征均值再单位化。source V 仅 known-only predicted-class risk 阈值校准，源42分组控制规则复用既有 `star_class_risk`；不以 novel/Q 选阈值、checkpoint、权重或候选。集合分数 `max relation`，被接收旧类取原默认头 argmax。源 novel 和开发 Q 的全部预测均执行，source unknown90不是额外入口。

每行 prediction 完整后，独立 CPU scorer 才连接 source novel 与 target Q 真值；训练和 producer config 不携带 truth/U/未知类数/配额。原 F0 基线为同一 P1 的原冻结默认头，另报候选自身 closed，区分闭集损伤与误拒损伤。主报逐 TX 宏 A_K/R_U/H-score，并列 C_K、总体/逐场景/逐RX/day/TX 和配对纠错损伤。

验收采用 targets v3：跨 RX 星地 C_K≥90%、R_U≥90%、A_K≥原同输入 F0 closed−2个百分点；H 不替代单项。当前 Q 的 high 场景未知计数0/4/2，缺一个未知身份，六场景完整认证必须保留 N/A。数据保持，不能重抽信道、删类或事后改 pooled 口径。该池为已暴露开发验证，正式 RX24-18 不消费。本轮仅 DA0_REG0，A_N/H_OW、无标签适应与匿名注册结果 N/A，不能完成整个 Phase3 目标。

## 执行与停止

唯一 owner `codex-phase3-star-open-episodic-20261030`；默认 N607，独占不可覆盖 run/release root。最多两行并发，每张 GPU 最多新增两个本任务训练／预测进程，每次启动核实实际 free≥8192 MB，优先不同卡。保留其他用户任务，不停止、修改或重启它们。每行自动训练冻结→全部源/目标预测→独立 CPU 评分；科学失败不停止健康控制或取消目标预测，不重试调参。

首步使用真实原 F0 checkpoint 的合成 G10 no-query smoke，验证 Sinc、开放分支和关系梯度、零增益原 logits 精确一致；评估另先进行真实冻结新 F、原 F0、DURA 的合成观测检查。只完成最小允许的8项流程：实际代码 commit、聚焦 negative/集成检查、一次直接 P0/P1 review、预登记、一次 preflight/归档比较/compile、独立 PID/CWD/argv/GPU/log增长和 truth-last。检查记录不替代真实成绩。

非有限、权限/祖先不符、输出碰撞或无法生成合法预测时，保留失败行，健康行继续。系统性技术失败需要停止时，仅由 owner 核实自己的 PID/CWD/完整命令后操作；未知启动状态先读回原 root/PID，不重复提交。具体执行 commit、命令、环境、路径、矩阵和产物唯一维护在本 run `experiment.json`，运行证据补入同 run 报告。
