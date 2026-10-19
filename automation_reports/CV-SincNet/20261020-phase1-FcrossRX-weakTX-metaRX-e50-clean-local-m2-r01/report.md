# F＋跨 RX 同 TX：弱发射机跨 RX 元训练（50 轮，clean）

run_id：`20261020-phase1-FcrossRX-weakTX-metaRX-e50-clean-local-m2-r01`。当前为 LOCAL_VERIFIED，尚无本轮性能结果。实际代码身份为运行时 `release_commit.txt` 中的 Git 提交；完整固定矩阵见 [experiment.json](experiment.json)。

## 固定方法与对照

复用原 F＋CROSS_RX50：三路 SIG_TIME_X3、joint160、统一居中物理 Sinc、RDrop alpha=1、旧门控＋ownEMA＋cos、普通增强、LS0.01。从零训练 50 轮（33L＋17U），每轮 222 步，共 11100 步。默认分类头随主干正常联合训练。无星地增强，测试只跑 clean。不继承历史 checkpoint；既有同配方本机 BASE 三种子的训练及 clean 结果只读复用。

首种子固定 2026092701，两行并行：EXTRA_RX_CE（临时更新 alpha=0，曝光匹配对照）和 META_RX（alpha=0.01，真实二阶可微临时特征更新）。弱 TX 由既有源 V 固定为 14-7、20-19，不使用本轮目标结果改集合。两行均保留原浅层／中层跨 RX 同 TX 对比损失和两次 RDrop 前向。

第 1110 步后每 16 步调用一次，预定 624 次。循环选择源 RX1/3/4/6/8 为 inner RX；inner CE 对当前 L 批中 anchor RX 的全部在场 TX 做类平衡，outer CE 对其余 RX 的两个弱 TX 做类平衡。只复用同一次实际增强后的 L128，不额外读取 IQ 或 U 真值。每次额外执行一次 L128 前向和一次 held 弱 TX 前向。

META_RX 在临时函数式副本中计算 p'=p−0.01·grad(inner)，保留二阶计算图；辅助损失为 0.5×(inner CE＋outer CE)，沿原 warm5/ramp10 到系数 0.1。EXTRA_RX_CE 使用相同曝光、损失和代码路径，仅 alpha=0。若当前批缺 anchor 或任一弱 TX，记录不适用并跳过该次辅助项；不适用率属于科学结果，不阻断首种子 clean 测试。

临时更新仅覆盖实际部署的无量纲特征参数。默认 W 在辅助项中 detach，主训练仍正常更新 W。两个物理 Sinc 坐标不做临时 SGD，但真实外层梯度仍可到达 Sinc；实际物理 AdamW 的学习率、epsilon、clip5 和含 origin 衰减保持原实现。克隆缓冲区、恢复缓存与训练标志，额外目标不消耗主 RNG；不添加模型参数、分类头、教师或真实优化器步数。

方法依据 [MLDG 原文](https://homepages.inf.ed.ac.uk/thospeda/papers/li2017metaDG.pdf) 的可微训练／验证更新思路；本实验是弱 TX、既有默认头和物理 Sinc 约束下的改编，不宣称标准 MLDG 复现。

## 源域扩种子规则（目标访问前固定）

先完成首种子两行，按同种子 BASE 独立决定各行是否原样扩展 2026092702/03，再访问任何本轮目标汇总。每个弱 TX 必须同时满足：源 V 的 out_RX 特征角度 margin q10 比 BASE 至少增加 0.002；RX spread／类间间隔至多为 BASE 的 1.05 倍；无零特征。默认头健康边界为总体不少于 BASE−0.2 个百分点、最差 RX 不少于−0.5 个百分点、任一 TX 不少于−1 个百分点。预定调用须为 624，适用率至少 99%。

该规则只作用于本轮未来实验，不追溯晋级以前失败的候选。几何来自同一次合法源 V 验证的原始 joint160；不拟合分类器或输出原型分类决策。所有源 RX 都已参与训练；out_RX 只排除描述性中心中的 RX，不是未见 RX 模型验证。科学规则失败仍完成首种子 clean，保留负结果，不按目标分数扩种子、重排或重跑。

## 数据、执行与停止

沿用已验证数据句柄：L_s=6300，U_s=56700（TX 真值隐藏），V=27000；源 RX1/3/4/6/8、day1/2/3、split392005。现有 source contract、metadata 和 VALIDATED_ONCE clean capsule 不重建。clean 测试 168000 个物理样本；每行 final50 冻结后使用 checkpoint 内默认 student 头预测，prediction 完成后独立 CPU truth-last 评分。

环境：本机 Windows，既有 ssr-gpu Python3.10.19、PyTorch2.10＋cu128、RTX5070Ti 16GB，FULL_FP32_NO_TF32，threads2／workers0。CWD 为独立只读代码发布根 LOCAL_RELEASE/cvs_F_weakTX_metaRX50_local_20261020_r01；输出为 LOCAL_RUN/20261020-phase1-FcrossRX-weakTX-metaRX-e50-clean-local-m2-r01，不覆盖已有目录。

命令：`LOCAL_BASE_PYTHON -X utf8 -u -m experiments.cvs_f_weaktx_metarx50.driver --profile LOCAL_PROFILE --metadata LOCAL_SOURCE_NATIVE_GATE_METADATA_R02 --base LOCAL_RUN/20261020-phase1-FcrossRX-weakTX-metaRX-e50-clean-local-m2-r01 --reference-base LOCAL_RUN/20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01 --commit RELEASE_COMMIT --max-workers 3`。

GPU0 总 source／predict worker 上限 3（计入其他 owner），派发前可用 RAM 至少 4.5 GiB；独立 CPU scorer 至多 1 个。保留所有健康任务。数据角色、来源、配置、输出冲突、非有限值、实际原生训练预算或 EMA／Sinc／特征路径异常、worker 技术失败时停止新派发并保留产物，不自动重启，不因性能停止。

预期产物：每行实际参数、初始化与无查询 smoke、逐步／逐轮训练记录、Sinc／RDrop／SupCon／meta 路径观察、final50 checkpoint、source 冻结、源几何与扩种子决定、clean prediction、独立评分、完整日志分析及同 row 汇总。Git 仅保留代码、配置和整理后的聚合报告；不上传 IQ、checkpoint、原始日志、逐包预测或本机私有状态。

## 已完成验证

[checks.json](checks.json)：8 项协议负例、精确二阶标量核对、624 次 RX 日历、逐弱 TX 源规则、原 F 初始化和 warmup 一致、真实 F CPU／CUDA 联合反传、辅助 W 无梯度而 Sinc／浅层可达、主 W 正常更新、EMA／推理与参数数量不变。一次真实匹配 final50 checkpoint 的零输入无查询 smoke 已通过，未用于初始化。额外目标 CPU／CUDA RNG 保持检查通过。这些均不代表源域或目标性能。

[review.json](review.json)：本候选一次独立 P0/P1 审查 PASS，无额外重复审查。既有跨进程训练存在微小数值非确定性，单种子小差异只能作为探索证据。本基准已有历史暴露，不宣称独立盲测。

尚待：提交和独立远端核对、一次发布归档比对、本机启动后进程／CWD／GPU／日志增长核实、完整训练及自动 clean／独立评分、同 row 结果提交推送。
