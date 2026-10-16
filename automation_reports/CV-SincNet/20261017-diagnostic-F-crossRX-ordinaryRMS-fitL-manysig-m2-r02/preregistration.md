# F＋跨 RX：同一普通增强 IQ 的 RMS 恢复源反证

状态：PLANNED，尚未启动。唯一矩阵见 [experiment.json](experiment.json)，依据与限制见 [研究说明](../../../analysis/cvs_ordinary_RMS_source_20261017.md)。本次只检验普通增强后的整体尺度是否造成可修复的源身份错误，不训练、不拟合头、不访问 target。

固定算子：x 为原生逐包 RMS 输入，o=A_old(x) 只生成一次，n=o×sqrt(sum(x²)/sum(o²))。FP64 求功率比、输出原 FP32；每包一个正实数，不读标签/RX/day/置信度，不夹增益，不重抽增强，不撤销 PA、噪声、多径、相位或时移形状。不修改 raw/As/U/正式测试 IQ，不声称提高 SNR、替换 RX 或已获得域泛化。旧 randomgain(raw) 失败保持关闭，此处是去除已存在 ordinary 输出尺度差异的独立问题。

实际 original native args、build_aug_base_cfg、configure_augmentor_for_epoch 在每个模型原不可变后端执行；固定有效 E13（原日程 min(epoch,13)，E13 至 E20 相同），保留旧全部 ordinary 操作。原增强配置仅模型 seed 随已登记模型变化，其他字段严格相等；native args 的非有限默认值只在日志副本写 null 并记录路径，执行参数保留。return_pa_strength 与原生默认调用先用相同合成 IQ/RNG 作 canonical 输出及 RNG 比较，不另生成第二份源 ordinary。

两份既有 own source-only scratch E20 final20 B00 默认头，seed 2026092701/02；原基线 F＋跨 RX 同 TX、三 SIG_TIME、统一 Sinc、joint160、NoExplicitCFO、default cos30 不变。逐模型仅加载一次 checkpoint，核对原初始化/完成/frozen/source_contract 与 payload、final-only/4440 更新、实际 L5040/U45360/V21600、RX1 held、默认头及无未知祖先/target/另训头。两个冻结模型原 U 训练合法，本次无 U 前向；不将诊断 checkpoint 用作下一训练初始化。

仅 fitL5040：RX3/4/6/8×day1/2/3×6TX，72 格各 70 包。既有 VALIDATED_ONCE 缓存反序列化原 L6300、未用源 V/历史特征，RX1 原 L1260 在推断前排除；U 仅角色 ID 元数据，无 IQ 或 TX 标签。不读 heldRX1/target，不把此折等同完整 L6300 或 L_train5580/L_dev720，不新增数据验证。

每模型 raw5040，四个固定 ordinary 抽样各 5040，再对同份 IQ 恢复 RMS 并推断 5040；共 45360 源 NN。四个 augmentation seeds 202610171701 至 04，独立 shuffle generators 202610171710 至 13，每抽样全 5040 各一次。模型 eval/RMS 不消费增强随机状态；两模型重建相同命名流，不冒称实际训练重放或额外独立样本。全两模型 90720 源 NN＋首模型真实 CP 合成 2 包 no-query smoke=90722 NN；40320 源 ordinary 生成、40320 RMS view；另 128 合成 ordinary 生成/0 NN 用于 canonical 检查。新 epoch/更新/教师/头拟合均 0。两参照原历史成本 40 model-E/8880 更新/9373356 NN 单列。

源数据访问前固定判据，两模型均须全部通过：RMS 相对原 ordinary accuracy≥+0.2 pp、macro-F1 不降；至少 3/4 抽样 accuracy 正增益、每抽样≥−0.2 pp；相对 raw accuracy/F1≥−0.5 pp，raw 正确保留≥99%；每 RX/TX/day 相对 raw accuracy≥−1 pp、相对 ordinary≥−0.5 pp；至少 5% 校正因子偏离 1 超过 1%；相对包功率误差≤1e−5，正实数形状误差≤1e−6。所有样本/抽样/分层保留。检查接线、非空或尺度变化不能代替身份收益。

科学 FAIL 仍完成预定双模型全部视图，再关闭 fullordinary-postRMS/E13/fourdraw 固定问题；不换归一化位置、部分系数、增益 clip、禁用某旧操作、抽样、RX 或 seed 救跑。PASS 只提供源尺度/身份依据，之后另登记三组 scratch E20（common raw.75＋ordinary.25＋原 SupCon；extra.25 raw／同一 ordinary／同 ordinary RMS，8L 前向/1.25 总尺度、原 U、默认头联合）；需同时超过两控制 clean≥+1 pp、F1 不降、每 practical/seenV≥−0.5 pp，再独立确认。该训练未实现/登记/启动，不自动训练或访问 target。

唯一 owner codex/root/F-ordinary-RMS-source-20261017-r02。现有 ordinary N607 CVS-RFFI Python，CPU/OMP/MKL/OpenBLAS 2、CUDA_VISIBLE_DEVICES 为空、GPU none；两个 worker 顺序，不干预健康旧任务。CWD PROJECT/releases/cvs_F_ordinary_RMS_fitL_20261017_r02；命令 CVS-RFFI/bin/python -u -m experiments.cvs_ordinary_rms17.dispatch，各 row runner 在 experiment.json。私有 runtime 的真实账户/路径不入 Git。

本地发布入口 F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_ordinary_rms17.publish，CWD 当前 managed codex/cvs-iq-timing-e20-20261016 工作树。Git commit 为代码身份，实际值保存在 release_commit.txt/resolved_config。输出根 PROJECT/runs/20261017-diagnostic-F-crossRX-ordinaryRMS-fitL-manysig-m2-r02 唯一不可覆盖；各 seed.stdout.log 与 release/source_ordinary_RMS.stdout.log。一个预检、一份 archive/一次传输 SHA 比较、一个 remote compile、launch 后 PID/CWD/argv/CUDA/log 增长或自然终态独立读回。smoke 为首 worker 第一项，PASS 立即继续；无 smoke token、额外签章或重复审查。

原输入/来源/角色/增强不符、输出碰撞、非有限或零能量、worker 异常保存 failure/partial 后停止，不自动重试。本地 19 输入/配置负例＋8 科学反例/96 合成 RMS 操作 PASS，0 真实 IQ/NN/更新；种子和日志 NaN 两项 P1 定向修复检查 PASS（执行参数未改）。一次独立 P0/P1 原审查及仅两项修复的定向复审见 review.json，不作额外审查层。

产物：root request/active/launch/result/completion、逐模型实际 native 配置/增强 callable/来源、raw 与四抽样完整配对 logits/判决、物理 ID/逐包尺度与 PA 统计、40 batch×4 draw×2 model 记录、完整日志、90720 源判决及 CE/F1/修复新错/全部分层/72 格/成本的独立复算、图、同 row 报告及登记。IQ、权重、原始判决、样本 ID、机器状态不提交 Git，只提交聚合报告元数据。

完整联合目标 ACTIVE：有效 A_g、稳定 A_s、完整契约 B00/B10/B01/B11 与交互项、跨 seed、正式 clean＋六 practical、low urban>60% 和 suburban 距自身 clean≤2 pp 探索目标均未完成。本候选区别于另一对话的 practical 正负噪声训练方向。

技术恢复说明：r01 在写配置日志时因原生 SatViewStage dataclass 不能严格 JSON 化而失败，源默认头预测为 0；旧根完整保留，不重试或覆盖。r02 仅将 dataclass 逐字段复制到日志并提前严格 JSON 校验，实际参数、日程、四抽样、模型、数据、全部科学判据与 NN 预算完全保持。原类 AST 复现与修复后字段 roundtrip、实际对象不变、原两项 P1 回归均 PASS；一次仅此运行 P1 的定向复审 PASS。r01 已发生 2 合成 NN/1 CP 加载/64 合成 ordinary，单列，不从 r02 90722 NN 中抹除。
