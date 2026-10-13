# 特征提取四组：历史旧门控＋EMA＋cos E50 预登记

状态：RUNNING，远端启动 VERIFIED。主方案 JOINT_MS50 的验收为三个固定种子 clean 准确率均值至少 84%。性能尚未证明，全部负结果保留。

用户已明确优化网络特征提取能力，分类头保持现有结构，并明确选择历史 batch_neighbor 伪标签门控＋EMA＋cos 完整配方作为主对照。此前纯 CE E50 保留为独立历史实验，不用它替代本轮基线。用户允许自行选择训练 epoch，原则不超过每模型 50；本轮全部从零 50 轮，以历史 130＋70 的阶段比例预先缩为 33 轮 label＋17 轮 pseudo。原 E200 仅为配方参考，不继承其权重或成本。

| 组别 | 特征提取改动 | 可训练身份参数（本地实际） |
|---|---|---:|
| OLD_GATE_COS50 | 原时序与 PA 特征块 | 177025 |
| TIME_MS50 | t1/t2/t3 多尺度深度卷积＋残差映射 | 237697 |
| PA_MS50 | pa_b1/pa_b2/pa_b3 多尺度卷积＋残差映射 | 291585 |
| JOINT_MS50 | 两个分支同时改造 | 352257 |

每组种子 2026092701、2026092702、2026092703，共 12 个模型。时序 kernel/dilation 为 (3,1)、(5,1)、(5,2)；PA kernel 为 3、5、7，逐层保留 dilation 1、2、4。保留原 norm、activation、pool、dropout、输出通道和最终 160 维。分类器始终为原 ResidualPhysicalHead、cosine6×160、scale30、原 branch LayerNorm/gain/融合接口；其参数按原损失正常训练，没有新增分类器、温度或校准规则。所有组一致保留既有 NoExplicitCFO 身份输入约束与内部逐包补偿。

同一历史训练配方应用于全部四组：EMA=0.999；rx/day quantile=0.86；batch_neighbor 时间门控，window2、最低 confidence0.8，强视图一致；lambda_u=0.16、lambda_ent=0.01、label_smoothing=0.01；AdamW lr0.0002、weight_decay0.0001、clip5，50 轮 cosine 最低0.000001；L batch128、U batch256、222 steps/epoch。每个模型固定 11100 个成功 optimizer/EMA updates，最后第 50 轮冻结。历史非星地弱/强视图及 EMA 执行顺序沿用原生实现；卫星拼接、卫星一致性和相关损失关闭。不能把本轮写成纯 CE 或与历史 E200 同成本。

使用原 ManySig 物理契约：L_s6300、U_s56700、V27000，源 RX1/3/4/6/8，原 label map、RX/day/TX 与物理 ID 划分不变。所有组都使用 U_s，TX 在原生 batch 返回前隐藏，禁止访问 U_s 真值。V 保留原只读健康诊断，没有拟合、选模或性能门槛；target 与 truth 不进入训练、结构或参数选择。

按最小研究流程先派发首种子四组。四组均完成固定完整预算并冻结，且每组最后 222 次更新的 source train/loss_tx_labeled 均值 ≤0.5、源产物合法后，才扩展任意组的另两种子；读取 source completion 和 freeze，不读取 V 精度或目标结果。这个粗源拟合条件不证明目标泛化，也不缩短首种子训练。失败时保留已运行行并记录 SOURCE_FALSIFICATION，不自动选择性重跑；健康活动行继续至固定闭环，不被终止。

每行完成与冻结后立即执行 clean 168000 条预测，不等待整批训练结束。该行完整 prediction 固定后通过 exec 启动全新独立评分进程，才接入 truth。每行报告 accuracy、Macro-F1、RX/day/TX 与混淆矩阵；全部 12 行完成后按三种子平均、样本标准差和固定配对差汇总。完整矩阵需有 2016000 predictions、216 指标记录，累计 133200 身份更新。主目标只由预定 JOINT_MS50 判定，不由控制组或最高单种子代替。测试基准已经在历史探索中暴露，本轮不是新的独立盲测；不按目标分数调参、排序、选模、停止或选择性重跑。

唯一 launch owner：codex/root/feature-multiscale50-oldgate-20261013。N607 普通账户现有 CVS-RFFI Python，CWD 为 releases/cvs_feature_multiscale50_20261013_r01；命令为 Python -u -m experiments.cvs_feature_multiscale50.dispatch。不可变 run 根为 runs/20261013-phase1-feature-multiscale-oldgate-cos-e50-clean-manysig-m12-r01；本地证据在 local_artifacts/cvs_feature_multiscale50_20261013_r01。全部绝对路径、逐行命令、环境与输出在 experiment.json。

Git commit 为代码/配置身份，发布时记录 actual release_commit.txt。一份 release archive 一次本地/远端 SHA 比对、一次普通账户预检和远端 compile，真实同契约 CE30 checkpoint 无 query smoke（不继承）及新结构检查 PASS 后立即训练；不要求额外 seal、receipt 或审批。每 GPU 最多3个总实验进程，包含其他 owner；至少12000MB空闲，保留其他健康任务。发布后一次 PID/CWD/argv/GPU 与日志增长核验。

系统性技术停止规则：错误数据/模型来源、隐藏 U TX 或 target 进入训练、错误 checkout/输出冲突、非有限值、不能运行或产生合法预测与独立评分、dispatcher 全局故障。停止新派发并保留所有产物，先核实后状态再恢复；不停止或重启健康旧任务。源端粗拟合失败按上面的预登记科学条件处理，不读取目标分数。

预期产物：resolved_config、resolved/actual_native_args、source_contract、initialization、初始无 query smoke、final_ssdg.pth、11100 step 和 50 epoch 完整日志、source completion/frozen、prediction/provenance/complete、独立 scores、row completion、全矩阵 analysis/final_analysis。数据、checkpoint、日志及预测只在原存储保留，Git 仅保存代码、配置、登记与报告。

本地已通过真实四种网络前向、所有新增分支有限非零 CE 梯度、classifier 与同 seed 原基线完全一致、参数优化器覆盖、strict state round-trip、EMA 副本与隐藏 U 批次、epoch51 和错误 update 预算拒绝；历史配方误改及错误输出/运行身份拒绝、首种子 source-only 扩展、prediction 不完整时 truth 拒绝、完整三种子主目标与 12 行闭环、隔离归档依赖导入和现有 NumPy 回退数值等价。独立 P0/P1 检查发现并修正原生 15 个 RX/day 域与预测模型 5 域不一致的问题；训练式 15 域模型至预测模型严格加载检查及四组首种子阶段语义均通过独立定向复审。未访问 query/truth。

每轮实现、配置和对应结果报告提交并推送到 Git，独立核对远端 OID。结果报告保留逐种子指标、固定配对差、负结果与缺项；原始权重、预测、日志和数据留在 run 根，不入 Git。

发布证据（2026-10-13）：实际运行代码 commit `5fa536f9ca2d90e7d30d547d58503a41b612d57f` 已推送，独立远端 OID 核对一致。一份归档比对、远端编译与真实同契约无 query smoke 均通过；首种子四组已完成第 1 轮并继续训练，PID/CWD/argv/GPU、日志增长及容量核验通过。实际 native args 的 EMA、U、batch_neighbor、强视图一致、cosine、50/33/17 与四组特征层位正确，分类头均为原 ResidualPhysicalHead。运行证据在 `local_artifacts/cvs_feature_multiscale50_20261013_r01/readback.json` 与 `runtime_verified.json`。尚无完整三种子目标成绩，不宣称达到 84%。

阶段执行核验：首种子四组分别到第 39/41/41/40 轮，均已进入 pseudo 阶段。实际 EMA 更新和非零无标签损失持续执行，U_s 真值可用标志为 0，非有限损失及梯度跳步均为 0。该源端记录证明本轮确实使用历史半监督配方；不等同于完整日志审核、源域终轮门槛通过或目标性能提升。证据为 `local_artifacts/cvs_feature_multiscale50_20261013_r01/pseudo_phase_verified.json`，其间未读取目标分数。其余 8 行仍等待四组完整 sourcefreeze，当前无失败记录。
