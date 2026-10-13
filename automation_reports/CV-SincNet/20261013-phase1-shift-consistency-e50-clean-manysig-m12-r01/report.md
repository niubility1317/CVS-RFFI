# E50 源时间偏移与浅中层统计一致性：预登记

当前状态：PLANNED；尚未发布或访问目标预测。目标为 SHIFT_STATS_SSL50 的三个主种子 clean 准确率均值至少 84%。未达标结果全部保留。

用户在当前对话允许自行选择训练长度，原则不超过 50 个 epoch。本候选预先固定每个模型从零训练 50 个 epoch。此次矩阵沿用已明确的普通 cosine 头、160 维身份表示、NoExplicitCFO、只测 clean。历史 E20、E30 的训练预算和记录不改写。本路线复用已引用任务明确的 native 身份网络；不把它误称为默认 residual_fusion。

| 组别 | L_s 损失及机制 | U_s |
|---|---|---|
| CE50 | clean 身份 CE | 关闭 |
| SHIFT_CE50 | clean 与同包非循环时间偏移 CE，各 0.5 | 关闭 |
| SHIFT_STATS50 | 上一组加浅中层共同有效区域统计一致性，权重 0.05 | 关闭 |
| SHIFT_STATS_SSL50 | 上一组加自身学生软概率 KL，一致性权重 0.25 | TX 隐藏 |

每组种子 2026092701、2026092702、2026092703，共 12 个模型。每个模型 50 × 222 = 11100 次更新，L_s 物理样本累计暴露 1420800 次。前 1110 次更新只做 clean CE；之后以固定循环 -2、-1、+1、+2 个采样点构造同一包的非循环平移。统计损失和 U_s 损失用 2220 次更新线性升至预定权重。CE50 有 11100 次 L_s 模型前向，其余组有 21090 次；半监督组另有 1278720 次 U_s 物理样本暴露、19980 次无梯度弱视图前向和 9990 次强视图前向。固定 epoch 和更新次数不代表计算成本相同。

统计约束位于实际 time_down 与 t2.pool 层，比较共同物理有效内部区域的逐通道均值和标准差，拼接后归一化。它不宣称分数步长逐点等变，也不消除全局 GroupNorm 对边界的依赖。半监督只使用当前学生，两个弱视图预测一致、最低置信度 0.95，且在相邻两次完整 U_s 遍历中稳定才接受。clean 软概率停止梯度；强视图为时间偏移加标准差 0.015 的噪声。无 EMA、预训练教师或隐藏 TX 真值。

训练固定 AdamW、峰值学习率 0.0002、最低 0.000001、50 轮 cosine 日程、weight decay 0.0001、batch 128、clip 10、label_smoothing 0，177025 个可训练参数。source_loader_config 中历史 E20 信息只提供既有合法数据角色及结构，实际训练长度由本候选 50 轮控制。V 仅核验角色，不前向、拟合或选模；不使用星地信道增强。训练初始化从零，没有 checkpoint 或其派生状态继承。一次真实 CE30 checkpoint 的无 query smoke 仅用于兼容性验证。

假设依据为固定源域六方向诊断 r03：三个冻结种子的单位时间偏移决策改变率分别为 2.2705%、1.9807%、3.3575%。该源现象支持检验偏移一致性，不证明目标提升。一般动机参考 [Zhang 2019](https://proceedings.mlr.press/v97/zhang19a.html) 和 [FixMatch](https://papers.nips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html)；本软 KL、历史门控实现并非原 FixMatch。其 RF 迁移有效性仍需本轮结果检验。本方案、参数和顺序不使用目标分数；现有测试基准已经暴露，不声称独立新盲测。

固定数据契约为原 ManySig source_contract：L_s 6300、U_s 56700（仅半监督组读取，TX 隐藏）、V 27000，源 RX 1/3/4/6/8；物理角色及 target 168000 个 clean 物理 ID 复用已核验契约，不重新划分。每 row 固定最后第 50 轮权重，自动进入 clean 预测。全部 168000 个预测固定并写入完成产物后，独立新评分进程才接入 truth。每 row 报告准确率、Macro-F1、RX、day、TX 和混淆矩阵，最终汇总三 seed 均值、样本标准差、四组固定配对差及实测成本。目标分数不回流调参、选模、排序或选择性重跑。

Git commit 为实际代码和配置身份，发布时记录于 release_commit.txt。环境为 N607 普通账户现有 CVS-RFFI Python；CWD 为发布目录 releases/cvs_shift_consistency50_20261013_r01。启动命令为 Python -u -m experiments.cvs_shift_consistency50.dispatch；逐 row 由同一 dispatcher 派发 worker。唯一 launch owner 为 codex/root/shift-consistency50-20261013。不可变输出根为 runs/20261013-phase1-shift-consistency-e50-clean-manysig-m12-r01，本地回读根为 local_artifacts/cvs_shift_consistency50_20261013_r01。完整绝对路径、命令及物理数据句柄见 experiment.json。

每 GPU 最多 3 个总实验进程，包含其他 owner；需要至少 12000 MB 空闲。当前健康任务不停止、不重启、不热改。发布只进行一次普通账户预检、一个 release archive 本地/远端 SHA 对比、一次远端 compile 和真实 checkpoint no-query smoke，PASS 后立即进入训练。发布后核验 PID、CWD、cmdline、GPU 和日志增长。

系统性技术停止条件：数据权限或 checkpoint 来源错误、目标或隐藏 TX 进入训练、错误 checkout、输出冲突、非有限值、无法运行或产生合法预测及独立评分、dispatcher 全局故障。保留全部产物，先核对后状态再恢复；不因性能低停止或选择性重跑。预期产物包括 resolved_config、source_contract、50 轮与 11100 step 全日志、identity.pth、source_frozen、prediction complete、预测文件、独立 scores、每 row completion 和全矩阵 analysis。

本地已通过：四组真实网络梯度与前向次数、浅中层统计梯度、教师停止梯度、非循环偏移与物理共同区域、完整 E50 计数、epoch51 和越权 U_s 分支拒绝、独立 truth-last 顺序及 12 row 汇总、归档选定依赖隔离导入。一次独立 P0/P1 缺陷检查待完成。

一次独立 P0/P1 检查 PASS，未发现阻断问题。配置与预登记、角色、U_s 门控、浅中层损失、fixed final50 和独立评分闭环一致。下一步按本固定矩阵发布。

实际发布：VERIFIED / RUNNING。代码 commit 319ed5bd5，与远端分支 OID 一致；单归档本地/远端校验、远端 compile 和真实同契约 checkpoint no-query smoke PASS。12 个模型已进入训练，逐 PID、CWD、argv、GPU 和非零 epoch 日志增长已独立核实；每 GPU 实际 1 至 2 个进程，当前无 failure。只有 SHIFT_STATS_SSL50 的 U_s 迭代和双弱视图启用，V 前向、EMA、星地生成和训练 checkpoint 继承均关闭。首次调用超时经后状态确认无归档/目录/进程，无远端重复发布；继续调用后观察到实际 submit 和训练产物。尚未完成矩阵和评分，84% 未达成。

完整收尾：VERIFIED / ANALYZED。12 models×50 epochs 全部训练、clean prediction、独立 truth-last scoring 完成；600 epochs、133200 steps、2016000 predictions、216 same-row records 全日志和原始指标独立重算通过。主方案三seed clean 80.0224%，84% 未达到。详细固定配对差和实际成本见 final_results.md。无失败或缺项，没有依据目标分数重跑。
