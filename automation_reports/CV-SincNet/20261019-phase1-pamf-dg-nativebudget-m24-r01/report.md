# PAMF Phase1 四种域泛化方法与自身CE：两GPU三种子

- run_id：`20261019-phase1-pamf-dg-nativebudget-m24-r01`
- group_id：`phase1-pamf-four-dg-nativebudget`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按 events.jsonl 及独立证据更新）

## 目的与对照

固定当前弱标签物理划分，四方法按各自配方及用户E100覆盖，从零训练与七视图truth-last评分。

本次固定 4 方法×full/自身骨干 CE×3 模型种子，共 24 行；唯一矩阵在 experiment.json，matrix.json 是发布用配置。模型种子为 2026092701/02/03，物理数据划分 seed392005 不变。PAMF 复用历史对应三个种子的 E200 clean 结果，训练成本单列，不复用作者全监督权重。

| 方法 | 训练预算 | 优化与冻结 | 依据 |
|---|---|---|---|
| RIEI-FD | E100，batch64 | SGD1e-4、momentum0、恒定 LR；full 按分类与解耦交替，固定末轮 | 用户明确 E100；原论文未披露总轮数；学习率/λ来自原文，short_stem1d 为历史项目实现选择 |
| DRIFT v2 | E100，batch256 | Adam1e-4、恒定 LR；原式负平方 L2 和，固定末轮 | 用户明确 E100；batch/LR/λ和公式来自 v2 |
| IB-RFF | 两阶段各最多 E200，batch128 | Adam1e-3；每阶段源 V NLL 降 LR、patience10 早停和选模；stage2 重置优化器 | 固定作者公开 ManySig 配方；α0/0.04，CE 对照为 0/0 |
| SDG TWC | E200，batch32 | Adam1e-3，源 V 最小 NLL；full 保留作者增强和对比目标 | 固定作者 WiSig S 配方；增强深度2/1、λ1/100 |

RIEI/DRIFT 由历史 E200 改为 E100，所有轮次安排按 1/2 解释；本次原恒定 LR 配方没有 warmup、milestone 或阶段启用节点，因此不凭空添加日程。IB-RFF 每阶段 patience5 的降 LR 与 patience10 的早停保持原值，不压成 E100。论文未披露轮数的两项不冒称“论文推荐 E100”。

CE 对照仅训练身份 CE，不计算 RX 诊断、域损失、原生 DG 增强或昂贵零权重分支；默认身份头与骨干联合训练。完整方法所必需的 RX 分类/对抗分支是算法训练组成，不是冻结后的额外诊断探针。SDG 移植到固定多 RX 源池；RIEI/DRIFT 保持本项目 equalized1/unit-RMS 输入。以上适配及非等计算量均在结果中披露，不称严格原论文完整表复现。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

仅物理 GPU0、GPU1；每卡一条本队列 row。队列等待外部 GPU 工作负载，不借用 GPU2 至 GPU7，不停止或热改任何其他任务。source/predict/scorer 为独立进程，训练和预测占用同一 row 的 GPU 槽；scorer 禁用 CUDA。CPU 数学库各限 2 线程。

旧 source_contract 直接复用，源角色为 RX1/3/4/6/8、day1/2/3，L_s6300/U_s56700/V27000；U_s 不导出不参与训练。source cache 只导出已有 L_s/V，物理 ID 按现有契约匹配，不新建数据校验链。数据代码复用不可变 SIG_TIME 发布及其 Git 提交 6408f29，外部模型/损失源码复用已固定的作者发布；这些路径及实际解释器保存在 local_artifacts/phase1_dg_nativebudget_20261019/runtime.json，不将私有运行位置或大产物提交 Git。

launcher 首先保存并重新加载真实 L_s batch 产生的模型 checkpoint，完成默认头无 query smoke，PASS 后直接进入队列。每 row 完成全部七视图 prediction 后独立 scorer 才连接原 truth；不因其他 row 未完成而等待，也不按已得成绩改变矩阵。七视图使用已有 VALIDATED_ONCE、168000 个物理包的 fixed received views，不重新生成信道。输出 root 不可覆盖。

只因数据/模型契约错误、输出碰撞、非有限源损失/梯度、无法启动或生成合法预测、错误 truth 连接等技术故障失败；两行技术失败后暂停未开始行，已健康运行行继续。低性能不停止、不扩种子、不选择性重跑。失败目录完整保留。

预期每行保留各训练阶段 epochs.jsonl、checkpoint、source freeze、七视图匿名预测、逐 RX/TX score。最终 summary.json/final_report.md 自动生成实际种子数、模型种子样本 SD、最差 RX 和 full−CE 配对差；未完成三种子明确标缺。代码提交以本次 package.json 和 release_commit.txt 读回为准。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 发布前验证

LOCAL_VERIFIED。24 行种子/方法预算检查、source 配置拒绝 truth/resume/teacher/target 输入、错误预算拒绝、prediction 不完整时 truth 保持关闭、四方法默认头 CE 梯度与实际更新、严格 checkpoint 往返及错误 source contract 拒绝均通过。另核对 DRIFT 原式在 256 维常数差上的负平方距离为 −256；SDG 原生两视图、两个对比项梯度和身份种子未被作者模块导入时的 seed2023 覆盖，检查通过。原生增强在隔离 NumPy1.23.5 下检查，未修改共享环境。

一次独立 P0/P1 正确性审查 PASS，无待修直接缺陷。选定 sixscene manifest 已读回为 VALIDATED_ONCE、168000 个物理包、完整六场景、固定六类和 source/target 互斥；旧三场景 capsule 仅供应相同的 clean IQ/原始 ID，二者不混淆。已发布并开始首行训练，尚无本次性能结果。

发布依赖核对发现 Git 内 RIEI 通用 trainer 带有未打包的卫星增强 import；本次发布改用独立 source-only 交替更新 helper，未改动其他任务的共享 trainer。新旧 source-only 更新后全模型张量逐元素一致检查通过，该单一 P1 的定点独立复审 PASS。未扩大为重复全量审查。

## 远端执行读回

VERIFIED：代码提交 ac5163f6cb59b49a657c3ee02089f8e556ae2279 已推送，远端分支 OID 与本地一致。一次发布归档传输 SHA 比较和远端 compile 通过。dispatcher PID2008297 的实际 CWD/argv 对应该不可变发布；源缓存完成，L_s6300/V27000，未导出 U_s。

真实 L_s batch 的八个 full/CE 默认头 checkpoint 往返 smoke 全部 PASS，未访问 query，检查权重已丢弃。首行 ibrff-full-s2026092701 在物理 GPU0 正式训练；GPU1 出现其他 owner 的健康任务，队列等待空闲，不借用 GPU2 至 GPU7，不干扰其进程。

独立证据保存在 local_artifacts/phase1_dg_nativebudget_20261019/postlaunch_third.json。本次尚无冻结后的测试结果，不作性能结论；所有行完成后自动生成同 row 七视图评分、三种子统计和 full−CE 配对差。

首行训练日志已从 0 增长至 10296 字节，stage1 已到 E27，损失和默认头源 V 输出有限，实际训练子进程为 PID2008921，GPU 暴露值为 0。证据：local_artifacts/phase1_dg_nativebudget_20261019/postlaunch_training.json；确认 RUNNING，不等同于矩阵完成。

## 首行完整产物（单种子，非矩阵结论）

VERIFIED：ibrff-full-s2026092701 已为 ANALYZED。源 V 规则使 stage1 在 30 轮、stage2 在 84 轮早停；默认头 best checkpoint 冻结后完成各 168000 包的七视图预测，prediction 明确 truth_read=false，独立 scorer 为 truth_last=true。当前实际完成 1/24 行，CE 配对行已自动开始。

| 视图 | 准确率（%） | Macro-F1（%） |
|---|---:|---:|
| clean | 64.7458 | 63.7213 |
| practical_high | 23.1482 | 16.5380 |
| practical_mid | 22.9357 | 16.4425 |
| practical_low_suburban | 22.4601 | 16.1526 |
| practical_high_urban | 22.7458 | 16.7536 |
| practical_mid_urban | 21.1226 | 17.0713 |
| practical_low_urban | 19.7619 | 16.7135 |

证据为 local_artifacts/phase1_dg_nativebudget_20261019/firstrow_artifacts.json，含 freeze、prediction complete、独立 scores 和 row complete 原始读回。保留负结果，不依据此分数修改后续训练。PAMF 历史对应三个种子的 E200 clean 基准可与后续同 seed 结果配对，但预算和已测视图单列。

## 后续读回与IB-RFF完整组三种子结果

最新独立读回确认dispatcher PID2008297及SDG首行worker实际存活，IB-RFF full／CE各三个seed共6行ANALYZED，SDG full首行到E115，其他行尚未开始；无FAILED。旧1/24描述保留为当时快照。

IB-RFF六行全部七视图预测已独立复算，42个总体CM及Accuracy／Macro-F1与原scorer逐值一致。[完整同row结果、日期分层与训练成本](ibrff_analysis.md)。clean full64.4792±2.9341%，CE63.4599±1.7316%，配对差+1.0192±1.8158个百分点，2/3正差，不能据此称稳定有效。

本次目标范围已明确排除PAMF，详细实施范围见docs/PHASE1_EXTERNAL_DG_EXECUTION_PLAN_20261019.md。新增162训练和6补测尚未发布，原24行配置未变。

## SDG首个seed独立核验（2026-10-19T12:26:50+08:00）

两行均完整训练200轮、39400更新，按源V最小NLL分别冻结E179和E39。七视图实际预测独立复算通过；仅一个seed，不报告三seed均值。见[首seed配对分析](sdg_seed01_analysis.md)。其余健康SDG训练继续，参数和队列顺序保持冻结。

## 第二seed SDG-full完成读回（2026-10-19T12:43:36+08:00）

实际E200训练、source freeze、168000包七视图prediction、独立truth-last评分及row complete齐全，原dispatcher正常回收并标ANALYZED，当前9行完成。原worker退出早于一次状态刷新，随后已读回正常闭合，未做重启。证据local_artifacts/phase1_external_dg_20261019/sdg_full_seed02_completion.json。其余健康任务继续；家族三seed未齐，不在这里形成最终均值结论。

## SDG第二seed独立配对核验（2026-10-19T12:47:27+08:00）

full与CE均完成E200、39400更新及七视图；独立复算CM／Accuracy／Macro-F1通过，实际冻结为源V最小NLL的E187／E32。见[第二seed配对分析](sdg_seed02_analysis.md)。目前SDG配对2/3，原队列至少10行完成；第三seed继续，不形成最终三seed结论。
