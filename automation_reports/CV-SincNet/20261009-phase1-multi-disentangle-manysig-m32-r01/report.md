# 旧门控＋cosine：独立线性、时间、接收机残差与交互多解耦四seed对照

- run_id：`20261009-phase1-multi-disentangle-manysig-m32-r01`
- group_id：`cvs-multi-disentanglement`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

一个身份骨干与多个独立解耦网络，分离辅助拟合和身份约束；8组×4seed scratch固定对照，全部冻结后全量7视图truth-last测试。

实现设计见 [DESIGN.md](../../../experiments/cvs_multi_disentangle/DESIGN.md)，逐项验收见 [traceability](../../../analysis/multi_disentangle_traceability.md)。这是用户确认的一个身份骨干、多独立解耦网络，不是一个域骨干的多输出头。

|实验臂|含义|
|---|---|
|legacy_cosine|原身份网络与旧门控、cosine 完整基线|
|fixed_interventions|相同 L/T 受控变化，使用观测变化，不学习解耦网络|
|unified|一个条件网络承担 L/T/R 三种任务，报告实测容量差异|
|linear|独立有效线性链路作用网络|
|temporal|独立包内指定时间变化网络|
|receiver|独立源域条件匹配 RX 组统计残差网络|
|multi|三个独立 L/T/R 网络共同约束身份骨干|
|multi_interaction|三个独立网络加独立、顺序明确的 LT 交互网络|

真实身份中间层 349 维由原网络的 base pre-LN 160 维、PA pre-LN 160 维和原前导观测 29 维组成；后段复用原融合与分类，不新增身份拼接或辅助 TX 分类器。这里的拼合只用于暴露原有中间状态，不增加身份信息来源。解耦网络均为训练辅助，部署时不分配辅助网络、优化器或源域组统计。

L/T 是接收后受控变化，不声称模拟完整原始通信链路。L 使用延迟 1、3 的复数弱抽头，系数尺度 0.06；T 使用常数、线性、二次相位基，尺度 0.08/0.10/0.06。作用码由各自的神经编码器从成对观测差异学习，已知系数只提供监督，不旁路替代网络。交互目标严格为 h11−h10−h01+h00，x11=T(L(x))。

R 仅使用 source L 中 TX/RX/接收质量分层的聚合充分统计，跨 RX 比较不是同一次发射的逐包反事实。在 multi/unified 中，其目标明确为组间总变化减去停止梯度的 L/T 预测作用；receiver 单支对照缺少 L/T，扣除项为零。身份约束使用 L+T+R 总作用保持跨 TX 的组间相对几何，不强制任意同 TX 包逐向量相等。组覆盖不足时记录跳过原因，不构造假配对。这是依赖模型的统计残差分解，RX 线性滤波和传播信道仍有不可辨识部分。

20 epoch 预热后每 4 个身份更新处理最多 32 个 source L 样本，单独 AdamW 拟合辅助网络；该阶段 EMA 锚点停止梯度。随后冻结辅助参数，用受限预测变化约束身份骨干。U 仍只使用原生旧门控/全 U 熵，V 不更新状态。主更新固定 44400 次，额外视图、拟合步骤、梯度、实际参与次数和计算成本单独记录。

不同路径数采用固定配方的总体归一化；固定变化对照只有 L/T，R 没有精确逐包真值。统一网络容量近似匹配而不是精确相等；三网络 39992 参数、统一网络 41552 参数，实际配置再次读回。加入交互额外使用有序组合视图，不能把其效果完全归因于参数数目或纯架构。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

所有 32 行从零训练，没有继承旧模型、teacher、优化器或域统计；teacher 只来自本行自身 EMA。model seeds 为 2026092701 至 2026092704，物理角色划分沿用既定契约；辅助初始化 seed=model_seed+11939，独立增强 seed=model_seed+39071。禁止 resume 和来源不明初始化。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

固定 32 行全部完成源训练后封存各自行 E200，再完成 32×7=224 组预测；每个视图 168000 个物理 query，独立 scorer 最后连接 truth。保留总体、7 个 RX、6 个 TX、4 天的指标，以及跨 4 seed 均值、SD 和配对差值。预计 3136 条总体/RX/TX 记录、896 条日期记录、208 条配对汇总。

使用现成 clean 与六种完整 practical 代理视图。这是已暴露基准，不是新盲测或真实在轨验证；Phase2/K/适应前后/新增类/H 为 N/A。测试结果不参与本矩阵重新选模、改超参或选择性重跑。

本地完整矩阵合成评分已通过：32×7 预测闭合、3136/896 指标及 208 配对统计，不完整预测会拒绝评分。该验证未读取真实 target；不构成性能结果。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 本地验证与独立审查

模型、训练接入和验证分别由三个子agent负责，主agent完成注册、评分及发布。独立审查闭环后剩余 P0/P1 为 0。CPU/CUDA 结构、真实 E/G 等价、独立网络梯度、零作用/界限、LT 顺序、R 残差目标、双向梯度隔离及仅身份推理检查通过。

原生训练链使用合成 L/U/V 在 1/21/22/23/24/80/131/132/200 epoch 执行各一次更新，覆盖八组正式损失、优化、EMA、日志和辅助状态保存；这不是完整 200 epoch 训练。CPU 新基线与旧路径全部模型状态逐位一致。GPU 注册策略不启用 cuDNN 确定性，原逐位断言失败的产物保留；聚焦重复诊断显示初始权重和全部 9 步原生 RNG 一致，旧旧最大权重差 1.1920928955e-7，旧新 2.9802322388e-8。GPU 差异处于观测到的原生重复波动内，不宣称 GPU 逐位可复现。

实际执行命令为 `python -m experiments.cvs_multi_disentangle.checks --device cpu/cuda:0 --output ...`、`python -m experiments.cvs_multi_disentangle.smoke --device cpu/cuda:0 --output ...`、`smoke --baseline-probe`、`smoke --summarize-gpu`，以及 `python -m experiments.cvs_multi_disentangle.evaluation_checks --output ...` 和交付器的独立重算。解释器为 `C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8`。小体积验证证据收录在 [evidence/local_verified.json](evidence/local_verified.json)。完整矩阵合成评分和交付复算通过，没有读取真实 target。

主体架构及训练职责按批准设计实现。统一对照容量为近似匹配；R 为统计残差而非可辨识的纯硬件成分；额外同因子双路径属于设计中的可选后续扩展，未加入首轮。正式训练完成、机制持续执行及测试性能仍待远端产物验证。

## 发布读回

VERIFIED。代码提交：`c037af5a19d5973486942a090bced9cbae070d38`；控制器 PID：2128714。登记 32 行，已启动 12 行，实际配置已读回 12 行，容量等待 0 行，尚在队列 20 行。全部 GPU 总实验进程数不超过 4。证据见 [launch_verified.json](evidence/launch_verified.json)。这是发布时快照；20 epoch 预热尚不能证明正式辅助机制已执行。

完整训练和测试尚未完成，性能提升未知。控制器自动执行固定 E200 冻结、全部 224 组预测和独立 truth-last 评分；本地只读观察器收取并独立复算产物后交付报告。

观察器 PID 68664 已通过独立 Win32_Process 与启动产物读回验证，见 [observer_verified.json](evidence/observer_verified.json)。N607 提交版本的 CPU 检查与八臂原生训练链预检均 PASS，见 [remote_preflight_verified.json](evidence/remote_preflight_verified.json)。代码推送 OID 证据见 [code_git_verified.json](evidence/code_git_verified.json)，草稿 [PR #8](https://github.com/niubility1317/CVS-RFFI/pull/8) 已更新，未合并。

## 完整测试交付

ANALYZED / VERIFIED。完整结果见[结果表](results/full_results.md)、[逐RX/TX](results/scores.csv)、[逐日期](results/day_scores.csv)。所有负结果完整保留。

## 本轮结果解释

32模型×7视图完整测试与独立复算已完成。全部场景、逐seed、Macro-F1、最差RX及关键配对差值见[完整汇总](results/complete_summary.md)。本轮L/T/R主方案未高于同轮基线，不能把已执行等同于有效解耦。全部32行实际机制记录VERIFIED，见[执行计数](results/mechanism_execution_summary.json)。
