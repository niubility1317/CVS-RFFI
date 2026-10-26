# NoSIG 双图标签质量：冻结源诊断

run：`20261026-diagnostic-PAMF-NoSIG-RGPL-sourceGraph-m1-r01`。用户要求先全面复盘，再用去 SIG_TIME 的网络研发兼顾原始 PAMF 与 DURA 配对性能的半监督方法。本行只检验新标签是否可信，不训练候选、不访问目标，也不代替三种子性能。

## 固定输入与方法

单个 model seed2026092701 的新 NoSIG-PL E30，来自本轮匹配基线的固定 `segment30`（本段10轮2220步，累计30轮6660步）。核实原 NoSIG E20 从零祖先、实际无 sequence_pool 参数、原6×160默认分类头、源数据合同一致和无 target／U truth／原生 EMA。G 为本契约原固定 G-LU，保持参数与运行路径。旧带 SIG 的教师权重、特征、概率不读入。

原 L6300、U56700、V27000 的物理划分不变。每包一份 raw 特征和12份 G 视图的归一化均值，12份来自四个预先固定的源信道实现×三景。U复用完整同G恢复 IQ 缓存；L/V按同命名空间生成。只用 L/U63000 节点分别建 raw/restored 两图，只有真实 L 标签是传播种子。不同 RX mutual20NN、边温度0.2、对称度归一化、传播系数0.9，最多200步，最大绝对残差≤1e−6。合并时排除自身 RX 的标签种子贡献。

仅处理冻结 raw confidence<0.95 的困难池。两图类别相同、各自 confidence≥0.95，且各图均有至少三个其他 RX 的非零传播组支持该类别，才认证。每组 argmax 非唯一时不计支持。V 只做独立20NN扩展，须进入源邻居原半径；源节点、度与传播量全冻结，无 V 节点或 V 间耦合。先保存全部 U/V 认证预测，再使用 V 标签计质量；U 真值始终缺席。

## 科学停止规则与输出

进入候选训练前的固定可行性条件：新增认证 V≥200，且实测 precision≥99.5%。同时报告完整 RX/day/TX 认证数、正确数、纠正数、引错数与净纠错。**不通过即停止这条候选，不调 k／阈值直到过线，不打开 U 真值。** 科学失败仍归档为完整诊断；通过也不自动启动训练，更不代表目标增益。

预期输出：真实 checkpoint L2 smoke；L/U/V 特征和无 U 真值的元数据；两图传播概率、收敛证据与图边数；冻结 U 伪标签、V 预测；质量分层与完整报告；全部 F/G 参数和注册 buffer 不变。主提取逻辑 F1170000包、新 G399600包、复用 U 缓存 G680400包；另计 smoke F4/G2包。零优化器、零新头拟合、零目标访问。实际运行时间和内存另外记录。

## 执行与技术停止规则

代码／配置唯一身份为执行时的 `runtime.code_commit`，本登记与代码同次提交。命令：`PYTHON -u -m experiments.pamf_channel_pl.rgpl_source_graph dispatch --config PROJECT/releases/20261026-diagnostic-PAMF-NoSIG-RGPL-sourceGraph-release-r01/runtime.json`。环境为 N607 既有非特权用户与 CVS-RFFI Python/Torch2.1、FP32/TF32off、CPU2线程；CWD 为同一 release；不安装依赖。

唯一 owner：`codex-pamf-nosig-rgpl-graph-20261026`。独立不可覆盖 run/release root。GPU0至7中取全局 CUDA+pending<2且空闲≥10000 MiB 的一张卡，只一条本行 GPU worker；保留所有健康任务。只进行一次 archive 传输比对及远端编译；真实 checkpoint smoke 为 worker 的第一步，通过即继续。独立核实 PID/CWD/argv/GPU/日志增长。

遇实际合同／模型身份／smoke／非有限／覆盖／状态变化／图不收敛或其他技术失败，该 worker 结束，保留所有产物，不自动重跑，不停止其他 owner。阶段状态和质量结果从实际文件核实，不用退出码单独证明完成。

方法设计：[PAMF-RGPL](../../../docs/goals/PAMF_NOSIG_RGPL_DESIGN_20261026.md)。复盘：[完整历史结果分析](../20261026-analysis-PAMF-SSL-retrospective-r01/report.md)。
