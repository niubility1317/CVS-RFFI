# reference_response 原 Phase1 机制叠加：LEO×MixStyle 四 seed 消融

- run_id：`20261004-phase1-reference-overlay-r1-manysig-m16-r01`
- group_id：`cvs-reference-original-phase1-overlay`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定用户指定 reference_response，从零进行 CE/LEO/MixStyle/LEO+MixStyle 的 2×2 消融；完整源域矩阵冻结后，全部固定行完成 clean 和卫星测试及独立评分。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 详细设计与发布依据

# reference_response逐步叠加原CVS Phase1机制

## 目标与边界

固定用户明确指定的`reference_response`身份网络，先回答原机制移植后是否提高识别性能，以及机制之间是否相互促进或抵消。性能优先，参数与计算成本作为次要指标。第一轮每个模型177025个参数，从零训练，不继承任何历史checkpoint、EMA、teacher、优化器或原型状态。

结构已经在历史暴露基准上探索，本次属于`BENCHMARK_INFORMED_FIXED_DESIGN`，不声称是全新盲测或完全未接触目标结果的架构选择。本轮测试结果只用于报告固定比较，禁止回流到选模、结构、超参数、seed剔除或选择性重跑。

## 第一轮：本次发布的16行

| 行组 | 网络 | LEO拼接 | MixStyle | 目的 |
|---|---|---|---|---|
| ce | reference_response | 关闭 | 关闭 | 本轮同预算对照 |
| leo | reference_response | 原采样器与物理实现 | 关闭 | 单独测LEO增益 |
| mixstyle | reference_response | 关闭 | 原实现、原位置 | 单独测MixStyle增益 |
| leo_mixstyle | reference_response | 开启 | 开启 | 测叠加与交互 |

每组model seed均为2026092701、2026092702、2026092703、2026092704。逐行配置在`configs/source-<arm>-s<seed>.json`及对应`predict-*.json`。launch_spec固定全16行，不根据源域名次取消测试对照。

### 固定数据与优化预算

- ManySig六个TX：14-10、14-7、20-15、20-19、6-15、8-20。
- 源RX索引1、3、4、6、8，day1、2、3；目标RX索引0、2、5、7、9、10、11。
- 复用既有source_contract物理ID，split seed392005，L/U/V为6300/56700/27000。第一轮仅L训练，V只读验证，U保留不用。
- equalized=1、中心256点、逐包RMS归一化、25MHz。LEO是其上的既有practical residual/post_sync/noeq变换。
- 200epoch，batch128，不drop_last，每epoch50步，共10000次优化更新。AdamW，lr2e-4、weight_decay1e-4、cosine最低1e-6；无梯度裁剪。
- 全FP32，关闭cuDNN/matmul TF32。model/loader/MixStyle随机性随model seed；LEO独立generator固定2027，receiver seed2027；evaluation seed为N/A，复用固定观测。
- 四组优化步数一致；LEO组从E1起额外前向6300个拼接样本/epoch，因此不宣称前向计算量一致。后续进入原U-loader训练时另设同预算桥接CE对照，不与本轮或历史完整CVS混称等计算量。

### LEO：沿用原机制，明确时序

直接调用原`ConcatSatChannelAugment`和`cvsrffi.practical_adapter.apply_practical`，不是另写简化增强。

| epoch | 批次应用概率 | 被选中的场景 |
|---|---:|---|
| 1至40 | 0.30 | practical_high |
| 41至90 | 0.60 | practical_mid、practical_low_urban |
| 91至200 | 0.80 | 上述三种场景 |

原采样器对整个batch做一次Bernoulli，并为被增强batch采一种场景。未命中时卫星视图仍是clean副本。每步拼接clean+view统一前向，但损失在E1至79只有clean CE；E80至200为`cleanCE+0.68*satelliteCE`。本轮consistency权重为0，避免同时加入另一机制。

早期额外前向并非完全无影响：它可能改变随机数消耗，且在叠加MixStyle时进入配对池。这是保留原fused concat行为的结果，将通过四组对照解释。实际channel命中样本与concat样本分别计数，不把未命中的副本写成真实信道增强样本。

物理ID由既有opaque source ID稳定映射为`source:<ID>`，session由RX/day决定。复用原算法和参数，不声称复现历史整数index所对应的同一随机信道实现。LEO种子2027在相关两组及四model seed间固定，以配对比较。

### MixStyle：保留原插入位置和配对约束

原`MixStyle1D`插入身份网络`time_down`和`t1`输出处；p=0.18、alpha=0.1、eps=1e-6、strength=0.70。仅同TX、跨RX/day配对，没有合格伙伴就跳过。验证和测试关闭MixStyle。网络仍输出真实融合身份特征及原分支特征，特征hook不留在模型内，不污染保存/重载。

本轮不打开域骨干、GRL、伪标签、teacher/EMA、DAOT、RC4、原型、开放世界或其他辅助损失。日志明确记录这些机制关闭，防止把“配置存在”当作“实际启用”。

## 冻结、测试与科学比较

1. 16行都完成源域训练；每行固定自身E200，不挑最高V的中间epoch。
2. 仅读取源域最终V及各源RX准确率，计算每组四seed均值`0.5*V+0.5*worstRX`，提前冻结下一轮起点。完全相同按ce、leo、mixstyle、leo_mixstyle顺序；无0.2个百分点成本容忍带。
3. 完整源矩阵冻结后，全部16个固定对照生成预测。复用现有VALIDATED_ONCE capsule，168000个物理query ID的clean及一一对应satellite观测；三种practical场景是satellite的互斥分层，不是三个各168000样本的独立测试集。
4. predictor不接收truth，只对每个样本进行六类竞争。全16行预测完整后，独立scorer连接truth。
5. 第二实现用bincount重新计算混淆矩阵、accuracy、Macro-F1并逐项对照；再输出按seed配对差值和交互项。

主报告包括四seed均值±样本SD、每seed、每视图、每RX、每TX、混淆矩阵、最差RX。报告LEO在两种MixStyle条件下的效应、MixStyle在两种LEO条件下的效应、both−CE，以及`both−LEO−MixStyle+CE`交互。所有负结果保留，不以测试提升决定补跑或剔除。

资源报告包括实际参数、梯度参与参数、实际epoch耗时、峰值显存、常驻buffer、checkpoint字节；另给合成CE-only前向/训练profile，注明它不含LEO/MixStyle完整开销。未测传输或内存项记N/A。本轮没有Phase2适应或新类，A/B/C、K、新类数、H均N/A。

## 后续分轮设计：本次不自动启动

后续每轮在新target访问前固定行、开关、继承合法性和源域选择规则。原则上继续从零训练；如果有继承需求，先按项目契约核对完整来源。每轮保留当前起点和同预算对照，机制启用必须由实际损失/梯度/样本计数证明。

| 轮次 | 加入内容 | 分离方法与必要验证 |
|---|---|---|
| R2 | 原两阶段训练与伪标签 | 在R1源域冻结起点上，先区分原训练调度/teacher与伪标签更新；标签阶段130epoch、伪标签阶段70epoch作为原配置起点。保持U标签不可读，记录采用率、阈值、有效loss与teacher来源。进入原U-loader步数时补同预算无伪标签桥接对照。 |
| R3 | 原域泛化组合 | 先域骨干+域分类/对抗的必要耦合组，再分别正交、跨视图一致性、groupCE、Fishr；不一次把整组开满。核对身份/域特征真实语义、梯度方向与每项实际权重。 |
| R4 | 原型与开放世界特征约束 | 分别评估prototype、身份compact、open-world feature，再加入proxy unknown、soft unknown mixup、source episode；最后按源规则测试预定组合。仅使用合法源数据/源合成未知，不接触target未知类标签。 |
| R5 | 后续扩展DAOT与RC4 | DAOT、RC4单独和叠加，形成2×2对照；沿用原实用版调度，记录hard/partial/identity分配、预算与校准实际生效。它们属于后续增强，不混称最初基础网络必含模块。 |
| R6 | 完整重组与删一消融 | 以源域规则冻结组合，验证完整机制栈、逐项移除、旧CVS同数据同预算对照；再讨论是否正式替换默认CVS入口。原有完整方法作为可追溯基准保留。 |

原配置库存来自`experiments/adv3b02_xuc/configs/rc4_practical_residual_noeq_20260918.json`：domain/adv/orth/cons/groupCE/Fishr系数分别1/.35/.05/.08/.16/.04；prototype .0032、open-world feature .0024、compact .032、proxy unknown .0045、soft unknown mixup .0045、source episode .0035。它们是后续移植起点，最终实际启用时间、warmup及调度由该轮resolved config与代码路径共同确认，本轮全部关闭。

## 执行与失败处理

唯一run：`20261004-phase1-reference-overlay-r1-manysig-m16-r01`；唯一release：`cvs_reference_overlay_r1_20261004_r01`；唯一owner：`codex/root/reference-overlay-r1-20261004`。

队列最多8个并行任务，优先低占用GPU，每GPU总训练数不超过2且空闲显存至少12GB。source与predict直接启动，避免包装进程重复计算GPU占用。空闲容量不足时等待，不停止其他任务。输出目录独占创建，不能重提交覆盖。

每行先做scratch真实checkpoint保存/重载无query检查。发布包在远端编译，并在远端实际环境运行四组原信道/前反向/保存重载检查。技术失败保留已有产物，不自动重跑；健康活动行继续，待它们结束后停止后续队列并记FAILED。低分不触发停机。只有预测、评分、独立复算全部完成后才标ANALYZED。

日志包括实际配置、每步完整JSONL、每epoch详细文本、紧凑JSONL及CSV；记录clean/sat CE、实际权重、LR、梯度、MixStyle调用与实际改变样本、LEO批次场景与命中数、V/各源RX及耗时。缺失项null并按阶段解释。

本地验证：8项针对性测试通过；四组CPU真实机制合成smoke通过，三LEO场景实际执行，checkpoint重载输出完全相同。独立P0/P1审查覆盖model/augmentation/source/predict/contract/prepare/dispatch/analyze，未发现阻断问题；远端发布与启动状态需另行读回。

## 发布与当前状态

2026-10-04 16:30（Asia/Hong_Kong）启动；独立启动读回VERIFIED。运行代码commit：`14a8196c711388f9d1266b3979e8468df72539de`，远端分支OID与该提交一致。dispatcher PID：3049566。

首批8行GPU0至7，已执行到epoch2至4；另外8行由同一dispatcher排队。全部首批PID/CWD/argv/实际177025参数、L/U/V计数、每epoch50步、无target访问、GPU进程及日志增长均核实。远端Torch2.1.0+cu121的四机制合成smoke通过，正式行各自scratch checkpoint无query重载通过。

| row | PID | GPU | 核实时epoch |
|---|---:|---:|---:|
| ce-s2026092701 | 3049585 | 0 | 4 |
| leo-s2026092701 | 3049594 | 1 | 2 |
| mixstyle-s2026092701 | 3049604 | 2 | 3 |
| leo_mixstyle-s2026092701 | 3049678 | 3 | 2 |
| ce-s2026092702 | 3049759 | 4 | 4 |
| leo-s2026092702 | 3049911 | 5 | 2 |
| mixstyle-s2026092702 | 3049987 | 6 | 3 |
| leo_mixstyle-s2026092702 | 3050001 | 7 | 2 |

LEO实际channel改变样本非零；MixStyle实际改变样本非零、每epoch两个位置合计100次调用；早期satellite_weight=0符合E80起效。当前没有训练完成、预测或评分结果，不能宣称实验全部完成。

下一步由既有dispatcher自动完成余下源训练、统一源矩阵冻结、16行clean/satellite预测、独立truth-last评分及bincount复算。恢复时先读`queue_state.json`、`completion.json`、`failure.json`及本次证据，禁止重复发布。最终结果应回填本报告并再次登记/Git交付。

## 2026-10-06下午进度与详细测试结果

2026-10-06T16:13:09.802859+08:00：新增矩阵训练完成28/136，R3另16行运行且自动补位正常；R2至R6尚无目标测试。R1全部16行完成预测、独立评分及复算，状态ANALYZED；本次已更正本地旧RUNNING登记。完整场景/seed/RX/TX指标、640条原始评分与当前队列见[详细报告](evidence/status_20261006_pm/report.md)。六个完整residual环境仍无测试结果，等待全部源训练冻结。
