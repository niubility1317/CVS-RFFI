# 冻结原 F 引导的星地 IQ 恢复目标

状态：ACTIVE，未达标。本文件仅跟踪本对话的识别器引导方向；另一对话的解析、物理恢复等方向保持独立，不向其发送指令，不接管其文件或任务。

## 用户要求与判定

- 目标测试集 low-urban 达到 70% 以上；其他五种 practical 接近同一原 F 的 clean。当前把“接近”暂按不超过 2 个百分点记录，这是 Agent 的操作解释，不能冒充用户指定数值。
- 分开报告两项证据：恢复 IQ 与同一物理样本 clean 的波形差距；恢复网络接入基座后，对真实 TX 的识别率、净纠错及 clean 保持。
- 性能结论以冻结后、完整目标预测和独立 truth-last 评分为准。源域增益、波形外观、NMSE 降低、程序检查通过均不能证明目标达成。
- 训练、结构、超参数、候选选择和重跑不使用目标分数反馈。保留全部负结果，不追认已经失败的候选。

原 F50 CROSS_RX 三套默认头的原始目标 clean 均值为 **85.113889%**。按上述临时口径，其他五种环境需达到 83.113889%；low-urban 单独要求至少 70%。目标未因实验困难或已有结果而缩小。

## 已完成的目标证据

同一不可变 168000 条物理样本，三套原 F 默认头分别 argmax 后取准确率均值；一个恢复器训练种子，不是三个恢复器种子。以下为已经完整评分的候选，不按单场景拼接为一个模型。

| 场景 | 原 R03 恢复 | correction-only 特征约束 | joint 弱决策约束 |同RX统计混合|跨RX统计混合|窄窗注意力|全窗注意力|时域滤波|频域滤波|标量和|单向投影|整批范数上限|逐帧范数上限|直接残差|四步流修正|全参数范数上限|分组件参数范数上限| 目标口径 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 84.743056% | 84.915873% | 85.016071% |85.054365%|85.064286%|85.078571%|85.058730%|85.089484%|85.006151%|85.062500%|85.063889%|84.939286%|85.061111%|84.903770%|85.000595%|84.942659%|84.934127%| 报告相对原 clean 的差值 |
| high | 77.575000% | 77.159921% | 77.025794% |77.175794%|77.473413%|77.253770%|77.267262%|77.335119%|77.207738%|77.240873%|77.228968%|77.463294%|77.926984%|77.760119%|78.406349%|77.552183%|77.564484%| ≥83.113889%（暂定） |
| mid | 76.491071% | 76.233333% | 76.102183% |76.173413%|76.464683%|76.239087%|76.258333%|76.328968%|76.183532%|76.230556%|76.241270%|76.530357%|76.995238%|76.729563%|77.347619%|76.649802%|76.638492%| ≥83.113889%（暂定） |
| low-suburban | 75.284127% | 75.170437% | 75.080159% |75.121627%|75.457540%|75.175397%|75.194841%|75.257143%|75.156548%|75.187302%|75.187698%|75.509722%|75.948016%|75.530159%|76.113294%|75.581349%|75.576389%| ≥83.113889%（暂定） |
| high-urban | 76.330952% | 76.224008% | 76.009127% |76.094246%|76.372024%|76.137698%|76.227183%|76.225198%|76.126786%|76.157738%|76.173016%|76.479365%|76.960119%|76.655952%|77.247421%|76.555754%|76.562500%| ≥83.113889%（暂定） |
| mid-urban | 70.116865% | 71.210317% | 70.841071% |70.747222%|70.975397%|70.675992%|70.845238%|70.741865%|70.698810%|70.735913%|70.727579%|71.363095%|72.014881%|70.953968%|71.285714%|71.388492%|71.373214%| ≥83.113889%（暂定） |
| low-urban | 63.392460% | **65.309127%** | 65.209325% |64.968056%|65.132937%|64.903968%|65.126190%|64.973016%|64.939087%|64.989484%|64.952778%|65.755357%|66.384127%|64.665675%|64.832738%|65.683532%|65.657937%| ≥70% |

low-urban 最好已测值为 66.384127%，距 70% 仍差 3.615873 个百分点；完整双判据见本轮结果，未按单环境拼接模型。joint 候选的 low-urban IQ NMSE 从输入的 1.987855 降至 0.078199，但这没有超过特征约束候选的识别率；波形与识别两项不能互相替代。

证据入口：

- [特征约束六环境目标结果](../../automation_reports/CV-SincNet/20261019-phase1-restoration-feature-target-practical-m3-r01/report.md)
- [特征约束 clean 结果](../../automation_reports/CV-SincNet/20261019-phase1-restoration-feature-target-clean-m3-r01/report.md)
- [joint 候选完整目标结果](../../automation_reports/CV-SincNet/20261019-phase1-restoration-joint-frozenF-source-s2026092701-m2-r01/report.md)

## 源域方向与当前实验

1. 既有 correction-only KL1 候选在源域改善城市识别，但波形 NMSE 增加 8.84%，未过当时固定 5% 界限；没有目标访问。保留失败，不追认。
2. joint 弱决策候选通过当时固定源规则，自动完成目标测试；源增益没有转化为新的目标最好值，见上表。
3. 三 teacher 平均/最差项、后验权重 0.1 的两个匹配 E10 候选均已完成源域。相对单 teacher joint E10，low-urban 分别下降 0.111111/0.444444 个百分点，未达到城市增益要求。两臂均未访问目标。
4. 三 teacher 均值、后验权重 0.3/1.0 的匹配 E10 消融均已完成源域且未通过。两臂 low-urban 为87.055556%/87.166667%，相对单 teacher 参考增加1.166667/1.277778个百分点；mid-urban 为93.444444%/93.166667%，仅增加0.111111/下降0.166667个百分点，均未达到固定+0.5要求。平均NMSE为0.03122304/0.03453391，1.0臂也超过新设的1.15倍原feature界限。clean均为98.5%，没有目标访问。新界限在评分前固定，没有更改历史5%失败或用户目标。

四个 E10 候选及两条更早持续引导的 joint30 新臂均已完成，负结果完整保留。joint30 直接从原 R02 coarse20＋全新零输出 correction 出发，原 F 与默认头冻结；方法 commit `672a36de61da32f3974472db97448b8977bbae71`。38 项聚焦检查、一次独立 P0/P1 审查、真实权重 no-query smoke 和实际启动证据均通过，所有训练/预测进程现已退出，没有新的目标访问。

|已完成 joint30 源候选|low-urban|相对 posterior03（百分点）|mid-urban|相对 posterior03（百分点）|clean|平均 practical NMSE|
|---|---:|---:|---:|---:|---:|---:|
|最终特征约束|87.111111%|+0.055556|93.055556%|−0.388889|98.500000%|0.03430840|
|增加中间层约束|86.888889%|−0.166667|93.166667%|−0.277778|98.555556%|0.03421370|

两臂均未达到预登记的两城市各 +0.5 个百分点要求，平均波形误差也超过 1.15 倍原 feature 界限，因此没有新目标测试。coarse 累计 50 轮、correction 30 轮，共 80 组件 epoch、串行继承 50 pass；单臂新增联合 30 轮/7410 步。两臂参数量同为 2140840，训练耗时分别为 3272.049/3469.647 s。旧 posterior03 是源指标参考，不是同计算成本对照；仅这两个新臂互相匹配。

post-hoc 源决策分析（代码 `9771481bb49a6cb814759500affa22a2d23b4831`，2 项计数检查通过）重用全部已完成预测，0 新模型前向/更新/目标访问。相对 posterior03，low-urban 最终特征臂纠正/损坏为 50/49，中间层臂为 47/50；mid-urban 分别为 29/36、31/36。六视图合并净变化为 −11/−12（10800 个重复视图决策，只有 1800 个物理样本）。错误有部分互补，但还不能证明融合有效。基于这个源依据，固定 IQ 平均已按独立新候选完成实际验证，见下。没有追认单个失败成员。

固定 0.5/0.5 IQ 平均实验也已完成（方法 commit `c369d4f2d6572a7f584f7675140187a675972bed`，28 项检查、一次独立 P0/P1 审查、真实成员 no-query smoke 和启动后态均通过）。两臂零新增训练，两个恢复成员合计 4281680 参数。mean_final 源 low/mid-urban 为 87.666667%/93.166667%，相对 posterior03 为 +0.611111/−0.277778 个百分点；mean_layerwise 为 87.388889%/93.166667%，变化 +0.333333/−0.277778。平均 NMSE 为 0.03152350/0.03147830，波形与 clean 保持通过，但两城市增益条件未通过。均已退出，没有新目标访问。不能由这些源结果推断目标收益。

已完成源汇总的分组诊断显示，参考 RX3 的 low-urban 为 79.722222%，其他四源 RX 为 86.111111% 至 90.833333%；其 clean 也较低（96.111111%），因此是关联而非因果证明。基于此依据的普通 observed、困难 top25% 与 RX 动态加权匹配实验现均已完成，方法 commit `63c9c3d2b81d92ebfee1623b54d942011e410f0a`。三臂同一原 R03 初始化、联合 E10/2470，参数/数据/顺序/优化器和三套冻结原 F/default heads 匹配，只改变额外 0.5×observed semantic 的聚合。

|已完成 observed 聚合源候选|low-urban|mid-urban|clean|平均 practical NMSE|固定源资格|
|---|---:|---:|---:|---:|---|
|mean_observed|86.833333%|93.444444%|98.500000%|0.03292334|未通过|
|cvar_observed|87.444444%|93.055556%|98.500000%|0.03605581|未通过|
|rxdro_observed|87.000000%|92.722222%|98.500000%|0.03280958|未通过|

所有原基础样本保留；额外选择曝光为普通/RX 各 630000、top25% 为 157500，RX 权重仅训练更新 2470 次，推断不使用 RX/q。各臂新增 L/U 优化曝光 63000/567000，coarse/correction 各累计 30 轮、60 组件 epoch、串行继承 50 pass，参数 2140840。30 项聚焦检查、一次独立 P0/P1 审查、真实 R03/三 F no-query smoke、实际 PID/CWD/命令/GPU/日志增长与完整 10 轮执行均已核实；所有训练进程现已退出。没有 L/U TX 真值或新拟合头。

三臂均未通过固定源规则，没有创建目标目录或新增目标 IQ/truth 访问。源负结果完整保留，不能称为目标负收益，也不能从源指标推断目标改善。实际目标最好值仍为 65.309127%，用户完整目标未达成。

一致伪身份/跨 RX 特征的匹配 E10 两臂现均已完成，方法 commit `623b3f3cb93f56fe77bf680e71570def9170ed2e`。29项检查、一次独立P0/P1审查、真实原R03/三F no-query smoke、实际进程和完整执行证据均通过；原F及默认头冻结，无L/U TX真值或新拟合头。

|已完成一致身份源候选|low-urban|mid-urban|clean|平均practical NMSE|固定源资格|
|---|---:|---:|---:|---:|---|
|consensus_native|86.888889%|93.333333%|98.500000%|0.03277007|未通过|
|consensus_crossrx|87.388889%|93.222222%|98.500000%|0.03295989|未通过|

两臂一致CE各曝光594020（L/U60740/533280），crossrx额外特征曝光594020，native为0；原基础曝光仍全量630000。三头高置信一致源记录59402/63000是伪身份接受率，不是准确率。无参数、各teacher独立坐标的其他RX均值只供训练，不作为分类器，推断不使用bank/RX。每臂新增联合10轮，coarse/correction各累计30轮、60组件epoch、串行继承50pass，参数2140840。两臂均未通过本轮预先固定low≥+0.5、mid不下降及其他/clean/波形要求，没有创建target目录或新增目标访问。过往失败未追认，实际目标最好仍65.309127%，用户目标未达成。


IQ摘要/冻结F语义条件E10两臂现均已完成，方法commit `1c869980ae9059e613ec29915d2b92db7444ed20`。25项聚焦检查、一次独立P0/P1审查、真实原R03/三F no-query smoke、实际启动及完整执行证据均通过；原F及默认头冻结，未使用L/U TX真值或新拟合头。

|已完成条件结构源候选|low-urban|mid-urban|clean|平均practical NMSE|固定源资格|
|---|---:|---:|---:|---:|---|
|iq_condition|86.833333%|93.055556%|98.500000%|0.03120911|未通过|
|semantic_condition|86.777778%|92.944444%|98.500000%|0.03114994|未通过|

两臂可训练参数均2244008，新四尺度FiLM从零输出初始化，首末轮权重非零。IQ条件为本帧received+coarse的4×40摘要，语义条件为原F1对本帧coarse的160维特征；只改变条件代码，原posterior03基础损失保持，无此前失败CE/prototype/focus。没有RX/TX/clean/其他帧运行输入或推断更新。每臂新增L/U曝光63000/567000、联合E10/2470；原coarse/correction各累计30轮、60原组件epoch，新增FiLM单独10轮，原F50不更新。独立语义恢复模型携带冻结F1，当前与外部原读出共享F1实例；训练参数匹配不等于总状态/计算匹配，实际耗时/显存详见报告。 iq_condition训练/末次保存1098.309s，独立恢复模型总参数2244008（其中冻结运行F0）。 semantic_condition训练/末次保存1107.328s，独立恢复模型总参数2477001（其中冻结运行F232993）。 两臂均未通过预先固定low≥+0.5、mid不下降及其他/clean/波形要求，没有target目录或新增目标访问；源负结果保留，实际目标最好仍65.309127%，用户完整目标未达成。


已完成行的 post-hoc 源分析发现：三 teacher 平均使整体与 tail NMSE 略低，但在原 clean 可识别样本中，六视图错误决策由 410 增至 419（10638 个重复视图决策，只有 1773 个物理样本）。错误组在 prefix 和 tail 均有较大误差；这是关联诊断，不是“只有 tail 导致错误”的因果证据，不改已冻结的选择或正在运行的配方。

- [三 teacher 平均/最差项源结果及分段分析](../../automation_reports/CV-SincNet/20261020-phase1-restoration-multireader-source-s2026092701-m2-r01/report.md)
- [决策约束强度消融的固定矩阵与进度](../../automation_reports/CV-SincNet/20261020-phase1-restoration-identity-priority-source-s2026092701-m2-r01/report.md)
- [coarse20＋全新修正器 joint30 的最终特征/中间层匹配实验](../../automation_reports/CV-SincNet/20261020-phase1-restoration-layerwise50-source-s2026092701-m2-r01/report.md)
- [固定 IQ 平均源结果及分组诊断](../../automation_reports/CV-SincNet/20261020-phase1-restoration-iqmean-source-s2026092701-m2-r01/report.md)
- [困难样本与 RX 分组加权的匹配实验](../../automation_reports/CV-SincNet/20261020-phase1-restoration-hardness-source-s2026092701-m3-r01/report.md)
- [一致伪身份与跨 RX 特征匹配实验](../../automation_reports/CV-SincNet/20261020-phase1-restoration-consensus-identity-source-s2026092701-m2-r01/report.md)
- [IQ摘要与冻结F语义条件的恢复结构对比](../../automation_reports/CV-SincNet/20261020-phase1-restoration-semantic-feedback-source-s2026092701-m2-r01/report.md)

## 统计混合固定比较（完整源与目标结果已返回）

本轮开始前的IQ摘要/语义FiLM两臂源low/mid均下降，未创建target目录；完整源负结果已保存，方法`1c869980ae9059e613ec29915d2b92db7444ed20`，结果`fb7c289d4186be9ba369c31f36232527858c9da7`。本轮不是因为波形界限而漏选识别候选：两臂波形与clean保持通过，但城市识别均不改善。

下一方向检验源接收机统计依赖：训练时在纠错器浅层和中层混合特征均值/标准差，候选配对仅来自三原头在source clean上一致且各置信度≥0.9的同伪身份、不同源RX记录；不读取L/U TX真值，不保证伪身份等于真实TX。无候选则跳过，不退回跨身份混合。拟定层位为原corrector encoder第0/2层，概率0.5、Beta(0.1,0.1)、epsilon1e-6；每条observed与配对clean使用相同物理配对/系数，但donor保持对应半批，不从clean半向observed半输入特征。均值/标准差停止梯度，逐条记录有效配对/实际混合、层位、L/U曝光。推断关闭混合，仅当前IQ，恢复网络原参数结构和三个原F/defaulthead保持。

拟定两行同伪身份同RX统计混合匹配控制、同伪身份跨RX统计混合，两行都保护伪身份。只对两种规则均有合法donor的物理记录启用混合，以相同随机抽取数匹配可配对集合、尝试概率和系数，只改变donor的RX条件；不使用旧失败checkpoint作为新目标候选。两行均从原R03 coarse20/correction20直接分叉，原posterior03三F基础损失、jointE10/2470、数据/顺序/预算匹配，不继承本轮或此前失败精修。接收机标签和伪身份仅用于source训练配对，原F和默认头冻结，不增加分类/诊断头或推断状态。

按用户“性能以目标测试集为准”的要求，新实验改用训练前预登记的固定目标对比矩阵：两行固定最终E10后均自动完整测试同一168000条clean＋六practical、三原默认头。source V用于诊断，不再用源涨分/NMSE资格筛出一行；这是新的固定比较，不追认任何旧失败，也不重测它们。每行全七视图预测固定后，独立CPU才接入对应truth；source/target CPU scorer串行。目标分数不改变任何行的配置、权重、预算、排序或选择性重跑，所有已登记行完整报告。

实现时保持逐行训练结束自动测试：控制器确认该source worker实际退出后再在同一卡启动该row target，避免源worker残留CUDA上下文与新预测worker并占；其他健康行不等待整批、也不因已出分行而变化。原R03/F来源、VALIDATED_ONCE数据、不可覆盖输出、唯一owner和最小发布流程保持。r01首步真实smoke因Torch2.1/NumPy2索引兼容错误退出，尚未启动训练worker或访问目标；失败产物保留。r02修复为显式long Tensor index_select，科学配方和固定双行目标矩阵保持，使用新的不可覆盖run/release，不重复全审。完整实现为`experiments/pamf_style_generalization`，run ID `20261020-phase1-restoration-style-generalization-source-s2026092701-m2-r02`。46项聚焦检查通过，涵盖同伪身份配对、同/跨RX匹配有效曝光和系数、半批隔离、统计停止梯度、真实浅/中层接线、推断原Core路径与逐帧独立、checkpoint预算/来源/权限拒绝、固定双行无源分数筛选以及全七视图/三F/有限IQ后truth。单次独立P0/P1审查发现进度文件独占创建会令第二次轮询失败，已修复并通过连续source→target→complete回归及该问题唯一一次定点复核；无未决P0/P1。

新两臂均2140840参数，jointE10/2470、相同原posterior03三F基础损失、原数据顺序/路由，mixseed2026102051；float32系数舍入为1的尝试单独记录并跳过无效混合，所有原基础记录保留。一次N607普通账号预检VERIFIED，GPU4/5各24124MB空闲无PID，空间≥8GB，新run/release不存在，原R03与source-only伪身份来源核实，其他owner保持。固定GPU4同RX、GPU5跨RX，每卡一条本run训练/预测进程，CPU scorer全run最多一个。真实原R03/三F/source Ls四条、两RX的no-query smoke已PASS，0V/0TX真值/0target/0optimizer更新；两臂已在GPU4/5实际训练，PID/CWD/命令/GPU/持续日志增长、两层混合应用和donor RX矩阵均核实，原F/defaulthead冻结。方法commit`99aa5bb976112f1469aacfefa2f1ee40f4fe537a`。两行已完整完成source V与目标七视图/三F独立truth-last评分，实际训练、目标预测、评分及controller均已退出；本轮仍未达到用户完整目标。完整命令、环境/CWD、矩阵、路径、资源、停止规则与产物见[本轮最小登记](../../automation_reports/CV-SincNet/20261020-phase1-restoration-style-generalization-source-s2026092701-m2-r02/report.md)。

## 固定边界

基座固定为原 `20261016-phase1-F-supcon-e50-clean-manysig-m9-r02` 的 CROSS_RX 三种子与原联合训练默认头，来源 commit `4df7b94d0fd8ce29056bd8236364e45afa92e07f`。不更新 F、重训或另拟合分类头。恢复训练使用同契约合法源 L/U 的原 R02 配对缓存，不读取 L/U TX 真值；F 提供已有监督知识，因此称识别器引导的自监督恢复，不称整个系统完全无标签。

此前 E10 消融从原 R03 coarse20/correction20 独立分叉，合计60组件epoch。本轮已完成的中间层实验直接使用原 R02 coarse20＋全新 correction，新增联合30轮，不继承 R03 correction20 或已测试的精修权重。两轮实验计算预算不同，不宣称等成本增益；匹配仅限本轮两臂内部，额外中间层参考前向耗时和显存按实际记录。原20轮配对缓存按1..20再1..10使用，不生成新IQ。

已完成的两批 E10 worker 均已退出。中间层实验曾使用 N607 普通账号、GPU4/5 每卡一条本 run worker，获取时计入其他 owner 且空闲显存至少 18 GB，源 CPU scorer 最多一个；原健康任务未改。两臂均已完成并退出，因源规则未通过，没有启动目标 worker。后续资源由新 run 预登记固定。科学选择只使用源证据，目标预测全部固定后独立 CPU 接入 truth；practical 推断不读取 paired clean、其他视图、目标真值或更新状态。确切 commit、命令、路径、停止规则和产物均在对应最小报告内，不另增审批、签名或哈希链。


## 统计混合固定矩阵的实际结论

同 RX 统计混合：目标low urban 64.968056%，clean 85.054365%。跨 RX 统计混合：目标low urban 65.132937%，clean 85.064286%。两臂均完成预登记E10/2470、原168000目标clean+六practical和三原头独立truth-last评分，实际source/target/controller均已退出。source V未作候选筛选，未修改/重排/选择性重跑任何行。完整逐环境/逐F/波形/净纠错/成本结果见[本轮报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-style-generalization-source-s2026092701-m2-r02/report.md)。本轮仍未达到用户完整目标，不由波形或训练检查推断性能。


## 时间上下文固定比较（完整源与目标结果已返回）

源域事后诊断显示，跨RX统计混合相对同RX主要改善RX1/6，但最弱RX3仍79.722222%，RX4下降；此前IQ摘要/语义FiLM源low/mid也未改善。下一候选依据这些source证据检验显式、输入相关时间交互，不依据目标成绩调整参数。已有卷积感受野较宽，GroupNorm也用帧内整体统计；不能称原网络没有全局信息，也不能把窄注意力mask解释为全模型只看局部原始IQ。

`experiments/pamf_temporal_context`两行window_attention/global_attention在纠错器最深128×32特征上增加同规格四头残差注意力，PreLayerNorm＋固定sin/cos位置编码、128→384 QKV、dense32×32 scaled-dot softmax、128→128输出投影，无dropout/FFN。两行只改变mask：半径2（154允许位置对）/全32（1024对），参数和随机初值完全匹配。输出投影从0开始，原R03全部旧state严格继承，初始完整IQ逐点等于原R03；不继承任何失败精修。新增66304参数，两行总2207144；仍单次当前IQ前向，无restorer内部运行F、信道迭代、RX/pseudo/bank/clean/其他帧输入或状态更新。另一用户对话当前四步/信道一致性路线仅仅读取确认，未发消息或修改其文件/任务。

原R03 coarse20/correction20独立分叉jointE10/2470，原posterior03三F冻结损失、原63000L/U缓存1..10、数据顺序/路由和seed保持，源V仅诊断；旧LR3e-5、新attention2e-4，共warm247/cos至各自.1、WD1e-4、clip1、FP32。原组件各累计30轮、60组件epoch、串行继承50pass，新attention另计10轮，0F/defaulthead更新或新头拟合。逐轮记录attention输出范数、QKV delta及实际非零梯度步数，避免把AdamW decay当作学习；初始零输出投影导致首步QKV梯度0是预期，第二步可学习已unit验证。

42项聚焦检查与一次独立P0/P1审查PASS；一次N607普通账号预检VERIFIED，原R03与数据契约合法、前序完成、新run/release不存在、GPU4/5各24124MB空闲无PID且空间≥8GB，其他owner保持。GPU4窄窗/GPU5全窗，每卡一条本run worker、source空闲≥18GB/target≥12GB、CPU scorer全run最多1。launcher首步实际原R03/三F/四Ls no-query smoke，PASS立即训练；每行实际source进程退出后自动同卡完整测试原168000 clean＋六practical及三原默认头，全部七视图预测固定后独立CPU才连接truth与评分clean，不作源分数筛选或目标反馈。方法commit`d3d0d90f16271204bb0f31a788557f21ad86655a`已实际发布，归档比较/编译、真实R03/三F/四Ls smoke均PASS；两条GPU4/5训练worker的PID/CWD/命令/GPU/持续日志增长、实际QKV非零梯度与attention曝光已核实。两臂已完整完成source V及目标七视图/三F独立truth-last评分，实际所有source/target/controller进程均已退出；本轮仍未达到用户完整目标。

Run ID `20261020-phase1-restoration-temporal-context-source-s2026092701-m2-r01`；[本轮最小登记与方法](../../automation_reports/CV-SincNet/20261020-phase1-restoration-temporal-context-source-s2026092701-m2-r01/report.md)包含命令、环境/CWD、矩阵、路径、资源、停止规则及产物。没有额外审批、签名或哈希链。


## 时间上下文固定矩阵的实际结论

窄窗注意力：目标low urban 64.903968%，clean 85.078571%。全窗注意力：目标low urban 65.126190%，clean 85.058730%。两行均完成原预登记E10/2470、168000物理目标clean+六practical及三原头独立truth-last评分。没有source分数筛选、目标反馈或选择性重跑，全部负结果保留。[完整逐环境/逐F/波形/净纠错/成本结果](../../automation_reports/CV-SincNet/20261020-phase1-restoration-temporal-context-source-s2026092701-m2-r01/report.md)。本轮仍未达到用户完整目标，不由新增结构/训练检查推断性能。


## 末端滤波固定比较（完整源与目标结果已返回）

对已有1800条source V固定IQ预测做事后误差分解（无新模型前向/更新/目标访问）：low urban原R03 NMSE0.068199，posterior03及两种时间注意力约0.07486至0.07490，多个有效频带均存在残差且精修普遍增加误差。该证据不证明某一频带是原因；原coarse已含频谱分支，本轮比较新增末端滤波的基底，不称首次FFT或已证明频率信息不足。候选依据source证据提出，不按目标成绩调整参数。

`experiments/pamf_basis_filter`两行time_basis/frequency_basis均从原合法R03完整继承coarse/correction，不继承失败精修。新增相同Core4→4、宽16/32/48/64的有界广义线性残差滤波，210628新参数、2351468总参数，张量及初值匹配。主通道幅度0.5至2、相位±π/2，镜像实/虚各±0.25；输出投影从0开始，先计算delta再加回原输出，因此两臂初始完整IQ逐点等于R03。time在当前received/恢复IQ上预测逐时刻系数并乘原IQ/共轭；frequency对两者完整orthoFFT并fftshift后预测逐频系数，ifftshift后乘原频谱/共轭镜像，再ifft残差。仅基底不同，不称算子数学等价或同计算成本（frequency每IQ多3次256点变换）。这些边界不保证身份保真或真实物理逆。

原posterior03三F冻结损失、原63000L/U与R02缓存1..10、顺序/路由/seed保持；joint E10/2470，旧LR3e-5/新filter2e-4、warm247/cos至.1、WD1e-4/clip1/FP32。原coarse/correction各累计30轮、60组件epoch、串行继承50pass，新filter另10轮，0L/U TX真值/F更新/新头拟合。记录新模块实际调用/IQ/系数位置/FFT和非零encoder梯度步数，区分学习与AdamW decay。推断单帧IQ、一次coarse/correction/filter，无运行F条件、信道oracle、RX/场景/clean/其他帧/迭代/状态更新。

46项聚焦检查及唯一一次独立P0/P1审查PASS，审查者复跑46项通过。一次N607普通账号预检VERIFIED：原R03合法、前序完整、新root不存在、GPU4/5各24124MB空闲无PID、空间≥8GB，保留其他owner。GPU4time/GPU5frequency，每卡一条本run worker，source≥18GB/target≥12GB/allowner<2/CPU scorer最多1。launcher首步真实R03/三F/四Ls no-query smoke，PASS立即训练；源V只诊断，固定最终E10两行都自动完整168000目标clean+六practical/三原默认头，各行七视图固定后独立CPU连接truth和pairedclean；实际source进程退出后同卡target，不作源筛选或目标反馈。方法commit`c68588016e6960e48ef3b320c8e5daeefc8f16ac`已实际发布，归档比较/编译、真实R03/三F/四Ls smoke均PASS；两条训练worker的PID/CWD/命令/GPU/持续日志增长及实际encoder非零梯度/滤波曝光已核实。两臂已完整完成source V及目标七视图/三F独立truth-last评分，实际所有source/target/controller进程均已退出；本轮仍未达到用户完整目标。

Run ID `20261020-phase1-restoration-basis-filter-source-s2026092701-m2-r01`；[最小登记与方法](../../automation_reports/CV-SincNet/20261020-phase1-restoration-basis-filter-source-s2026092701-m2-r01/report.md)含命令、环境/CWD、矩阵、路径、资源、停止规则和预期产物，source_error_profile.json只含聚合误差。另一用户对话只读确认采用解析观测量损失/展开结构路线，未发消息或改动其文件/任务。


## 输出梯度固定比较（完整源与目标结果已返回）

source误差分解显示精修在多个有效频带增加波形误差，末端time/frequency源low urban87.055556/86.888889也未明显超过posterior03。由此提出梯度冲突假设，尚未证明；新方案不根据新basis的目标成绩调整，原basis预测继续保持。`experiments/pamf_gradient_projection`两行scalar_sum/waveform_projection从原合法R03完整相同state独立开始，无新模块、2140840参数，原两项loss数值/三冻结F/缓存/顺序/预算保持。均计算每恢复IQ帧2×256坐标的wave/identity sum-loss梯度gw/gi；控制相加，投影臂仅在dot<0且norm²(gw)>1e-20时去除gi沿gw的冲突分量，合并除以实际IQ批量后一次原模型backward。记录实际conflict/applied/obs-clean/dot/norms，不把零冲突作为技术失败。

这是单向IQ输出空间投影，借鉴[梯度投影论文](https://arxiv.org/abs/2001.06782)，不称完整参数空间PCGrad，也不保证共享Jacobian/AdamW后的参数步或最终波形收益。44项聚焦检查及一次独立P0/P1审查PASS；源Ls实际R03/三F no-query smoke为launcher首步，PASS立即训练。原R03 coarse20/corr20+jointE10/2470，每组件累计30轮、60组件epoch、串行50pass，0L/U TX真值/F更新/新头拟合。sourceV只诊断，两行固定最终E10自动完整168000目标七视图/三原头，七视图固定后独立CPU truth/pairedclean；不以source或目标成绩改参数/预算/顺序/选择性重跑。

一次N607普通账号预检VERIFIED：GPU6/7各24124MB空闲无PID、原R03合法、新root不存在，前一basis在GPU4/5的实际target PID/CWD/cmd LIVE并保留。新GPU6scalar/GPU7projection各一worker，重叠期间本owner最多4GPUworkers、每卡1且allowner<2；新sourceCPUscorer等前basis完成再执行，全run独立scorer最多1。方法commit`9f2d8f65e507809de396af5fdb89f11b749759b1`已实际发布，归档比较/编译、真实R03/三F/四Ls smoke均PASS；GPU6/7两条source worker的PID/CWD/cmd/GPU/日志增长和实际gradient conflict/merge计数已核实。两臂已完整完成source V及目标七视图/三F独立truth-last评分，实际所有source/target/controller进程均已退出；本轮仍未达到用户完整目标。[最小登记与方法](../../automation_reports/CV-SincNet/20261020-phase1-restoration-gradient-projection-source-s2026092701-m2-r01/report.md)含命令、环境/CWD、矩阵、路径、资源、停止规则和产物；另一用户对话未被发送指令或改动。


## 末端滤波固定矩阵的实际结论

时域滤波：目标low urban 64.973016%，clean 85.089484%。频域滤波：目标low urban 64.939087%，clean 85.006151%。两行均完成原预登记E10/2470、168000物理目标clean+六practical及三原头独立truth-last评分。没有source分数筛选、目标反馈或选择性重跑，全部负结果保留。[完整逐环境/逐F/波形/净纠错/成本结果](../../automation_reports/CV-SincNet/20261020-phase1-restoration-basis-filter-source-s2026092701-m2-r01/report.md)。本轮仍未达到用户完整目标，不由新增结构/训练检查推断性能。


## 梯度幅度固定比较（完整源与目标结果已返回）

合法源训练scalar_sum E5/1235/630000IQ的累计identity/wave输出梯度RMS范数比15.426531（不是逐帧/step比，也不证明原因）。据此检验幅度上限，未读取该run新target分数，不热改其或投影行。`experiments/pamf_gradient_balance`新global_cap/per_frame_cap均从原合法R03完整相同state开始，2140840参数，无新增模块，原数据/顺序/预算/三F冻结loss保持；gw+s×gi、s=min(1,sqrt(wave norm²/identity norm²))，分别按整批总和或每IQ帧计算，只缩小identity、不投影冲突。identity norm²≤1e-20时s=1；非零identity/wave0时s=0，停止scale梯度后除以实际IQ批量反传。不是参数GradNorm/PCGrad，不保证参数步/最终波形或身份改善，实际scale/计数/dot/norms完整记录。

科学矩阵为前run scalar_sum固定E10控制（已登记并健康训练，直接复用、不重训/重预测/从其初始化）与两新norm-cap行；配置和matrix含external_control引用。每新行jointE10/2470、原coarse/corr各30轮、60组件epoch、串行50pass，0L/U TX真值/F更新/新头拟合。sourceV仅诊断，两新行固定最终E10自动完整168000目标七视图/三原头，全部固定后独立CPU truth/pairedclean；控制技术缺项如实报告，不静默声称匹配完整。

50项聚焦检查及唯一一次独立P0/P1审查PASS。一次N607普通账号预检VERIFIED，新GPU4/5各24124MB空闲无PID、原R03合法、新root不存在；GPU6/7现有source实际PID/CWD/cmd LIVE并保留。重叠最多4本owner worker，每卡1/allowner<2，CPU scorer先等前run complete/failure再执行，实际source退出后同卡target。方法commit`9866f24dab7097294d585069bfcfe529d3346ade`已实际发布，归档比较/编译、真实R03/三F/四Ls smoke均PASS；GPU4/5两条source worker的PID/CWD/cmd/GPU/日志增长和实际norm-cap/scale计数已核实。两臂已完整完成source V及目标七视图/三F独立truth-last评分，实际所有source/target/controller进程均已退出；本轮仍未达到用户完整目标。[最小登记与方法](../../automation_reports/CV-SincNet/20261020-phase1-restoration-gradient-balance-source-s2026092701-m2-r01/report.md)含命令、环境/CWD、矩阵、路径、资源、停止规则和产物；另一用户对话未被发送指令或改动。


## 输出梯度固定矩阵的实际结论

标量和：目标low urban 64.989484%，clean 85.062500%。单向投影：目标low urban 64.952778%，clean 85.063889%。两行均完成原预登记E10/2470、168000物理目标clean+六practical及三原头独立truth-last评分。没有source分数筛选、目标反馈或选择性重跑，全部负结果保留。[完整逐环境/逐F/波形/净纠错/成本结果](../../automation_reports/CV-SincNet/20261020-phase1-restoration-gradient-projection-source-s2026092701-m2-r01/report.md)。本轮仍未达到用户完整目标，不由新增结构/训练检查推断性能。


## 确定性残差流固定比较（完整源与目标结果已返回）

旧R03随机残差流在其AsF/sourceV设置有7.03125%三采样TX切换，未通过身份稳定性；本轮不加载其权重。新`pamf_residual_flow`两行direct_residual/rectified_flow4从同一原合法R03完整冻结state开始，新增相同Core7→2 widths16/32/48/64零输出网络，210882可训练/2351722总参数。训练velocity目标clean-anchor，direct t0；flow半t0/半U[0,1)，源point=anchor+t(clean-anchor)，两行F/logpower代理均anchor+velocity。推断只用当前IQ/R03anchor，1/4步固定Euler，无Gaussian/F/clean/真实信道/TX/RX/场景/其他帧/更新；不把source训练代理当完整采样，不称PMRF复现/最优性或身份保证。

原数据/缓存1..10/顺序/路由/seed和三原F冻结loss保持，newfield E10/2470、AdamW2e-4，旧coarse/corr各20不更新、新field10，总组件/串行50。40项检查及一次独立P0/P1审查PASS；GPU6/7一次预检空闲，当前GPU4/5范数约束target存活保留。launcher首步真实R03/三F/四Ls no-query smoke PASS立即训练；两行sourceV只诊断、固定最终E10自动完整168000/七视图/三原头独立truth-last，CPU scorer先等范数run终态。方法commit`9625f878ed9659ce053911bb4b033d7261cfc7d2`已实际发布，归档比较/编译、真实R03/三F/四Ls smoke均PASS；GPU6/7两条source worker的PID/CWD/cmd/GPU/日志增长和实际field/time/encoder非零学习计数已核实。两臂已完整完成source V及目标七视图/三F独立truth-last评分，实际所有source/target/controller进程均已退出；本轮仍未达到用户完整目标。[最小登记](../../automation_reports/CV-SincNet/20261020-phase1-restoration-residual-flow-source-s2026092701-m2-r01/report.md)。


## 梯度范数上限固定矩阵的实际结论

整批范数上限：目标low urban 65.755357%，clean 84.939286%。逐帧范数上限：目标low urban 66.384127%，clean 85.061111%。两行均完成原预登记E10/2470、168000物理目标clean+六practical及三原头独立truth-last评分。没有source分数筛选、目标反馈或选择性重跑，全部负结果保留。[完整逐环境/逐F/波形/净纠错/成本结果](../../automation_reports/CV-SincNet/20261020-phase1-restoration-gradient-balance-source-s2026092701-m2-r01/report.md)。本轮仍未达到用户完整目标，不由新增结构/训练检查推断性能。


## 参数梯度固定比较（完整源与目标结果已返回）

依据已有源输出梯度RMS比15.426531及Jacobian不保持任务范数的限制，新增`pamf_parameter_balance`全参数与coarse/correction分组参数cap两行。仍从原合法R03完整相同state开始、2140840参数、jointE10/2470与原数据/路由/seed/三冻结F loss保持，任务mean loss分别求参数梯度、各组s=min(1,sqrt(w2/i2))、detached合并回p.grad，再clip1/AdamW，无二次batch除法。记录每更新步/组的scale/conflict/norms/dot；原始Euclidean wave方向性质不保证AdamW或最终效果，不称GradNorm/PCGrad。

复用既有gradient_balance/per_frame_cap作为operator结构对应的output-space控制，不按分数选、不重训/预测/初始化；两新行反传工作不同于该控制，成本如实报告。60项检查及一次独立P0/P1审查PASS；新GPU4/5一次预检空闲，GPU6/7流修正target存活保留。launcher首步真实R03/三F/四Ls no-query smoke PASS立即训练；sourceV只诊断，两固定最终E10都自动完整168000/七视图/三原头独立truth-last，新CPU评分等流修正run终态。方法commit`fd9c8fee5370f2b3ca7a2744862e99efb59968fd`已实际发布，归档比较/编译、真实R03/三F/四Ls smoke均PASS；GPU4/5两条source worker的PID/CWD/cmd/GPU/日志增长和实际参数组/step cap计数已核实。两臂已完整完成source V及目标七视图/三F独立truth-last评分，实际所有source/target/controller进程均已退出；本轮仍未达到用户完整目标。[最小登记](../../automation_reports/CV-SincNet/20261020-phase1-restoration-parameter-balance-source-s2026092701-m2-r01/report.md)。


## 确定性残差流固定矩阵的实际结论

直接残差：目标low urban 64.665675%，clean 84.903770%。四步流修正：目标low urban 64.832738%，clean 85.000595%。两行均完成原预登记E10/2470、168000物理目标clean+六practical及三原头独立truth-last评分。没有source分数筛选、目标反馈或选择性重跑，全部负结果保留。[完整逐环境/逐F/波形/净纠错/成本结果](../../automation_reports/CV-SincNet/20261020-phase1-restoration-residual-flow-source-s2026092701-m2-r01/report.md)。本轮仍未达到用户完整目标，不由新增结构/训练检查推断性能。


## 当前源 clean 局部敏感度比较（两臂实际训练中）

固定`feature_tangent`与`posterior_tangent`，分别用同物理源clean处冻结原F的六坐标Jacobian二次误差替换一个语义项，另一项、原R03初始化、jointE10/2470、数据/路由/seed/AdamW与per_frame_cap保持；无新NN/头、0TX标签/目标训练。特征使用固定Rademacher探针，后验使用六原类sqrt概率坐标，J训练缓存，推理仍单帧原网络。源V仅诊断，两臂都自动完整168000×七视图×三原默认头独立truth-last；GPU6/7，前序参数target4/5保留，评分等前序终态。50项检查PASS，额外J准备成本单列，方法commit`2afa1cdbd31e1b2d56214499a797bac00ab64c5c`已发布，归档比较/编译及真实R03/三F/四Ls局部J smoke PASS；GPU6/7两条source worker的PID/CWD/cmd/GPU/日志增长和实际源J准备进度已核实，缓存进度不冒充训练完成。尚无本轮新目标表现。[最小登记](../../automation_reports/CV-SincNet/20261020-phase1-restoration-tangent-guidance-source-s2026092701-m2-r01/report.md)。


## 参数梯度范数上限固定矩阵的实际结论

全参数范数上限：目标low urban 65.683532%，clean 84.942659%。分组件参数范数上限：目标low urban 65.657937%，clean 84.934127%。两行均完成原预登记E10/2470、168000物理目标clean+六practical及三原头独立truth-last评分。没有source分数筛选、目标反馈或选择性重跑，全部负结果保留。[完整逐环境/逐F/波形/净纠错/成本结果](../../automation_reports/CV-SincNet/20261020-phase1-restoration-parameter-balance-source-s2026092701-m2-r01/report.md)。本轮仍未达到用户完整目标，不由新增结构/训练检查推断性能。


## 当前三 F 梯度一致性比较（两臂实际训练中）

固定`frame_agreement/time_agreement`，逐个求三原F的相同语义loss对恢复IQ的梯度，仅在整帧512维或每时刻I/Q二维三对dot全≥0处保留mean3F，再沿用per_frame_cap。无教师梯度归一化/拟合权重/轴signmask；算子对教师置换/正交I-Q变换不变，原始identity方向性质不保证AdamW或最终效果。旧源码只保留平均后梯度，尚未实测冲突，不以当前局部J的目标分数定义候选。

原合法R03同state/2140840参数、jointE10/2470、数据/路由/seed/AdamW/三冻结F原loss保持，无新NN/头。记录帧或复采样单位的agreement/activeagreement/pairconflicts/cosines/raw-agreednorms，零一致不设资格gate。54项检查及一次独立P0/P1审查PASS；GPU4/5一次预检空闲，GPU6/7局部Jsource LIVE保留。首步真实权重smoke PASS立即训练，sourceV仅诊断，两固定最终E10自动完整168000×七视图×三原头独立truth-last，CPU评分等前序局部Jrun终态。方法commit`d30b6092eb489f20d512e69f231354981cc93e78`已发布，归档比较/编译及真实R03/三F/四Ls smoke PASS；GPU4/5实际PID/CWD/cmd/GPU/日志增长与逐F方向一致/逐帧cap计数已核实。尚无三 F 一致性新模型的目标表现。[最小登记](../../automation_reports/CV-SincNet/20261020-phase1-restoration-teacher-agreement-source-s2026092701-m2-r01/report.md)。
