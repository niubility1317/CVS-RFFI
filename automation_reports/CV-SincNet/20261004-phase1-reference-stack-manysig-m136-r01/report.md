# reference_response原Phase1剩余机制：R2至R6共136行

- run_id：`20261004-phase1-reference-stack-manysig-m136-r01`
- group_id：`cvs-reference-original-phase1-overlay`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；发布VERIFIED（2026-10-04独立读回，每卡2行、R2共16行运行，R3至R6等待源域依赖；详见末尾调度变更证据）

## 目的与对照

用户授权一次发布剩余全部实验；依源域冻结顺序推进两阶段/伪标签、DG、开放世界特征、DAOT×RC4、完整重组及删组消融。全部136行源训练完成后统一测试。

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

# reference_response剩余Phase1机制实验：R2至R6

用户授权：2026-10-04“发布剩余的所有实验”。唯一launch owner：`codex/root/reference-stack-20261004`。

本批run：`20261004-phase1-reference-stack-manysig-m136-r01`。136行均从零训练，不加载上一轮或上一阶段权重。复用的是原Phase1实现与经源域选择的机制配置。除R6的原生网络对照外，身份骨干固定为reference_response；原域分支保留。性能优先，成本作为次要比较。

## 矩阵与递进规则

每组均使用模型seed2026092701、2026092702、2026092703、2026092704。先完整执行一个阶段的全部源训练，依据四seed均值`0.5×源V准确率+0.5×最差源RX准确率`选择下一阶段起点；完全并列时取下表顺序靠前组。只继承机制配置，每行仍从零训练。每阶段实际生效配置写入`<stage>_resolved_matrix.json`，然后启动。

| 阶段 | 固定实验组（顺序也是完全并列时优先级） | 行数 | 主要问题 |
|---|---|---:|---|
| R2 | bridge、twostage、ema、pseudo | 16 | 原生训练预算桥接→130+70两阶段→EMA状态→伪标签 |
| R3 | base、domain、orth、cons、group_ce、fishr、all_dg | 28 | 域分类/GRL及原DG损失的单项与组合效果 |
| R4 | base、proto、compact、owfeat、proxy、softmix、episode、all_open | 32 | 原型、类内紧致和开放世界特征训练机制 |
| R5 | base、daot、rc4、both | 16 | 在共同MUSE M3桥接下做DAOT×RC4二因素比较 |
| R6 | selected、full、native_full、no_leo、no_mixstyle、no_pseudo、no_dg、no_proto_compact、no_open_boundary、no_daot、no_rc4 | 44 | 源域选中栈、固定完整栈、原骨干对照及删组消融 |
| 合计 | 34组×4seed | 136 | 全部预登记固定对比均进入最终测试 |

R1已冻结的源域选择是leo。本批只读取该源域冻结记录；R1测试分数不参与本批设计或选择。R2的bridge包含原RF增强、原生优化和阶段权重调度、保留的域分支，不是只增加步数的纯CE对照。R2的ema组只维护teacher状态，最终使用student验证和预测，因此不能解释为独立EMA推理收益。pseudo组加入原伪标签目标。

R3的domain同时开启域CE和GRL；orth、cons、group_ce、fishr各自以domain为共同依赖。R4六个单项在相同父配置上比较，all_open一次叠加六项。R5各组共同开启原MUSE M3、EMA和伪标签，这是RC4所需的实现依赖；R5与R4跨阶段比较包含训练机制桥接，DAOT/RC4净效应以R5内部对照为准。

R6的selected从零重建R5源域选中配置；full固定叠加全部本方案机制，不依赖选中结果；native_full在相同完整训练配置下使用原生身份网络。原生网络保留原有共享结构，reference_response替换后身份分支与原域分支不共享原Sinc/hf前端，比较解释保留这一差异。

R6删组含义：no_leo仅移除有标签concat卫星增强，DAOT/MUSE各自视图保留；no_dg只移除有标签DG损失组，MUSE/RC4域信号保留；no_proto_compact移除有标签原型/紧致项，MUSE bank仍在。no_pseudo同时移除伪标签、MUSE及依赖MUSE的RC4，保留有标签DAOT，属于依赖组消融。其余删组删除对应声明机制。不能将这些依赖组结果声称为某个孤立损失的净效应。

## 固定训练与输入

- E200，每轮按U_s的222批步数循环有标签loader，共44400次成功更新；L批128且drop_last，U批256且保留尾批。R2 bridge为200轮有标签阶段，twostage/ema/pseudo为130+70；下游按继承配置使用相同总预算。
- 原生AdamW、学习率/损失warmup、梯度裁剪和RF强视图调度保留。`lambda_*`默认全部清零，再只打开声明项、分类项和适用LEO/伪标签项。PAIC guard、feature masks、TX/RX geometry、Phase2 prototype导出统一关闭。FP32且TF32关闭，25MHz。
- DG权重：domain CE=1、GRL=0.35、orth=0.05、cons=0.08、group_ce=0.16、Fishr=0.04。开放特征权重：proto=0.0032、compact=0.032、owfeat=0.0024、proxy=0.0045、softmix=0.0045、episode=0.0035。原生调度对这些基础权重的阶段变化完整记录。
- 六类TX：14-10、14-7、20-15、20-19、6-15、8-20。复用原source_contract，物理ID角色精确核对：L/U/V=6300/56700/27000，源RX索引1、3、4、6、8，split seed392005。U标签和TX metadata在dataset边界隐藏；旧路径只取得-1哨兵，伪标签真实正确率诊断记不可用。
- 每行checkpoint/EMA仅来自自身scratch student；checkpoint来源为空。固定E200 student作为唯一最终权重，不做目标选模。模型参数、实际生效配置、数据角色和完整来源写入每行记录。
- 不同机制会增加每步多视图前向/反向次数；等更新数不等于等算力。保留原生资源汇总、实际参数、训练耗时/显存；未测量项写N/A。

## 测试与报告

必须等待全部136行源训练完成、五阶段选择均冻结，才允许任何本批query读取。测试所有固定组，不按测试表现筛选或重排。复用既有VALIDATED_ONCE胶囊：168000个clean及配对satellite观测；卫星按practical_high、practical_mid、practical_low_urban分层。逐样本六类竞争，无query拟合。

全部预测写盘后由独立`comparison_suite.score`连接truth；另用bincount复算全部混淆矩阵、Accuracy和Macro-F1。输出34组×5视图的四seed均值/样本SD、每RX/TX结果、最差RX、固定同seed对照差值、R5交互效应，保留全部负结果。R2对照bridge，R3/R4/R5对照base，R6对照full。

这是已接触过的基准上的固定设计，声明`BENCHMARK_INFORMED_FIXED_DESIGN`，不声称全新盲测。仅Phase1，无Phase2适应/新增类；K、三阶段旧类指标、新类准确率及H均N/A。

## 执行、故障与验证

队列最多同时8行，每GPU总训练进程不超过2且启动前至少12GB空闲。唯一调度器依次R2→R3→R4→R5→R6→预测→评分→独立复算。其他健康任务不受影响。

每行启动显示实际参数，保留原生详细文本、完整逐步gzip JSONL、去大数组的epoch JSONL/CSV、原生资源汇总。非有限loss/gradient立即使该行失败，保留已写产物；封存前日志步数和成功更新数必须均44400。任何技术失败暂停后续发放，已运行健康行自然完成，不自动重试、不因低分停止。

本地11项定向测试通过。五组原生训练集成冒烟使用合成输入，实际执行第1、41、80、131、181轮路径，覆盖伪标签、开放世界损失、MUSE、DAOT/RC4和原骨干。无正式数据或目标访问。独立P0/P1审查发现的不足预算晋级问题已修复并定向复核；发布脚本沿用已有不可覆盖发布路径，另做N607同环境CPU冒烟与启动读回。

唯一配置维护位置：本run的experiment.json；逐行模板指向运行前生成的真实配置。源码规则：`design.py`。只有完整预测、评分和独立复算均完成，才能登记ANALYZED。

## 发布与当前状态

2026-10-04T20:33:39.086489+08:00独立读回：发布VERIFIED，运行RUNNING。代码commit：`0c73c904c8331f26254761042516f1ae43ed9c98`；发布前已独立核对Git远端OID。dispatcher PID：3157203。

136行均已提交至唯一调度器：R2首批8行GPU0至7运行，R2另8行排队，R3至R6共120行等待源域前序依赖。首批PID/CWD/argv、实际844754主模型参数（含原域分支）、L/U/V物理角色一致、scratch来源、完整FP32和GPU进程均核实；日志增长及首轮完成已验证。

| row | PID | GPU | 核实时epoch |
|---|---:|---:|---:|
| r2-bridge-s2026092701 | 3157224 | 0 | 2 |
| r2-twostage-s2026092701 | 3157233 | 1 | 2 |
| r2-ema-s2026092701 | 3157243 | 2 | 2 |
| r2-pseudo-s2026092701 | 3157316 | 3 | 2 |
| r2-bridge-s2026092702 | 3157397 | 4 | 2 |
| r2-twostage-s2026092702 | 3157541 | 5 | 2 |
| r2-ema-s2026092702 | 3157553 | 6 | 2 |
| r2-pseudo-s2026092702 | 3157631 | 7 | 2 |

远端CPU合成集成冒烟覆盖5组×第1、41、80、131、181轮，全部PASS；无正式数据或query接触。正式行自身初始化checkpoint无query重载通过。当前本批无最终预测/评分，不能宣称性能提升或实验全部完成。

后续由已运行dispatcher顺序完成R2至R6源训练与源域选择，再统一136行clean/satellite预测、独立truth-last评分和第二实现复算。不创建额外定时任务。恢复先读queue_state/completion/failure，禁止重复发布。最终评分后应回填本报告、登记和Git；技术失败保留原产物，不自动重试。

## 每卡两实验调度变更

2026-10-04用户明确要求“一张卡两个实验”。计划将总并发由8改16，保持每GPU最多2个训练任务和启动前12GB空闲显存检查。仅替换本run调度器，按PID/启动时间/CWD/argv接管原训练进程；不重启训练、不改变实验配置与训练代码。后续阶段继续按原源域选择依赖推进。

4项定向测试通过：接管与完成判定、既有配置禁止覆盖、原8行保留并补满每卡2行、故障后禁止补位。独立P0/P1审查未发现阻断问题。当前尚待远端接管读回；以本节后续证据为准。

### 每卡两实验已生效

2026-10-04T20:44:49.299449+08:00独立读回VERIFIED：GPU0至7各2个本run训练进程，R2共16行运行，无排队和故障。原8个训练PID及启动时间保持一致，日志继续增长；新增8行均完成首轮。旧调度PID 3157203已退出，新调度PID 3163411接管成功。

训练代码仍为`0c73c904c8331f26254761042516f1ae43ed9c98`，调度代码为`27729a13cc659d075025840f9c0a6bcbe056dada`。只停止旧调度器，没有停止、重启或修改训练worker。后续自动补位仍受每卡2行和可用显存限制，源域选择与统一测试流程不变。活动owner见远端`dispatcher_active.json`，初始`dispatcher.json`和原submit作为历史证据保留。

| row | PID | GPU | 核实时epoch |
|---|---:|---:|---:|
| r2-bridge-s2026092701 | 3157224 | 0 | 15 |
| r2-twostage-s2026092701 | 3157233 | 1 | 15 |
| r2-ema-s2026092701 | 3157243 | 2 | 15 |
| r2-pseudo-s2026092701 | 3157316 | 3 | 15 |
| r2-bridge-s2026092702 | 3157397 | 4 | 15 |
| r2-twostage-s2026092702 | 3157541 | 5 | 15 |
| r2-ema-s2026092702 | 3157553 | 6 | 15 |
| r2-pseudo-s2026092702 | 3157631 | 7 | 15 |
| r2-bridge-s2026092703 | 3163432 | 0 | 1 |
| r2-twostage-s2026092703 | 3163441 | 1 | 1 |
| r2-ema-s2026092703 | 3163451 | 4 | 1 |
| r2-pseudo-s2026092703 | 3163461 | 5 | 1 |
| r2-bridge-s2026092704 | 3163534 | 2 | 1 |
| r2-twostage-s2026092704 | 3163608 | 3 | 1 |
| r2-ema-s2026092704 | 3163682 | 6 | 1 |
| r2-pseudo-s2026092704 | 3163757 | 7 | 1 |

本次证据：evidence/capacity16/verification.json；完整进程、GPU、接管前后快照见同目录readback.json。实验仍为RUNNING，尚未完成最终测试。

## 进度与七视图测试补充

2026-10-04T22:40:54.929533+08:00核实：本批已完成0/136行，R2有16行运行，R3至R6有120行等待前序依赖；当前第107至119轮，每行预算200轮，仍每卡2个。已完整解析当前16行全部已落盘epoch耗时记录；本次只作进度/耗时估算，不作收敛诊断。

近期每轮约75至79秒；考虑伪标签阶段增加计算，R2预估尚需2至3小时。全批训练暂按2至4天排期，后续DG/MUSE/DAOT/RC4尚无本批实测速度，区间是粗估而非承诺。额外七视图评估暂留1至3小时，待首批推理实测校准。

用户明确追加clean与六个residual完整星地环境。原三场景分层测试保留；新增依赖评估run：`20261004-phase1-reference-stack-sixscene-manysig-m136-r01`，等待父run全部ANALYZED后自动执行136行×7视图×168000个物理query。训练与源域选择流程不改。是否已经成功排队以新增run的启动证据为准。

## 后续训练执行加速（2026-10-05）

用户授权优化后续训练速度。诊断、限制和实现见[加速诊断报告](../20261004-diagnostic-reference-stack-speed-synthetic-m12-r01/report.md)。拟只将未启动R3–R6路由到独立执行release；健康R2保持原PID及代码。正式科学配置、预算、每卡两行和clean+六residual自动测试不变。当前发布状态以本段后续独立读回为准。

首次发布r01在交接前身份检查处FAILED：旧controller通过sys.argv自报的参数不包含解释器-u，而/proc和原启动凭据包含。独立读回确认原PID3163411及全部16个worker继续运行，没有发送停止信号。远端Torch2.1五项CPU兼容检查已VERIFIED。修复为核对原启动凭据的完整argv、PID、CWD，并让新controller记录/proc真实argv；4项回归测试通过，使用新release r02，保留r01产物。

2026-10-05T00:22:30.985953+08:00独立读回VERIFIED：新controller PID 3261036、release `cvs_reference_stack_fast_20261005_r02`、代码commit `7d29968511875f8b0b2bb724977f435dfdc485b0`。原controller3163411已退出，全部16个R2 worker的PID、启动时间、CWD和argv保持一致，原加速开关仍关闭；R3至R6共120行后续自动使用新入口。当前R2为172至199轮，完整完成0/136（该时刻快照），无失败。六场景等待进程3219697仍为WAITING_PARENT_COMPLETION。

R2按当前慢行约100至109秒/轮估计还需约45至60分钟，随同卡其他行完成可能缩短。阶段内自动补位，阶段间需要前一阶段全部候选完成并冻结源域选择；不得在冻结前抢跑依赖未知的下一阶段。R3至R6尚未在N607实际使用加速入口，因此当前验证确认的是接管和未来路由，不声称已获得正式训练加速比例或测试结果。

本地去预热后的短窗口耗时降低范围0.5%至6.6%，复杂R5/R6收益较小；原生完整训练仍有大量小算子和信道计算。没有更改训练轮数或步数来制造加速。保留原始完整日志、科学预算和自动truth-last测试收尾。
