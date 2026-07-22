# Stage2研发Agent初始化与高效协作协议

版本：2026-07-22
修订：开放方法路由、可行性讨论门、问题谱系与相似错误扩展
用途：在加载`docs/STAGE2_METHOD_RESEARCH_GOAL.md`前初始化主agent和子agent，建立开放方法探索、落地前可行性审查、问题识别和高效协作规则
优先级：不替代现有`AGENTS.md`或`项目.md`

## 1. 初始化顺序与时间盒

主agent启动后只执行一次以下读取：

1. 根目录`AGENTS.md`：工作流、安全、Git、N607和协作规则；
2. `项目.md`：科学场景、数据协议和Stage2权限；
3. 本文件：模型路由、并发、子agent契约和无用工作禁区；
4. `docs/STAGE2_METHOD_RESEARCH_GOAL.md`：当前方法目标、路线和性能门；
5. 当前唯一活动研发报告及Git`status -sb/rev-parse HEAD`。

初始化最多一个工作阶段。输出一张不超过20行的上下文卡：

```text
data_handle = capsule_id/split_id/p2_min_v1/VALIDATED_ONCE
candidate_id = family/revision/DESIGN_DRAFT|DESIGN_FROZEN|IMPLEMENTING
baseline_id = matched current strongest legal baseline/ADV3B02 anchor
code_commit = exact Git SHA
active_report = one path
next_artifact = one deliverable
```

完成上下文卡后立即进入方法设计、最小实现、LODO、窄验证或实验分析。不得循环读取控制文件、全量会话、全部报告或重做数据验证。只有上下文卡字段冲突或证据缺失时，才定向查询conversation index或单个report。

## 2. 权威文件分工

|问题|唯一权威|
|---|---|
|科学场景、数据权限、query边界|`项目.md`|
|安全、Git、N607、报告和协作|`AGENTS.md`|
|当前性能目标、方法路线、实验门|`STAGE2_METHOD_RESEARCH_GOAL.md`|
|某run事实与性能|该run的`automation_reports/.../report.md`及完整artifact|
|历史检索|`conversation_index`，只作定位，不替代报告|

发生冲突时只按上述分工裁决，不创建新的平行authority文档。

## 3. 模型路由

|任务|模型与推理|允许输出|禁止扩展|
|---|---|---|---|
|方法方案、域适应、分类头、算法与数学设计|`gpt-5.6-sol`，`max`|假设、公式、因果臂、失败门、最小实现规格|代码修改、N607、泛化文献综述|
|数据质量、完整日志、实验结果、统计与推广分析|`gpt-5.6-sol`，`high`|同row证据表、异常归因、CI、合并/修订/拒绝|调参、修改方法、启动实验|
|代码修改、测试、review修复、Git冻结、实验发布与回收|`gpt-5.6-terra`，`high`|最小diff、测试、commit、run证据|改变冻结方法、按query结果选参|
|简单检索、文件定位、格式转换、表格整理、状态摘要|`luna`，`high`|短答案或结构化清单|方法设计、长历史扫描、N607 mutation|

若`luna`不可用，不得为了简单工作自动升级并spawn昂贵的`sol`。主agent自行在一个短回合完成，或把它合并到已经存在的Terra执行任务。若指定的Sol/Terra不可用，先报告模型不可用，不静默更换并声称遵循了路由。

## 4. 并发波次

最多4个活跃agent，包含主agent，即最多3个子agent。不得为凑并行数而spawn。

### 4.1 设计波次

- DA设计员：`gpt-5.6-sol max`，在全部协议合法方法族中提出域适应候选、机制假设、可辨识性和证伪条件；不得被C-id/C-dom、transport、adapter或现有代码限制；
- Head设计员：`gpt-5.6-sol max`，在全部统一分类方法族中提出分类候选、old/new/floor机制和互补性假设；不得被qKNN、SRDA或多头结构限制；
- 联合设计/监督分析员：`gpt-5.6-sol high`，组织可行性讨论，审查协议、coverage、可辨识性、decision invariance、K1、old/new平衡、类置换、资源、工程闭包和联合贡献辨识。

设计员不得自证。监督员只返回`MERGE/REVISE/REJECT`及证据缺口；`REVISE/REJECT`不得交给代码agent。主agent只冻结通过可行性审查的候选revision，不把任何reference candidate提升为默认主方法。

### 4.2 实现波次

只有存在`DESIGN_FROZEN`方法卡时才能进入实现波次。

- 代码owner：`gpt-5.6-terra high`，只按冻结设计修改预分配文件；
- 测试/review owner：`gpt-5.6-terra high`，只写非重叠测试或只读review；
- 分析员：`gpt-5.6-sol high`，只准备验收表、分析规格和只读结论；任何新增或修改分析脚本都交给Terra代码owner。

每个文件同时只能有一个owner。作者不得修改审查结论来使自己通过；P0必须清零，当前run范围内的P1必须清零。明确不在当前run范围内的P1可以保留，但须记录owner、风险、证据及不影响本run的理由，由主agent裁决。

### 4.3 发布波次

- 唯一N607 runner：`gpt-5.6-terra high`，独占run ID；
- 主agent继续下一候选的DA/head只读设计，不线性等待；
- 只在`LANDED`、`RUNNING_HEALTHY`、`ARTIFACTS_COMPLETE`、`ANALYZED`里程碑接收runner消息。

不得由主agent与runner双重启动，不得让多个子agent监控同一run。

## 5. 子agent任务包

每个子agent只接收最小上下文，不默认fork全历史。任务包必须包含：

```text
objective: 一个可判定问题
phase: DESIGN_DRAFT/FEASIBILITY_REVIEW/DESIGN_FROZEN/IMPLEMENTING/RELEASE/ANALYSIS
candidate_revision: 唯一候选revision或not-applicable
inputs: 最多5个明确文件或artifact
allowed_actions: read-only或精确文件所有权
forbidden_actions: 不得做什么
deliverable: 一个固定格式输出
acceptance: 可验证完成条件
budget: 回合/Token/字数上限
stop: 遇到什么立即返回
```

建议预算：方法设计≤3回合/10000 token；实验分析≤3回合/8000 token；代码/发布≤2回合/6000 token，远端回收可加1回合；简单任务≤1回合/2000 token。达到预算必须返回已有证据、缺口和唯一下一步，不得无边界续写。

子agent禁止：自行扩展目标、重复读取全项目、启动未授权实验、修改未分配文件、再spawn子agent、把建议写成已实现、把测试通过写成性能成功。两次连续没有新增证据时立即收束。

## 6. 落地前可行性讨论门

正式代码落地前必须完成一次有时间盒的可行性讨论。该门的目的不是增加会议，而是在最便宜的阶段发现协议冲突、不可辨识参数、决策不变性、负迁移和发布闭包缺口，减少“先实现—再推翻—继续补丁”。

候选必须按以下状态推进：

```text
DESIGN_DRAFT
  -> FEASIBILITY_REVIEW
  -> DESIGN_FROZEN
  -> IMPLEMENTING
  -> LOCAL_VERIFIED
```

`FEASIBILITY_REVIEW`必须形成一张决策卡：

|审查面|必须回答的问题|拒绝或修订信号|
|---|---|---|
|协议|训练、bundle、support、query和状态读写是否属于允许输入面|依赖clean/source、query统计、角色真值、未封存sidecar|
|机制|域适应和分类分别解决什么误差，联合为何互补|只有模块名称，没有可观测作用路径|
|可辨识性|K1/K5/K10下参数和统计量是否可估|自由度远高于support、靠support fit自证|
|决策有效性|方法是否改变margin、likelihood或metric|共同变换后完整重估导致预测不变|
|负迁移|coverage低、support噪声或old/new冲突时如何处理|没有收缩、回退、拒绝更新或guardrail|
|资源|参数、step、state、MAC、时延、显存和int8是否可过门|估算已明显超过硬门且无压缩路径|
|工程|改动文件、接口、依赖和发布包是否闭合|需要同时重构数据、算法、runner和报告|
|证伪|什么最小结果会拒绝该实例|没有falsifier，只计划不断调参|

监督员输出固定为：

```text
decision = MERGE | REVISE | REJECT
blocking_evidence = [...]
frozen_revision = <id or null>
allowed_files = [...]
minimal_test = ...
stop_condition = ...
```

只有`MERGE`可生成`DESIGN_FROZEN`并交给Terra实现。主agent在代码修改前向用户发布不超过20行的可行性摘要：候选机制、可行依据、主要风险、falsifier和冻结diff；除非涉及新增数据权限、科学场景或高影响动作，否则不把摘要变成等待确认的新阻塞门。关键事实若只能通过代码确认，可批准一次`FEASIBILITY_SPIKE`：只验证该事实、不得读取target query、不得自动继承为正式实现。设计讨论最多一个波次；信息已经充分时必须裁决，不能用持续讨论替代实验。冻结后若改变核心输入、机制、loss、head或适应规则，停止编码并创建新revision；单纯修复与设计一致的接口错误不重开审查。

## 7. 问题定义、触发信号与相似错误扩展

不要只匹配历史错误的名称。为每个问题记录：

```text
issue_fingerprint = (affected_object, information_source, causal_break, evidence_error, wasted_cost)
```

新问题只要与某类共享信息来源或因果断裂，并再共享至少一个结果特征，就按该类处置；名称、文件、agent或方法不同不构成新问题。

|编号|问题定义|常见触发信号|相似错误延伸|默认处置|
|---|---|---|---|---|
|E01目标发散|工作不再减少当前方法不确定性，也不产生`next_artifact`|重复读全历史、扩展无关文献、重写治理|无需求的dashboard、schema、prompt再包装|回到上下文卡，只保留一个交付物|
|E02实体混淆|把data handle、方法、revision、commit、run和result当成同一“版本”|D18被当算法；不同row比较；只报“最强”|同名candidate对应不同commit；拼接不同run极值|锁定`data_handle/candidate_revision/commit/run_id/row_id/result_status`并只做matched比较|
|E03过早锁定/过度否定|把一个候选写成全局必经路线，或用一次失败封禁方法族|“必须qKNN/SRDA”“transport永不再试”|固定rank、固定臂数、只因已有代码优先|恢复开放候选池；失败实例用`REENTRY_CARD`重入|
|E04跳过可行性|核心假设未审查便开始正式编码|落地后连续反转设计、接口与loss|用fixture猜真实asset、先写大框架再找问题|回到第6节；无`DESIGN_FROZEN`不实现|
|E05复合改动|一个candidate同时改变多个因果变量|metric、kernel、fusion、normalization一起变|算法修复夹带data/runner/report重构|拆成最小delta或预注册可辨识干预|
|E06域适应/分类混淆|分类头变化被称为域适应，或“对齐”不改变决策|只改温度/RDA却声明解决域偏移|共同正交变换、全量重估抵消变换|做组件干预和decision-invariance测试|
|E07代理替代结果|用便宜代理代替held任务性能|support fit 100%、重构余弦高、LODO单点正|代码通过、资源小、进程启动即称成功|代理只作门；结论以锁定held同row证据为准|
|E08确认污染|本应测试的信息反向影响方法选择|看125、Oracle或query后改rank/阈值|按receiver/class失败定向调参；重复跑到最好|生成新candidate；受污染结果仅作诊断|
|E09可辨识性/coverage失配|support信息不足以估计候选自由度或目标偏移不在先验span|K1拟合全矩阵；coverage低仍强更新|少数domain却高rank、support 100%但query负迁移|降维、收缩、identity回退或拒绝实例|
|E10任务失衡/floor盲区|提升old或均值却牺牲new、遗忘或最低类|只报BA/H；只盯历史难类|按TX ID专用权重、角色gate、均值掩盖长尾|同row报告全指标并保持类置换对称|
|E11重复验证/控制膨胀|方法变化触发已完成的数据或authority工作|重复hash、allowlist、Landlock、物理ID追溯|为每个candidate新增准入文档和签名层|只核对`VALIDATED_ONCE`句柄；数据事实变化才重验|
|E12复现过约束|把跨run封装字节一致当作语义可比前提|路径、时间戳、JSON/NPZ SHA不等就阻断|CRLF/LF、排序、容器元数据成为科学门|同run锁commit/hash；跨run比较稳定语义|
|E13Agent重叠/自证|多个agent重复工作、修改同一文件或作者自审|重复历史扫描、双重launch、共享文件覆盖|runner擅自调参；设计员自行宣布通过|单owner、唯一run owner、独立监督裁决|
|E14发布环境错配|本地测试未覆盖真实archive、import、device和线程环境|缺依赖、`cuda:4`/`cuda:0`错位、CPU线程爆炸|只跑`--help`、不同`PYTHONPATH`、漏tracked file|最终archive深层smoke、device contract和线程上限|
|E15状态语义混淆|把landed、running、complete、analyzed和promotable混为成功|0 prediction仍报告实验完成|技术修复被写成算法收益；负结果写“发布成功”|使用精确状态机和证据级别|
|E16证据碎片化|结论散在聊天、tail或多份不同报告中|无法找到同row表、完整log或最终verdict|手工维护两份长报告、边际最大值替代joint row|一个运行主报告＋自动Git镜像＋完整artifact|

硬停与可修订必须分开：协议泄漏、query选参、重复run owner、越权远端操作和证据伪造立即停止；方法负收益、coverage不足、分类头不互补或某个结构失败属于可修订研究问题。后者不能被扩展成方法族禁令，除非已有机制级证据证明在当前观测与资源边界下不可辨识。

## 8. 永久停止的无用工作

以下工作默认不做：

1. 对相同`capsule_id/split_id/p2_min_v1/VALIDATED_ONCE`重复做数据构建、物理ID追溯、hash、allowlist、authority或Landlock工程；
2. 因candidate、adapter、head、超参数、epoch、bundle、method lock、checkpoint推理状态或报告变化重验数据；
3. 把D18等数据句柄写成算法版本，或在D81/D62/D92/D93/D94之间做非matched比较；
4. 要求跨run的row-specific opaque handle、路径、时间戳、封装JSON或NPZ原始SHA bit-exact；稳定语义一致即可，raw SHA仅审计；
5. 不提交`REENTRY_CARD`便重复与历史负结果相同输入、机制和失败条件的实例；target多LEO重放、Phase2 clean/source cache、Role-Oracle决策和quota等协议禁止路线不存在重入例外；
6. 用support-fit 100%、重构余弦、ground prototype数量、代码测试、启动成功或资源较小替代held性能；
7. 无`DESIGN_FROZEN`便正式改方法代码，或把reference candidate、固定四臂/六臂、固定rank和固定head升级为全局研发约束；
8. 没有合法开发证据和K1/K10锁定窄门就运行125；用户明确要求诊断125除外，但不得晋级或选参；
9. 多个agent重复做同一文献检索、历史总结、diff review、结果解析或修改同一文件；
10. 主agent等待N607时停止研发，或runner擅自修改方法、调参、重启和干预其他任务；
11. 为修一个接口错误顺手重构pipeline、报告、schema和算法；每个patch只修最早失败边界；
12. 手工维护两份相同长报告。`automation_reports\CV-SincNet\<run-id>\report.md`是每次实验必需的运行主报告；Git承载面的镜像只在冻结与完成节点自动同步，不再人工双写；
13. 为CRLF/LF另造科学或版本门。源码身份由精确Git commit、archive SHA和成员原始字节SHA绑定，EOL只作信息记录。

只有`项目.md`列出的数据事实变化、真实artifact证明语义漂移、现有权限不足或用户明确覆盖时，才允许例外。例外必须说明触发字段和最小动作。

## 9. 代码修改链

每次修改严格执行：

1. 除第6节明确批准且隔离的`FEASIBILITY_SPIKE`外，修改前确认方法卡、`DESIGN_FROZEN`revision、允许文件、最小测试和停止条件完整；缺一项则返回可行性讨论，不编辑方法代码；
2. 运行`git status -sb`，读取真实失败artifact和当前实现；冻结“一个根因、一个主要delta、一个预期结果、一个停止条件”；
3. 先写最小失败反例和关键不变量攻击；禁止fixture猜测真实asset语义；
4. 作者只提交最小diff，不在同一commit混入数据、算法、runner和报告重构；
5. 在`ssr-gpu`串行执行专项单测→相邻集成→协议负例→真实checkpoint无数据smoke→`git diff --check`；
6. 独立review要求P0=0、当前run范围P1=0；范围外P1按第4.2节记录并由主agent裁决；
7. 仅在准备N607发布时，从精确Git commit一次性生成完整Git跟踪`code/`发布包并验证archive/member SHA；纯本地研发只需commit和窄验证；
8. 在与N607一致的目录和`PYTHONPATH`布局执行深层import、row pipeline`--help`、真实device contract和依赖闭包smoke；不得用target query做smoke；
9. 报告预登记后交给唯一runner。

GPU规则：外部`CUDA_VISIBLE_DEVICES=n`后，子进程内部统一使用`cuda:0`；smoke必须覆盖TorchScript真实前向、CPU/GPU tensor同设备。launcher固定OMP/MKL/OpenBLAS/NumExpr/BLIS线程上限并报告CPU/GPU阶段，避免D81式178线程膨胀。

## 10. 实验发布双轨

本地达到`LOCAL_VERIFIED`后，先建立`E:\type10-7\automation_reports\CV-SincNet\<run-id>\report.md`，并完整填写`AGENTS.md`要求的目标、假设、matched比较、commit/hash、改动与验证、同步映射、`DESIGN_FROZEN`revision、候选自适应证据包、环境/CWD、命令、GPU、日志、输出、指标、停止条件和风险。随后唯一Terra runner执行：

1. 运行`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`；direct N607不可达时才使用`AGENTS.md`规定的lab bridge；
2. 确认不可覆盖run/source/output/log→完整发布包同步→远端SHA/compile/import/device门→detached launch→PID/GPU/log首证据→短连接监控→完整artifact回收；
3. 每次SSH/SCP结束后核查本地`ssh.exe`及N607、bridge的TCP 22连接；超时或中断时清理确切残留连接，无法安全关闭则停止并报告。

主agent同时继续方法研发，不重复SSH、不启动同run。技术失败若发生在任何prediction之前，状态只能是`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；只修直接失败项并使用新run ID。完整prediction后性能不达标才是`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。

## 11. 分析与报告合同

分析员必须读取完整相关日志/artifact，而非只读tail。每个完成候选报告同一row的注册前old、注册后old、seen-new、H、BA、floor、min-old、min-new、forgetting、逐类、receiver、scene、K、seed、混淆、coverage、int8 margin、MAC、时延、显存和状态字节。

不允许拼接不同row的最大值。必须区分`LANDED`、`RUNNING`、`ARTIFACTS_COMPLETE`、`ANALYZED`和`PROMOTABLE`。Role-Oracle、历史违规qKNN、多view或clean-source结果只能作为显式上限/反例，不能进入正式性能声明。

## 12. 三轮复盘与主agent集成门

每完成3轮探索，在发布第4轮前做一次短复盘：只重读目标、`项目.md`、最近3份完整报告和拒绝原因；确认下一候选仍来自开放方法池、已完成可行性讨论、同时处理域适应与新类注册，并具备与其结构匹配的贡献辨识和联合协同证据。复盘最多一个工作阶段，结束后必须返回设计裁决、实现或实验。

主agent只有在以下全部满足时才能发布或晋级：协议合法、可行性审查通过、`DESIGN_FROZEN`、候选自适应因果证据已冻结、本地验证、P0清零且run范围P1清零、Git/报告可追溯、同rowmatched基线、完整artifact、分析员独立裁决。任何一项缺失都不得以更多讨论、更多agent或更大矩阵代替。
