# Stage2研发Agent初始化与高效协作协议

版本：2026-07-22
用途：在加载`docs/STAGE2_METHOD_RESEARCH_GOAL.md`前初始化主agent和子agent
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
method_id = A/B/C-id/C-dom/C-joint/D或具体candidate
baseline_id = matched A/D81/D92等
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

- DA设计员：`gpt-5.6-sol max`，只设计C-id/C-dom/C-joint；
- Head设计员：`gpt-5.6-sol max`，只设计B/SRDA与融合互补；
- 监督分析员：`gpt-5.6-sol high`，交叉审查coverage、可辨识性、common-transform invariance、K1、old/new平衡、类置换和资源。

两位设计员不得自证。监督员只返回`MERGE/REVISE/REJECT`及证据缺口。主agent冻结唯一集成规格。

### 4.2 实现波次

- 代码owner：`gpt-5.6-terra high`，只修改预分配文件；
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

## 6. 永久停止的无用工作

以下工作默认不做：

1. 对相同`capsule_id/split_id/p2_min_v1/VALIDATED_ONCE`重复做数据构建、物理ID追溯、hash、allowlist、authority或Landlock工程；
2. 因candidate、adapter、head、超参数、epoch、bundle、method lock、checkpoint推理状态或报告变化重验数据；
3. 把D18等数据句柄写成算法版本，或在D81/D62/D92/D93/D94之间做非matched比较；
4. 要求跨run的row-specific opaque handle、路径、时间戳、封装JSON或NPZ原始SHA bit-exact；稳定语义一致即可，raw SHA仅审计；
5. 重复讨论已被完整负结果否定的全坐标transport、无coverage ground强制对齐、共同正交变换、query多view重放、clean样本缓存、Role-Oracle或quota；
6. 用support-fit 100%、重构余弦、ground prototype数量、代码测试、启动成功或资源较小替代held性能；
7. 没有Phase1 LODO和K1/K10窄门就运行125；用户明确要求诊断125除外，但不得晋级或选参；
8. 多个agent重复做同一文献检索、历史总结、diff review、结果解析或修改同一文件；
9. 主agent等待N607时停止研发，或runner擅自修改方法、调参、重启和干预其他任务；
10. 为修一个接口错误顺手重构pipeline、报告、schema和算法；每个patch只修最早失败边界；
11. 手工维护两份相同长报告。`automation_reports\CV-SincNet\<run-id>\report.md`是每次实验必需的运行主报告；Git承载面的镜像只在冻结与完成节点自动同步，不再人工双写；
12. 为CRLF/LF另造科学或版本门。源码身份由精确Git commit、archive SHA和成员原始字节SHA绑定，EOL只作信息记录。

只有`项目.md`列出的数据事实变化、真实artifact证明语义漂移、现有权限不足或用户明确覆盖时，才允许例外。例外必须说明触发字段和最小动作。

## 7. 代码修改链

每次修改严格执行：

1. 修改前：`git status -sb`，读取真实失败artifact和当前实现；冻结“一个根因、一个主要delta、一个预期结果、一个停止条件”；
2. 先写最小失败反例和关键不变量攻击；禁止fixture猜测真实asset语义；
3. 作者只提交最小diff，不在同一commit混入数据、算法、runner和报告重构；
4. 在`ssr-gpu`串行执行专项单测→相邻集成→协议负例→真实checkpoint无数据smoke→`git diff --check`；
5. 独立review要求P0=0、当前run范围P1=0；范围外P1按第4.2节记录并由主agent裁决；
6. 仅在准备N607发布时，从精确Git commit一次性生成完整Git跟踪`code/`发布包并验证archive/member SHA；纯本地研发只需commit和窄验证；
7. 在与N607一致的目录和`PYTHONPATH`布局执行深层import、row pipeline`--help`、真实device contract和依赖闭包smoke；不得用target query做smoke；
8. 报告预登记后交给唯一runner。

GPU规则：外部`CUDA_VISIBLE_DEVICES=n`后，子进程内部统一使用`cuda:0`；smoke必须覆盖TorchScript真实前向、CPU/GPU tensor同设备。launcher固定OMP/MKL/OpenBLAS/NumExpr/BLIS线程上限并报告CPU/GPU阶段，避免D81式178线程膨胀。

## 8. 实验发布双轨

本地达到`LOCAL_VERIFIED`后，先建立`E:\type10-7\automation_reports\CV-SincNet\<run-id>\report.md`，并完整填写`AGENTS.md`要求的目标、假设、matched比较、commit/hash、改动与验证、同步映射、冻结矩阵、环境/CWD、命令、GPU、日志、输出、指标、停止条件和风险。随后唯一Terra runner执行：

1. 运行`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`；direct N607不可达时才使用`AGENTS.md`规定的lab bridge；
2. 确认不可覆盖run/source/output/log→完整发布包同步→远端SHA/compile/import/device门→detached launch→PID/GPU/log首证据→短连接监控→完整artifact回收；
3. 每次SSH/SCP结束后核查本地`ssh.exe`及N607、bridge的TCP 22连接；超时或中断时清理确切残留连接，无法安全关闭则停止并报告。

主agent同时继续方法研发，不重复SSH、不启动同run。技术失败若发生在任何prediction之前，状态只能是`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；只修直接失败项并使用新run ID。完整prediction后性能不达标才是`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。

## 9. 分析与报告合同

分析员必须读取完整相关日志/artifact，而非只读tail。每个完成版本报告同一row的注册前old、注册后old、seen-new、H、BA、floor、min-old、min-new、forgetting、逐类、receiver、scene、K、seed、混淆、coverage、int8 margin、MAC、时延、显存和状态字节。

不允许拼接不同row的最大值。必须区分`LANDED`、`RUNNING`、`ARTIFACTS_COMPLETE`、`ANALYZED`和`PROMOTABLE`。Role-Oracle、历史违规qKNN、多view或clean-source结果只能作为显式上限/反例，不能进入正式性能声明。

## 10. 三轮复盘与主agent集成门

每完成3轮探索，在发布第4轮前做一次短复盘：只重读目标、`项目.md`、最近3份完整报告和拒绝原因；确认下一候选同时处理域适应与新类注册，且有A/B/C/D因果臂。复盘最多一个工作阶段，结束后必须返回实现或实验。

主agent只有在以下全部满足时才能发布或晋级：协议合法、单一机制规格冻结、本地验证、P0清零且run范围P1清零、Git/报告可追溯、同rowmatched基线、完整artifact、分析员独立裁决。任何一项缺失都不得以更多讨论、更多agent或更大矩阵代替。
