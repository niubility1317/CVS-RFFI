# CVS-RFFI工作约定

## Checkpoint复用硬禁令（2026-09-08）

- 正式CVS实验必须执行数据协议“Checkpoint训练数据一致性与继承污染禁令”。禁止加载训练数据契约不一致、上游来源不明、或曾利用目标数据训练/选模的checkpoint及其派生状态；同名数据集、split、架构或历史“成熟基座”不能证明合规。
- 在加载任何初始化/resume/teacher/EMA/蒸馏来源前，核对实际数据角色与物理ID、完整继承来源、选模依据，并在现有run报告记录结论。部分加载、冻结骨干、重置头/optimizer、改seed或重做LEO增强均不能绕过。
- CHECKPOINT_DATA_CONTRACT_MISMATCH、CHECKPOINT_PROVENANCE_UNVERIFIED、CHECKPOINT_TARGET_CONTAMINATED均阻止使用该来源；无合规权重时仅在已有授权内按本次契约从零训练，不得静默回退旧权重。历史产物保留，污染结果不得用于干净泛化或晋级。
- 本检查属于最小流程第1项“输入权限”，不是REJECTED_EXTRA_GATE；checkpoint兼容性与VALIDATED_ONCE数据复用分别判断。不新增数据重验、签名/receipt链、固定审批或重复审查。旧规则中“不因checkpoint变化做provenance检查”仅限重复数据验证，不得解释为免查模型来源。

## 任务与权限
完成用户已授权的目标；常规、可逆的实现选择自行决定，说明必要假设并继续。缺少会改变目标或权限的关键信息时才询问，同时推进不依赖答案的工作。已有授权在其任务范围内有效。
用户明确要求优先于技能建议；技能、旧报告和工具默认值不能扩大权限或增设审批。若技能实际导致暂停，指出文件、原文和具体原因。
工具任务开始、重要变化、阻塞及恢复时给出简短进展；持续工作每60秒至少更新一次。最终说明结果、验证和未完成项。中文正文按中文排版，不在中英文/数字间加额外空格；保留英文短语和代码内部空格。

## 按需读取
- 每次新建/启动实验及查找历史实验：使用[experiment-management](.agents/skills/experiment-management/SKILL.md)，覆盖CVS、外部对比、联邦、消融、诊断、seed扫描及独立评估。先查[实验总索引](experiment_registry/README.md)，再读命中的report/config/artifact；只有需要对话原因或授权时才查conversation_index。
- 科学场景、数据协议、实验设计或结果解释：先读工作区`E:/type10-7/项目.md`。独立克隆使用`docs/PROJECT_PROTOCOL.md`；本机以当前`项目.md`为准。科学协议与活动目标分离，性能目标/矩阵/预算来自本次目标和报告。
- N607设计、发布、监控、修复或评分：使用仓库技能[执行入口](.agents/skills/cvs-experiment-workflow/SKILL.md)，按任务读取[实验流程](tools/optimizer_workflow_contract.md)和[N607操作](docs/workflows/n607.md)。
- 工作流或控制面维护：读[职责映射](tools/optimizer_control_manifest.md)。历史快照、备份prompt和状态changelog仅是证据。
- 仅修改文档无需加载实验矩阵、全量日志或执行GPU测试。实验记录先搜索experiment_registry；需要对话历史时搜索现有`conversation_index`，索引不存在或过期才串行运行`tools/conversation_index.py build`。

## 实验登记与交接（2026-09-16）
- 所有启动路径（手动命令、launcher、队列、授权自动化）在既有预登记中维护`automation_reports/CV-SincNet/<run_id>/experiment.json`、`report.md`和`events.jsonl`；统一字段与命令见[管理规范](docs/EXPERIMENT_MANAGEMENT.md)。每次新实验记录名称/说明、稳定group_id、不可覆盖run_id、逐行配置、数据/seed角色、checkpoint完整来源、执行环境/commit/命令、输出/log/预期artifact位置和唯一launch owner。
- 启动前填实际配置，启动后登记PID/GPU/log与resolved config证据，状态变化和评分完成后更新原记录与索引。六类seed分别记录model/split/data/augmentation/support/evaluation，不凭单个seed推断所有随机性；矩阵每row独占输出。说明和模板是用户要求的既有第5项预登记，不另设审批、哈希链或数据重验。
- 新对话先搜索索引、读取命中记录，再核实必要的实时证据；禁止从旧RUNNING、目录日期或已有配置推断当前运行/完成。历史失败、停止、替代、source-only、仅配置和partial均保留，未知字段不猜填；不为整理移动、删除或重命名原数据、checkpoint、日志和输出。
- 登记、模板、工具和索引元数据随本次相关报告进入Git；大体积数据与原始产物保持原存储并用路径引用。训练/选模不能读取含历史目标评分的索引来调参或选择性重跑，评分权限仍按项目协议。

## Windows执行
使用Windows原生环境，不通过WSL转接Windows项目。`pwsh/pwsh.exe`保持禁用，除非用户经有界兼容性测试重新授权。
优先原生可执行文件和已验证Python；必要时用短小的`powershell.exe -NoProfile -NonInteractive`调用，多步操作用已审阅的.ps1。
在PowerShell/.ps1、Conda、中文/JSON、SSH、UAC、注册表/服务/网络、递归扫描或归档操作前，读全局[故障目录](E:/codex/home/skills/using-git-bash-on-windows/references/powershell-failure-catalog.md)，同一任务不重复读取。
仅明确需要.sh、POSIX或MSYS2时使用`using-git-bash-on-windows`：精确选择`C:/Program Files/Git/bin/bash.exe`、`login:false`，先确认`MSYSTEM=MINGW64`；若被路由为System32 Bash/WSL，停止Bash载荷。全局旧技能的“所有Windows操作必须经Git Bash”不适用于本项目。
项目代码测试使用激活后的`ssr-gpu`；非交互调用可串行`conda run -n ssr-gpu ...`，核对解释器和环境。禁止并发Conda包装。结构化文本使用显式UTF-8；读回验证编码。失败先定位命令层，取得新证据后再重试。

## Git与交付
`E:/type10-7`当前不是有效Git仓库；主承载面为`github_publish/CVS-RFFI-repo`。修改前运行`git status -sb`，保护其他人的暂存/未暂存内容；需要隔离时从已核实基线创建工作树。
所有正式代码、配置、脚本、报告和控制面修改进入Git。根目录交付物镜像到Git承载面后提交。显式stage本次路径；不要`git add -A`、强推、重写共享历史或清除无关内容。
检查diff并完成与变更相关的验证后，提交并立即push；无upstream时使用`git push --set-upstream origin HEAD`。独立比较远端分支OID与本地HEAD，检查无ahead/behind。失败保留提交并报告FAILED或UNKNOWN。
GitHub发布/仓库治理默认分支与PR流程；不擅自合并。记录变更用途、验证、提交和交付路径。仓库自动push hook若已安装可复用，仍需远端读回。
仅实际行为变更需要相应测试；文档检查编码、链接和规则一致性。聚焦验证通过后，仅新失败或未解决风险才扩大测试。单纯修订报告不触发实验复审或重新验证数据。

## 不可削弱的实验边界
`项目.md`定义科学权限，包括外部对比方法的显式例外。Stage2主方法的query及其view只读、逐样本面对全部注册类，不使用truth/role、真实类别数量、配额或全局重排；prediction固定后才由独立scorer连接truth，结果不回流调参/选择/重跑。
匹配`p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`的数据跨方法复用。只有received IQ、物理ID、receiver/TX、scenario、K、support/query或schema变化才重验；方法、checkpoint、预算、报告变化不触发重验。
实验仅允许[八项最小流程](tools/optimizer_workflow_contract.md#八项最小流程)中的直接正确性要求阻断；白名单外记录`REJECTED_EXTRA_GATE`并继续。禁止另加哈希/封存/receipt/authority链、重复审查、完整125早期门槛或smoke许可。数据builder的一次性职责不转嫁给方法研发。
保护数据集、checkpoint、日志、指标、报告和输出；删除/覆盖必须有明确范围授权。N607管理员`N607-admin/szu2310433034`默认禁用：当前任务的登录授权与每项状态修改授权分别取得，任务结束失效。
远端默认只读，修改先本地Git验证再SCP。监控请求不授权干预健康任务；每GPU最多两个训练实验。只有授权的新实验可使用剩余名额。低性能不能停机；技术故障仅按预登记规则处理所属run，保留产物，不影响其他进程。
联邦WiSig训练比例固定0.1；默认epochs=200、fl_rounds=200、fl_client_key=receiver，除用户明确覆盖相应默认值。

## 协作
独立P0/P1实验审查按最小流程执行。其他并行子任务仅在当前运行环境允许且能节省时间时使用，分配互不重叠的责任；不按固定角色数派生Agent，不重复审查。主Agent负责整合与科学解释；一个run只有一个launch owner。
恢复长任务先读当前交接，核实已有产物和进程再继续；不因对话压缩、SSH超时或中途提问重复启动。交接只保存目标、已完成、当前run/commit、证据路径、阻塞和下一步。
