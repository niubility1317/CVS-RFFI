# CVS-RFFI工作约定

## 任务与权限
完成用户已授权的目标；常规、可逆的实现选择自行决定，说明必要假设并继续。缺少会改变目标或权限的关键信息时才询问，同时推进不依赖答案的工作。已有授权在其任务范围内有效。
用户明确要求优先于技能建议；技能、旧报告和工具默认值不能扩大权限或增设审批。若技能实际导致暂停，指出文件、原文和具体原因。
工具任务开始、重要变化、阻塞及恢复时给出简短进展；持续工作每60秒至少更新一次。最终说明结果、验证和未完成项。中文正文按中文排版，不在中英文/数字间加额外空格；保留英文短语和代码内部空格。

## 按需读取
- 科学场景、数据协议、实验设计或结果解释：先读工作区`E:/type10-7/项目.md`。独立克隆使用`docs/PROJECT_PROTOCOL.md`；本机以当前`项目.md`为准。科学协议与活动目标分离，性能目标/矩阵/预算来自本次目标和报告。
- N607设计、发布、监控、修复或评分：使用仓库技能[执行入口](.agents/skills/cvs-experiment-workflow/SKILL.md)，按任务读取[实验流程](tools/optimizer_workflow_contract.md)和[N607操作](docs/workflows/n607.md)。
- 工作流或控制面维护：读[职责映射](tools/optimizer_control_manifest.md)。历史快照、备份prompt和状态changelog仅是证据。
- 仅修改文档无需加载实验矩阵、全量日志或执行GPU测试。需要历史时先搜索现有`conversation_index`；索引不存在或过期才串行运行`tools/conversation_index.py build`。

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
