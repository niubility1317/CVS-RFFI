# CVS项目资产治理进度

## 2026-08-13

- 读取`AGENTS.md`、`项目.md`及适用技能，确认协议、Git、N607和删除授权边界。
- 刷新项目会话索引，共1661条记录。
- 完成本地根目录、主要资产面、Git承载面和在制工作树的只读盘点。
- 完成N607直连预检、根级资产盘点、主要运行/日志/发布面统计，并验证SSH完全断开。
- 用户选择并批准方案A：清单先行、零搬迁。
- 从提交`b2e0f30c96b4586cff625a63b3a3a976c2f8e896`创建独立分支`codex/project-governance-20260813`和工作树`E:\type10-7\code\snapshots\project_governance_20260813_wt`。
- 已创建书面设计及任务专属`task_plan.md`、`findings.md`、`progress.md`；当前正在自检。
- 完成第一轮人工自检并修复资产粒度、实验身份、归档条件、删除候选条件和大清单分流门槛的歧义。
- 占位符扫描无命中，敏感值扫描无命中；四个文档存在且工作树除本任务新目录外无其他变化。
- 首次暂存检查发现设计元数据的Markdown硬换行尾随空格，已改为项目符号，等待重新验证。
- 重新暂存后`git diff --cached --check`通过，暂存区仅包含四个本任务文档；进入设计提交与用户复核门禁。
- 设计文档和工作日志已提交到独立分支；当前等待用户书面复核，不进入实现。
- 尚未实现扫描器、生成正式清单、移动或删除资产，也未重新连接N607。

## 2026-08-17

- 用户确认书面设计通过。
- 重新读取当前`AGENTS.md`、UTF-8版`项目.md`、Git Bash安全技能和实施计划技能，确认本轮只制定计划。
- 已验证首个终端外层为`C:\Program Files\Git\bin\bash.exe`且`MSYSTEM=MINGW64`；未执行`pwsh`。
- 核对现有工具、测试、N607进程盘点脚本和pytest配置，冻结治理工具包的文件边界与测试策略。
- 创建`docs/superpowers/plans/2026-08-17-project-asset-governance.md`，覆盖本地/N607只读采集、Git归属、实验索引、保留分类、待审批清单、输出体积分流、断连验证和两阶段Git提交。
- 完成实施计划自检：11个任务连续、文件职责一致、UTF-8读取通过、未保留未决实现选择，`git diff --check`通过。
- 当前仍未实现扫描器、生成正式清单、连接N607、移动、覆盖或删除任何资产。

## 2026-08-18

- 用户选择子代理驱动实施；主代理按冻结计划集成Tasks1–8，并由独立代理分别执行规格和代码质量复审。
- 完成本地资产采集、Git归属、实验索引、保留分类、N607只读采集协议、不可覆盖报告器和顶层CLI；默认路径不请求N607，`--print-plan`不创建输出、不连接网络。
- 主实施提交为`5eb740bea4cdca86912bc784459f6437351f2d2a`（`feat: add read-only project asset governance inventory`）。
- 关闭远端错误退出码、实施Git提交选择、原始N607错误证据、N607尝试级收据、结果状态闭合和有界stderr等复审问题；修复提交为`92b0ac5ead32200a68433f49096db4234c0f675b`（`fix: close governance CLI evidence gaps`）。
- 关闭N607根scope可被伪造为`NOT_PRESENT`且仍成功的问题；提交为`b20dee2a79bf3b500c08448726f6c02367d2d4d8`（`fix: reject non-verified N607 roots`）。
- 关闭本地配置根缺失被误报成功，并在CLI同时增加LOCAL/N607根唯一`VERIFIED`防御；提交为`dae859c8d85f0d159f0f461d653b81f0d969459e`（`fix: fail closed when local root is missing`）。
- Task8规格复审达到`P0=0、P1=0、P2=0`；代码质量复审达到`P0=0、P1=0、P2=1`，剩余总输出缓冲问题已记录为正式扫描后的非阻断优化项。
- Task9静态搜索已逐项审查：未发现破坏性执行路径、管理员N607路由、持久SSH连接或不安全主机密钥选项。
- 修复后由主代理在`ssr-gpu`环境串行复验8个根边界用例和8个治理/N607测试文件：定点8项通过，完整256项通过；随后执行`compileall`，退出码为0。
- 对`2a9cfe96..92b0ac5e`累计差异执行`git diff --check`，退出码为0。该范围含并行DOCX提交，不能据此把DOCX归入治理实现；治理代码采用逐文件提交和提交级复审。
- Task9独立审查发现并关闭正式执行计划中的P1路径缺陷：手工预检改用具有相邻host-local配置的根权威脚本，未复制或修改SSH配置、身份文件和密钥。
- 记录一个非阻断P2：收据只显式记录扫描时Git HEAD，未单列主实施与修复提交；本次文档提交会在该HEAD的Git历史中绑定上述4个完整hash，后续可增加受控`implementation_commits`字段。
- Task9最终独立总审查为`APPROVE，P0=0、P1=0、P2=3、Minor=0`；除显式commit列表外，另两个P2是本地外部命令总输出可能整体缓冲和最终收据非原子发布，均转入扫描后优化清单。
- 当前没有执行真实`powershell.exe`、`netstat.exe`、SSH、SCP、N607预检或网络连接，也没有移动、覆盖、删除、停止任务或远端写入。
- 现有DOCX修改、DOCX删除状态和全部`.docx_qa*`目录保持未暂存、未触碰；独立总审查门禁已通过，下一步是预览后执行一次正式只读盘点。
- 生成首次正式scan ID`PGOV_20260817T185755Z`并完成零写入预览；预览范围正确，两个目标目录保持不存在。
- 根权威N607预检通过，验证普通账号、direct配置/identity、服务器时间、项目根和8张RTX3090可见；预检后无活动`ssh.exe`或`ESTABLISHED`连接。
- 唯一正式扫描进程启动后未取得明确exit、终态收据或输出目录；该ID已封存为`UNKNOWN / NO_RECEIPT`且不得重跑。源资产变更、移动、覆盖和删除均为0。
- 已把长运行会话必须持续轮询到明确exit的规则补入实施计划；提交该操作性修复后才生成新的不可覆盖scan ID。
- 第二个不可覆盖ID`PGOV_20260817T191013Z`也未形成终态收据；最终两个ID均只留下0文件Git目录，external目录不存在，且Windows原生进程表确认没有活动Python/Conda扫描进程。两个ID均封存为`UNKNOWN / NO_RECEIPT`，空目录原样保留。
- 纠正监控假阳性：`ps -ef`的scan ID子串匹配命中了监控脚本自身命令行，不能作为存活证据。后续必须保留原始执行session到明确exit；fallback只能使用启动时记录的精确Windows PID、父子绑定和原生进程表交叉核验。
- 两次扫描终态均确认无`ssh.exe`和两个固定TCP22端点的`ESTABLISHED`连接；资产移动、覆盖、删除、权限修改、任务停止和远端写入均为0。
- 已启动真实规模性能修复：用TDD消除实验索引中的全资产/全组重复遍历，并增加阶段与终态可观测性审查。修复、回归、独立复审和Git提交完成前不启动第三次扫描。
- 完成实验索引预计算与缓存优化，提交`08cd32fa`；独立复审发现并关闭Windows盘符根/UNC根假绑定，修复提交`ea2f4fca`，最终复审`APPROVE，P0=0、P1=0、P2=0`。
- 完成不可覆盖阶段journal、精确Windows PID和N607 child/proxy liveness、收据三态、单次终态追加、progress限额闭合和诚实终态JSON，提交`5890cbfe`。
- 终态证据经过三轮独立故障注入复审，最终达到`APPROVE，P0=0、P1=0、P2=0`；两个非阻断Minor随后也已关闭。
- 主代理在`ssr-gpu`环境串行复验8个治理/N607测试文件，共301项通过；`compileall`和`5d8abc3b..5890cbfe`差异检查通过。工作树只保留未触碰的用户DOCX修改/删除和`.docx_qa*`目录。
- 已核对正式Git输出目标位于本地采集深度4，超过控制证据深度上限3，不会把进行中的progress自我纳入清单。下一步提交本日志后生成全新不可覆盖ID，执行零写入预览和一次正式只读扫描。
- 第三个不可覆盖ID`PGOV_20260818T023221Z`在界面执行会话丢失后仅保留3条progress记录，最后状态为`GIT/STARTED`；精确Windows PID`30260`已不存在，收据与external目录均不存在，SSH及两个固定端点连接为0。该ID封存为`UNKNOWN_ABRUPT_EXIT / NO_RECEIPT`，目录原样保留。
- 第四个不可覆盖ID`PGOV_20260818T030132Z`采用独立后台执行、精确Windows PID和原子退出码文件，成功完成零写入预览及根权威N607预检；预检验证普通账户、项目根和8张RTX3090可见。
- 第四次扫描在`GIT/STARTED`阶段暴露确定性性能缺陷：本地有66个直接Git工作树标记，实施仓库有72个linked worktree，旧实现会对每个候选重复展开同一common Git目录，最坏约执行`66×72×4=19008`次工作树元数据命令。该扫描尚未接触N607，已按精确PID仅终止本任务Python进程；退出码`143`已持久化，无收据，进度目录、runner日志和退出码文件均保留，未删除或覆盖任何资产。
- 按TDD新增真实linked-worktree回归：旧实现准确失败为`worktree list`调用2次；修复按规范化common Git目录缓存展开结果后GREEN。Git专属9项、完整治理/N607 302项、`compileall`及差异检查全部通过。
- Git归属性能修复提交为`eb42737af4b186d02dc308f5351c3e330675d3dd`（`fix: deduplicate linked Git ownership discovery`），提交精确只含`collect_git.py`及其测试；用户DOCX、`.docx_qa*`和所有扫描证据均未暂存。
