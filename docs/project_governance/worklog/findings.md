# CVS项目资产治理已验证发现

## 当前事实快照

- 本地根目录`E:\type10-7`不是有效Git仓库；其中`.git`为空目录，缺少`HEAD`和`config`。
- Git发布承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`。盘点时受跟踪文件干净，但有290个折叠后的未跟踪状态项，主要包括pytest临时目录、`local_artifacts`和实验配置。
- `E:\type10-7\code\snapshots\ground_proto_da_rd_wt`存在1个受跟踪修改和40个折叠后的未跟踪状态项，必须视为用户在制工作。
- 本地根级资产共166项，其中99个文件、67个目录，根级文件合计约6.19GB。
- 本地`automation_reports\CV-SincNet`有1589个顶层项，`code\snapshots`有785个顶层项。
- N607普通账号项目根为`/home/szu2070436088/2510044040/CV-SincNet`，不是Git仓库。
- 2026-08-13只读盘点时，N607的8张RTX 3090均无计算进程，项目根所在磁盘约11TB、已用3.0TB、可用7.4TB。
- N607项目根有185个顶层项，其中119个文件、66个目录，根级文件合计约5.75GB；包含23个`.tar`和23个`.out`。
- N607的`runs`有1307个顶层项，`logs`有487个顶层项，`releases`有159个顶层项。
- N607存在零字节输出与异常名称，但它们只能进入人工审查，不能仅凭名称或大小判定可删除。
- 所有本次SSH命令结束后，本机`ssh.exe`和到N607/实验室桥接TCP22的连接均为0。

## 设计结论

- 第一阶段必须用索引实现“整理”，不能通过搬迁原件实现目录整齐。
- Git归属、实验状态和保留级别必须分开建模；“非Git”不等于“无价值”。
- 大文件首轮只记录元数据；只有疑似重复或进入待审批删除表的对象才补充内容哈希。
- 删除清单必须包含精确路径、理由、依赖、可恢复性和预计释放空间，并保持`AWAITING_USER_APPROVAL`。
- 方法与性能优化是治理闭环后的独立子项目，不与资产分类同时推进。

## 书面设计自检发现

- “资产总表”需要明确治理粒度，否则可能误解为递归枚举全部数据样本；设计已改为根级全覆盖、主要承载面直接子项全覆盖、控制证据深度受控发现。
- `experiment_id`、历史归档和删除候选原本存在解释空间；设计已补充稳定身份、非mtime归档规则及删除候选的全部必要条件。
- Git内完整清单与外部大清单的分流门槛原本不够精确；设计已固定单文件10MiB、单次提交50MiB和外部artifact路径。

## 2026-08-17实施规划发现

- 用户已确认书面设计通过；该确认授权制定实施计划，不构成任何删除、移动、覆盖或远端写入授权。
- 当前治理工作树存在不属于本任务的`.docx_qa_cvs_ntn*/`系列目录；规划期间该系列由`_v2`继续增加到`_v4`。`tools/build_cvs_ntn_scenario_docx.py`曾作为未跟踪文件出现，后续状态中已不再列出。这说明工作树有并行用户活动；本任务未触碰这些内容，并要求所有暂存操作只列出本任务精确文件。
- 顶层安全规则现要求所有Windows终端任务使用Git Bash且硬性禁止`pwsh`。实施计划据此使用精确Git Bash外壳；仅现有N607预检脚本可由Git Bash窄调用`powershell.exe`，不新增PowerShell实现。
- 现有`tools/n607_training_inventory.py`已经提供远端进程识别的可复用分类逻辑，但正式资产采集仍需独立的路径深度、NDJSON、审批和断连证据边界。
- 工程测试同时使用`pytest`与`unittest`，根`pytest.ini`已把`.`和`code`加入导入路径；新治理工具适合放在`tools/project_governance`并由根`tests`做纯标准库fixture测试。

## 2026-08-18实施与验收发现

- Tasks1–8已实现稳定资产身份、受控本地采集、精确Git归属、证据驱动实验索引、保留分级、N607有界NDJSON采集、不可覆盖报告输出和默认不接触N607的CLI编排。
- 正式收据会记录实施工作树的精确Git提交、N607结果与路由、预检状态、每次尝试的child/proxy PID与退出证据、断连状态、残留端点、活跃训练观察和原始`SCAN_ERROR`证据；任一未知远端结果优先返回退出码3，已证实扫描错误返回2。
- 所有删除候选只能处于`AWAITING_USER_APPROVAL`和`NOT_AUTHORIZED`；实现不提供删除、移动、覆盖、权限修改、停止进程、Git暂存、Git提交或Git推送执行路径。
- 静态安全搜索中的危险词命中均来自负向测试、显式安全说明、字符串标准化、不可变数据替换或内存缓冲裁剪；管理员账号、`ControlMaster`、`ControlPersist`和不安全主机密钥选项无命中。
- Task9修复后完整治理/N607聚焦套件共256项测试通过，`compileall`通过，累计实施差异的`git diff --check`通过。
- Task8规格复审结论为`P0=0、P1=0、P2=0`；代码质量复审为`P0=0、P1=0、P2=1`。残余P2是受信预检stdout总量及本地`netstat`断连探针总输出仍可能整体缓冲；单行读取和stderr尾部已有界。该项不增加远端写入、删除、自动终止或路由歧义，记录到正式扫描后的工程优化清单。
- 主实施提交为`5eb740bea4cdca86912bc784459f6437351f2d2a`，证据闭合修复为`92b0ac5ead32200a68433f49096db4234c0f675b`，N607根验证修复为`b20dee2a79bf3b500c08448726f6c02367d2d4d8`，本地根与CLI防御修复为`dae859c8d85f0d159f0f461d653b81f0d969459e`；正式扫描收据HEAD必须在Git历史中绑定这些版本事实。
- 并行存在的DOCX修改、DOCX删除状态和`.docx_qa*`目录不是本任务资产；实施提交和修复提交均未暂存、恢复、删除或改写它们。
- Task9独立审查发现Task10手工预检原先指向隔离工作树脚本，但该位置没有相邻`n607_ssh_config`，照计划会在本地安全失败。计划已改为根权威`E:/type10-7/tools/n607_ssh_preflight.ps1`；生产采集器本来就固定到同一脚本及相邻配置，不复制配置或密钥。
- 当前收据schema只显式写入扫描时`implementation.git_head`，未单列主实施和修复提交。后续文档提交会把上述完整hash固化在`progress.md`，正式收据的HEAD会在Git历史中绑定该文档；显式`implementation_commits`数组作为非阻断P2列入后续工程优化。
- Task9总审查发现并关闭两项根范围P1：本地配置根缺失现在产生`SCAN_ERROR`和退出码2；N607 NDJSON根scope必须唯一且为`VERIFIED`。可选承载面缺失仍保留`NOT_PRESENT`，不会被误判为错误。
- Task9最终独立结论为`APPROVE，P0=0、P1=0、P2=3、Minor=0`。3个P2是预检stdout与`netstat`总输出仍可能整体缓冲、收据未显式保存实施commit列表、最后写入的收据不是原子发布；它们不增加删除、移动、覆盖、远端写入、自动终止或路由歧义。

## 2026-08-18首次正式扫描尝试

- scan ID`PGOV_20260817T185755Z`的零写入预览通过，明确包含本地根、10个本地承载面、N607根、7个N607承载面及`n607_contact=true`；预览前后两个输出目录均不存在。
- 手工根权威预检最终exit0，普通账号`szu2070436088`、direct配置与identity、服务器时间、项目根和8张RTX3090均可见；首次未整体引用Windows路径的调用exit127且未触达N607，已归因为Git Bash路径引用错误。
- 唯一正式扫描进程曾启动，但执行控制器没有保留到明确exit，stdout/stderr均无终态信息，且Git/external输出目录均未创建。因此不能推断成功或失败，必须把该ID标为`UNKNOWN / NO_RECEIPT`并禁止重跑。
- 扫描后未见`ssh.exe`客户端或到两个固定TCP22端点的`ESTABLISHED`连接；仅N607直连端点存在PID0的`TIME_WAIT`闭合套接字，bridge端点无连接记录。
- 本次尝试没有形成删除候选或任何可执行授权，原资产移动、覆盖、删除、权限修改和任务停止均为0；DOCX和`.docx_qa*`未触碰。
- 具体执行缺陷是长运行命令会话没有持续轮询到明确process exit。计划已增加“保留同一session并轮询到终态”的规则；该规则提交后才能以新ID重试，不能把空输出或子进程消失当作完成证据。

## 2026-08-18两次正式扫描终态纠正

- 首个ID`PGOV_20260817T185755Z`和第二个ID`PGOV_20260817T191013Z`最终均无活动Python/Conda进程。两个Git输出目录存在但均为0文件，两个external目录均不存在，`scan_receipt.json`均不存在；由于原始执行会话和退出码已丢失，两个ID都只能封存为`UNKNOWN / NO_RECEIPT`，不得重跑同一ID。
- 先前长期报告的`process=PRESENT`是假阳性：监控命令使用`ps -ef`按scan ID匹配时，命中了监控脚本自身命令行中的`for run_id in ...`。最终由Windows原生`ps -W`和`tasklist.exe /FO CSV`交叉确认无Python/Conda扫描进程。后续不得用包含目标ID的监控命令行做单一存活证据；必须保留原始执行session，或使用启动时记录的精确Windows PID及父子绑定。
- 终态独立核验同时确认`ssh.exe=0`、到`172.31.111.215:22`和`172.31.105.18:22`的`ESTABLISHED=0`。两次尝试均未形成删除候选或授权，原资产移动、覆盖、删除、权限修改、进程停止和远端写入均为0；两个空目录作为失败现场保留，未经用户明确同意不删除。
- 静态性能审查发现`index_experiments.py`在显式run root、expected artifact、同commit binding和known run ID关联上存在多处全量笛卡尔遍历；真实资产规模下会放大为数千万至上亿次Python路径/字符串比较。源码未见无限循环，但该实现不能继续用于第三次正式扫描。下一步先以TDD改为等价的路径索引、binding缓存和token倒排，再用新ID启动唯一一次正式扫描。

## 2026-08-18规模性能与终态证据修复闭合

- 实验索引已改为一次性规范化路径索引、证据binding缓存、commit+token倒排和known run ID去重。180资产规模回归中，旧实现触发9580次路径包含判断；新实现受确定性调用计数约束，不依赖墙钟计时。性能提交为`08cd32fa7e705494877b46bc06f58c75269c9527`。
- 独立复审发现首版路径索引对Windows盘符根和UNC共享根产生新增后代假绑定。修复复用旧LOCAL包含谓词，保持盘符根、UNC、普通LOCAL、POSIX根和N607反斜杠/大小写语义；修复提交为`ea2f4fca97012cfc0f6b8d93ea448a261a4df7ec`，最终复审`P0=0、P1=0、P2=0`。
- CLI现在在安全校验后独占创建`scan_progress.ndjson`，以UTF-8 NDJSON、flush和fsync记录固定阶段、当前Windows PID、受控N607 child/proxy退出事实和组合liveness。既有空目录、错误token、额外文件、symlink及任一终态journal都不能被重新用于写出。
- 收据仍最后写，保留既有`receipt_file`字段，并记录`terminal_state`及冻结progress的字节数和SHA-256。完整、部分和不可判定收据采用`MATCH/PARTIAL/UNKNOWN`三态；只有明确部分写入才允许追加失败终态，回读未知一律不再改写journal并返回退出码3。
- progress计入Git单文件和总量上限；token固定为48位小写十六进制。主child和全部proxy只有都证实退出时才记`EXITED`；timeout或proxy未证退出保持`LIVE_CHILD_UNKNOWN/UNKNOWN`。终态追加只尝试一次，异常处理本身不能遮蔽原始失败或制造重复终态。
- 终态修复提交为`5890cbfe`。独立故障注入经过三轮复审后达到`APPROVE，P0=0、P1=0、P2=0、Minor=2`，随后两个Minor也已关闭。主代理在`ssr-gpu`环境串行复验8个治理/N607测试文件，共301项通过；`compileall`和`git diff --check`通过。
- 当前Git输出目录位于`code/snapshots/<worktree>/docs/project_governance/<scan_id>`。相对`code/snapshots`承载面，progress文件处于受控证据深度4，超过配置上限3；普通目录本身也不作为控制证据记录。因此本次进行中的progress不会被本地采集器自我纳入或产生变化中哈希。

## 2026-08-18 Git归属展开性能修复

- `PGOV_20260818T023221Z`的持久化证据只到`GIT/STARTED`，精确PID消失且没有退出码或收据，只能封存为`UNKNOWN_ABRUPT_EXIT / NO_RECEIPT`；该结论不等于扫描成功或资产清单完成。
- `PGOV_20260818T030132Z`首次把扫描与界面会话解耦，精确Windows PID为`25176`。本地阶段完成后，Git阶段长时间无新事件；只读静态核验发现`discover_repositories()`会为每个linked-worktree候选重新执行`worktree list`，再对同一组全部工作树重复读取common dir、branch、HEAD和status。
- 当前本地快照面有66个`.git`标记，实施仓库有72个linked worktree；旧路径最坏会放大为4752次工作树展开、约19008次Git元数据命令。这是确定性重复计算，不是数据规模本身或N607故障。
- 扫描在尚未进入N607时按progress绑定的精确PID有意终止，退出码`143`由独立wrapper持久化；无收据、无删除候选、无SSH连接、无远端写入，原资产移动、覆盖、删除、权限修改均为0。
- 修复仅用common Git目录键去重仓库展开；不同工作树仍分别保留branch、HEAD和status，独立仓库不合并，资产分类和公开schema不变。真实Git工作树测试证明同一common目录的`worktree list`由2次降为1次，两个工作树status各保留1次。
- 修复提交`eb42737af4b186d02dc308f5351c3e330675d3dd`通过Git专属9项、完整治理/N607 302项、编译与差异检查。下一次正式扫描必须使用新ID并绑定该提交之后的Git HEAD。
- `PGOV_20260818T032541Z`在common目录去重后仍于Git阶段持续占用单核。10秒累计CPU增加10秒且无Git子进程，把问题进一步收敛到Python内部资产归属匹配，而不是`git status`或服务器I/O。
- `_repository_for_path()`原实现对每条资产遍历仓库时都重新调用`_resolved_path(repository.repository_root)`。真实Git fixture的80资产、2工作树测试测得171次解析；缓存仓库根后降为常数级不超过20次，同时保持最长根优先和原有Windows包含语义。
- 第二项修复提交`04dcce542313301496eef2438548d56449a4ea07`通过Git专属10项、完整治理/N607 303项、编译与差异检查。该扫描在N607前被精确终止，退出码`143`，无远端连接、无收据、无删除候选和资产变更。

## 2026-08-18首次全链到达N607后的闭合修复

- `PGOV_20260818T035415Z`证明LOCAL、GIT、INDEX和RETENTION可在真实规模下完成；N607根权威预检为`DIRECT_READY`，但远端只读采集在旧45秒整段总时限处返回`UNKNOWN`。该时限同时覆盖7个承载面的深度3遍历和受限控制文件读取，而SSH连接建立另有10秒限制，因此失败证据指向采集预算不足，不是身份、路由或连接建立歧义。
- timeout尝试记录的主child PID`18132`和proxy PID`16640/21764/31512/35228`均在事后精确核验中不存在；本机无`ssh.exe`，到`172.31.111.215:22`与`172.31.105.18:22`的`ESTABLISHED`连接均为0。不得把timeout当作成功；本次正式结果保持`UNKNOWN`，也没有改走非授权路由。
- 同一正式扫描在EMISSION阶段因`git status --porcelain=v2 -z`的NUL分隔状态进入CSV writer而失败。`PGOVD_20260818T041946Z`用本地只读诊断准确复现`_csv.Error: need to escape, but no escapechar set`；修复只对CSV单元格转义NUL，JSON仍保留原值，提交为`92cb63eb`。
- N607整段采集预算由45秒改为15分钟，仍是一次性有界命令，仍保持`ConnectTimeout=10`、流式NDJSON校验、无远端文件、无自动重试和终态/断连证据要求。TDD先使旧值精确失败，再以提交`e702fcec`转绿；完整治理/N607 304项测试通过。
- `PGOV_20260818T035415Z`与诊断ID均无最终收据，不能作为资产总表。所有progress、runner log和exit证据保留；删除、移动、覆盖和清理任何失败现场仍需用户明确批准。

## 2026-08-18真实规模输出与N607传输修复

- `PGOV_20260818T044859Z`首次让N607采集自然运行到returncode0和明确断连。预检与DIRECT两次attempt均无timeout，主child和全部proxy均`EXITED`，断连为`VERIFIED`；因此N607结果`FAILED`不能归因于连接残留。
- 独立只读诊断`PGOVN607D_20260818T052813Z`直接保留collector的错误文本，确认唯一失败为流式NDJSON超过256MiB总量上限。远端payload此前会为特定控制文件携带最多2MiB`evidence_text`，但CLI的`AssetRecord`转换完全不读取或保存该字段，实验索引也不消费它；这属于无效传输放大。
- 提交`47b06b01`删除未消费正文的wire字段与解码路径，仍保留资产身份、相对路径、显示/转义名、类型、大小、mtime、访问状态、哈希状态、受限SHA、证据角色、scope、process、SCAN_ERROR和COLLECTION_COMPLETE。无效UTF-8控制文件现在按原始字节做受限SHA，不再因仅用于丢弃的文本解码而制造SCAN_ERROR。
- 同一正式扫描的external侧已完整写出8项约2.1GiB文件；旧Emitter随后又把CSV完整复制为Git分片，Git侧写出101项约895MiB后才执行50MiB最终检查，所以没有收据。该失败正确阻止了伪成功，但分片策略在真实规模下不可完成。
- 提交`44d5988d`在写分片前比较完整CSV字节与总预算，并为不可能容纳的情况选择`EXTERNAL_COMPLETE_WITH_GIT_SUMMARIES`；external完整表、绝对路径、字节数和SHA仍进入收据，Git不再复制注定超限的完整分片。新增400行规模测试由旧实现准确RED为`325265>50000`，修复后收据、摘要、外置完整表和总量均闭合。
- 所有失败目录和external文件均作为诊断证据保留，未删除、移动或覆盖；它们不是最终资产总表，也不包含任何删除授权。

## 2026-08-18最终正式资产治理清单

- 移除未消费正文后，真实N607 wire仍为329443890字节、436217条记录，说明剩余超限来自高基数资产元数据中的确定性重复字段，而不是正文、路由、超时或断连。ASSET wire省略`asset_id/location/root_id/display_name/escaped_name`后，由本地基于canonical relative path重建并对旧格式矛盾值失败关闭；真实wire降至192735445字节，记录数不变，256MiB总限额未放宽。
- 最终扫描`PGOV_20260818T062450Z`的N607结果为`VERIFIED`，普通账户直连、主child和全部proxy退出、断连均有闭合证据。20个本地/N607配置scope全部`VERIFIED`，证明服务器内容已实际进入清单，而非仅完成连接预检。
- 最终资产总表包含619283项：本地183077项、N607 436206项。N607侧为2027个目录、434054个文件和125个符号链接，全部436206项`access_status=OK`；433673项受控文件形成SHA256，25项因大小上限不读取内容。
- 唯一N607`SCAN_ERROR`是`proc_partial_visibility`：普通账户无法读取1037个`/proc`条目，并保留10个样本PID。该错误影响“未观察到活动训练”的完备性，不影响资产记录、目录scope或断连闭合；因此收据保留N607`VERIFIED`同时CLI保守返回exit2。
- 实验索引618904项，没有`RUNNING/ACTIVE`记录或项目绑定进程证据；2395项为`OPEN_INCOMPLETE`、581398项为`ORPHAN_REVIEW`、35111项为`SCAN_ERROR`。这些状态是待人工整理的证据分类，不等同于可删除判定。
- 保留决策中851项为`KEEP_ACTIVE`，618432项为`REVIEW_REQUIRED`；所有619283项建议动作均为`REVIEW`。删除候选表只有表头，候选数与授权数均为0；本轮没有任何可执行删除范围。
- Git归属覆盖71个仓库，135419项为受跟踪Git资产、9453项为工作树未跟踪资产、97项为可再生忽略项、38108项为非Git证据、436206项为远端非Git资产，Git归属错误为0。
- 完整外置清单8项合计4203811483字节，Git摘要与progress/receipt合计11682字节。所有收据列出的字节数和SHA均由落盘文件逐字节复算一致，收据最后写入；该清单可以作为后续人工整理和方法优化的稳定基线。
- 删除、移动、覆盖、源资产变更、远端写入、任务停止和管理员访问全部为0。失败扫描证据、用户DOCX状态及`.docx_qa*`目录继续原样保留，未经用户明确批准不得清理。
