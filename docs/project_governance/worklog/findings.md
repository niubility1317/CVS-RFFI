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
