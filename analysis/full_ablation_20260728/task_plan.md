# CVS-RFFI全量消融目标计划

## 目标

完成`CVS_FULL_ABLATION_DESIGN_PHASE1_PHASE2_20260728.md`定义的分层消融闭环：先满足T0实现与协议门槛，再依次完成T1主消融、满足晋级条件的T2内部消融、T3敏感性/生命周期/资源实验和T4非晋级诊断；所有N607运行均使用不可覆盖run ID、完整日志、同row指标、独立scorer和可追溯Git版本。

## 完成判据

- 设计报告中的每个机制族、arm和公共证据要求均有`verified/deferred/rejected/blocked`之一，且非`verified`项有明确科学或执行原因。
- T0聚焦测试、协议负测试、真实checkpoint无query smoke、完整矩阵dry-run和独立审查均达到`P0=0,P1=0`。
- Phase1至少完成报告规定的30次第一层训练；Phase2主消融先完成75-row screening，满足预登记晋级条件的核心arm完成900-row fresh confirmation。
- 8张GPU按每卡最多2个训练进程调度；任何时刻不超过上限，且不干预非本run任务。
- 每个正式run保存设计报告第11节要求的manifest、预测、score、资源、退出和同row指标artifact。
- 最终报告区分技术完成、诊断结果、正式性能和不可晋级证据，不使用历史/Oracle/partial结果替代fresh confirmation。

## 当前阶段

`PHASE_2_PHASE1_T1_IMPLEMENTATION / IN_PROGRESS`

## 阶段

- [x]0.读取控制面、设计、历史索引、Git状态并完成实现就绪审计。
- [x]1.冻结首批T1范围、fresh seed/draw注册表、paired manifest和8×2调度矩阵。
- [ ]2.补齐并验证Phase1统一arm factory、参数匹配、内部开关和指标/资源artifact。
- [ ]3.补齐并验证Phase2 arm factory、fallback闭合、same-row scorer、连续状态和资源路径。
- [ ]4.在`ssr-gpu`串行执行聚焦测试、协议负测试、真实checkpoint无query smoke和完整矩阵dry-run。
- [ ]5.完成独立审查`P0=0,P1=0`、Git提交和N607报告预登记。
- [ ]6.由唯一实验runner执行N607预检、占用审计、精确同步、远端校验和分波次发布。
- [ ]7.持续完成启动/首row/首worker wave/终局健康检查并回收完整artifact。
- [ ]8.完成T1分析与晋级判定，再执行符合条件的T2/T3/T4。
- [ ]9.完成paired统计、论文表图数据、反向追踪审计和最终报告。

## 运行安全边界

- N607普通账号优先；不使用`N607-admin`。
- 发布前不连接服务器做写操作；本地实现、验证、提交和报告预登记必须先完成。
- 每个GPU最多2个训练进程；已有任务计入上限。
- 性能差不是停止理由；只按预登记的协议/执行/零预测/重复异常指纹规则停止精确run进程树。
- 不覆盖或删除任何现有dataset、checkpoint、log、metrics、report或run输出。

## 错误记录

|时间|错误|处理|
|---|---|---|
|2026-07-28|创建goal时提示已有未完成goal|读取后确认现有goal与本次请求完全一致，直接接续|
|2026-07-28|`session-catchup.py`无输出且状态1|直接读取现有根目录计划文件并以Git内独立计划承载本目标|
|2026-07-28|广扫历史Git worktree超时|停止全量扫描，只检查发布仓库和当前候选worktree|
|2026-07-28|非初始化PowerShell中`conda activate ssr-gpu`失败|串行使用`conda run`；后续直接使用登记环境Python规避中文输出编码问题|
|2026-07-28|Conda包装搜索因GBK编码失败|定位`ssr-gpu`实际路径并直接调用环境Python，索引搜索成功|
|2026-07-28|PowerShell在子表达式内组合Git退出码时语法错误|改为逐文件执行后读取`LASTEXITCODE`，检查成功且未改变仓库|
|2026-07-28|首次聚焦pytest导入`code.cvsrffi`失败|改用`cvsrffi`并显式把仓库`code`目录加入`PYTHONPATH`，7项通过|
|2026-07-28|fresh seed历史精确搜索第一次覆盖面过宽而超时|缩小到Git跟踪面、自动化报告控制文件和项目对话索引，三处均无精确值命中|
|2026-07-28|A0首次参数匹配多出48个参数|定位为`lite_d`双分支共享Sinc/HF stem被重复计数；改为只补偿相对完整模型真正移除的唯一参数|

## 重大决策

|决策|依据|
|---|---|
|不立即在N607启动“全量”矩阵|设计报告第9.1和第13节明确T0门槛及缺失实现项|
|设计提交`790da982`作为本目标源文档|它是用户桌面设计报告的Git镜像，已提交且内容一致|
|历史D92结果只作回归/诊断线索|不是本次fresh confirmation，且既有正式结论为未晋级|
|先以T1主消融形成首批发布|设计规定T2仅在T1整体作用稳定后执行，避免事后扫参挽救主张|
|`P2-F3`与`P2-FULL`逻辑保留、物理去重|二者设计状态完全相同；重复执行不能增加因果证据，只会制造伪独立样本|
