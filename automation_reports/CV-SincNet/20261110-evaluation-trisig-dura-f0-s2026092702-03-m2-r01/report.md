# Tri-SIG-TIME SSL 与 DURA v1.0 的三种子匹配基线补测

- run_id：`20261110-evaluation-trisig-dura-f0-s2026092702-03-m2-r01`
- group_id：`trisig-dura-main-matched-baseline`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用种子01，冻结原三视图模型补测种子02/03，提供完整SSL与DURA七景配对基线。

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

## 本次范围与核验

用户将 DURA 七景列为主要测试。复用既有 2026092701 的默认 DURA＋Tri-SIG-TIME 七景结果；对原同契约固定 E50 三视图模型 2026092702/03，各自只重用既有 raw 决策并读取同一 DURA v1.0 恢复缓存，不重新训练。主总体为 cross_rx 126000 与 cross_both 42000，先各自统计后合并；每种子均为同一 168000 个物理包。无弱信道测试。全部两种子七景预测固定、双方原始 ID 与缓存 ID 匹配、完整预检后才由独立 CPU scorer 读取真值。逐 seed／场景／TX／RX／日期及同包纠错与错失用于后续与同包候选比较。现有 DURA-L 为不同历史恢复器，只作为其他 owner 实验的辅助读出，不等同本次默认 DURA v1.0。

原普通 PL 和本次完整方法使用不同时间顺序和训练机制，比较只支持整体方法差异；原基准已曝光，不称独立盲测。补测针对两份缺失模型，种子 01 保持原报告身份。局部负例和物理 ID 顺序、评分屏障的 4 项检查通过，Python 编译通过；一次独立 P0/P1 审查无阻断发现。真实 checkpoint 无查询 smoke 为调度器首项，必须 PASS 才预测；当前没有新增预测或评分。

2026-11-10：状态 VERIFIED／QUEUED。执行代码 `7fa5dae67a78b4a0e8f454ea5b2314b9e94f4906` 已推送并独立对照远端 OID；单发布包传输一致、远端编译通过。已独立读取 controller PID、CWD、命令及 `WAITING_FOR_SMOKE` 状态；GPU 0、1 各有两条其他任务，调度器等待空位。真实 smoke、预测和评分均尚未开始，不把排队当作实验完成。
