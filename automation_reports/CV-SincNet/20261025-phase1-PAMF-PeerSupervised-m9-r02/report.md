# PAMF 源域共识与监督机制三组九模型

- run_id：`20261025-phase1-PAMF-PeerSupervised-m9-r02`
- group_id：`pamf-peer-supervised-source-consensus`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

F0 三方案×三学生种子：冻结其他两seed同契约E20确认自身.95硬标签，U权重2及完整PAMF机制，并有普通CE/权重1消融。保留现有健康训练，旧21扩展暂缓，全部显卡容量允许时优先本轮。

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

## 本轮固定方案与源域依据

本轮 3 方案×3 学生种子（2026092701/02/03），共 9 个学生。主方案 `PeerT1U2` 使用学生自身 raw 置信度≥0.95 的标签，且同契约另外两 seed 的冻结 E20 均同意该标签且置信度≥0.95。门控不替换学生标签。不使用 As 置信度筛选。两个消融分别保留共识但只用普通 U CE（`PeerPL2`），以及保留完整 U 机制但把 U 权重从 2 降至 1（`PeerT1U1`）。三个方案和顺序在本轮 target 访问前固定，全部报告，不按 target 结果更换 checkpoint。

完整 U 机制复用原 PAMF 六前向：ordinary 两次、ordinary/As 混合两次、raw 两次；ordinary/raw KL=1，As 独立 KL=0；仅 accepted 伪标签参与浅层 `time_down` 和中层 `t2.pool` 的跨 RX 对比。CE 总剂量按 1.25 归一，U 总权重 2 或 1；物理半组样本的 As CE 占比 40%。L 机制、骨干、默认联合分类头、优化器、分段日程均保持。无 EMA、风格混合、外推、梯度保护、另训头或 G 训练。

已完成的源域 CPU 诊断为 `20261025-phase1-PAMF-PL-consensus-source-m6-r01`。F0 冻结 E20 三方共识精度 99.971406%、覆盖 90.670370%；label01 为 99.778176%、86.822222%，错误 52 条，单模型错误为 139～207 条。该诊断不更新参数/缓冲区、不打开 U 真值或 target；它是来源和机制依据，不是本轮学生的真实 U 正确率或 target 性能。本轮训练中学生会更新、peer 始终为 E20，实际接受率与曝光量逐步记录。

每学生仍为自身同契约 ordinaryPL E20＋独立 E10＋E20，累计 E50/11100 次更新、13L＋37LU；前 7LU 已为普通 PL，后 30LU 为本轮机制。两名 peer 必须在 native trainer 作用域和学生 RNG 重置前加载，冻结 eval/no_grad，不加入学生模块或优化器。最终推理仅一个学生默认头。单学生训练来源包含自身 50＋其他 2×20；三个种子共享 3 份 E20，单方案唯一训练为 150 轮；本轮 3 方案共享前缀后为 60＋270=330 轮，新增训练 270 轮，另外报告 peer 前向成本。

## 全显卡与队列优先级

用户已授权全部 GPU0～7。0/3/4 对应主方案和普通 CE 消融的 seed01/02/03；权重 1 消融在 1/2/7 对应 seed01/02/03。每卡本任务最多 2 条，入场时总 CUDA/pending 也限制 2 条且空闲≥10000 MiB，保留其他 owner。GPU1/2 原任务继续，GPU5/6/7 现有 6 条 ChannelPL 健康训练及其自动预测/评分继续。

发布的第一步为一次真实 E20 的 L/U 有限反传、默认头和冻结 peer 无 query smoke，零优化器更新。PASS 后核实旧 controller 精确 PID/CWD/完整参数，仅停止该 controller 单 PID，保持 detached 子进程。新 owner 从原 99 个任务及精确进程信息接管，复用已完成结果；21 个未开始训练及其依赖暂缓，待新 9 个学生全部训练、预测、评分完成后原配置续排。科学 release、种子、模型参数、预算和输出路径不变。实际接管数量以独立读回为准。

## 固定评测与目标

每学生固定 final E50 后立即预测原 cross_date、cross_rx、cross_both 三个人群；各 clean＋六种 practical，raw PAMF 与同一份同契约 G-LU＋PAMF。该 row 所有人群/视图/路线预测完成后，独立 CPU scorer 才连接 truth。监督版、普通 PL、REPL 分数从已有合法完成产物按实际 job 路径复用，不猜路径或挑结果。

原目标为 F0 168000 个目标 RX 物理样本（cross_rx126000＋cross_both42000）的 raw PAMF 三学生 seed 均值：超过监督版和普通 PL 中较高者，clean 至少 1 个百分点、六景均值严格超过 1.5 个百分点。G-LU 为必报参考，单独列增益。另保留三个独立人群及 RX/day/TX/逐景细节。该基准已多次公开，保留自适应研发背景，不声明独立盲测或把三个 seed SD 当作显著性检验。

本地相关协议负测和训练算术/默认头/冻结教师/精确接管身份检查已通过。r01 在 prepare 发现缺少 `reference_score_jobs`，尚未启动 smoke、GPU worker 或旧 controller 接管，保留 NO_PERFORMANCE_RESULT。r02 补齐已核实的最新历史 score jobs 句柄，并在本地发布前和 prepare 入口检查实际消费的输入。新增的缺失字段回归检查通过，科学配方、9 行与预算不变。尚无本轮新学生 target 结果，目标尚未达成。

缺失句柄的定点独立 P0/P1 复审 PASS。原 45 项相关检查加 1 项缺失句柄回归通过，共 46 项已验证检查；未增加第二次全审或额外科学门槛。

## 调度器技术状态与恢复

真实同契约 E20 的 L/U 有限反传、默认头、冻结 peer 检查 PASS；零优化器更新、零 query 读取。27 个科学任务已固定，尚未启动 GPU worker。随后旧 controller 精确单 PID 退出时读取 environ 遇到 PermissionError，新 controller 退出；旧 6 个 detached 健康训练继续。修复只调整退出存活检查并由独立 control-r03 接管，复用本 run 原配置、原 release、原输出和 PASS smoke，不重复训练或重新选模。证据见 [handoff_failure.json](handoff_failure.json)。

## 启动独立读回：VERIFIED／RUNNING

新 control 的 PID/CWD/完整参数和实际 GPU 已独立核实，archive 单次传输校验、远端编译 PASS。恢复已接管 6 个原健康 worker，未停止训练子进程；原 21 个未启动扩展已暂缓。科学配置和预算零改动。实际 GPU0～7 均在使用，观察到每卡总 CUDA 进程最多 2 个。

新矩阵 9 个学生中 8 个已启动且均完成首个新增 epoch（累计 E21/4662）；剩余 PeerT1U1 seed2026092703 等待 GPU7 空位。两次独立读回之间，全部 8 个新学生日志均增长。已读取当前段所有可用完整结构化记录共 3564 条；该快照不等于全程结束分析。每行首个新增 epoch 共 222 步，U_seen=56700、peer_eval_views=113400；full U 方案 student_views=340200、physical_U=28350。实际 U 权重、L 每步 768 前向包、As 独立 KL0、teacher/EMA 零更新、零非有限跳步均与登记一致。主方案实际 U 接受率约 90%，不是 U 标签正确率。源域执行明细见科学 run 的 early_source_execution.csv。

原训练自动测试已继续：完成的旧行直接进入预测队列，不等待整批。新学生也将在固定 E50 后自动完成 raw PAMF 与同一 G-LU＋PAMF、三人群、七视图的预测和独立 truth-last。当前没有本轮新学生 target 指标，性能目标尚未证明。证据见 [launch_evidence.json](launch_evidence.json)。

## 新九学生全部启动

原等待 GPU7 的 PeerT1U1 seed2026092703 已启动，已独立观察到全部 9 个新学生的精确进程身份和实际 GPU：每组均为三个种子。原科学版本、参数和预算不变。目前是源训练进度，没有新学生 target 性能结论。证据见 [all_nine_running.json](all_nine_running.json)。

## 九学生持续执行快照

已读取全部 9 行 segment30 当前可用完整结构化记录，共 18456 条；三条 PeerPL2 已闭合 E30/6660 并健康进入原 E50 段，其余完整机制行继续 E28～E29。U 权重、peer 两次冻结前向、L/U 实际曝光、As KL0、零 EMA/peer 更新、零非有限跳步保持。该快照只覆盖 segment30，不代表 E50 或 target 完成。early_source_execution.csv 已扩展为全部 9 行。另有独立 label01 三学生使用空位运行，原九学生的科学 release 和配方不变。

## 九行 E30 闭合与实际 E50 续训

九行均已闭合自身 E20+10；读取全部 9×2220=19980 条结构化记录，检查实际 loss、U 曝光与接受计数、固定权重、冻结 peer 更新数、L768 包、As KL0 和非有限跳步，并与每段 peer_supervised_state.json 逐行对应。每行 U_seen=567000、peer_eval_views=1134000。未发现 EMA/peer 更新、U 真值、V 训练、G 训练或非有限跳步。

已读取各自 segment50/initial_smoke.pt 中实际原生初始化配置，确认 parent_source 指向同 seed、同方法自身 E30，offset30/6660、再训练20轮，原两名 E20 peer、源合同和固定机制不变。原默认头保持6×160，九个原生源域无 query smoke 均 PASS。该核对只分析已有源域产物，没有新优化器更新、target 读取或选择 E30 作为最终 checkpoint；仍固定 final E50 后自动完整测试。明细见 [source_E30_and_resume.csv](source_E30_and_resume.csv)，证据见 [all_nine_E30_resume_evidence.json](all_nine_E30_resume_evidence.json)。
