# PAMF 共识监督机制在1%标注下的三种子验证

- run_id：`20261025-phase1-PAMF-PeerSupervised-label01-m3-r01`
- group_id：`pamf-peer-supervised-source-consensus`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

源域证据预选label01：复用同契约ordinaryPL E20，每学生own10+20，固定PeerT1U2主配方；3学生/3GPU独立队列，原9F0健康训练与参数不变。

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

## 固定科学配方与选择依据

追加 label01 三学生，是对原 11 设置中的低标注契约进行同机制验证，依据为已完成、无状态更新的源域诊断：冻结三 seed E20 共识的 source V 伪标签精度 99.778176%、覆盖 86.822222%，错误 52 条，单模型错误为 139～207 条。该诊断不是 U 真值，也不是新学生 target 性能。当前 9 个 F0 学生尚无 target 结果，本扩展不依据其结果改参数、选 checkpoint 或重排。

数据为 label01 原合同：L_s=630、U_s=62370、V=27000，L/(L+U)=1%；RX/day/TX、表示和物理角色保持。同契约自身 ordinaryPL E20 与另两 seed 的 E20 在加载前核实完整来源和实际角色。禁止借用 F0 E20、peer 或 G-LU。每学生依原 20+10+20 到 E50/11100，前 13L+7ordinaryLU、后 30peerLU，默认头随骨干联合训练。总成本同时披露其他两 peer 的 E20；三个学生共享 3×20 前缀，加 3×30 新训练，唯一训练来源为 150 轮、新增 90 轮。

完整原 PeerT1U2 配方固定：own raw .95，另两 frozen E20 同标签且均 .95；只筛选、不替换 own 标签；U 总权重 2，六次 U 前向、CE 归一 1.25、ordinary/raw KL1、As 双 CE/独立 KL0、accepted 浅中层跨 RX 对比。无 EMA、风格混合、另训头或 G 训练。L 目标与原预算保持，不因 label01 调超参数。

## 运行与评测

独立不可覆盖 owner 使用 GPU5/6/7，各卡本 run 最多 1 条，所有 CUDA/pending 总数≤2，启动空闲≥10000 MiB。与现有健康 worker 共享容量，不调用 handoff 或任何终止信号。原 F0 九学生的 release、参数和预算保持。第一步执行一次真实 label01 E20 的 source L/U 有限反传、原默认头与冻结 peer 无 query smoke，零优化器更新；PASS 后立即按固定矩阵继续。

每行 final E50 冻结后自动预测原 cross_date/cross_rx/cross_both、clean+六种 practical、raw PAMF+同一 label01 G-LU＋PAMF。该行所有预测固定后独立 CPU truth-last。监督版/普通PL/REPL 的匹配历史分数按实际完成 job 路径复用。全部 3 学生均报告，不依 target 选择性重跑。

本设置为 secondary，结果无论涨幅多少，均不能替代原 F0 raw168000 的 clean≥1pp、六景>1.5pp 目标。27 项本地相关检查通过，包含同契约 peer、数据句柄隔离及“secondary 大增益不能误报原 F0 目标”的检查。新增协议映射与独立入口的一次 P0/P1 审查 PASS。

## 实际启动：VERIFIED／RUNNING

真实 label01 E20 的 L/U 有限反传、原默认头、冻结 peer 检查 PASS；L 前向包 768、U_seen256、peer_eval512、full U student1536，零优化器更新和 query 读取，peer 参数/缓冲区保持。随后三个学生均已在 GPU5/6/7 启动，controller 和各 worker 的精确身份、实际 GPU、两次独立读回间日志增长均已核实。观察到每卡总 CUDA 进程不超过 2 个，原九学生也继续运行，合计 12 个新学生。未调用终止或接管，无 G 训练。单模型和同契约 G-LU、三人群七景在固定 E50 后自动测试；本轮尚无 target 结果，原 F0 目标仍未证明。证据见 [launch_evidence.json](launch_evidence.json)。
