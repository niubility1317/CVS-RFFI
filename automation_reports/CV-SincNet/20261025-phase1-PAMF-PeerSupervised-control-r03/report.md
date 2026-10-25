# PAMF 九模型健康进程接管与退出竞态修复

- run_id：`20261025-phase1-PAMF-PeerSupervised-control-r03`
- group_id：`pamf-peer-supervised-source-consensus`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅替换调度器，复用r02原27科学任务/配置/release/output/预算和PASS smoke；接管旧6健康worker，暂缓21，所有新旧模型均不重复训练。

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

## 唯一修复与复用边界

原科学 run 为 `20261025-phase1-PAMF-PeerSupervised-m9-r02`，科学执行 commit 为 `7eb8d21752b5fab6f1366e1cb96c2266bd46c2a7`，9 个学生/27 个阶段全部沿用原配置、原路径、原 release 和原 E50 预算。本 control 没有新增科学模型，修复后的代码只由调度器消费；模型训练/预测仍执行原科学 release。原真实 E20 的 L/U 有限反传 smoke 已 PASS，零优化器更新、零 query 读取，直接复用。

原 controller 在已验证身份的旧 ChannelPL controller 单 PID 退出时读取 environ，出现 Linux 退出瞬间 PermissionError。该旧 controller 确已停止，6 个 detached worker 仍健康训练，未停止或重启。退出等待与 adopted worker 轮询改用 proc stat；身份读取异常只有在独立确认已死时才视为退出，仍存活但不可核实的进程不接管。中断接管只接受原 r02 的匹配 PID/code/owner 快照与已死状态，不再发送第二次终止信号。

4 项直接回归通过：正常与中断接管均保留全部 99 个旧科学任务及 6 个健康 worker；退出环境读取异常采用独立存活证据；原 27 个新科学配置和 PASS smoke 原样复用。该缺陷的独立定点 P0/P1 复审 PASS。

GPU0～7、每卡本任务及总 CUDA/pending 均≤2、空闲≥10000 MiB、全局单 scorer 保持。用户授权暂缓的 21 条未启动旧扩展，待原 9 学生全部训练/预测/评分完成后按原配置续排；旧健康训练及其自动测试立即继续。三方案/三种子、raw 与固定同契约 G-LU、三人群七景、独立 truth-last 均不变。目标仍须由真实完整测试证明。

## 启动独立读回：VERIFIED／RUNNING

新 control 的 PID/CWD/完整参数和实际 GPU 已独立核实，archive 单次传输校验、远端编译 PASS。恢复已接管 6 个原健康 worker，未停止训练子进程；原 21 个未启动扩展已暂缓。科学配置和预算零改动。实际 GPU0～7 均在使用，观察到每卡总 CUDA 进程最多 2 个。

新矩阵 9 个学生中 8 个已启动且均完成首个新增 epoch（累计 E21/4662）；剩余 PeerT1U1 seed2026092703 等待 GPU7 空位。两次独立读回之间，全部 8 个新学生日志均增长。已读取当前段所有可用完整结构化记录共 3564 条；该快照不等于全程结束分析。每行首个新增 epoch 共 222 步，U_seen=56700、peer_eval_views=113400；full U 方案 student_views=340200、physical_U=28350。实际 U 权重、L 每步 768 前向包、As 独立 KL0、teacher/EMA 零更新、零非有限跳步均与登记一致。主方案实际 U 接受率约 90%，不是 U 标签正确率。源域执行明细见科学 run 的 early_source_execution.csv。

原训练自动测试已继续：完成的旧行直接进入预测队列，不等待整批。新学生也将在固定 E50 后自动完成 raw PAMF 与同一 G-LU＋PAMF、三人群、七视图的预测和独立 truth-last。当前没有本轮新学生 target 指标，性能目标尚未证明。证据见 [launch_evidence.json](launch_evidence.json)。
