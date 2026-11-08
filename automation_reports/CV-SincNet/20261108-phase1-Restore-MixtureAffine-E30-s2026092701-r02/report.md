# 仿射混合恢复器：第 22 轮技术续训

- run_id：`20261108-phase1-Restore-MixtureAffine-E30-s2026092701-r02`
- group_id：`source-mixture-affine-scratch30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原任务触发资源保护，从最后完整 E22 恢复原模型、AdamW 和随机状态，继续至 E30 并完成原定地面测试。

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

## 技术恢复与固定测试

原 r01 在 GPU 0 总计算进程数短时达到 4 后触发上限保护，只停止本任务；E22/5434 步完整，原源域训练及预测产物保留，未开始目标预测。新版本严格恢复同一模型、AdamW、CPU/GPU 随机状态及源顺序，续训 E23–E30，累计 E30/7410。保留原 L_s/U_s、源混合先验、原冻结 Tri-SIG-TIME SSL 连续特征、损失、数据与种子。

GPU 6 是用户明确授权总进程最多 5 的单卡；本任务最多一个进程，入场最多 3 个其他计算进程且显存空闲≥12288 MiB，运行中总数超过 5 或三次无法核实则仅停止本任务并保留产物。启动先进行实际源 2 样本/F0 与 E22 恢复检查；固定最终 E30 后自动预测原 Main 两总体全部七视图，全部固定后独立 truth-last。

11 项定向测试与一次独立 P0/P1 审核通过；一次 N607 预检、发布归档校验和远端编译通过。原运行及未参与本任务的进程均未被修改。

## 运行读回

VERIFIED：控制器 PID 1367768、训练 PID 1367915，实际执行 release、工作目录和原 F0 均已核对。GPU 6 入场时有 3 个其他计算进程、空闲 18531 MiB；训练启动后共 4 个。真实源样本与原 F0 检查通过；父 r01 E22/5434 的模型、AdamW 和随机状态已恢复。结果跟踪 PID 83980 已启动；训练完成后继续原定测试及独立评分。当前尚无测试结果。

## 独立评分恢复

原评分因缺少评分模块在连接 truth 前失败；固定预测已在独立 run `20261108-evaluation-Restore-ScoreRepairAffine-s2026092701-r01` 使用原评分器完成 truth-last 评分与完整配对复核。原失败状态和产物保持；逐场景结果见该 run 的 analysis.md。
