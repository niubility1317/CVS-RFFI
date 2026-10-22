# T1＋DURA 按负载均衡，GPU0/6 试增至四进程

- run_id：`20261022-phase1-T1-DURA-multiprocess-control-r03`
- group_id：`phase1-cric-dura-complete-system`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Resource-only handover of existing769 jobs. Baseline3/card; GPU0/6 fourth F only when admitted; global GPU capacity26, preserve all healthy scientific workers.

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

## 已确认的资源方案与停止规则

本次只更换资源调度器，原 769 个执行任务、数据句柄、三种子预算、源域冻结、七视图预测与独立 truth-last 全部保持。科学 worker 始终使用原 3e880690 release、完整命令和配置；现有健康 PID 不迁移、不重启，完成行直接复用。

基础容量仍为每卡 3 条。仅 GPU0、GPU6 在至少 5 个观测点覆盖 120 秒且平均 GPU 利用率低于 85%、三条现有任务全为本 run F 训练、无外部进程时可补第 4 条 F；全机计算进程总量最多 26。按滚动 GPU 利用率从低到高安排新任务，其他卡不得加第 4 条，G/G_student 每卡最多 1 条，正式时延独占且排除 GPU2/3。

准入要求 CPU<75%、可用 RAM≥64 GiB、IOwait<5%，且启动前空闲显存≥12000 MiB。逐卡按 F=4096、G=10240、预测=2048 MiB 预留峰值，另留 3072 MiB 余量；不能按当前 G 的低显存占用忽略启动和后续阶段。遥测失败只暂停新 GPU 派发，健康任务继续。

在第 4 条启动前记录原三条 F 的同阶段轮次耗时；每条至少返回三轮同阶段新记录后，以 (4/3)/耗时比中位数作为吞吐代理。代理≤1 时不再补该卡第 4 条，让进程自然回落。这是工程保护代理，不是严格受控的因果加速证明；阶段改变或样本不足保持待测，不读取目标评分。

34 秒基线采样：CPU53.4%，可用RAM358.1 GiB，IOwait0.036%；GPU0/6平均利用率77.6%/81.8%。F显存3.28至3.63 GiB、约1逻辑核、RSS1.80至1.97 GiB；当前G阶段显存0.93至1.80 GiB、RSS1.99至6.60 GiB，但此前启动峰值约9.22 GiB。预测显存约0.86 GiB但计算负载不轻，不能把它当成轻计算任务。

新增 11 项资源负测与既有 9 项所属进程保护测试已通过，共 20 项。独立审查发现的 P1（未进 CUDA 的自有 PID 与外部 CUDA PID 不能用数量 max 计数）已用失败回归复现并修复：使用实际 PID 并集，额外的实时容量变化保守计入。普通本地 Python 没有 pytest，验证使用标准库 unittest；不因此安装依赖或运行无关科学测试。科学输入与 worker 未变，复用原实际 checkpoint 无 query smoke PASS 和 VALIDATED_ONCE，不新增数据验证或 smoke 许可层。发布前仅对该 P1 定点复审。

发布入口：experiments.cric_dura_comparison.publish_load_balance，参数指向隔离 Git 检出、原 r02 runtime 和本次独占 local_artifacts/cric_dura_load_balance_20261022。一次 release 归档只校验一次本地/远端 SHA，远端编译一次；发布回执不代表接管成功，随后独立核实 owner、原健康 worker、GPU 并发和日志。
