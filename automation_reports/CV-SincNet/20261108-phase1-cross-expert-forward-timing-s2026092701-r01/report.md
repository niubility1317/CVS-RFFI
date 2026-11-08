# 大型时频网络源域 GPU 前向分段计时

- run_id：`20261108-phase1-cross-expert-forward-timing-s2026092701-r01`
- group_id：`phase1-cross-expert-forward-timing`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

以非零输出头检查完整模型前向字节一致性，定位输入处理、FFT、时域与频域网络和 IFFT 的 GPU 同步耗时。

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

## 前向阶段计时

执行代码 commit `873023c03f780f5afcc2360a9c35ad3a25ccadf5`。GPU2 启动前空闲 18,501 MiB，另有两个其他 owner 计算进程；队列 PID 1283935、GPU worker PID 1283946、实际工作目录及真实 DURA 源样本 smoke 已核实。四项相关检查、独立 P0/P1 审查及其修复定点复核通过；发布包两端一致并在远端编译通过。

固定 128 条合法源记录，采用确定性 FP32、关闭 TF32 和固定种子。为使两个分支的输出参与字节级核验，独立诊断模型的两个输出头均以标准差 0.001 的固定正态随机值初始化；这不是在训 checkpoint。热身后各计两次输入、FFT、时域分支、频域分支及 IFFT/合成，每次与原 forward 完整输出按字节比较。计时阶段强制同步，GPU 与其他任务共享；不读取 F/TX/V/目标，不更新或保存模型。结果仅用于定位短程前向耗时，不能当作全训练加速或最终识别性能。

## 计时结果

VERIFIED / ANALYZED：两次完整输出逐字节匹配，时域分支约 46.17 / 46.22 秒、频域约 21.69 / 22.62 秒；FFT 与 IFFT 均为毫秒级。GPU 共享和阶段同步限制结论，详见 [analysis.md](analysis.md)。
