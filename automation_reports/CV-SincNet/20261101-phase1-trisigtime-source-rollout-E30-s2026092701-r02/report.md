# Tri-SIG-TIME SSL 引导的生成轨迹星地恢复

- run_id：`20261101-phase1-trisigtime-source-rollout-E30-s2026092701-r02`
- group_id：`phase1-trisigtime-flow`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

只用 Phase1 源域训练数据，从接收 IQ 出发完整训练两条四步恢复轨迹，冻结 Tri-SIG-TIME SSL 并对 Phase1 测试集进行七视图预测。

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

## 已核实状态（2026-11-01）

代码版本 `42eede565ba414513f6b9b7f5a2193e0c4fcb9d0` 已发布至 N607，归档 SHA-256 两端一致。实际控制器 PID 2176487；单进程 GPU worker PID 2176595，实际位于 GPU 1，与另一同账号进程共享，启动时空闲显存 20275 MB。PID、工作目录、命令及 GPU 归属均已独立读回。

真实基座和两条源训练记录的 smoke 为 PASS：恢复 IQ 和冻结基座特征指导均产生非零有限梯度，基座保持冻结，checkpoint 严格重载通过，未读取 TX 标签和测试 IQ。源数据 L_s／U_s 物理 ID 与 Tri-SIG-TIME SSL 源契约一致。

已完成第 1／30 轮、493 次更新，训练日志逐轮增长。完整训练、七视图预测与独立评分尚未完成，当前没有新恢复网络性能结果。

## 训练完成与冻结检查

Phase1 源训练已完成 E30／14790 次更新，最终记录显示 Tri-SIG-TIME 基座始终冻结、训练阶段未读 TX 标签或测试 IQ；生成网络 160200 个参数。最终 checkpoint 的 SHA-256 为 `f54bc1a64db40f24dc63d096ba004c7049fc8a25d9549312c0cead132941136b`。

独立 CPU 进程用相同源码严格重载该 checkpoint 和原基座，在两条 Phase1 源训练记录上验证了输出与分类分数有限、重复推理逐项一致、生成网络确实改变输入，且两网络保持冻结。此检查不是测试性能评估。原 GPU worker 已开始依预登记产生七视图预测，尚无完整预测或评分。

## Phase1 七视图独立评分结果

状态：**ANALYZED，目标未达成**。模型固定 E30，Phase1 ground test 每视图 168000 条，原始输入与恢复输入均使用同一冻结 Tri-SIG-TIME SSL 基座。原始 clean 准确率 86.5387%。以下差距为原始 clean 准确率减对应恢复准确率，单位为百分点。

| 视图 | 原始准确率 | 恢复准确率 | 对原始 clean 的差距 |
|---|---:|---:|---:|
| clean | 86.5387% | 86.2345% | 0.3042 |
| practical_high | 80.6155% | 75.5155% | 11.0232 |
| practical_mid | 78.1077% | 74.0077% | 12.5310 |
| practical_low_suburban | 75.7446% | 72.5411% | 13.9976 |
| practical_high_urban | 77.0500% | 72.1893% | 14.3494 |
| practical_mid_urban | 62.7958% | 60.5143% | 26.0244 |
| practical_low_urban | 51.6048% | 51.0363% | 35.5024 |

六个星地视图均未达到距 clean 不超过 3 个百分点，且恢复准确率均低于各自未恢复输入；clean 损失 0.3042 个百分点，满足预登记的 1 个百分点 clean 保护线。最困难视图相对 clean 的降幅为 41.0249%，也未满足相对降幅不超过 3% 的更严格口径。

独立 CPU 进程在核验全部十四份完整预测后，按物理 ID 连接真值，复算全部总体、TX、RX、day 统计和 Macro-F1，与原评分一致。原始七视图的所有 168000 条预测还与原 Phase1 Tri-SIG-TIME SSL 预测按物理 ID 逐项对照，七个视图均零差异。仅一个模型 seed；固定测试属于已知开发测试集，非独立盲测。

该结果说明本候选没有证据表明显著提升识别，不能把源训练损失下降当作星地达标。两个基于源训练诊断、在本结果出现前已固定的后续候选继续按原计划；同基座 DURA 对照单独运行。
