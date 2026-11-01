# Phase1 受限边界校正学生恢复网络

- run_id：`20261101-phase1-boundary-student-E30-s2026092701-r02`
- group_id：`phase1-bounded-boundary-student`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；七视图预测和独立复算已完成，实际状态见 events.jsonl 和结果记录。

## 目的与对照

冻结原 DURA 和 Tri-SIG-TIME SSL；仅在源训练中蒸馏特权输入梯度反事实，推理由受限学生从接收和 DURA IQ 预测校正。

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

## 重新发布与启动核验

固定源码 `f72f9618859fce30a1983a3b4b10c8c869b2dbba`，远端发布归档 SHA-256 `e56b85dee8c19ac06afc4499df305a759e822abdcaca744806d8f04e60364ae6`，真实权重检查在源数据上进行。CPU 合成中断续训逐位一致，独立复核未发现明确 P1；真实 GPU checkpoint 和续训尚待验证。

真实 F/DURA 的 CPU 2 条 L_s 源记录检查通过，冻结模型未变；源档案物化 L128/U384 IQ，不读 TX 标签、V 或测试 IQ。控制器遵循共享容量限制等待 GPU，正式训练尚未开始。

控制器 PID 24645；尚无本 run GPU worker。 CPU/GPU 小批次检查不是星地性能结果。

## GPU 检查与正式训练

GPU 4 worker PID 50451 独立核验，真实 F/DURA 的 128 条 L_s 源记录 GPU 检查通过；7 条教师判断不一致，特权训练目标到达 7 条 clean 教师类别。模型冻结，无 TX 标签、V 或测试 IQ；临时学生权重丢弃，正式 E30 从头初始化，首轮进行中。检查数字不是可部署学生准确率。

## 固定训练完成

已完成预先登记的 30 轮和 14790 次更新。独立使用真实冻结 Tri-SIG-TIME SSL 和 DURA 在两条源记录上重载最终学生，确认输出确定且满足 5% 校正约束；最终检查点 SHA-256 为 `951bd2dc49753db8b21af0cba6afcf0124ea05f43d59203957a0dd603b8eb0a7`。训练未读取 TX 标签、V 或测试 IQ。

七个 Phase1 视图的预测全部固定后才接入真值；所用测试此前已被观察，不属于独立盲测。

## Phase1 测试结果

七个视图的预测全部固定后接入真值，并独立复算每个视图、TX、RX 和 day 分组。每个视图均为同一组 168000 条物理记录。冻结 Tri-SIG-TIME SSL 对原始 clean 的准确率为 **86.5387%**；按相对下降不超过 3% 计算，各场景需达到约 **83.9425%**。

| 场景 | 学生恢复准确率 | 相对 clean 的下降（百分点） | 相对下降 | 同记录 DURA 准确率 |
| --- | ---: | ---: | ---: | ---: |
| 高信号 | 84.0083% | 2.5304 | 2.9240% | 84.3435% |
| 中信号 | 82.4006% | 4.1381 | 4.7818% | 83.0161% |
| 低信号郊区 | 80.4696% | 6.0690 | 7.0131% | 81.3220% |
| 高信号城市 | 82.5637% | 3.9750 | 4.5933% | 83.0494% |
| 中信号城市 | 74.6881% | 11.8506 | 13.6940% | 75.7060% |
| 低信号城市 | 67.5452% | 18.9935 | 21.9479% | 68.8012% |

只有高信号场景满足相对下降不超过 3%，其余五个场景未达到目标。六个受损场景的学生准确率均低于相同物理记录和冻结基座下的 DURA。clean 经学生恢复后的准确率为 86.5179%，相对原始 clean 下降 0.0208 个百分点，满足预设 clean 防护。

七视图原始基座决策与原 Tri-SIG-TIME SSL 预测逐条一致，168000 个物理 ID 对齐。独立复算见 [independent_rescore.txt](independent_rescore.txt)，与 DURA 的逐条修复和损伤见 [paired_dura_comparison.txt](paired_dura_comparison.txt)，基座决策核验见 [baseline_decisions.txt](baseline_decisions.txt)。预测完成记录、评分摘要和摘要哈希分别为 [prediction_complete.json](prediction_complete.json)、[test_summary.json](test_summary.json)、[sha256.json](sha256.json)。

这些结果使用了此前已观察的单随机种子 Phase1 开发测试集，不是独立盲测。第 30 轮是预先固定的检查点，测试结果未用于更新此模型。
