# CVS 频点内时间关系：源域选中四 seed clean 与 44 个冻结控制

- run_id：`20261003-phase1-cvs-spectral-relation-clean-manysig-m48-r01`
- group_id：`cvs-clean-spectral-relation-confirmation`；类别：`cvs`；阶段：`Phase1-spectral-relation-clean-confirmation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

只测试源规则选中的新候选；4 个新 clean 预测与原 44 个预测统一独立评分；不反馈调参。

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

## 已核实的启动依据

源训练8×E200全部结束，80000步与全V标量已独立复算。固定16源记录选中relation_frequency_energy；本轮仅4份选中模型生成clean预测，44份既有预测保持原路径。完整48份预测固定后由独立scorer连接truth，生成384条ALL/RX评分。

173项条件clean测试及唯一独立P0/P1审查通过。实际checkpoint来源、FP32参数和固定Hann窗在读取query前逐模型核验。仅clean，无LEO、support适应、额外训练或候选重排。

[冻结来源](evidence/source_freeze_reference.json) · [独立审查](evidence/conditional_clean_independent_review.json)。实际发布版本、PID和GPU在启动读回后补录。
