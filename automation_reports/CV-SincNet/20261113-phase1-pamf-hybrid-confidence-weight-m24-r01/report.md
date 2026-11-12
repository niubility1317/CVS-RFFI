# Hybrid 全置信样本辅助权重对照

- run_id：`20261113-phase1-pamf-hybrid-confidence-weight-m24-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

四种配置、三种子比较全置信 Hybrid 辅助权重 0.25 与 0.5，并配对既有全权重。

覆盖接收机划分 2、10 类、1% 标签、6 个源接收机；每个配置各 3 个同配置同种子原 Tri E20，追加 10＋20 轮，共 24 行。原 Hybrid、普通 PL 以及正在运行的全权重置信度 Hybrid 均按相同数据和种子复用，不重训。

全部置信度至少 0.95 的源 U 样本都进入 Hybrid 辅助路径；仅将新增 fusion／branch 辅助损失整体乘以 0.25 或 0.5，不使用分支投票或原型。若无可用不同伪类别，则保留同包余弦关系。原物理强度 taper 和其他损失不变；由于全部置信通过样本都进入物理视图，原混合 U 路径也会受到影响。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。固定完成每行训练、clean 和六场景预测及独立 truth-last 评分；同种子按样本数合并跨接收机与跨接收机及日期，再按三个种子汇总。测试集已经曝光，比较是探索性的，结果不反馈到训练矩阵。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
