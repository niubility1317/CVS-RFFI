# PAMF 仅置信度筛选的分支辅助实验

- run_id：`20261113-phase1-pamf-confidence-only-m9-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

三个完整源数据配置（10 类、1% 标签、6 个源接收机）、三个种子；全部置信度至少 0.95 的 U 参与辅助训练，不使用分支投票。原普通 PL 与原 4/5 分支原型对照直接复用，已完成对照不重训。

辅助分支覆盖原 U 批次的全部置信度通过样本，包括原物理半批之外的样本；因此原混合 U 路径也会接收这些新增物理视图。延续原物理强度 taper；如果原型正类缺少跨日期有标签参考，以该包原始视图的 detached 特征作为正样本；如果没有可用负类，则用同包余弦项，保证所有通过样本的原型损失有定义。这些回退都计数报告，不使用 U 真值。

## 数据、seed与模型来源

严格沿用每种设置的完整源 L/U/V 角色，不留出开发集；每行从相同设置和种子的原 Tri E20 独立续训 10＋20 轮。完整数据、权重来源和逐行配置见 experiment.json。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。每行冻结后自动预测 clean 和六个场景，再独立 truth-last 评分；按样本数合并 cross_rx 与 cross_both，再按种子汇总。训练过程记录通过数量、类别分布、缺参考回退和源 L 抽样正确率。测试集此前曝光，比较属于探索性结果，不反馈改变固定训练矩阵。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
