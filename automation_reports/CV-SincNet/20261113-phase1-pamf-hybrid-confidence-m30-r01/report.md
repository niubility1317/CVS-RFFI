# Hybrid 仅置信度辅助训练十配置实验

- run_id：`20261113-phase1-pamf-hybrid-confidence-m30-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

十种原始数据配置、三个种子；所有置信度至少 0.95 的 U 参与 Hybrid 辅助训练。

本组不使用分支原型损失，也不对 U 做分支投票。通过阈值的全部无标签样本，包括原物理半批之外的样本，均参与 Hybrid 的 fusion／branch 辅助项。若一个批次只有一个被接受的伪类别，以同包原始／扰动视图的余弦关系补足缺失的对比负样本情形。扩展物理路由也会改变原有混合 U 路径；物理强度沿用原 taper。

## 数据、seed与模型来源

保持 Main split、接收机划分 1／2、10 类、1%／5% 标签和 2／3／4／6 源接收机的原始完整 L/U/V 角色，不留出源域开发集。每行从同配置同种子的原 Tri E20 独立续训 10＋20 轮；原 Hybrid 和普通 PL 同配置同种子对照直接复用。U 的真实类别隐藏。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。每行训练冻结后自动预测 clean 和六个场景，独立 truth-last 评分；按样本数合并 cross_rx 与 cross_both，再按三个种子汇总。测试集已曝光，结果是探索性比较，不反馈修改既定矩阵。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
