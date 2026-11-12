# 10 类 PAMF 分支覆盖率与可靠性实验

- run_id：`20261112-phase1-pamf-coverage-tenclass-m9-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原始完整训练数据上比较参考库、3/5 低权重分层及两者结合，3 个匹配种子。既有原 4/5 全权重和普通 PL 对照直接复用；正在进行的同配置 `ten_class-prototype-gate3` 三种子作为 3/5 全权重对照，不重复训练。该对照不阻塞新 run 的训练和评分。

## 数据、seed与模型来源

保持原 ManyRx 训练角色：有标签 6533、无标签 58828，原 V 不变，不留出源域开发集；各模型从原数据相同 seed 的 Tri E20 独立续训 10＋20 轮。U 的真实类别隐藏。完整权重来源和逐行配置见 experiment.json。

参考库只存有标签训练样本，每类别／日期最多 8 条，每 20 步刷新一次模型特征；每个待判断样本排除同日期参考。强档两视图均至少 4/5，弱档均至少 3/5 且不属于强档；弱档分支辅助损失为原强度的 25%。现有 U 物理视图路径会接收新加入的弱档样本，这一附带变化与辅助损失权重一并报告。

记录参考缺失、强弱档覆盖和伪类别计数、强弱档损失、抽样辅助梯度冲突；每 50 步用训练中的有标签数据做跨日期分层可靠性诊断。此诊断并非独立留出准确率。类别无关一致性未纳入此轮，以便单独分辨参考库与分层的作用。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。每行训练冻结后自动预测 clean 和六个场景，再独立 truth-last 评分；同 seed 合并 cross_rx 与 cross_both。测试集已曝光，结果属于探索性比较，不作为独立盲测，也不反馈改变固定训练矩阵。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
