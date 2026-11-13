# 第二轮条件方差分支源训练样本敏感度诊断

- run_id：`20261107-diagnostic-Restore-conditional-variance-E02-sourceLU-s2026092701-r01`
- group_id：`frozen-source-training-variance-sensitivity`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定E02、原训练draw02、5源RX×7scene各前2条，比较当前/零/初始/0.1方差比例，检查损失和局部梯度。

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

## 固定源诊断已完成

VERIFIED / ANALYZED。固定E02源训练70条（实际全部L_s，五RX×七scene×2），两种源先验×四种固定方差，共56单元完成且独立重算。当前分支与零附加方差的损失差仅约1e-8量级，当前softplus平均斜率约2.2e-7；统一增大方差在global55/70、other-RX44/70条上局部不利，但约43%/46%的坐标仍有负梯度方向。结果支持表达限制与梯度衰减并存，尚不证明永久失效或Main识别收益。源2检查、3项针对性测试、一次独立P0/P1和实际CPU执行通过，零模型更新、零TX/V/OE/target读取。GPU6当前已完成3/30轮，原final30和Main七景矩阵保持。

详细机制分析
