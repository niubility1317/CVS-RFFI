# 源接收机协方差差异的固定基表达能力

- run_id：`20261107-analysis-Restore-source-covariance-capacity-s2026110707-r01`
- group_id：`source-covariance-representation-capacity`；类别：`analysis`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

只读取既有源矩，分析top64独立缩放、全512独立缩放、top64耦合和当前有界signed64的Frobenius表达能力。

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

## 全部源统计分析已完成

VERIFIED / ANALYZED。只读既有源矩完成10单元，零新IQ/标签/V/OE/target/拟合/模型读取。global本底五RX等权：独立64约75.70%，独立512约75.91%，耦合64理想空间约98.35%的协方差差异Frobenius能量；不是识别率或IQ恢复比例。10个矩阵直接重建独立复算通过且耦合替换均正定。源端相对方差谱也超出当前0.1至1.9范围。支持后续研究方向间耦合与更充分的正定伸缩，不以该上限代替网络学习或正式Main收益。

[完整矩阵几何分析](analysis.md)
