# DURA 源非线性残差可预测性诊断

- run_id：`20261107-phase1-dura-nonlinear-residual-source-s2026092701-r01`
- group_id：`phase1-dura-nonlinear-residual-source`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED，详见[源域结果](analysis.md)。

## 目的与对照

比较源拟合线性与三次复 IQ 特征对按物理 ID 留出样本的 DURA 残差解释能力。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

12288 条留出源视图中，三次分组平均 IQ NMSE 相对 DURA 下降约 0.053%，略差于线性分组；拟合系数未保存，未访问目标。

## 交接

该固定三次特征证据不足以作为新恢复网络；已登记的多视图学生、流匹配及其他候选继续按原计划训练与七景测试。
