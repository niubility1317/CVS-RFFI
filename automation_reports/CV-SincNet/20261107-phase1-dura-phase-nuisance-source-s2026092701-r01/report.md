# DURA 源信号相位旋转响应诊断

- run_id：`20261107-phase1-dura-phase-nuisance-source-s2026092701-r01`
- group_id：`phase1-dura-phase-nuisance-source`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：源诊断已完成。固定 DURA 真实 checkpoint 的源样本 smoke 通过，672 条源记录 × 四种输入旋转已完成，结果与限制见[分析](analysis.md)。

## 目的与对照

对固定源接收 IQ 做四种整体旋转，比较 DURA 对固定与同步旋转 clean 的响应。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

已完成预登记的源 IQ 诊断；没有新恢复模型、目标预测或识别准确率结论。七景测试要求适用于未来实际发布的恢复器，不适用于本次未训练、未部署的诊断。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
