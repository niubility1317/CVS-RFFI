# 条件相位源域理想旋转诊断

- run_id：`20261107-phase1-dura-conditional-phase-oracle-source-s2026092701-r01`
- group_id：`phase1-dura-conditional-phase-oracle`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：672 条源记录的理想相位诊断已完成，真实权重源 smoke 通过；见[源域分析](analysis.md)。

## 目的与对照

使用源 clean 比较单记录受限理想相位旋转和训练完成的相位分支。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

固定条件相位模型和 DURA，比较逐记录源 clean 参考下的 ±90° 范围最优整体相位，以及已学习整体分支的有效相位和波形误差。候选训练 ID 与 pilot 互斥；原 DURA 可能见过 pilot，pilot 也已被探索。诊断不读取 F、TX 标签、V 或目标，不更新模型，也不改变已固定七景测试。

真实 checkpoint 源 smoke、四项聚焦测试与独立 P0/P1 审查通过；理想相位使用部署时不可用的 clean，并非实际方法或身份性能上界。完整数值及边界见[源域分析](analysis.md)。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
