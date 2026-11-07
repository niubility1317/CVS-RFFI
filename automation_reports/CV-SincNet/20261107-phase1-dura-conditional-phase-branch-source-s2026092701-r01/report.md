# 条件相位双分支源域贡献诊断

- run_id：`20261107-phase1-dura-conditional-phase-branch-source-s2026092701-r01`
- group_id：`phase1-dura-conditional-phase-branch`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：N607 GPU 0 容量队列已启动，PID 408689；正式源诊断等待 GPU 容量。实时状态以远端进程、日志和产物为准。

## 目的与对照

固定最终条件相位恢复器与 DURA，测量源 IQ 的逐分支贡献及波形误差。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

已固定训练完成的条件相位模型和原 DURA，将在 672 条源 pilot 上分解局部与整体相位分支的贡献、混合权重及源波形误差。候选训练 ID 与 pilot 互斥；原 DURA 可能见过 pilot，pilot 也已被探索。该诊断不读取 F、TX 标签、V 或目标，不更新模型，也不改变已固定的七景测试。

聚焦测试和独立 P0/P1 审查通过；分支求和与模型前向须一致。远端真实权重源 smoke 将在获得容量后自动执行。指标不是身份识别准确率；诊断尚未运行，暂无线索可说明哪一分支改善了识别性能。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
