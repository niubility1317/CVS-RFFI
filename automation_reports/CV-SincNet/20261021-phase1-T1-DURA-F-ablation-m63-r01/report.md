# 最终T1-E50识别机制：21设置×3种子

- run_id：`20261021-phase1-T1-DURA-F-ablation-m63-r01`
- group_id：`phase1-cric-dura-complete-system`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Final T1 correction: 20(13L+7LU)+own10LU+own20LU, AsCE/KL0, sixL, nativeEMA/U, defaulthead. Existing F0 final T1 three seeds reused. Old wrong33L17U unstarted179 withdrawn; all healthy old workers/results preserved as historical. 150mainF+63mechanismF;35same-contract fullG+21Gstudents;249prediction/scoring rows;2exclusive timing panels.

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

Final version correction and exact matrix: see docs/goals/CRIC_DURA_COMPARISON_AND_LATENCY_20261021.md. Final T1 is13L37U, not old33L17U. Current state PLANNED until independent PID/log/CUDA evidence.

**VERIFIED / RUNNING**：8张卡均有本矩阵真实CUDA worker，F源训练/完整G谱系/G消融实际步数增长；不接管旧错误配方健康worker。当前首段F监督约85至90秒/轮，U和G后续阶段仍待校准；整批暂估3至7天。统一时延r01兼容失败已独立r02恢复等待独占卡，主队列不变。
