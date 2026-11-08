# DURA 与融合网络源域微扰响应

- run_id：`20261108-phase1-direct-mix-source-response-s2026092701-r01`
- group_id：`phase1-direct-mix-source-response`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

比较 DURA、独立网络、固定各半和两个冻结门控对四类源域人工微扰的响应；无 F 和目标读取。

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

## 后台源域实验

比较 DURA、直接恢复、固定各半与两版门控对四种相对 RMS 1% 人工微扰的响应。112 条既有源 pilot 按 14 个 RX/视图组各取 8 条，均与直接网络及两门控训练记录不重叠；原 DURA 是否见过源记录未知。微扰通过原记录的 DURA 估计物理作用注入，受扰后重新估计上下文，指标只反映合成代理，不能代表发射机身份或目标准确率。

不读取 F、TX 标签、V 或目标，不更新权重。代码 commit `e0cb4bab03804e4f12df7d79a5235aa87db0905b`；八项针对性检查、独立 P0/P1 审查、归档传输与远端编译通过。CPU 后台 PID 1160532，真实源前 8 条检查通过。

已完成 112 条源 pilot 微扰响应计算；分组均值与解释边界见 [analysis.md](analysis.md)。
