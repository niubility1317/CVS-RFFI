# Tri 实际更新保护与 U 强度因子对照

本批预登记四行：十类、六源各使用同 seed 2026092701、同合法源域原 Tri E20，接续自身 E21–E50 30 轮。两种因子均保留当前完整 L 目标的实际 AdamW 更新保护及原 Tri 其他机制：`step_weight3` 将 U 总损失权重从 6 降为 3、原梯度上限为 1；`step_guard05` 保持权重 6，将原 U/L 梯度上限从 1 改为 0.5。分别与既有不加实际更新保护的相同权重或上限候选以及当前权重 6 的更新保护候选比较。

登记依据是源域记录中 t2 多数步原 U 范数接近或超过 L 范数、频繁限幅；半权重和半上限是预定的两个因子，并非根据任何后续目标分数选择。历史目标及 seed 已曝光，全部结论只能是探索性，不得以目标结果改变本矩阵、候选顺序、模型或超参数。

训练在 N607 普通用户、GPU 2/3 按每卡含其他用户总进程不超过 5 的调度进行；实际代码为提交后发布的 Git commit，命令与路径在 experiment.json，控制器启动前真实 E20 四行 no-query 冒烟。每行记录每步 L/U 梯度、原梯度投影/限幅、复制当前段 AdamW 状态的实际增量投影及回退、源验证、接受率和不含真值的逐样本伪标签。每行 E50 源冻结后独立离线 U 真值评分；预测只做 clean 的 cross_date/cross_rx/cross_both，先预测固定，后独立 truth-last 评分。禁止用 U 真值或目标结果回流训练或选择；不进行星地和 weak 测试。发生协议、非有限或系统技术错误时停止新调度，保留其他健康进程和已有产物。

独立 P0/P1 审查已完成，未发现直接阻断问题；根据审查修正冒烟更新计数，并在发布前执行针对性配置和既有保护测试。实际预检、传输、smoke、PID/GPU、日志增长及最终测试以独立产物为准。

# Tri 实际更新保护与 U 强度因子对照

- run_id：`20261116-phase1-pamf-repl-step-factor-clean-m4-r01`
- group_id：`phase1-pamf-repl-weight-clean`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

十类和六源各测试权重 3 与梯度限幅 0.5，两者均保留完整 L 实际更新保护。

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
