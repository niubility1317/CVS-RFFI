# DURA单阶段联合训练的源端成本诊断

- run_id：`20261030-diagnostic-DURA-single-stage-cost-s2026092701-r01`
- group_id：`phase1-dura-single-stage-efficiency`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

CPU2zerooptimizerupdates;onlineDURAteacher/cachedteacher/randomsinglejointsourceobjective;B32/B256warm2timed6;fullDURAgraph266951params;noF/TX/V/target

用户新增训练效率要求后固定此小型成本诊断，不改当前两行健康E20续训。旧DURA已有约120个组件epoch祖先，当前续训又追加20；仅报告新增E20会掩盖研发成本。新的候选结构仍为完整266951参数两阶段推理图，但从随机参数开始，在一次源联合更新中拟合最终IQ、两个内部prior、原谱目标和source模拟器作用/方差；不读取学习教师或祖先权重。这不是两个推理展开阶段分别预训练。

本run仅比较CPU单步成本：在线冻结DURA教师、同batch缓存教师、随机单阶段联合目标；B32/B256各两次预热加六次计时。固定R07epoch01中前256个Ls角色位置，同一批源IQ反复用于计时，0TX/F/V/target。计时含forward/objective/backward/clip，不含optimizerstep、IO、传输、初始化或教师缓存生产；不能将缓存组数值当总训练开销或CPU时间当GPU吞吐。无优化器更新、无新权重保存、无分类prediction，模型完整state前后相同。

本地单步梯度回归通过：所有模型参数梯度有限，source clean/oracleaction/variance目标不接收梯度；编译通过。一次独立P0/P1审查PASS。launcher第一步会对原实际DURA checkpoint及随机完整图执行两条真实Ls无query smoke后立即计时。后续直接训练候选仍须固定E20、完整七景与独立truth-last，不能以本诊断宣布效率/性能目标完成。

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
