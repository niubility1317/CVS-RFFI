# DURA 四阶统计与真实模拟信道变化保持七景测试

- run_id：`20261109-phase1-dura-cumulant-true-response-ground-test-s2026092701-r01`
- group_id：`phase1-dura-cumulant-true-response`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定 DURA 和大容量统计条件恢复器，源模拟器信道生成同记录人工扰动，训练变化保持；部署只读单条接收 IQ。

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

## 固定方案与权限

在相同初始化、同一 32768 条合法源物理 ID、同一 epoch21/22/23 三视图及 20 轮预算下，对 14966274 参数的 DURA 条件恢复网络增加 0.01 倍源微扰响应损失。每条源 clean 构造共轭、幅度三次、相位斜坡或相邻差分中的一种 1% RMS 扰动，通过同记录的源模拟器作用加入 received，重新计算冻结 DURA 上下文；模型原输入与受扰输入的输出之差对齐已知扰动。原 IQ、DURA 改变量和有限窗统计损失分别保持原权重 1、0.02、0.1。

模拟器作用仅参与源训练构造，部署逐条只使用当前 received 及其 DURA 上下文。源 G 训练不读取 TX 标签、F 权重或输出、V、目标或目标真值；F 只用于独立源无查询 smoke 和冻结后正式测试。四种方向是合成代理，不能称为真实发射机身份扰动。最终第 20 轮固定，不依据目标或源诊断结果选择 checkpoint。

GPU4 资源调度先让已登记的源域配对诊断结束，再按不少于 8192 MiB 空闲、少于 3 个总计算进程及本任务同卡最多一 worker 执行；该顺序仅为避免资源竞争，不使用配对诊断结果作训练或测试门槛。训练含每批两次网络前向和扰动上下文准备，应单独记录耗时。错误时保留本 run 日志和产物，不停止其他 owner。

冻结后对同一 Tri-SIG-TIME SSL Main split 默认头执行 clean 和六种 practical，各 168000 条固定物理记录；全部预测完成后独立 truth-last，报告与 DURA 的净纠错、TX/RX/day 和相对 raw clean 的 3% 及绝对 3 个百分点两种口径。历史测试已曝光，不能称为新盲测或真实在轨成绩。

注册命令、固定路径、stop rule 和预期产物见 experiment.json；代码版本在发布前回填。

代码 commit：`5d9c453a3ebcf09dcae6ea1488adef964015e776`。源记录对齐、模拟器参数不得在推理读取、异常样本拒绝等 3 项检查通过；真实 DURA 与 64 条合法源记录训练 smoke、112 条源记录冻结识别器 smoke 由队列执行。一次独立 P0/P1 审查及其依赖定点复核完成；实际远端 smoke 和正式训练尚未完成。

发布归档本地与 N607 SHA-256 均为 `b2b5ad6d67271f44f425d9f135ae36a16c263b87e98a7ed0c1a77c64a4e8a902`。远端编译、3 项源对齐及部署边界检查、runner 和原 DURA 固定依赖导入均通过。真实数据／权重 GPU smoke 尚未执行，须由队列首步完成。

后台队列 PID 2509552 已核实存活、运行于固定发布目录，日志显示等待已有源配对诊断完成后使用 GPU4。启动前 GPU4 空闲 19336 MiB、已有 1 个其他 owner 计算进程；当前本版本尚未开始真实权重 smoke、训练或正式预测。

## 模拟扰动的解释边界

训练中的受扰观测由原始 received 加上缓存中匹配的源模拟信道作用于扰动得到；信道与接收处理参数固定，原观测中已有噪声随之保留。该构造没有对改变后的源波形重新抽取噪声，也没有重新运行完整接收机、锁定判断或 AGC。因此它是条件固定的源端合成变化，不等同于独立重跑完整模拟器，更不证明真实在轨扰动分布。
