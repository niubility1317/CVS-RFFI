# 源域逐时滞严格中心化统计对比

- run_id：`20261109-phase1-strict-cumulant-source-geometry-s2026092701-r01`
- group_id：`phase1-strict-cumulant-source-geometry`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

对比全窗中心化近似与逐滞后子段分别中心化的二二共轭四阶统计，仅进行源记录配对检索。

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

## 固定源域对比

同一合法源训练池，前 512 条物理 ID 的 clean IQ 作为标准化样本，后 512 条不同物理 ID 作为匹配记录，并在三个固定模拟源视图中分别比较 received 与冻结 DURA。对比当前模型实际使用的全窗中心化矩差、每个时滞分别对子段中心化的二二共轭四阶统计，以及两者各自与功率特征等权组合。匹配率是同物理记录检索，不是 TX 身份准确率；clean 参考不可作为目标端部署库，DURA 可能见过这些源记录。

此为 CPU 两线程源域诊断，不读取 F 权重或输出、TX 标签、V 或目标，不训练模型；不会改变正在运行的四阶模型或其正式七景评测。代码 ba1425cf4f58344143ec8b905dfaf0d66e541fca，输入、输出、命令和停止规则见 experiment.json。严格公式的局部数值检查与独立 P0/P1 静态审查通过，服务器执行待验证。

## 服务器执行与结果

发布包两端 SHA-256 均为 2ecdc7b67898cf15e960db246ec86023262f0793849ffba5f0e2d127d049b3fd；服务器编译和导入通过。CPU 两线程任务完成 8 条源记录冻结 DURA smoke 与全部源域对比，完整日志和 result.json 已核实。任务较短，结束前未捕获独立的存活 PID 快照；逐视图比较与边界见 [源域分析](analysis.md)。
