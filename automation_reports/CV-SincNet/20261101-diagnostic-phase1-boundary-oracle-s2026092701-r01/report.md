# Phase1 源训练样本受限分类边界可达性反事实

- run_id：`20261101-diagnostic-phase1-boundary-oracle-s2026092701-r01`
- group_id：`phase1-source-boundary-reachability`；类别：`cvs`；阶段：`Phase1-source-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

对固定源样本中 DURA 与 clean 基座判断不一致的记录，研究受限波形改动是否能到达 clean 判断。

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

## 源训练样本边界反事实结果

VERIFIED／ANALYZED_SOURCE_ONLY。固定 512 个 Phase1 源训练物理记录，在源视图 01 和 20 中分别处理原 DURA 输出与 clean 信号的 Tri-SIG-TIME SSL 判断不一致的记录。网络参数冻结，输入梯度方向共 12 步，逐条 L2 扰动幅度分别限制为配对 clean L2 范数的 2% 和 5%；没有读取 TX 标签、V 或测试 IQ。

| 源视图 | DURA 与 clean 判断不一致 | 2% 半径内到达 clean 判断 | 5% 半径内到达 clean 判断 | 2% 前／后平均 IQ NMSE |
|---|---:|---:|---:|---:|
| 01 | 21/512 | 20/21 | 21/21 | 0.099025／0.098468 |
| 20 | 20/512 | 20/20 | 20/20 | 0.115561／0.115702 |

独立进程核对逐组样本数、目标到达数、有限性及扰动预算；2% 半径的最大归一化平方扰动为 0.0004。两个视图共享同一批物理记录，不能当作独立样本。

这是使用 clean 样本类别及 clean 范数的特权反事实，真实接收时不可获得，不能部署，也不是 TX 真实准确率或可达性能上界。梯度可能产生欺骗分类器的对抗性扰动，不能推断物理保真、泛化或 3% 目标达成。结果仅提示可继续研究无 clean 输入的摊销校正，并需要源域之外的严格验证；本开发测试集已被观察，不能称盲测。
