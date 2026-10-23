# Full practical residual SDG原生增强数组兼容恢复r06

- run_id：`20261023-phase1-practical-residual-control-r06`
- group_id：`phase1-full-practical-residual-factorial`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

恢复原作者ragged片段object转换语义，替代两条失败SDG及四消费者，接管健康G/IB。

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

## SDG 原作者片段打乱 NumPy 兼容恢复 r06

两条 F0-sdg-A（seed2026092701/02）在原作者 `permutation` 的 `np.random.permutation(splits)` 处失败。256点分为3段时长度不等，NumPy1.24不再自动将该list转换为object array；源侧与局部回归均复现，2同指纹触发停新派发。没有完整训练轮日志，旧模型/输出/失败日志全部保留，不能当正式负结果。

兼容修复保留作者permutation的函数code和其余6个算子，仅该函数的局部NumPy代理对不等长list显式构造原object array，再调用原RandomState.permutation。全局NumPy不改。equal/random两模式×40seed共80例，与按片段索引排列的参考实现逐元素输出及完整RNG状态一致。实际Torch1.11/NumPy1.24、12条合法源L、作者AugDataset两原生视图及Full默认头联合更新一次（disposable smoke）PASS，正式训练更新0、target/Utruth读取0。17项聚焦测试及一次限定直接问题P0/P1审查PASS。

新不可覆盖control-r06只替代2失败SDG F+4阻塞消费者6阶段，130未启动SDG F只换执行release。原G、健康IB、producer保留原PID/CWD/config，已完成F和恢复缓存不重跑；此前HSIC已在真实stage2验证修复。三次技术替代链各6阶段保留排除，历史1898、有效1880，原600新F/4完整G和627读出不变，预算/数据/seed/增强概率不变。只用GPU0–4，每卡总compute<=3并计入其他owner，5/6/7保留。必须独立核实真实SDG轮次/日志增长后才称本次恢复完成。

## 实际恢复核验

状态 **VERIFIED / RUNNING**，执行 `07ada6b251f049573c0c4013c66a6c91830919a3`，owner PID 1984936，旧r05 controller正常交接退出。原G和CPU producer的PID/CWD/argv保持。两条恢复SDG已产生完整轮日志和有限原生Full项，均越过原数据取样故障；169.2秒观测间隔内13条相同PID、12条日志增长。无未解决指纹。当前有效状态 `{"RUNNING": 13, "COMPLETE": 11, "QUEUED": 1856}`，本任务GPU分布 `{2: 3, 1: 2, 0: 3, 3: 3, 4: 2}`；仅0–4，5/6/7保留。

原先IB的stage2真实恢复证据在control-r05/runtime_fix_verification.json，两个新技术故障均已明确替代，原失败历史/部分成本保留。600新F/4G、627读出和1880有效阶段不变。执行版本按current jobs.json逐任务解析，不能把07ada标成全部旧健康worker的版本。尚未全部训练或评分，不宣称新性能/整链加速倍数。
