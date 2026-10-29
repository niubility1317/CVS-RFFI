# PAMF第三阶段：原F参照的源已知风险校准

- run_id：`20261029-phase3-source-baseline-risk-s2026092701-r01`
- group_id：`pamf-phase3-source-baseline-risk`；类别：`cvs`；阶段：`Phase3-source-original-baseline-risk-calibration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Frozen两套E51＋200步readout，只由knownV与originalF6参照，按原总损伤2pp和FRR5%选择最大合法全局scalar阈值；无unknown校准。

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

实际code/config commit：`a18daf5080bca7e76c3837907dcfdefd97fbc5bd`。9项聚焦tests及4subtests通过，一次独立P0/P1PASS；最大合法阈值同brute枚举一致。真实冻结readout两条L特征smoke为首步，PASS后knownV+原F同IQ baseline校准；每RX及总体总损伤≤2pp、FRR≤5%，novel只在threshold冻结后预测评分。全部状态冻结，encoder/head/readout新增更新0，历史222/200。N607一次preflightPASS，CPUonly、GPU新增0、foreignstop0，源数据句柄与产物复用，不重验raw或传数据。新source候选不改变此前NO_PROMOTION、95%unknown/旧类保护及正式target禁读。

## 实际结果

VERIFIED / ANALYZED_SOURCE_ONLY，NO_PROMOTION。MLP监督88.0635%／OE92.6667%，OE较父90.7619增加1.9048pp；FRR2.0296%、open97.3852%、相对原F总损伤1.0815pp。所有known保护通过，但整体及RX3/4/6/8未知仍未95。14阈值均由known-only确定且下一候选不可行；监督RX6/OERX3的2pp损伤预算恰绑定，不按未知改阈值。新F/head/readout更新0，4完整日志及2L冻结readout smoke齐全、foreignstop/target0。首次独立post时已结束，未捕获liveCWD/GPU观察，不伪造或重跑。完整结果见 docs/PAMF_PHASE3_SOURCE_BASELINE_RISK_RESULTS_20261029.md；同novel重复source开发，非正式跨RX星地/适应/注册证据。
