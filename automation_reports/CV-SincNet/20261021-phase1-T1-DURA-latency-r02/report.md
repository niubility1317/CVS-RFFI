# 最终T1＋DURA统一时延：基线导入冲突恢复

- run_id：`20261021-phase1-T1-DURA-latency-r02`
- group_id：`phase1-cric-dura-complete-system`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Only unified timing recovery; original failed namespace retained, all healthy scientific workers/controller unchanged. Add immutable external baselines path to already-imported native namespace. Wait truly idle GPU0/1/4/5/6/7; 13GiB recorded reservation keeps all >=12GB sharing schedulers away. Reservation not model memory; baseline/incremental separately recorded.

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

**VERIFIED / RUNNING**：GPU7、真实CUDA PID/CWD/命令均核实，已保存26个case-round。冻结F/G及8个外部Full/CE均实际前向，恢复后已进入正式计时。显存预约13GiB单列，不当作模型显存；不提前用部分计时填完整表。其余7卡科学训练继续，无健康进程被停止。

## 已完成正式时延

**VERIFIED / ANALYZED**：20 calls × 5 batches × 5 rounds 全部 500 个单元完成。已取回全部 140000 次 GPU wall 和 10000 次传输测量，逐轮重算 median/P95/吞吐与原 summary 一致。完整结果见 [latency_results.md](latency_results.md)、[latency_summary.csv](latency_summary.csv)、[latency_verification.json](latency_verification.json)。

最终 T1＋DURA 默认 B1 为 4.541187 ms，P95 4.725135 ms；H2D/CPU 类别输出边界 median 4.611311 ms。同权重 native 为 37.189997 ms，执行路线加速 8.189 倍。500 个单元的完成不代表未来机制面板完成，旧失败行保留且仅由本 r02 替代。
