# D92-SGJoint冻结方法rx3透明重复基准

- run_id：`20260928-phase2-d92-sgjoint-repeat-rx3-m4-r01`
- group_id：`d92-fixed-phase1-sgjoint-repeated-benchmark`；类别：`cvs`；阶段：`Phase2-repeated-benchmark`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定4个Phase1模型与已验证received capsule；复用原D92预测和冻结目标特征，在新输出拟合SGJoint并与原D92配对。目标已经评分过，只声明重复基准。

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

Frozen SGJoint repeated benchmark authorized by user. Cached input preflight VERIFIED; 74 reuse-runner/scorer tests and 31 summary tests passed. One focused reuse delta review found no P0/P1 blockers. Both cohorts execute completely; no score downloads until both terminal. Core formula unchanged from 7a5118629. Await actual launch evidence; no live run claimed.

VERIFIED RUNNING at readback_1790608341.json; supervisor2441154 and four CPU children bound to intended release. Continue same run; never relaunch due to observation timeout.


<!-- SGJOINT_FINAL_ANALYSIS_20260928 -->

## 最终结果与完整核验

当前结果已完成评分与分析（SCORED / ANALYZED），证据状态 VERIFIED。上文的启动、运行中及待读分说明是历史记录。实际发布 commit 为 `a794b71e1c93d6d4e9f3cfe1f2ccc591a685bdea`。本批次全部 7236 条评分通过混淆矩阵独立复算，3600 次拟合日志完整核验。D92-SGJoint-v1 未晋级，整体优化目标尚未完成。

下表均为候选减 D92，单位为百分点。联合列只含有新类的任务；最后一列含 old-only，按预登记用于旧类退化约束，阈值为不低于 −1 个百分点。

| K | Δ旧类（联合） | Δ新类 | ΔH | Δ旧类（全部注册任务） |
|---|---:|---:|---:|---:|
| 1 | +0.409 | -2.555 | -1.735 | +0.878 |
| 5 | +5.600 | -1.175 | +0.831 | +6.149 |
| 10 | +2.326 | -1.907 | -0.760 | +2.256 |
| 20 | +0.262 | -0.055 | -0.008 | -0.037 |

本批次累计实际拟合计时 836.284 秒，平均 0.232301 秒/任务；K1 固定配置 900 次，support 内部交叉验证 2700 次。这是 N607 CPU 运行中的 fit_seconds 累加，包含该拟合函数内的工作，不等同于并行运行墙钟时间或星载硬件速度。分类头采用闭式求解，温度由有界标量求解选择，不声称迭代训练收敛。

记录到的单进程峰值 RSS 为 604016640 至 605163520 字节，不能称为纯模型状态。实际分类头状态为 7728 至 53456 字节，固定摘要算子为 204800 字节，持久数值状态合计 212528 至 258256 字节。

既有摘要数值数组为 4,410 字节，注册/schema 数组为 670 字节，数组合计 5,080 字节。压缩 NPZ 与 manifest 合计 8,191 至 8,383 字节/模型。冻结配置 `summary_already_deployed=false`，因此该文件包仍计入交付字节；本轮新增地面统计为 0 字节，不代表既有摘要交付为 0。source_rows_read 与 query_rows_used_for_fit 均为 0，Phase1 保持冻结。

本批次不能单独替代完整联合结论。四个 RX 按真实配对单元等权，两个批次权重为 3∶1。复用数据此前已经评分，结果只支持透明重复基准比较，不支持新的独立泛化声明；共享 query 的 support 抽样不作为独立模型重复。结果不用于本候选调参或选择性重跑。

证据：[产物索引](results/artifacts.json)、[本批次完整汇总](results/summary/report.md)、[拟合审计](results/fit_audit.json)。

Both cohorts terminal and jointly analyzed; no relaunch; complete fixed results preserved.
