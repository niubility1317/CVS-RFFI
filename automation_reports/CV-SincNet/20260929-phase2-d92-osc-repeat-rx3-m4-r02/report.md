# D92-OSC冻结方法rx3透明重复基准

- run_id：`20260929-phase2-d92-osc-repeat-rx3-m4-r02`
- group_id：`d92-fixed-phase1-osc-repeated-benchmark`；类别：`cvs`；阶段：`Phase2-repeated-benchmark`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定Phase1及既有received四相位特征；当前任务全部注册类support进行物理medoid循环对齐与共享收缩协方差估计，query逐样本对四种相对相位边缘化；复用原D92预测。 Technical recovery of cache metadata binding only; fixed algorithm, data, model and matrix unchanged.

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

PLANNED technical recovery only: legacy origin metadata omits SHA; expected SHA remains required in D92 startup and BNNA producer metadata/cache. Formula, matrix, source provenance, data and model unchanged.

Technical repair VERIFIED: 90 local tests and readonly full binding checks on all eight real caches passed; no IQ/truth/scoring/fit access. Evidence osc_cache_schema_recovery_diagnostic_20260929.json. Ready for recovery release.

RUNNING independently VERIFIED at readback_1790616709.json. Four CPU workers, actual fit progress; no restart or score reading until both cohorts finish.


<!-- OSC_FINAL_ANALYSIS_20260929 -->

## 最终结果、核验与成本

当前终态 SCORED / ANALYZED，证据 VERIFIED；上文启动说明保留为历史记录。实际 release commit：`19a38b714f26599b6a9a074ccfedacc1c8df464b`。完整 7236 条评分，算术与训练日志审计通过。

四个 K 中，联合旧类、新类、H 分别有 3、3、3 个提升；全部任务旧类保护条件有 4 个通过。预登记重复基准条件未通过。本归档不自动判定晋级，独立泛化与整体目标完成尚未确认。

下表为候选减 D92，单位为百分点。前三列仅含新增类数大于 0 的联合任务；末列含 old-only，用于旧类保护条件。

| K | Δ旧类（联合） | Δ新类 | ΔH | Δ旧类（全部任务） |
|---|---:|---:|---:|---:|
| 1 | +0.167 | -1.592 | -0.892 | +0.559 |
| 5 | +8.104 | +1.720 | +4.194 | +8.135 |
| 10 | +4.822 | +1.271 | +2.568 | +4.618 |
| 20 | +1.732 | +3.589 | +3.176 | +1.579 |

全部 3600 次任务拟合中，K1 固定解析规则 900 次，support 留出诊断任务 2700 次。共 11700 次解析拟合，其中 fold 内 8100 次、最终拟合 3600 次。留出诊断不参与选模或温度校准；没有优化器更新、学习率、梯度或 epoch，也不声称迭代收敛。

本次新增特征提取 0.000 秒、核心拟合 308.972 秒、query 打分 119.713 秒、预测写出 16.049 秒，任务总计时 464.199 秒。阶段累计值不等于并行墙钟时间或星载速度；不叠加已经包含的计时。

每条物理观察有 4 个固定相位视图，仍只计一个样本。编译后的 W+b 为每类 8,200 字节，持久数值状态合计 49200 至 213200 字节；类 ID、审计容器另计。复用既有冻结 cache，不加载模型或继承适应状态。历史缓存提取耗时与本次计算分列；接收端 cache、临时数组和进程 RSS 不算地面传输。

完整训练 checkpoint 包为 15,992,872 至 15,992,936 字节，不是经裁剪核实的最小推理包。模型部署状态未知，新增模型传输字节为 null；新增源域 payload 为 0 字节，不读取地面摘要，query/source 拟合行数均为 0。Phase1 保持冻结。

联合验收按真实单元等权，rx3∶rx1 为 3∶1，不能用单批次替代联合结果。目标数据此前已评分，本次不构成新增独立泛化验证；不作统计显著性声明。结果不回流调参或选择性重跑，完整原始产物保留。

见[产物索引](results/artifacts.json)、[本批次汇总](results/summary/report.md)、[训练审计](results/fit_audit.json)。

ANALYZED VERIFIED: full paired matrix complete, all arithmetic/fold/baseline/interpretation audits passed. K5/10/20 improve new/old/H; K1 new/H lower. No promotion, no new independent-data validation, goal ACTIVE.
