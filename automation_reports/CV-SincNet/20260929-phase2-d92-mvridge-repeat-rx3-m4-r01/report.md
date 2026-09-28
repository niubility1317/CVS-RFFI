# D92-MVRidge冻结方法rx3透明重复基准

- run_id：`20260929-phase2-d92-mvridge-repeat-rx3-m4-r01`
- group_id：`d92-fixed-phase1-mvridge-repeated-benchmark`；类别：`cvs`；阶段：`Phase2-repeated-benchmark`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定Phase1与既有四相位received特征；全部注册类support以每物理样本总权重1拟合共享ridge判别头，单位ridge正则，无old/new组权或teacher。query逐样本四视图均值推理。

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

MVRidge PLANNED: frozen physical-weighted four-view supervised ridge1, complete matrix, source-free/cache-only CPU. Resource/path preflight VERIFIED. No experiment launched; root sole launch owner.

MVRidge local validation VERIFIED: 9 core tests, 13 predictor tests, 57 collector tests, and relevant orchestration/scoring/summary tests passed. Independent P0/P1 review found no blockers. Frozen method and full matrix are ready for committed publication; not launched at this entry.

RUNNING VERIFIED by readback_1790618402.json; release 71e4bef490bfae3acf907cc3f488b9ca62172a0f. Full matrix CPU run uses frozen pure received features and zero new source payload. No duplicate launch; await both cohorts before reading results.


<!-- MVRidge_FINAL_ANALYSIS_20260929 -->

## 最终结果、核验与成本

当前终态 SCORED / ANALYZED，证据 VERIFIED；上文启动说明保留为历史记录。实际 release commit：`71e4bef490bfae3acf907cc3f488b9ca62172a0f`。完整 7236 条评分，算术与训练日志审计通过。

四个 K 中，联合旧类、新类、H 分别有 1、0、0 个提升；全部任务旧类保护条件有 2 个通过。预登记重复基准条件未通过。本归档不自动判定晋级，独立泛化与整体目标完成尚未确认。

下表为候选减 D92，单位为百分点。前三列仅含新增类数大于 0 的联合任务；末列含 old-only，用于旧类保护条件。

| K | Δ旧类（联合） | Δ新类 | ΔH | Δ旧类（全部任务） |
|---|---:|---:|---:|---:|
| 1 | -1.907 | -1.275 | -0.799 | -2.236 |
| 5 | +2.496 | -0.876 | +0.401 | +2.131 |
| 10 | -1.000 | -3.044 | -2.283 | -1.270 |
| 20 | -1.989 | -3.027 | -2.561 | -2.192 |

全部 3600 次任务拟合中，K1 固定解析规则 900 次，support 留出诊断任务 2700 次。共 11700 次解析拟合，其中 fold 内 8100 次、最终拟合 3600 次。每个物理样本的四个 view 各权重 1/4，总质量为 1，ridge 系数固定为 1。留出诊断不参与选模或温度校准；没有优化器更新、学习率或 epoch，记录求解后的真实梯度残差，不声称迭代收敛。所有解析拟合的最大梯度残差为 3.36529e-13，仅用于数值求解审计。

本次新增特征提取 0.000 秒、核心拟合 89.555 秒、query 打分 47.687 秒、预测写出 17.503 秒，任务总计时 169.164 秒。阶段累计值不等于并行墙钟时间或星载速度；不叠加已经包含的计时。

每条物理观察有 4 个固定相位视图，仍只计一个样本。编译后的 W+b 为每类 2,056 字节，持久数值状态合计 12336 至 53456 字节；类 ID、审计容器另计。复用既有冻结 cache，不加载模型或继承适应状态。历史缓存提取耗时与本次计算分列；接收端 cache、临时数组和进程 RSS 不算地面传输。

完整训练 checkpoint 包为 15,992,872 至 15,992,936 字节，不是经裁剪核实的最小推理包。模型部署状态未知，新增模型传输字节为 null；新增源域 payload 为 0 字节，不读取地面摘要，query/source 拟合行数均为 0。Phase1 保持冻结。

分类分数不是校准概率，support OOF NLL 仅作同一固定模型的留出诊断，不用于跨方法优劣比较。不同 N、C 任务的损失总量只核验分解恒等式，不合并解释为平均训练质量。

联合验收按真实单元等权，rx3∶rx1 为 3∶1，不能用单批次替代联合结果。目标数据此前已评分，本次不构成新增独立泛化验证；不作统计显著性声明。结果不回流调参或选择性重跑，完整原始产物保留。

见[产物索引](results/artifacts.json)、[本批次汇总](results/summary/report.md)、[训练审计](results/fit_audit.json)。

ANALYZED VERIFIED: complete paired matrix, original D92/DG exact, all fit/arithmetic/interpretation audits passed. MVRidge does not meet the full objective. No live experiment; raw evidence retained. User-authorized repeated benchmark only; no independent-generalization claim or candidate promotion.
