# D92-BNNA冻结方法rx3透明重复基准

- run_id：`20260929-phase2-d92-bnna-repeat-rx3-m4-r01`
- group_id：`d92-fixed-phase1-bnna-repeated-benchmark`；类别：`cvs`；阶段：`Phase2-repeated-benchmark`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定Phase1和既有received IQ及support/query；当前任务support的4个确定性相位视图训练有界非线性适应器；完整物理样本fold重拟合全部状态；复用原D92预测。

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

BNNA formula frozen; support-only full physical folds; metadata ancestry reused; preflight verified; launch pending local implementation verification.

LOCAL_VERIFIED: frozen formula; core14 and full entry/audit integration passed; independent P0/P1 no blocker. Model deployment and incremental model transmission unknown. Existing checkpoint ancestry remains exact source-only scratch final200. Real native synthetic smoke runs before received IQ is opened.

VERIFIED RUNNING at readback_1790613570.json; actual release 0fce1af680992fac2c343762af0f07915e10b647. Both cohorts execute unchanged; no scores may be downloaded before both reach terminal state.


<!-- BNNA_FINAL_ANALYSIS_20260929 -->

## 最终结果、核验与成本

当前终态 SCORED / ANALYZED，证据 VERIFIED；上文启动说明保留为历史记录。实际 release commit：`0fce1af680992fac2c343762af0f07915e10b647`。完整 7236 条评分，算术与训练日志审计通过。

四个 K 中，联合旧类、新类、H 分别有 1、0、0 个提升；全部任务旧类保护条件有 1 个通过。预登记重复基准条件未通过。本归档不自动判定晋级，独立泛化与整体目标完成尚未确认。

下表为候选减 D92，单位为百分点。前三列仅含新增类数大于 0 的联合任务；末列含 old-only，用于旧类保护条件。

| K | Δ旧类（联合） | Δ新类 | ΔH | Δ旧类（全部任务） |
|---|---:|---:|---:|---:|
| 1 | -1.278 | -1.407 | -1.243 | -0.940 |
| 5 | +0.458 | -6.451 | -4.185 | +0.680 |
| 10 | -5.440 | -10.565 | -9.151 | -5.118 |
| 20 | -9.328 | -13.389 | -12.468 | -9.277 |

全部 3600 次任务拟合中，K1 固定训练 900 次，support 交叉验证任务 2700 次。实际 Adam 更新 716288 步，其中 fold 内 518400 步、最终拟合 197888 步。每个非零 rank 的训练调用固定执行 64 步；rank0 或最终选择 identity 时按预登记跳过对应更新。64 步预算不代表已证明收敛。

累计特征提取 350.907 秒、核心拟合 4239.120 秒、query 打分 135.697 秒、预测写出 16.306 秒，任务总计时 4467.502 秒。阶段累计值不等于并行墙钟时间或星载速度；不叠加已经包含的计时。

每条物理观察有 4 个固定相位视图，仍只计一个样本。原型头 12288 至 53248 字节，基矩阵 0 至 10240 字节，门控系数 0 至 64 字节，持久数值状态合计 12288 至 63552 字节。接收端 cache、临时拟合数组和进程 RSS 不算地面传输。

完整训练 checkpoint 包为 15,992,872 至 15,992,936 字节，不是经裁剪核实的最小推理包。模型部署状态未知，新增模型传输字节为 null；新增源域 payload 为 0 字节，不读取地面摘要，query/source 拟合行数均为 0。Phase1 保持冻结。

联合验收按真实单元等权，rx3∶rx1 为 3∶1，不能用单批次替代联合结果。目标数据此前已评分，本次不构成新增独立泛化验证；不作统计显著性声明。结果不回流调参或选择性重跑，完整原始产物保留。

见[产物索引](results/artifacts.json)、[本批次汇总](results/summary/report.md)、[训练审计](results/fit_audit.json)。

ANALYZED: complete joint4800 fits/9648 scores and958784 Adam steps verified. BNNA did not meet comprehensive old/new/H criteria. Preserve all evidence; no rerun or promotion.
