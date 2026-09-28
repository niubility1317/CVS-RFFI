# D92-BranchRidge冻结方法rx3透明重复基准

- run_id：`20260929-phase2-d92-branch-ridge-repeat-rx3-m4-r01`
- group_id：`d92-fixed-phase1-branch-ridge-repeated-benchmark`；类别：`cvs`；阶段：`Phase2-repeated-benchmark`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定Phase1，使用原始received单view的身份、FFT及融合前时域/频域/PA分支，当前任务全部真实support拟合固定ridge1解析分类头；所有K同式。

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

PLANNED: single support-justified BranchRidge-v1, full benchmark reuse. Exact original source-only checkpoint lineage unchanged and previously verified; live resource/path/capsule metadata preflight VERIFIED. No experiment launched; final focused entry tests and unique P0/P1 review pending.

Implementation ready: 40 core/probe, 23 exporter/entry, 61 orchestration and 24 summary tests passed. Native synthetic no-query smoke is built into export before any received forward; singleton inference, immutable state and truth-last barrier preserved. No experiment launched yet.

RUNNING VERIFIED readback_1790622276.json and readback_1790622356.json: live owned supervisor, matching export argv/CWD and growing logs. Actual runtime commit 01e93386736f1919ee3f009fb1b16fb04f9a463d. Successful native synthetic smoke recorded separately from received forward counts. Do not repeat launch; no new scores read.


<!-- BRANCH_RIDGE_FINAL_ANALYSIS_20260929 -->

## 最终结果、核验与成本

当前 SCORED / ANALYZED，证据 VERIFIED；原文保留运行历史。release commit：01e93386736f1919ee3f009fb1b16fb04f9a463d。完整 7236 条评分，原始 D92/DG 逐记录保持一致。

四个 K 中，联合旧类、新类、H 分别有 4、4、4 个提升；全部任务旧类保护条件有 4 个通过。预登记重复基准条件通过。本归档不自动判定晋级，独立泛化与整体目标完成尚未确认。

下表为候选减 D92，单位百分点；前三列仅新类存在任务，末列含 old-only。

| K | Δ旧类（联合） | Δ新类 | ΔH | Δ旧类（全部任务） |
|---|---:|---:|---:|---:|
| 1 | +5.410 | +4.409 | +5.374 | +4.926 |
| 5 | +12.431 | +7.101 | +9.247 | +12.023 |
| 10 | +8.402 | +5.803 | +6.838 | +7.827 |
| 20 | +5.343 | +6.107 | +6.008 | +4.824 |

共 3600 次全 support 解析拟合，含 900 次 K1；每任务一次分解，所有 K 均无运行时 OOF、网格或选参。拟合调用累计 16.988 s，query 打分累计 25.795 s，提取累计 405.211 s；本 cohort 墙钟 473.063 s，累计进程时间不等于墙钟。 received forward 61776 次，合成 smoke 4 次分列。数值头为 W[736,C]+b[C]，5896C B，C=6 至 26 对应 35376 至 153296 B，不含 Python/registry/audit 开销。最大记录梯度残差 5.25e-13；没有优化器更新，不作迭代收敛声明。新增 source payload 与 ground statistics 均为 0；模型文件是已有完整训练 checkpoint 包，不是最小推理包，部署状态和增量模型传输未知。RSS 为单进程高水位，不能相加冒充并发峰值；所有实测分量和字节见产物索引。

验收采用完整四 RX 等单元权重，不能以本 cohort 替代联合结论。本轮为已评分数据透明重复基准，开发使用授权 support 诊断；没有新的独立泛化证据。结果不回流选参或选择性重跑。

见[产物索引](results/artifacts.json)、[本 cohort 汇总](results/summary/report.md)。

ANALYZED VERIFIED: all K and all four model seeds improve old/new/H; complete full-matrix audits and baseline equality passed. Candidate held fixed for independent confirmation. Current repeated benchmark does not establish new independent generalization. Full artifacts and limitations retained.
