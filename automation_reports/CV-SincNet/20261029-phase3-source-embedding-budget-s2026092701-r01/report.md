# PAMF第三阶段：十轮开放集特征增量与匹配监督

- run_id：`20261029-phase3-source-embedding-budget-s2026092701-r01`
- group_id：`pamf-phase3-source-embedding-budget`；类别：`cvs`；阶段：`Phase3-source-ten-epoch-open-embedding-budget`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

BothfromoriginalE50 independentcold CENTER_UNITS nativeL arms, fixed2220steps/total13320; OE21λ.1, new8branches+original4OE200readout, known originalF risk; no earlystop/noveltune.

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

实际code/config commit：`6ffccc4ac1bb80d2d47a77e08a8a288012fa1353`。36项聚焦tests、38个subtests通过，一次本候选独立P0/P1PASS。原222兼容保持，新2220支持10个native逻辑epoch、连续随机流；两行原E50独立cold启动，全局11101..13320，固定末步无V/novel早停。原L2/OE2实际Sinc/原head梯度smoke无step恢复state/RNG后即训练，原头joint、Uoff；冻结后ownL2与40652新8分支导出、各自L-only几何＋原4OE读出200。knownV仅校准原F总损伤2pp/FRR5%，阈值先于novel；7methods完整预测后独立truth-last、onlyMLP/完整两行判95及old保护。N607一次preflight、source21/4数据复用、不raw重验/新传数据，每卡新增≤2、不停foreign；单row失败保留健康另一行。

## 实际启动核实

VERIFIED_RUNNING。原E50实际L2梯度smoke两行PASS、OE2额外PASS，CPU/GPU路径与immutable root符合；driver2486087，train监督2486111/GPU1、OE2486112/GPU2，独立CWD/argv/CUDA/PID已读回。两份日志从73/138字节增长到19757/21283字节，观察到210/204步，owned显存各1496MB（观测非峰值）。无foreign停止，当前无性能结果；后续冻结、导出、读出、预测与评分由同一流水线自动完成。
