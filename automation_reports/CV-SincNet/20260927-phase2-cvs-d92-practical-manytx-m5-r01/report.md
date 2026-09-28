# CVS最终clean/星地测试与D92 E0真实256维注册

- run_id：`20260927-phase2-cvs-d92-practical-manytx-m5-r01`
- group_id：`phase2-cvs-d92-true256-practical-matched`；类别：`cvs`；阶段：`Phase1-final-and-Phase2`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定Phase1五种子第200轮权重，复用与对比相同的已验证received IQ；D92 P2-256-FULL全2100划分，每seed另有21个frozen DG。source ground仅本次L生成。

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

## 固定版本与完成条件

D92固定P2-256-FULL，identity160＋FFT96真实256维；ground为本次最终模型在全部source L上重新生成的rank3低秩v2组件，4096-bin P90半径与D89可靠性谱。旧v1只用于内存中原codec输入，不是部署组件。无新类使用同一执行器旧类装配；新2类仅在原协方差实现中增加注册规模8，数学不变。详见docs/CVS_MATCHED_EXPERIMENTS_20260927.md。

CPU五lane等待各自Phase1完成。先核对final200来源并做真实checkpoint无query前向，再source L准备、v2ground、最终clean/星地预测、固定received特征、D92全部2100划分与21DG；全部五lane固定预测后才独立评分。训练失败或子进程异常不自动重跑、不读取部分目标评分来修方法。现有训练进程不改动。

验证：本地7项pytest通过（旧类回退、新2/K5、原新5实现数值一致、query分批/顺序不变、v2/P90ground、L物理角色与污染拒绝、Torch/NumPy旧桥禁用）。native来源负测通过，真实scratch checkpoint合成fixture的严格重载及clean/satellite无truth预测通过。纯合成D92 runtime smoke本地PASS，远端发布时执行一次，不依赖pytest。独立P0/P1审查PASS；唯一远端pytest依赖已改为工具smoke子命令并定点确认。

当前状态为LOCAL_VERIFIED，尚未提交远端依赖队列，不等同于已产生D92结果。

## 远端依赖队列

VERIFIED：release `cvs_d92_matched_20260927_r01`来自已push并独立读回的commit `7b5809a755bc641b129f5a703e82b95d422610d2`。远端Torch2.1/NumPy2.2环境的v2ground＋new0K1/new2K5真实256维合成smoke PASS。dispatcher PID1276770及五个CPU worker的PID/CWD/argv均已核实，五row为WAITING_SOURCE。待各自final200完成后自动执行；当前尚无D92性能结果。证据见evidence/launch_readback.json。

## 2026-09-28完成核查

PARTIAL_TECHNICAL_FAILURE。详见docs/CVS_RESULTS_STATUS_20260928.md及evidence/completion_status_20260928.json。五个Phase1完成200轮；D92三组完成预测、两组发生LDA部署系数一致性错误，未评分。

## Phase1独立评分修复（2026-09-28）

用户授权“phase 1的目标测试结果呢，修复问题”“继续完成”。在本次既有登记内完成已冻结的Phase1评分，不改变训练权重、seed、预测或数据。五row的final_eval均已完成168000条，旧dispatcher与五worker已退出。评分只依赖全部Phase1预测完成及其ID、类型、形状和类别范围校验；校验通过后才读final_truth。Phase2失败仍保留且不评分部分矩阵。

启动入口tools/publish_cvs_phase1_score_repair.py；发布cvs_phase1_score_repair_20260928_r01，CPU独立评分，launch owner为codex/root/phase1-score-repair。沿用experiment.json的全部路径与seed。输出原run/phase1_final_results.json，独占创建；启动参数与commit记录在phase1_score_repair_startup.json，日志phase1_score_repair.log。本地回归覆盖Phase2失败不阻断完整Phase1、Phase1未完成/ID错误时禁止读取truth、正确accuracy及已有结果禁止覆盖。
