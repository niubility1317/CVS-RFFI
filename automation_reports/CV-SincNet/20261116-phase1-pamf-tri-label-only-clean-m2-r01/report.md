# Tri 固定重排仅有标签监督 clean 对照

## 预登记对照

用户指定十类和六源设置、seed `2026092701` 的纯监督对照。两行均从零训练、仅打开 L 和 V，固定 E20+10+20，共 11100 步；三视图、固定重排 `[0,5,3,6,2,1,4,7]`、默认头、原监督 CE/对比/KL 及监督信道增强保留。没有 U 样本、伪标签或半监督损失。逐段检查 `U_actual/PL_seen/PL_accepted=0`。历史原 Tri E20 只用于隔离的无 query smoke，不作为训练初始化或祖先。

每行冻结 E50 后自动预测 clean 的 cross_date、cross_rx、cross_both，预测完成后独立 truth-last 评分，按同物理 ID 与既有普通 PL/原 Tri 比较。用户明确 clean-only，不跑星地。两种设置与单 seed 由用户指定，目标曾曝光，结果属于探索性比较；历史 PL 使用自然顺序，本对照采用 Tri 固定重排，解释差异时必须披露。

独立 run/output root；N607 GPU 2/3/4/5/7，发布时核查剩余显存及其他 owner 的进程，总进程不超过每卡 5。protocol/非有限/系统技术故障停止新调度，不干扰其他健康任务。

## 启动证据

**VERIFIED / RUNNING**：执行 commit `40bc82f9c311d388a1f967847f37ef145fc4bccd`；归档传输和远端编译通过，真实 checkpoint 无 query smoke 通过。独立检查 controller 与两个训练 worker 的 PID、CWD、命令和 GPU，十类在 GPU 2、六源在 GPU 3；初始日志均记录 `init=scratch`、U 训练样本 0、伪标签阶段 0。两行预测与评分尚未完成，必须等待实际冻结及独立评分。

## 既有自然顺序监督与 PL 对照

读取历史 `20261024-phase1-PAMF-REPL-completion-m201-r01` 的同 seed 冻结模型与独立 truth-last clean 评分；监督和 PL 同设置分别使用相同源合同、source RX/day 和样本量，均为 E50/11100 步。监督模型的记录为 `U_enabled=false`、U seen=0。以下主指标按 cross_rx 与 cross_both 的正确数及样本数加权；cross_date 单列。这些历史模型记录 `historical_posthoc_selection_disclosed=true`，而本 seed 也根据已曝光目标选取，因此仅作探索性比较。

|设置|自然顺序纯监督|普通 PL|监督减 PL|监督跨日期|PL 跨日期|
|---|---:|---:|---:|---:|---:|
|十类|83.9995%|84.1994%|−0.2000 pp|87.3813%|87.6403%|
|六源|86.6764%|86.4285%|+0.2479 pp|94.1139%|94.2667%|

该结果表明在所选单 seed、六源跨接收机合并口径中，普通 PL 本身已比纯监督低；不能预设只要超过 PL 就超过监督。十类自然顺序监督比 PL 低约 0.20 pp。新 run 的监督模型采用 Tri 固定重排，与历史自然顺序监督不同，待冻结评分后单列结果，不用历史分数替代其结果。原始评分、完成状态和源合同元数据副本在 `local_artifacts/pamf_tri_label_only_20261116/`。

- run_id：`20261116-phase1-pamf-tri-label-only-clean-m2-r01`
- group_id：`phase1-pamf-repl-weight-clean`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

十类及六源各一个 seed，从零全程仅 L 训练 E50；自动 clean 测试。

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
