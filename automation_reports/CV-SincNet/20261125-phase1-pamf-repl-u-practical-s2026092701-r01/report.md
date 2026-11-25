# PAMF Main split 无标签 practical 单因素对照

- run_id：`20261125-phase1-pamf-repl-u-practical-s2026092701-r01`
- group_id：`phase1-pamf-repl-u-practical`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定三视图 PAMF 标准主法，仅替换完整 REPL 的 U As 半批 IQ。

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
## 变量与基线

Main split 的原三视图 Tri-SIG-TIME SSL，固定重排 [0,5,3,6,2,1,4,7]，模型种子 2026092701。对照复用同一 E20 权重起点、同一数据合同的既有 PAMF E50。E20/4440 步来自从零训练且未读目标的既有行。候选沿用原分段优化语义续跑 E21–30 和 E31–50，总计新增 30 轮 6660 步、累计 50 轮 11100 步。

REPL 原交替半批改用与 PAMF 源 U_s 物理 ID 逐项一致的 STAR 同包 practical residual。三场景为 low suburban、mid urban、low urban，缓存生成概率为 0.40/0.35/0.25，每个 ID 对应固定的模拟视图。辅助槽覆盖 E21 以后原 As 半批，每步复用原两次 CE 前向。保留 0.95 门控、原伪标签、MixStyle、原梯度保护、同一默认头、监督端 As 和原六次 U 前向。U 的 TX 真值保持隐藏。DURA-LU 只用于训练冻结后的读出。

## 测试与判定

本轮复用原 Main split 的 cross-date 30000、cross-RX 126000、cross-both 42000 条物理样本，以及匹配的 DURA-LU 输出缓存。候选固定 E50 后完成 raw 和 DURA-LU 的 clean 与六种 Practical，全部预测完成才独立 CPU 评分。报告同 seed、同 ID 的准确率、Macro-F1 和逐 RX、TX、日期结果。

首轮判断条款使用 cross-RX 的固定范围。raw low urban 至少 +2 个百分点，raw 六景均值不降，raw clean 和 DURA-LU 六景均值至多各降 0.5 个百分点。cross-date 和 cross-both 单列。一次种子筛查尚不足以确立跨种子稳定性。

## 发布前核对

本地检查通过同一物理 ID 取数、ordinary 半批完全一致、场景计数、拒绝含 TX 的 U 元数据、拒绝错位缓存及预测端拒绝真值字段。一次独立 P0/P1 审查未发现阻塞。N607 冒烟以真实 E20 权重完成一批仅源域的有限反向，随后立即进入正式续训。
