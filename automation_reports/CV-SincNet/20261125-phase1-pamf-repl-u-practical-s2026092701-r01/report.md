# PAMF Main split 无标签 practical 单因素对照

- run_id：`20261125-phase1-pamf-repl-u-practical-s2026092701-r01`
- group_id：`phase1-pamf-repl-u-practical`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定三视图 PAMF 标准主法，仅替换完整 REPL 的 U As 半批 IQ。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

最终评分和门槛判定见文末。
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

## N607 发布读回

VERIFIED。代码提交 `df585e9aa9ccdeb4a2674bf047bcad9ebbf069c4`，发布归档校验与远端编译通过。真实 E20 权重源域冒烟通过，零目标读取、零优化器更新，128 条 practical As 覆盖三个场景。GPU 7 的正式训练已从 E21 开始，核对了运行目录、命令行和 GPU。

读回 E21 已记录的前 52 个优化步，6656 条 U As 输入实际使用 practical，原始 IQ As 计数为 0。已登记的冻结预测和独立评分将在 E50 后自动依次执行。

## 最终结果

VERIFIED，N607 于 2026-11-25 23:45（香港时间）完成 E50/11100 步冻结、三个人群的两路径七景预测和独立 truth-last 评分。种子 2026092701。新增 6660 步中，850500 条 U As 输入实际使用同包 practical 缓存。

| 跨接收机指标 | 基线准确率 | 候选准确率 | 差值（百分点） | 门槛结果 |
| --- | ---: | ---: | ---: | --- |
| Clean 直接识别 | 88.11% | 87.87% | −0.24 | 通过 |
| Low urban 直接识别 | 51.95% | 54.75% | +2.80 | 通过 |
| 六景直接识别均值 | 71.39% | 73.62% | +2.23 | 通过 |
| DURA-LU 六景均值 | 80.26% | 79.02% | −1.24 | 未通过 |

跨接收机直接识别六景逐项增益为 +1.79、+2.13、+2.18、+2.00、+2.46、+2.80 个百分点。场景顺序为 high suburban、mid suburban、low suburban、high urban、mid urban、low urban。接入 DURA-LU 后六景逐项差值为 −1.41、−1.28、−1.22、−1.46、−1.15、−0.91 个百分点。跨接收机直接识别六景 Macro-F1 从 71.18% 升至 73.35%，DURA-LU 六景 Macro-F1 从 80.19% 降至 78.98%。

跨日期直接识别六景均值提高 2.10 个百分点，DURA-LU 下降 0.07 个百分点。双跨直接识别六景均值提高 2.49 个百分点，DURA-LU 下降 0.04 个百分点。

原预登记四项条件通过三项，单种子整体晋级门槛未通过。结果支持 practical U As 增强公式下的直接识别收益，恢复后路线的跨接收机均值未保住。N607 原 run 的 `paired_scores.csv`、`complete.json` 和逐行评分文件保留完整证据。
