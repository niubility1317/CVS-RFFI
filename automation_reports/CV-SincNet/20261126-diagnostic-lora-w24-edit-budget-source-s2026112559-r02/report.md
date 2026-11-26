# LoRa W24 同等剂量与修正上限源域诊断（仅 Full）

- run_id：`20261126-diagnostic-lora-w24-edit-budget-source-s2026112559-r02`
- group_id：`lora-w24-performance-mechanism`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Compare frozen learned half readout with source-only exact-noise controls at learned magnitude, at model cap, and with complete center-only subtraction.

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


## 运行与评分结果

**VERIFIED。** 固定 W24 E104 和 40,000 步首尾联合损失恢复器，对 300 个源 V 包执行 Full 的 clean 与六种 practical。启动时用真实源 U IQ 和原 checkpoint 复核归档输入。直通和学习半量与原冻结源预测逐包一致。全量预测固定后，独立进程连接源 V 标签。没有目标读取、参数更新或候选选择。精确噪声仅用于隔离对照。先前双路线 r01 已按用户最新范围指示停止，保留检查和日志，r02 实际只处理 Full。

| 读出 | 六景 / 1,800 | 低城区 / 300 | Clean / 300 |
| --- | ---: | ---: | ---: |
| 直通 | 1,394 | 150 | 297 |
| 原首尾联合损失半量 | 1,441 | 177 | 297 |
| 同等强度、精确噪声投影方向 | 1,524 | 190 | 297 |
| 精确方向、27.5% RMS 上限 | 1,586 | 205 | 297 |
| 精确投影方向、放开 RMS 上限 | 1,673 | 255 | 297 |
| 完整扣除中心新增噪声 | 1,680 | 259 | 297 |

同等强度对照相对学习半量，六景增加 83 次，按 RX 分层 95% 配对区间为 [59，108]。低城区增加 13 次，区间 [1，26]。1,800 个受扰观测的等强度方向均可定义，仅 2 个会跨过投影后噪声自身强度。clean 的新增噪声为 0，隔离对照恒等，仅报告 clean 正确数。

同为精确投影方向时，放开 27.5% 上限使六景增加 87 次，区间 [69，105]；低城区增加 50 次，区间 [37，64]。再允许扣除逐 chirp 跟踪坐标中原先保留的直流分量，六景和低城区分别增加 7、4 次，两个区间均跨 0。中心完整精确扣噪只用于说明这一源面板的可恢复信息。

低城区实际编辑强度中位数为实收中心 RMS 的 17.77%，投影噪声需要 29.13%。300 个包中有 154 个投影后所需强度超过 27.5%，实际学习编辑有 86 个已达到上限的 95%。学习编辑与理想投影方向的余弦相似度中位数为 0.9634。证据支持优先检验按实收质量放宽回写强度，同时在轻扰动和 clean 包保留小编辑。实际学习方向仍有可测误差，达到精确扣噪实验的正确数需要超出强度调整的改进。

源域产物位于 N607 本运行的 `full_native` 子目录，本机复核保存 `local_artifacts/20261126-diagnostic-lora-w24-edit-budget-source-s2026112559-r02/source-scores.json`。
