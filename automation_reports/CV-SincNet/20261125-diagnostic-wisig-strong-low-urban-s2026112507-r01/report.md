# 强 Practical Full 低城区的源 V 独立复核

- run_id：`20261125-diagnostic-wisig-strong-low-urban-s2026112507-r01`
- group_id：`wisig-strong-urban-estimation`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在预留的 60 个源 V 包对比上一版、十类全包版和从源 U IQ 学习先验的平均估计。带强中城区和默认低城区对照。

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

## 保留源 V 独立复核

N607 状态 VERIFIED。新评估覆盖 60 个未在十类实验使用的源 V 物理包，按源接收机与日期分层。三套方法的完整估计输出都固定后，独立读取模拟器无噪波形和方差。开发只读取 128 个源 U 配对 IQ。

下表顺序为上一版 Guarded → 十类全包版 Adaptive → 低信噪比先验版。IQ-only 校准选出的先验混合系数为 0。专门先验版保留全包估计路径，因此同包总体和尾部结果一致。

| 条件 | 作用 NMSE 中位数（%） | 作用 NMSE P90（%） | 方差误差中位数（dB） | 方差误差 P90（dB） |
|---|---|---|---|---|
| 强 Practical / Full / Low Urban | 14.8 → 3.96 → 3.96 | 60.4 → 18.4 → 18.4 | 0.52 → 0.274 → 0.274 | 1.56 → 0.591 → 0.591 |
| 强 Practical / Full / Mid Urban | 0.37 → 0.0804 → 0.0804 | 34.1 → 12.3 → 12.3 | 0.643 → 0.174 → 0.174 | 1.55 → 0.568 → 0.568 |
| 默认 Practical / Full / Low Urban | 3.68 → 1.06 → 1.06 | 17.7 → 4.79 → 4.79 | 0.568 → 0.257 → 0.257 | 2.14 → 0.802 → 0.802 |

主场景中，全包版相对 Guarded 的作用中位误差和 P90 分别改善 3.73 和 3.28 倍，方差中位误差改善 1.90 倍。物理包与十类主验证互斥。
