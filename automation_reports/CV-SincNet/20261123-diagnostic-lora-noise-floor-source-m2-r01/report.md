# LoRa 源域噪声估计与复谱收缩归因

- run_id：`20261123-diagnostic-lora-noise-floor-source-m2-r01`
- group_id：`lora-observable-satellite-recovery`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定原前端、门限、物理包和冻结识别器，隔离对比外带估计、精确总功率、精确逐频点功率及精确扣噪。

复用原局部 Wiener 的 900 个同序源 V 物理包、原源 U 阈值、256 kS/s 窄窗追踪和 Tri-set seed 2026092702。保持原五路冻结预测。额外诊断保持相同的触发包、坐标、短时谱分组与协方差窗。第一路只把外带估计换成逐包真实总 STFT 功率。第二路用真实逐频点功率先白化再求解，保留相同的公共和独有方差比例。后两路分别精确扣除非载波噪声与全部逐点噪声，用于核查加性损伤可解释的范围和保载波限制。

逐点噪声和功率只在该诊断进程内可见，结果仅用于归因。前端、门、恢复权重和原预测矩阵均无更新。频带诊断以扣去真实加性噪声的接收波形为参照，保留原信道作用，用相同的观测载波与 RMS 坐标核对复幅相。输出按 1–8、9–32、33–64 和 65–128 kHz 的双侧频带报告每包归一化复误差、幅度误差和归一化复相关实部的中位数。预测全部冻结后，由独立 CPU 使用源 V 真值评分。 [原局部 Wiener 对照](../20261123-phase1-lora-local-wiener-source-m2-r01/report.md)

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
