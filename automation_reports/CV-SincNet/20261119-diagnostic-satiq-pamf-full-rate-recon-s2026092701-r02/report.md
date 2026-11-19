# SatIQ PAMF full-rate waveform reconstruction diagnostic

- run_id：`20261119-diagnostic-satiq-pamf-full-rate-recon-s2026092701-r02`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Compare a full-rate branch waveform reconstruction auxiliary to original PAMF-Metric on the identical first source epoch and packet order.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

重构解码器从全速率的单位 160 维向量输出 704 维 I/Q，覆盖 11 个符号，每个符号每通道保留 32 个有序平均值。训练目标取同一个实测包的 RMS 归一化波形。推理和验证仍使用 320 维身份向量和 3,103 类分类输出。随机组批、物理样本顺序、共享参数初值及其它损失均与正式 PAMF-Metric 对齐。

N607 实测三个源 L 批次，比较初始化及正式 PAMF-Metric 的三个冻结检查点。辅助 MSE 权重为 1 时，投影层梯度范数为原监督目标的 0.108% 至 0.131%。在启动前锁定固定权重 128，对应抽查时 13.8% 至 16.7% 的量级。该校准只使用 L 上的梯度。正式训练仍从零初始化，共享参数逐张量核对相等。

## 交接

N607 物理 GPU 5 的 PID 1975395 已通过实包检查，固定 28,225 步开始执行。日志独立记录重构 MSE、各项 CE、KL 和三元组损失，源验证点为 5,000、20,000 和 28,225 步。[正式对照](../20261119-phase1-satiq-ccs2023-sup-s2026092701-r03/report.md)

## 第 5,000 步源验证

| 方法 | 同星 AUC K1／K16 | K1／K16 Top-1 | 分类头 CE |
|---|---:|---:|---:|
| PAMF-Metric | 0.5012／0.4908 | 0.094%／0.063% | 8.0375 |
| 全速率重构 | 0.5011／0.4919 | 0.102%／0.086% | 8.0420 |

重构训练 MSE 从首个 100 步窗口的 0.1428 降至第 5,000 步的 0.0196。固定 512 个源验证包的抽查中，全速率向量的不同包平均余弦从 0.999603 降至 0.989123，但归一化表示的有效维度约为 1.09。当前重构引入了波形变化，尚未带来可辨认的波束收益。后续检查点继续执行。

## 第 20,000 步源验证

重构版同星 AUC 为 K1 0.5072、K16 0.5155，分类头 CE 为 7.8826。相同步数的原 PAMF-Metric 为 0.5170、0.5121 和 7.8161。K16 略升，K1 和 CE 较差。模型仍在执行固定一轮诊断。
