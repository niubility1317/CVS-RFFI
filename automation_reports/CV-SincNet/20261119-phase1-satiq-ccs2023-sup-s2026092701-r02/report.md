# SatIQ CCS 2023 supervised 100-epoch beam comparison

按用户要求，三组采用 100 个完整数据 epoch。PAMF-SUP 适应波束标签，保留共用输入尺度和收发端角色隔离。原 20,000 步版本的 GPU 派发已在启动训练前被替换，原 CPU 准备继续生成三组共享的缓存。[数据与方法](../../../docs/PAMF_SATIQ_CCS2023_SUPERVISED_20261119.md)

一轮为 28,225 个 8 类 × 4 包批次，共 903,200 条互异物理记录。三组用相同的逐轮样本顺序。单行训练预算为 2,822,500 次更新。首轮内的第 5,000 和 20,000 步记录源验证，之后在配置的轮末继续验证。增强关闭，归一化使用逐包 I/Q 联合 min–max。

三组为 SATIQ、PAMF-SUP 和 PAMF-Metric。PAMF-SUP 将公共同步字段与全速率分支分别单位化再融合，使用同星竞争 CE 和同包两路 KL，另记录融合与分支 CE。PAMF-Metric 另加半难三元组。SATIQ 使用论文自编码器重构 MSE 和半难三元组，不接 CE 头。全部从零初始化。

2026 年 11 月 19 日 18:50（香港时间）读回确认 N607 监督调度进程 `1925785` 已建立，工作目录为 `/home/szu2070436088/2510044040/CV-SincNet/releases/20261119-phase1-satiq-ccs2023-sup-s2026092701-r02`。正式 GPU 训练等待已验证的完整缓存。旧 CPU 准备进程 `1920076` 继续向公共数据目录写缓存，GPU 派发入口以 `STOPPED_BEFORE_GPU.txt` 保留替代证据。[登记](experiment.json) · [状态](events.jsonl)

- run_id：`20261119-phase1-satiq-ccs2023-sup-s2026092701-r02`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Paper SatIQ vs SatIQ-adapted PAMF-SUP and metric control, all supervised and on the same 100 source data epochs.

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
