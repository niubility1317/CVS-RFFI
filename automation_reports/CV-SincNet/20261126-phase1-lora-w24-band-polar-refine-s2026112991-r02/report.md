# LoRa W24 回写幅相细化与直角坐标对照（Full）

- run_id：`20261126-phase1-lora-w24-band-polar-refine-s2026112991-r02`
- group_id：`phase1-lora-w24-restoration-performance`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（Full 源域细化）

## 目的与对照

Freeze the source backbone. Compare equal-size Cartesian and polar narrow-band post-MAP refiners at fixed edit energy.

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


## 预登记对照

源域隔离在固定能量下分别验证了相位与相对幅度的作用。本次固定合法源 U 训练的首尾 DURA 40,000 步权重，冻结两步 MAP、原生前端和接收估计精度。仅在最后一次 MAP 后增加两个同参数量的匹配头。输入是实收频谱、原 DURA 修正频谱及实收相干度。每条 chirp 的正负 9–32 kHz 分开做相同卷积，始终使用当前包。两组分别输出直角坐标复乘子和对数幅度／相位乘子，输出层从零开始。回写在实收 IQ 尺度中按包保持原 9–32 kHz 能量，带外与原生边缘保持。仍以中心 0.5 倍修正作为候选读出。

两头同种子、共同合法训练划分、通用扰动和 1,200 步预算，保留原损失，各加 0.25 倍的带内逐系数复误差，配对目标归一到原可用的回写能量。新增损失仅用源 U 同条 clean 的两次通用扰动，排除恒等与合成自然差异探针。不读取实用信道透传真值、W24、V 或目标。

源域使用同一固定 Full 七景面板，父版盲预测须复现。幅相头半量六景须超过直通、从零长训半量、首尾父版半量与直角坐标半量，低城区至少持平四者，clean 至少持平直通。通过即自动完成 Full 三个正式分片，预测冻结后独立 truth-last。本次另有并行的第二次 MAP 先验时域／频域对照，当前等能量末级头属于独立机制。

## 发布核验

执行提交 `847e6710d418fbb49adaa4fe5eae5210ab727a3d`。r01 的双视图批维错误已复现、修正并通过定向复核，20 项相关测试通过。发布归档和编译 VERIFIED。控制器 PID 847184，训练 PID 847607 和 847727，工作目录与完整命令均匹配，物理 GPU 分别为 6 和 7，各占用 1,830 MiB。两组真实源 U IQ 无查询冒烟通过，总参数 1,277,248，仅 6,210 个频带头参数可训练。父恢复器冻结，合法 40,000 步父权重已严格加载。W24、TX、V、目标、实用信道和轨迹读取为 0。两臂日志分别增长至 150 和 50 步。本机独立评分等待完整正式预测。
