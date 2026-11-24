# WiSig 上 LoRa 两版恢复网络的跨协议与源域重训对照

- run_id：`20261124-phase1-wisig-lora-recovery-comparison-m4-r01`
- group_id：`wisig-lora-recovery-comparison`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定 LoRa 扩容和秩四弱扰动门各做跨协议原权重诊断及 WiSig 源 U 重训；同包对比直通及冻结 R69，分别标注模型和数据祖先。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 本次方法与接收信息

原权重诊断将同一 WiSig 256 点窗口复制至六个 LoRa 输入位置，依原 LoRa 去扫、门控、秩四或展开求解规则输出，再对六个恢复结果取均值。六份只是同一观测的计算表示；不视为六个独立窗口。原 LoRa 数据契约与 WiSig 不同，这两行作为独立跨数据诊断，不标成 WiSig 匹配源训练。

两个 WiSig 重训模型从随机参数开始，使用 52,604 个 TX 隐藏源 U 包和其独立通用 IQ 扰动，每臂 40,000 步、batch 16。4,096 个互斥源 U 包校准门限。扩容组保留两阶段先验和求解，以实收 WiSig 短训练字段的周期差分估计有效方差；秩四组以单个观测做秩四频谱条件更新，并用源 U 训练 WiSig 弱扰动门。两组均以 0.5 恢复增量读出。WiSig 窗口副本不增加物理样本数，重训秩四条件更新不将副本当作独立噪声观测。

训练、门限和推理只读取源 U 或当前固定接收 IQ，以及由 IQ 计算的估计量；不提供模拟器的真实信道、真实噪声方差、逐点噪声或轨迹。DURA v1.0 为冻结对照，其历史上游作用估计器曾使用模拟响应与噪声方差监督，必须随结果披露。固定 Full 和 Residual 两条路线、两个 Main split 划分、clean 与六种 practical、三个冻结识别器种子；完成 16 个候选预测分片后独立 truth-last，与同包 R69 和直通比较。新模型各自仅一个恢复器训练 seed，三个是识别器 seed。

## 准备状态

代码处于本地准备和验证阶段。真实源权重及真实源 IQ smoke、训练、目标预测和评分均尚未执行。N607 的 8 张 GPU 已观察到其他计算进程；资源共享策略等待用户选择。本实验没有读取目标真值，没有启动 N607 任务。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
