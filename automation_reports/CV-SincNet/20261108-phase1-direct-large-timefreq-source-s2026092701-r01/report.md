# 独立大容量时频恢复 GPU 训练

- run_id：`20261108-phase1-direct-large-timefreq-source-s2026092701-r01`
- group_id：`phase1-direct-large-timefreq`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；N607 GPU 2 源训练运行中。

## 目的与对照

从单条 received IQ 直接恢复；时域、频域联合全窗口注意力。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

预登记：从单条 received IQ 直接输出恢复波形。时域局部卷积与归一 FFT 频域卷积各形成 64 个 token，8 层、宽度 384、8 头全窗口注意力联合处理；全采样局部路径与注意力输出解码为有界残差。15557090 个参数从零训练，输出头零初始化回放 received，不继承 DURA 权重或训练输出。模型只读取单记录 received；DURA 在最终权重冻结后用于源 pilot 和地面配对对照。

固定合法 32768 个源 ID × 三视图，20 轮、batch 32、AdamW 学习率 0.0003，损失为 clean IQ 归一化误差加 0.02 倍输出相对 received 的改变量；训练不读取 F、TX 标签、V 或目标。第 20 轮固定后自动使用冻结 Tri-SIG-TIME SSL Main split 默认头进行 clean 和六种 practical 场景预测，每景 168000 条，七景全部固定后独立 truth-last，并报告与 DURA 配对及 TX/RX/day。GPU 2 预检空闲 20381 MiB、已有两个其他计算进程，队列再次检查容量，不影响其他 owner。发生数据契约错误、非有限值、错误 checkpoint、目标泄漏或不可恢复技术故障时，仅精确停止本 run 并保留证据；测试结果不回流选模。

八项聚焦检查、真实源 IQ 上的确定性 GPU 六步训练检查以及一次独立 P0/P1 审查通过。模型规模和源训练损失不构成识别性能结论。

远端 64 个源 ID × 三视图训练 smoke 已通过，确认训练阶段 DURA、F、TX 标签、V 和目标读取均为零；controller PID 542083、trainer PID 542283 已核实，正式训练日志已开始增长。原始输出仍按独立 run 保存。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
