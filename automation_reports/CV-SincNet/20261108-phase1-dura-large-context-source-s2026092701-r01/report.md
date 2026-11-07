# DURA 大容量全局上下文 GPU 训练

- run_id：`20261108-phase1-dura-large-context-source-s2026092701-r01`
- group_id：`phase1-dura-large-context`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；N607 GPU 0 源训练运行中。

## 目的与对照

冻结 DURA，全局注意力及全采样局部分支联合生成波形修正。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

预登记：冻结 DURA，使用单记录 received、DURA 输出、物理残差和可观测方差；全采样局部卷积与 64 token、8 层、宽度 384、8 头的全窗口注意力融合，生成噪声门控残差。新分支 14792706 个可训练参数，原 DURA 266951 参数冻结；零初始化严格回放 DURA。依据源域长距离相关检查提出，不以目标成绩选取方案；部署不依赖 clean、TX 标签或多记录输入。

固定合法 32768 源 ID × 三视图、20 轮、batch 32、AdamW 学习率 0.0003，最终轮冻结。训练不读 F 权重／输出、TX 标签、V 或目标；七景预测使用同一冻结 Tri-SIG-TIME SSL Main split 默认头，每景 168000 条、全部固定后独立 truth-last，与 DURA 逐条配对并统计 TX/RX/day。GPU 0 预检空闲 20889 MiB、2 个其他计算进程，队列再次检查容量；其他 owner 不受影响。发生数据契约错误、非有限值、目标泄漏、错误 checkpoint 或不可恢复技术故障时，精确停止本 run 并保留证据；不依据测试结果选模或中途终止。

9 项聚焦检查和真实源 IQ／DURA checkpoint 的确定性 GPU 六步检查通过；独立 P0/P1 审查发现的线性插值反向问题已在发布前改为确定性重复上采样并通过针对性复核。N607 的 64 ID × 三视图真实 DURA 源 smoke 已通过，确认 F/TX/V/目标读取为零、DURA 冻结；controller PID 519409、trainer PID 519508 已核实。正式训练已启动，尚无识别性能结果；模型大小不作为上限或效果证明。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
