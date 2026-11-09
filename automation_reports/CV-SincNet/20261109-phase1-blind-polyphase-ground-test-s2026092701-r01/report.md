# 多相加速盲信道恢复七景测试

- run_id：`20261109-phase1-blind-polyphase-ground-test-s2026092701-r01`
- group_id：`phase1-blind-polyphase`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

独立训练与盲信道展开恢复同结构、同源配置的多相卷积实现；浮点轨迹和最终预测需要独立验证。

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

## 多相执行与固定实验

本运行从零独立训练，与原始盲信道恢复使用相同结构、源数据、种子、20 轮、batch32、优化器和损失；其中 23 个扩张卷积使用数学等价的多相计算。原运行保持健康并继续，不加载或覆盖其权重及输出。推理仍只接收当前 IQ，训练不读取 DURA、F、TX 标签、V 或目标。

GPU3 的合成随机输入对比，在热身后各测 4 步：原始实现平均 2.5634 秒／步，多相实现平均 0.0958 秒／步，约 26.8 倍；单次对比输出最大绝对差 2.39e-7，梯度最大绝对差 1.50e-8。这是合成批次计时，并非完整训练或真实测试端到端加速；浮点归约不同，不保证训练轨迹、最终权重或正式预测完全相同。

固定 GPU3，每阶段至少 8192 MiB 空闲、少于 3 个 GPU 计算进程、本任务同卡最多一 worker；真实源数据与权重 smoke、112 条源记录冻结基准无查询 smoke 通过后，自动 20 轮训练、七景逐条预测和独立 truth-last。固定 Tri-SIG-TIME SSL Main split，每景 168000 条，与 DURA 配对并报告 TX/RX/day；已曝光地面模拟不声称盲测或真实在轨。

代码 cd0687eed0fa69db343bb6000b486143698fe797，运行命令和停止规则见 experiment.json。局部输出／梯度和错误来源拒绝检查通过，一次独立 P0/P1 审查无阻断；服务器执行待验证。
