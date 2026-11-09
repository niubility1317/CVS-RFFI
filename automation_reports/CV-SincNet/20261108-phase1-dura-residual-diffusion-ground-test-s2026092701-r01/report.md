# DURA 条件残差扩散七景测试

- run_id：`20261108-phase1-dura-residual-diffusion-ground-test-s2026092701-r01`
- group_id：`phase1-dura-residual-diffusion`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

以冻结 DURA、当前接收 IQ 与物理创新为条件，训练 clean−DURA 残差的噪声去除网络；固定八步、四条反向配对轨迹平均成单条 IQ。

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

## 方法与固定实验矩阵

冻结 DURA v1.0，独立初始化 1790786 参数的残差去噪网络。源训练中，以 clean−DURA 的归一化残差为目标，随机抽取时间和高斯噪声，使用余弦方差保留过程生成带噪残差；网络读取当前 received、DURA、物理创新、方差、带噪残差和时间，直接预测 clean 残差。目标为源 IQ 归一化误差加 0.02 倍 DURA 改变量。部署从四个固定反向配对噪声开始，各做 8 次固定反向更新并平均为单条 IQ；不声称有限轨迹构成校准后验或保证身份保真。

固定 32768 条合法源物理 ID × 三个源视图 × 20 轮，batch32，AdamW 0.0003，最终轮冻结；源训练不读取 TX 标签、F 权重／输出、V、目标 IQ 或真值。模型 seed 2026110817、源记录 seed 2026110611、顺序 seed 2026110813、时间 seed 2026110818、训练噪声 seed 2026110819、固定推理噪声 seed 2026110820。相同记录的三个视图不计为独立物理样本。

GPU0 每个 GPU 阶段需要至少 8192 MiB 空闲、总计算进程少于 3，本任务同卡最多一个 worker；真实源 IQ 和 DURA 的六步 GPU 预检、112 条源记录冻结 Tri-SIG-TIME SSL 无查询检查均通过后自动训练。固定后按 Phase1 Main split 原默认头预测 clean 和六个 practical 场景，每景 168000 条；全部预测固定后独立 truth-last，报告与 DURA 的配对纠错、逐 TX/RX/day、raw clean 的相对 3% 与绝对 3 个百分点门槛。目标结果不回流训练；测试已曝光，不是新盲测或真实卫星。

源权限错误、输出冲突、真实权重 smoke 失败、非有限值或系统技术失败时停止本 run 并保留产物。代码 commit c48f5bebc921b4e2fd97341cb8eebcf6a627dbe2，精确命令、环境、路径和预期产物见 experiment.json。五项聚焦检查通过，包括正反向时间轨迹、固定噪声及逐记录独立性、错误 checkpoint 拒绝和 GPU 梯度；一次独立 P0/P1 审查无阻断。服务器真实预检和正式结果尚待执行。

2026-11-08：真实源数据 GPU 预检和 112 条源记录冻结识别器检查通过；独立源训练已启动，固定最终轮后自动七景预测和独立评分。暂无正式识别结果。

2026-11-08：条件扩散模型最终第 20 轮已冻结，原控制器自动启动七景预测。N607 GPU0 预测 PID 1741893 的工作目录、命令和 GPU 占用已核实；其他 owner 的同卡任务保持运行。七景尚未全部预测和独立评分，目前没有正式识别结果。

## 七景最终评分

七景各 168000 条预测固定后，已完成独立 truth-last 与 DURA 配对评分。六个受损场景均低于 DURA，均未达到相对下降 3% 门槛。逐场景、配对纠错及 TX/RX/day 分组见 [结果分析](analysis.md)。测试为已曝光的 Phase1 地面模拟结果。
