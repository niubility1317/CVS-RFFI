# DURA 四阶统计同记录排序七景测试

- run_id：`20261109-phase1-dura-cumulant-rank-ground-test-s2026092701-r01`
- group_id：`phase1-dura-cumulant-rank`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（七景预测）；具体进度见 events.jsonl。

## 目的与对照

匹配四阶统计基线架构与预算，在合法源配对 IQ 上额外优化功率与四阶统计的同物理记录排序。

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

## 固定方案与比较

与同期四阶统计恢复器使用相同 14966274 参数架构、模型种子、32768 源物理 ID、每条三视图、数据顺序、20 轮、batch32、AdamW 和 DURA 基准；本模型重新独立初始化，不加载对照权重。原 IQ、DURA 改变量和四阶统计损失保持；额外加入 0.002 倍源物理记录排序损失，温度 0.5。

用固定的前 512 条合法源干净 IQ 分别标准化 32 频带功率和 21 维多时滞四阶特征；训练中恢复波形匹配同一条源物理记录的干净特征，出现多个视图时均作为正样本。该方法区分的是源记录，不是 TX 身份；其他不同记录可能来自同一 TX。参考特征只在源训练计算，推理不使用干净参考。训练不读取 TX 标签、F 权重或输出、V、目标 IQ 或目标真值。

GPU7 每阶段至少 8192 MiB 空闲且总计算进程少于 3，本任务同卡最多一 worker；真实源 IQ smoke 及 112 条源记录冻结识别器无查询 smoke 后自动训练、七景预测和独立 truth-last。固定 Tri-SIG-TIME SSL Main split 默认头，每景 168000 条，报告与 DURA 配对纠错、TX/RX/day 及相对 raw clean 3% 和绝对 3 个百分点门槛。已曝光 Phase1 地面模拟结果不是新盲测或真实在轨。

代码 1e83c8f23c837cf6ff68c218d2934cad4403411b；精确命令、数据、输出及停止规则见 experiment.json。5 项聚焦检查通过，一次独立 P0/P1 静态审查无阻断；服务器预检和识别结果尚待核实。

## 正式预测启动

最终第 20 轮冻结后，七景预测 PID 2522111 已核实在固定发布目录和 GPU7 运行；全部预测完成后才独立 truth-last。当前没有可报告的识别准确率。
