# 困难信道加权的 Tri-SIG-TIME 语义恢复

- run_id：`20261101-phase1-trisigtime-hard-semantic-E30-s2026092701-r01`
- group_id：`phase1-trisigtime-hard-semantic-flow`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

相同 Phase1 源域与生成轨迹模型，训练时以模拟源噪声方差识别困难信道并加权，同时使用冻结 Tri-SIG-TIME 分类分布指导。

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

## 已核实状态（2026-11-01）

Phase1 R07 第 1 个源视图的源训练方差最高四分位阈值为 0.004452520050108433；对应 15750 条源记录，包含全部 10276 条源模拟 BLOCKED 和 8949 条源 practical_low_urban 中的 6223 条。该统计仅用于设计源训练，不代表 Phase1 测试分布。

候选对源 received variance 达到上述固定阈值的样本赋予权重 3，其余为 1；当输入替换成 clean 时权重强制为 1。每批按权重和归一化；推理只接收信号，不接收 variance、scene、state、lock 或配对 clean。其余语义双轨目标、种子及 E30/14790 预算与上一候选对应。

实际 release 版本 `c67eb148d92bf0fa46443daf9f76767c70631efd`，归档两端 SHA-256 一致。真实 Tri-SIG-TIME SSL 与源样本 smoke 通过：语义 KL 梯度范数 39.1355，基座冻结，checkpoint 严格重载，权重及 clean 排除规则通过检查；未读取 TX 标签或测试 IQ。

控制器 PID 2194052 已独立读回，正在等待前序语义候选的控制器和 GPU worker 退出；本候选尚未开始训练和测试。自动队列按固定配置继续，不基于前序测试指标调参。

## 固定困难权重的源视图覆盖核查

只读检查 R07 的 20 个源训练视图 `variance.npy` 和 `groups.npz`，未读取 IQ 或标签。固定阈值在各视图覆盖全部源样本的 24.62% 至 25.18%，覆盖各视图源 BLOCKED 样本的 100%，覆盖源 practical_low_urban 样本的 69.18% 至 71.05%。25% clean 输入替换后按实现重新设为权重 1，因此实际每轮加权 received 数量以训练日志为准。该检查只支持训练设计覆盖，不证明测试性能。

## 实际训练启动

前序分类分布候选及其 GPU worker 退出后，预先登记的困难样本加权候选已在 GPU 1 开始源训练；控制器 PID 2194052、worker PID 2259761 和 GPU 归属均独立核实。第一轮完成 493 次更新，其中 11849 条受扰样本达到预先固定的源方差阈值并获得权重 3；该权重不在推理中使用。冻结基座为 Phase1 Tri-SIG-TIME SSL，配置和 30 轮预算维持登记值。

此前 Phase1 开发测试结果已被观察，后续同一测试集不属于独立盲测；当前没有本候选的测试性能。
