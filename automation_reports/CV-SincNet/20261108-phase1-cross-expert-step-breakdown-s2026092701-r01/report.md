# 大型时频恢复网络源域 GPU 训练分段计时

- run_id：`20261108-phase1-cross-expert-step-breakdown-s2026092701-r01`
- group_id：`phase1-cross-expert-step-breakdown`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定真实源输入和完整时频网络，测量数据搬运、前向、损失、反向、裁剪和更新的同步分段耗时。

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

## 后台训练计时

固定代码 commit `592e2cad2287fbd29e218d383da2fbf3afa1d9ea`；四项相关测试、编译和一次独立 P0/P1 审查通过，发布包两端一致、远端编译通过。GPU2 启动前空闲 18,501 MiB，存在两个其他 owner 计算进程；队列 PID 1270462、GPU worker PID 1270476 及实际工作目录已核实。

该基准只使用合法源缓存及冻结 DURA/direct，不读取 F、TX 标签、V 或目标，先运行真实 checkpoint 源样本检查，随后热身一步并将四个训练步分成数据搬运、前向、损失、反向、梯度裁剪和参数更新分别计时；不保存模型。每阶段同步会增加额外开销，GPU 共享负载也会影响结果，分段耗时不能冒充未插桩的全训练速度或训练加速效果。

## 计时结果

VERIFIED / ANALYZED：四步同步计时前向 200.329 秒、反向 2.245 秒、数据索引与传输 0.056 秒。共享 GPU 且阶段同步，不能推断完整训练耗时或归因到具体算子；详见 [analysis.md](analysis.md)。
