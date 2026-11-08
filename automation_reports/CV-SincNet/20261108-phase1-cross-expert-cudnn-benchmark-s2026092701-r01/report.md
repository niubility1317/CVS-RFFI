# 时频网络确定性卷积算法训练对照

- run_id：`20261108-phase1-cross-expert-cudnn-benchmark-s2026092701-r01`
- group_id：`phase1-cross-expert-cudnn-benchmark`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定模型、源输入、批次和精度，比较 cuDNN 算法自动选择关闭或开启的短程训练耗时与逐值训练状态。

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

## 后台训练对照

固定代码 commit `5fd6f3008776ab93a09a32168198f44cc0e7a95f`。真实 DURA 权重与固定合法源 IQ 冒烟通过；N607 GPU2 启动前空闲 18,501 MiB，存在两个其他 owner 计算进程。队列 PID 1239510 与 GPU worker PID 1239524、实际工作目录和日志已核实。三项针对性检查和独立 P0/P1 审查通过，发布包两端校验一致、远端编译通过。

对照关闭或启用 cuDNN 算法自动选择，同一 512 源 ID × 三视图、相同种子和批次，分别热身并完成两对、每条件四步；逐值比较损失、模型、优化器以及 CPU/GPU/采样随机状态。模型不保存，不读取 F/TX/V/目标。该实验不同于仍在 GPU0 排队的数据驻留对照；不会修改任何已有训练或预测。共享 GPU 的其他负载会影响计时，短程状态相同也不等于完整训练和测试输出相同。

## 完成结果

VERIFIED / ANALYZED：两组四步对照均记录损失、模型、优化器及随机状态逐值一致；关闭→开启算法自动选择的耗时分别为 161.502→161.046 秒和 138.891→203.599 秒，顺序相反。共享 GPU 条件下没有稳定加速证据，不改变在训模型。详见 [analysis.md](analysis.md)。
