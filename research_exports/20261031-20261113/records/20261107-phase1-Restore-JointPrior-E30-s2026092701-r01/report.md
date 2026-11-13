# 条件均值与耦合协方差联合恢复器，从零单阶段 30 轮

- run_id：`20261107-phase1-Restore-JointPrior-E30-s2026092701-r01`
- group_id：`joint-source-mean-coupled-covariance-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在固定耦合协方差结构上增加共享上下文的130参数均值输出头，按原先验逐坐标标准差限幅，仍经同一物理后验恢复IQ。

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

## [计算环境] 联合先验恢复器发布与实际排队

VERIFIED / QUEUED。联合均值/协方差版已发布，独立读回实际控制器PID/CWD/命令、CUDA为空、无子进程、等待日志增长及零GPU占用。代码4bba9df3、发布包SHA对比与远端编译通过，11项针对性检查及一次P0/P1审查通过。固定等待Coupled版最后GPU终态，再按GPU1正常总数最多3个、已有最多2个且剩余至少12000 MiB准入；不读取其分数，不影响其他owner。新增130参数的先验均值头与原耦合协方差共用上下文，初始输出一致，均值按原先验每坐标1个标准差限幅。原L_s/U_s、实际带噪IQ/STFT/连续冻结F0损失、15global+15otherRX、从零单阶段30/7410保持。真实source2/F0/均值及协方差梯度/E0检查仍待GPU训练首步，PASS后直接继续完整训练与Main七景；尚无本版性能结果。 发布后独立证据

## 联合先验版真实训练启动

VERIFIED / RUNNING。联合先验版已按GPU1正常总数最多3个准入，实际worker的PID/CWD/原科学命令/GPU与首轮日志增长核实。真实原F0、完整后验、E0重载、梯度隔离、协方差恒等初始化和均值零初始化检查通过；源连续特征损失对均值头产生非零梯度。观察时已完成1轮/247步。原从零单阶段30轮/7410步和完整Main七景保持，不以首轮损失宣布识别收益。 启动证据

## 完整30轮与原Main七景独立复算完成

VERIFIED / ANALYZED。固定scratch30/7410、全部源日志/顺序和final=E30核实；原Main42单元全部预测后独立truth-last，4阶段exit0且进程均已退出。1302组/868raw配对及DURA/clean配对复算PASS，最大差1.29e-14，baseline判决差0。六景84.717857/83.111905/81.377976/83.225000/75.116071/66.709524%，1/6；clean两个总体净+187/+13，联合验收False。原科学代码/模型/数据/预算保持，默认DURA不变，其他已登记候选继续。 完整结果
