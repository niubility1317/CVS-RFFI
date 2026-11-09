# DURA 强扰动变化保持源训练

- run_id：`20261109-phase1-dura-cumulant-response10-source-s2026092701-r01`
- group_id：`phase1-dura-cumulant-response10`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

匹配 1% 真实模拟信道变化保持版，唯一改变固定人造扰动为 clean RMS 的 10%。

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

## 固定幅度对照

对照在跑的 1% RMS 源人工扰动版：相同 32,768 个合法源记录、21/22/23 三个信道视图、初始化种子、网络、优化器、20 轮及损失权重；只把四种人工方向的幅度从 clean RMS 的 1% 调到 10%，匹配模拟器作用和响应目标同步缩放。原始接收噪声、模拟信道和接收机状态固定，不重跑完整模拟或 AGC。

源域先前可见度诊断中，1% 人工方向在低城市场景相对观测残差的中位数约为 −38 至 −33 dB。固定线性/共轭线性 action 下幅度放大 10 倍会使此能量比增加约 20 dB；残差含噪声与模拟误差，该量不是 SNR。10% 人工扰动未证明是真实身份差异，也可能偏离实际分布，必须以七景测试判断。

代码 d3f60d4f6570c032ad1fb76df65ac464f9f1dbed；远端两项扰动 RMS 和 checkpoint 拒绝检查通过，独立 P0/P1 静态审查无阻断。GPU0 可用显存与总计算进程数在每阶段检查，保留其他 owner 进程；首步执行真实模型源样本 smoke。源训练不读 F/TX/V/目标，全部七景预测完成后才进行独立 truth-last。
