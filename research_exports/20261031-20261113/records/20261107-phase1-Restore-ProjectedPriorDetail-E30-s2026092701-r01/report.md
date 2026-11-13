# 后验约束的条件 IQ 先验，从零单阶段 30 轮

- run_id：`20261107-phase1-Restore-ProjectedPriorDetail-E30-s2026092701-r01`
- group_id：`source-supervised-observation-projected-prior-detail`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

细节网络提出条件先验均值，由物理后验投影再修正 IQ。源训练以同算子的无噪收缩误差监督。

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

## [计算环境] 后验约束条件先验版排队已核实


## 使用常规 GPU5 的原方案资源交接

原科学代码、数据、网络和从零30轮/7410步保持，独立正常GPU5容量允许与相位版并行；总进程最多3，启动前已有最多2且剩余显存至少12000 MiB。只在核实原队列无子进程、未提交任何训练或预测时交接；不读取其他候选得分来调度。原真实源检查、完整七景和独立truth-last照常执行。

控制代码：`d38f2dfc0378ae269f733b6f9a3ea039ddf11163`；科学代码：`8cf3669c4a5d98f6c4930117952203afd56f2f6e`。命令：`[已省略路径] -u -m experiments.pamf_phase3_restoration.projected_prior_gpu5_dispatch --root [已省略路径] --legacy-release [已省略路径] --cache [已省略路径] --commit d38f2dfc0378ae269f733b6f9a3ea039ddf11163`。

## 原匹配对照已实际训练

VERIFIED / RUNNING。原匹配对照已在正常GPU5从零启动，准入时已有2个计算进程、20381 MiB可用，总数上限3。真实原F0/完整后验、梯度隔离、严格E0重载和源2检查通过；原科学进程/CWD/GPU及日志增长已独立核实。只接替无子进程且未提交训练的旧队列，30轮与自动完整七景保持，尚无识别成绩。 运行核验

## 第 1 轮源训练核实

VERIFIED / RUNNING。日志记录63000条源记录、247步及63000次源F0特征前向；损失组成复算差8.02e-10。该轮无噪投影近似读出IQ NMSE为0.075417，实际带噪恢复读出IQ NMSE为0.883437；计算对象不同，差值不能直接解释为噪声的单独因果贡献。模型仍在初期，不能用一轮源损失预测正式识别结果。本轮共享负载计时519.83秒，不作为独占速度比较或预测后续耗时。 首轮标量

## 完整30轮与原Main七景独立复算完成

VERIFIED / ANALYZED。固定scratch30/7410、全部源日志/顺序和final=E30核实；原Main42单元全部预测后独立truth-last，4阶段exit0且进程均已退出。1302组/868raw配对及DURA/clean配对复算PASS，最大差1.2e-14，baseline判决差0。六景83.817262/82.515476/80.863690/82.642857/75.964286/68.738095%，1/6；clean两个总体净+175/+29，联合验收False。原科学代码/模型/数据/预算保持，默认DURA不变，其他已登记候选继续。 完整结果
