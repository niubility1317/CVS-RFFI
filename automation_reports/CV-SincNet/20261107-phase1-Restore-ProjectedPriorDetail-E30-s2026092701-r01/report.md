# 后验约束的条件 IQ 先验，从零单阶段 30 轮

- run_id：`20261107-phase1-Restore-ProjectedPriorDetail-E30-s2026092701-r01`
- group_id：`source-supervised-observation-projected-prior-detail`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
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

## N607 后验约束条件先验版排队已核实

VERIFIED / QUEUED。实际代码 8cf3669c4a5d98f6c4930117952203afd56f2f6e，控制器 PID 3965491 在线，实际工作目录、命令、CUDA 空环境和日志已核对。发布包 SHA 一致，远端编译通过。等待主路相位监督版最后一项 GPU 阶段终止，再按 GPU4/6 原准入条件启动。真实源检查、训练和 Main 均待执行。

## 使用常规 GPU5 的原方案资源交接

原科学代码、数据、网络和从零30轮/7410步保持，独立正常GPU5容量允许与相位版并行；总进程最多3，启动前已有最多2且剩余显存至少12000 MiB。只在核实原队列无子进程、未提交任何训练或预测时交接；不读取其他候选得分来调度。原真实源检查、完整七景和独立truth-last照常执行。

控制代码：`d38f2dfc0378ae269f733b6f9a3ea039ddf11163`；科学代码：`8cf3669c4a5d98f6c4930117952203afd56f2f6e`。命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.pamf_phase3_restoration.projected_prior_gpu5_dispatch --root /home/szu2070436088/2510044040/CV-SincNet --legacy-release /home/szu2070436088/2510044040/CV-SincNet/releases/20261107-phase1-Restore-ProjectedPriorDetail-E30-s2026092701-r01-release --cache /home/szu2070436088/2510044040/CV-SincNet/runs/20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01/cache --commit d38f2dfc0378ae269f733b6f9a3ea039ddf11163`。
