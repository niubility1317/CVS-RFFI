# 整包时变复数逆滤波恢复器，从零单阶段 30 轮

- run_id：`20261107-phase1-Restore-PacketInverseFilter-E30-s2026092701-r01`
- group_id：`source-scratch-packet-position-wise-inverse`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

前导重复参考条件化。包首、中段、后段独立产生复数主路、共轭路及 IQ 残差。

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

## N607 整包时变逆滤波版排队已核实

VERIFIED / QUEUED。实际代码 92d0dcbe0f25927d5a1cf4d98669ded45a410a5b，控制器 PID 4001257 在线，实际工作目录、命令、CUDA 空环境和日志已核对。发布包 SHA 一致，远端编译通过。等待后验约束条件先验版最后一项 GPU 阶段终止，再按 GPU4/6 原准入条件启动。真实源检查、训练和 Main 均待执行。

## GPU5 常规资源调度预登记

新资源控制器 `42e22b1f7f8667a3f4b11b87a4fe6e6cf20b164e`，所有训练/预测/评分仍执行原 `92d0dcbe0f25927d5a1cf4d98669ded45a410a5b` 和原release。GPU5正常总数最多3个（已有最多2个、剩余至少12000 MiB），每阶段一个本run GPU子进程；取消等待ProjectedPrior最后GPU阶段的纯资源依赖。架构490866参数、原L_s/U_s、连续F0指导、数据顺序/draw/seed、从零单阶段30轮/7410和原Main双总体七景均不变。真实源2/实际G与F0/梯度及严格E0重载由原训练器首步执行，PASS后连续训练；final30后原预测、独立truth-last和报告继续。交接只涉及经完整PID/CWD/cmdline/CUDA空/无子进程/无run root核实的旧等待控制器，确认它退出后才启动新owner。保留旧release/logs/提交记录；原run输出root独占创建，不覆盖任何产物。原run真实stage终态继续驱动原MismatchedPrior后继队列，不能伪造完成。4项针对性检查和一次资源范围P0/P1通过，技术失败保留、不自动重启。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.pamf_phase3_restoration.packet_filter_resource_dispatch --root /home/szu2070436088/2510044040/CV-SincNet --legacy-release /home/szu2070436088/2510044040/CV-SincNet/releases/20261107-phase1-Restore-PacketInverseFilter-E30-s2026092701-r01-release --cache /home/szu2070436088/2510044040/CV-SincNet/runs/20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01/cache --commit 42e22b1f7f8667a3f4b11b87a4fe6e6cf20b164e`。CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/20261107-phase1-Restore-PacketInverseFilter-E30-s2026092701-r01-GPU5-dispatch-release`。
