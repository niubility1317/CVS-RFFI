# 源接收机先验失配训练，从零单阶段 30 轮

- run_id：`20261107-phase1-Restore-MismatchedPrior-E30-s2026092701-r01`
- group_id：`source-prior-mismatch-exposure-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同一条件先验网络，奇数轮原全源先验，偶数轮用排除当前源接收机的配额匹配先验。

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

## N607 源先验失配训练排队已核实

VERIFIED / QUEUED。实际代码 14e7315d6da5d6fe242e7568f20c97ade064e8d9，控制器 PID 4033035 在线，实际工作目录、命令、CUDA 空环境和日志已核对。发布包 SHA 一致，远端编译通过。等待整包逆滤波版最后一项 GPU 阶段终止，再按 GPU4/6 原准入条件启动。真实源检查、训练和 Main 均待执行。

## GPU5 串行常规资源预登记

资源控制器 `bf6ab827408f1d20f4e8bbcfcf511d1a118cbc41`，全部科学阶段仍为原 `14e7315d6da5d6fe242e7568f20c97ade064e8d9` 与原release。GPU5正常总数最多3个、已有最多2个且剩余至少12000 MiB；两版按modulation、mismatched顺序，每阶段一个本run GPU子进程。mismatched只等待modulation真实最后GPU终态，不读取成绩作排队条件。源L_s/U_s、架构/损失/seed/原顺序及draw、从零30轮/7410、原Tri-SIG-TIME SSL Main双总体七景和独立truth-last保持。只交接经完整身份、CUDA为空、无子进程、无提交阶段和训练产物核实的旧等待控制器，独立确认退出后才发新owner。原输出骨架、launch.json、release及日志保留；新resource_controller.json独占创建。真实源模型检查仍为原训练器首步，PASS后直接继续；技术失败保留，不重启其他健康任务。9项针对性检查与一次资源范围P0/P1通过。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.pamf_phase3_restoration.registered_restoration_dispatch --candidate mismatched --root /home/szu2070436088/2510044040/CV-SincNet --legacy-release /home/szu2070436088/2510044040/CV-SincNet/releases/20261107-phase1-Restore-MismatchedPrior-E30-s2026092701-r01-release --cache /home/szu2070436088/2510044040/CV-SincNet/runs/20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01/cache --commit bf6ab827408f1d20f4e8bbcfcf511d1a118cbc41`。CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/20261107-phase1-Restore-MismatchedPrior-E30-s2026092701-r01-GPU5-registered-dispatch-release`。

## GPU5 调度实际状态核实

VERIFIED / QUEUED。先验失配版的新控制器在线，CWD/命令/科学与调度代码/CUDA空环境/无子进程及等待日志已核实。它当前没有GPU训练或真实源模型检查，固定等待调制版真实最后GPU终态，再重新核实GPU5正常总数最多3个和至少12000 MiB。不读取调制版分数作为排队条件；原L_s/U_s、连续源F0指导、先验切换、从零单阶段30轮及完整Main矩阵保持。

[实际执行与资源证据](resource_launch_evidence.json)

## GPU5 原先验失配训练实际开始

VERIFIED / RUNNING。先验失配版已按预登记的纯资源顺序在GPU5实际开始训练，准入已有2个进程、剩余19187 MiB，正常总数上限3个。一个本run训练子进程的PID/CWD/命令/父子关系/CUDA/GPU占用与日志增长已核实，父控制器CUDA为空。真实source2、原F0、完整后验、物理与细节梯度隔离、F0冻结及严格E0重载通过。实际训练继续使用原14e7315d科学代码、原L_s/U_s和连续源F0指导，从零单阶段30轮/7410，15轮全源先验与15轮排除当前源RX先验。目标、V、OE和TX标签不参与G训练；final30后原Main双总体七景与独立truth-last保持。调度不读取前一模型分数。

[实际训练证据](resource_training_started.json)

## 完整30轮与原Main七景独立复算完成

VERIFIED / ANALYZED。固定scratch30/7410、全部源日志/顺序和final=E30核实；原Main42单元全部预测后独立truth-last，4阶段exit0且进程均已退出。1302组/868raw配对及DURA/clean配对复算PASS，最大差1.2e-14，baseline判决差0。六景83.727976/82.407143/80.827976/82.618452/76.281548/69.490476%，1/6；clean两个总体净-37/-46，联合验收False。原科学代码/模型/数据/预算保持，默认DURA不变，其他已登记候选继续。 [完整结果](analysis.md)
