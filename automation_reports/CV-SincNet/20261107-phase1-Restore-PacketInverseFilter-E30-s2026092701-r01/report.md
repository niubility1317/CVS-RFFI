# 整包时变复数逆滤波恢复器，从零单阶段 30 轮

**六景0/6；两个clean总体净+500/-92。** [完整结果](analysis.md)。

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

原scratch30训练、Main双总体七景与独立复算已完成，详细性能和修复/损伤见analysis.md。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## N607 整包时变逆滤波版排队已核实

VERIFIED / QUEUED。实际代码 92d0dcbe0f25927d5a1cf4d98669ded45a410a5b，控制器 PID 4001257 在线，实际工作目录、命令、CUDA 空环境和日志已核对。发布包 SHA 一致，远端编译通过。等待后验约束条件先验版最后一项 GPU 阶段终止，再按 GPU4/6 原准入条件启动。真实源检查、训练和 Main 均待执行。

## GPU5 常规资源调度预登记

新资源控制器 `42e22b1f7f8667a3f4b11b87a4fe6e6cf20b164e`，所有训练/预测/评分仍执行原 `92d0dcbe0f25927d5a1cf4d98669ded45a410a5b` 和原release。GPU5正常总数最多3个（已有最多2个、剩余至少12000 MiB），每阶段一个本run GPU子进程；取消等待ProjectedPrior最后GPU阶段的纯资源依赖。架构490866参数、原L_s/U_s、连续F0指导、数据顺序/draw/seed、从零单阶段30轮/7410和原Main双总体七景均不变。真实源2/实际G与F0/梯度及严格E0重载由原训练器首步执行，PASS后连续训练；final30后原预测、独立truth-last和报告继续。交接只涉及经完整PID/CWD/cmdline/CUDA空/无子进程/无run root核实的旧等待控制器，确认它退出后才启动新owner。保留旧release/logs/提交记录；原run输出root独占创建，不覆盖任何产物。原run真实stage终态继续驱动原MismatchedPrior后继队列，不能伪造完成。4项针对性检查和一次资源范围P0/P1通过，技术失败保留、不自动重启。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.pamf_phase3_restoration.packet_filter_resource_dispatch --root /home/szu2070436088/2510044040/CV-SincNet --legacy-release /home/szu2070436088/2510044040/CV-SincNet/releases/20261107-phase1-Restore-PacketInverseFilter-E30-s2026092701-r01-release --cache /home/szu2070436088/2510044040/CV-SincNet/runs/20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01/cache --commit 42e22b1f7f8667a3f4b11b87a4fe6e6cf20b164e`。CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/20261107-phase1-Restore-PacketInverseFilter-E30-s2026092701-r01-GPU5-dispatch-release`。

## GPU5 实际训练与真实模型检查通过

VERIFIED / RUNNING。整包逆滤波恢复器已在GPU5实际训练，准入已有2个进程、剩余18501 MiB，独立读回总数3、本run仅一个GPU子进程，使用原总数最多3个限制。旧CUDA空且尚无GPU阶段/输出root的本run等待控制器已精确交接，其他健康任务与旧release/logs保持。调度代码 `42e22b1f7f8667a3f4b11b87a4fe6e6cf20b164e`，实际训练/预测/评分仍原 `92d0dcbe0f25927d5a1cf4d98669ded45a410a5b` 和原release。真实两条源、实际F0输入梯度、G初始恒等及严格检查点重载通过，训练日志已增长。490866个参数从零，原L_s6300/U_s56700、连续源F0指导、seed/顺序/draw/单阶段30轮7410更新及原Main双总体七景保持。这次只改变纯资源等待，不依据任何源/目标分数选择、取消或重排科学候选；默认DURA和其他已登记训练不变。当前无本版完整轮次和正式成绩。

[资源准入、执行代码与实际源检查](resource_launch_evidence.json)

## 前6轮全部源曝光与实际更新核实

VERIFIED / RUNNING。已完成6/30轮、1482/7410次更新；原63000条源记录每轮均为完整排列，累计378000次曝光，其中L_s37800、U_s340200。已逐一分析全部6轮记录，并核实E0/E06科学代码和配置一致、模型与优化器有限、AdamW步数1482，包首/中段/后段三组head权重均实际更新。技术状态用于运行核实，冻结点保持final30，不使用中间checkpoint选模。 平均38.508秒/轮，具体计时范围与限制见[完整训练分析](training_progress_E06.md)。

## 原单阶段30轮及Main七景完成并独立复算

VERIFIED / ANALYZED。整包时变逆滤波恢复器490866参数从零单阶段30轮/7410完整结束，全部30轮日志/源顺序/预算、E0/E30有限状态、三组区域head实际更新、final逐值等于E30技术状态已核实。原Main两总体168000条x七景，三流42单元全部预测后独立truth-last；4阶段exit0、控制器及子进程均已退出。独立复算1302组/868raw配对及DURA/clean配对通过，最大差1.2e-14、raw/DURA复用判决差0。六景依次82.230357/80.775000/78.887500/80.596429/72.464286/63.987500%，0/6；clean两总体净+500/-92，联合验收未通过。实际训练阶段1161.801秒，完整两阶段预测246.631秒，均为共享负载整阶段计时，不代替B1时延或主性能。原source-only数据、连续F0指导和科学代码保持，资源调度未改变模型或测试范围；其他已登记候选继续，不按本项结果取消、组合或重排。默认DURA保持。
