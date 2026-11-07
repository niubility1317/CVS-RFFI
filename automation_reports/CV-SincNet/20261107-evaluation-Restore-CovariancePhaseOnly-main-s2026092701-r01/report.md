# 冻结 CovarianceMAP 的整体相位校正，原 Main 七景

**六景0/6；clean两总体净-1900/-133。** [完整性能与净纠错](analysis.md)。

- run_id：`20261107-evaluation-Restore-CovariancePhaseOnly-main-s2026092701-r01`
- group_id：`frozen-CovarianceMAP-phase-only-main`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

沿用源域从零单阶段 E30 权重，新增零参数相位边缘化推理。完整原 Main 双总体七景，独立 truth-last。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

原Main双总体七景与独立复算已完成，完整性能、逐组和修复/损伤见analysis.md。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## N607 正式对照排队已核实

VERIFIED / QUEUED。控制器 PID 3845271 在线，实际工作目录、命令、代码和 CUDA 空环境已核对。发布包 SHA 一致，远端编译通过。固定等待 C4 最后一个 GPU 阶段终止，再按原许可 GPU4/6 条件开始。快照时尚无本版 GPU 预测或真实源检查。完全复用原 E30 权重，只新增整体相位的冻结推理。

## GPU3 资源调度预登记

资源控制器代码 `5ba8f9b2649e81efd58d5fd8597dbb97f09c4dea`，预测与评分仍执行原 `d17c50dceaba5aed58c9eeaafc433b0a5fda5a20`。已冻结的相位对照没有新训练，取消其等待C4最后GPU阶段的资源依赖，改用GPU3原总数最多3个、剩余至少12000 MiB的准入。保留原run/root/launch及所有产物，仅在独立核对原控制器完整身份、CUDA为空、无子进程、无已提交GPU阶段后交接该控制器。原模型/输入/参数/两总体七景/控制组/先预测后独立truth评分完全保持。新控制器一个GPU预测子进程逐阶段运行，评分独立CPU且使用原共享锁。原run里真实cross_both终态继续服务已有后继任务，不伪造阶段完成或复制其他run的预测。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.pamf_phase3_restoration.phase_only_resource_dispatch --root /home/szu2070436088/2510044040/CV-SincNet --legacy-release /home/szu2070436088/2510044040/CV-SincNet/releases/20261107-evaluation-Restore-CovariancePhaseOnly-main-s2026092701-r01-release --cache /home/szu2070436088/2510044040/CV-SincNet/runs/20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01/cache --commit 5ba8f9b2649e81efd58d5fd8597dbb97f09c4dea`。CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/20261107-evaluation-Restore-CovariancePhaseOnly-main-s2026092701-r01-GPU3-dispatch-release`。预期新增resource_controller.json，原stage submitted/terminal、原row完整预测与结果。实际source2模型检查仍在每个总体预测首步；5项针对性测试和一次资源范围P0/P1通过。技术失败保留，不自动重跑或改动其他健康任务。

## GPU3 实际预测和原路径检查通过

VERIFIED / RUNNING。GPU3原准入已有2个计算进程、剩余19873 MiB，符合总数最多3个和至少12000 MiB条件。精确核实后仅交接旧CUDA空、无GPU阶段的本run控制器，旧控制器已终止；原launch.json和输出保持。新调度器 `5ba8f9b2649e81efd58d5fd8597dbb97f09c4dea` 已启动一个实际GPU预测进程，预测和后续评分从原release执行 `d17c50dceaba5aed58c9eeaafc433b0a5fda5a20`，真实原BF/F0/两条L_s和相位路径等价检查通过，日志已增长。固定的原Main双总体七景继续，全部预测后才独立truth-last；当前尚无正式成绩。既有后继队列仍读取本run真实GPU阶段终态，不伪造完成。

[资源交接与真实模型证据](resource_launch_evidence.json)

## 原 Main 七景完成并独立复算

VERIFIED / ANALYZED。原Main两总体168000条x七景、四流56单元全预测后独立truth-last，三阶段真实exit0且当前控制器/子进程均已退出。全部1736组、1302raw配对及BF/DURA/clean额外配对独立复算通过，最大指标差1.24e-14，三条控制流复用判决差0。六景依次83.427976/81.999405/80.531548/82.244048/75.323214/67.878571%，0/6；clean跨接收机净-1900、跨接收机且跨日期净-133。本对照六景与两个clean联合验收未通过。低档城区67.878571%，距140345条门槛少26309条正确。这是同一冻结原BF权重的整体相位推理对照，0新训练；实际预测/评分仍原代码，资源调度变化没有改变模型、数据或矩阵。后续在训候选保持原计划，默认DURA保持。
