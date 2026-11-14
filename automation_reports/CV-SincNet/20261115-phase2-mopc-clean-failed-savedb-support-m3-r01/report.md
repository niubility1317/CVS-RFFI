# Clean ProtoNet→MoPC 故障的同 B 支持诊断

原 full run 的 `0061-hybrid-protonet_cda` 在 P01/MIX03/14-7、seed 2026092702、K10/N20 的公开仓库 MoPC 第 3 阶段第 2 epoch 第 6 次迭代出现非有限 loss、CE、层级正则和 18 个梯度张量；原 11 份预测及 B-K1/K5/K10 保留。已完成 432 项，调度器已失败排空；旧 run 不重启。

本诊断只加载该任务同一次保存的 B-K10、原 clean 新类支持集与原 C seed；不读 query、truth，不产生预测。GPU 5、6、7 分别执行原公式无裁剪、全局梯度裁剪 1、裁剪 10。裁剪明确改变优化更新，仅做假设诊断；结果不得冒充原仓库方法，也不表示允许正式方法变更。每卡一个进程，启动前至少 6000 MiB 空闲，其他 owner 不受影响。

入口 `python -m experiments.pamf_phase2_practical.clean_mopc_savedb_diagnostic --runtime <r01 hybrid runtime> --job <r01 failed job> --k 10 --n 20 --device cuda:0 --variant <original|clip1|clip10>`；运行目录 `runs/20261115-phase2-mopc-clean-failed-savedb-support-m3-r01`，日志各自独占。记录首个观测非有限边界及最终状态；无法复现不等于根因修复，禁止盲重跑故障项。

2026-11-15：三卡执行结束，每支完成 320 次更新，未观察到非有限，包括无裁剪原始分支。诊断附加逐层检查并使用其他 GPU，实际运行路径与故障时序不完全相同；原失败未复现，根因未知。两个裁剪分支改变优化器行为，不能据其完成宣称修复。提交 `5b840b0a5d80432844976cd1bbc8fc1bc9a23331`，原任务产物保持。
