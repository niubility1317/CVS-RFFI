# r15 保存 B 的论文公式与裁剪诊断

使用 r15 任务 `protonet_cda-s2026092702-c0195` 对应的真实 r11 保存 B-K10、P02/MIX05/8-8 原 received K10/N20 support 与原 C seed。两条独立诊断均先检查源 checkpoint，仅使用新类 support，无 query、truth、预测或评分。GPU 6 使用 paper 公式和原优化器；GPU 7 对同一论文公式使用全局梯度范数 1.0 裁剪，属于明确改变优化器更新的诊断变体，不称作者仓库原版。其余阶段预算、输入和 seed 相同。两条在新不可变输出中记录首个观测非有限边界和完成状态；不重发旧隔离，不改变任何旧输出。

本诊断运行当前 release 的 FP64 梯度范数观测与逐层 instrumentation；r15 原运行是 paper 语义及旧 FP32 范数 guard，instrumentation/GPU 不同，不能视为逐操作完全重放。若原变体未复现，不证明真实故障修复；裁剪成功亦不证明论文公式根因，且不能冒充原优化器。

入口 `python -m experiments.pamf_phase2_practical.group2_savedb_candidate_diagnostic --runtime <r15 runtime> --job <r15 jobs/00002.json> --k 10 --n 20 --device cuda:0 --variant <original|clip1>`；按物理 GPU 设置 `CUDA_VISIBLE_DEVICES`，每卡一进程，启动前确认他人任务与至少 6000 MiB 余量。运行代码版本见下方。

VERIFIED：GPU 6 PID 3510259 的无裁剪分支和 GPU 7 PID 3510260 的裁剪分支均已退出；六个真实源模型 smoke PASS，各完成 320 次更新，观察边界均未见非有限。两支均只读取源和 support，无 query/truth/预测。因为未裁剪分支没有复现故障，这次无法判定裁剪是否消除了根因；裁剪改变优化器，未用于正式 pilot。实际运行代码 commit `bf8c4a8a4ed08d2ae237db197e86616bcc5c2a31`，日志位于本 run 的 `original.log` 和 `clip1.log`。
