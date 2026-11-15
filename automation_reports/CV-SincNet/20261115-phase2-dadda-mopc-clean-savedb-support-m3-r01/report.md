# DADDA→MoPC clean 故障同 B 支持诊断

仿射 ProtoNet 接续 run 的 `0086-hybrid-dadda_sda` 在 P01/MIX04/1-1、seed 2026092703、K1/N20，第 3 阶段第 8 epoch 第 8 次迭代出现 loss、cross_entropy 和 gradient_norm 非有限，且多个梯度张量元素非有限。任务已保留 3 份预测、B-K1 和日志；不能作为单纯有限梯度范数溢出。原 run 已失败排空，其余完整结果保持。

从该次实际 B-K1 及同一 clean 新类支持、原 C seed，只做三支诊断：GPU5 仓库原公式无裁剪，GPU6/7 分别仅诊断全局梯度裁剪 1/10。裁剪改变 SGD 更新，不得冒充仓库原版；无 query、truth 或预测，旧产物不可覆盖。使用 `clean_mopc_savedb_diagnostic --runtime <r01 runtime> --job <0086 job> --k 1 --n 20 --device cuda:0 --variant <original|clip1|clip10>`，每卡一个进程，启动前至少 6000 MiB，保护其他 owner。若未复现不宣称修复；首次观测非有限边界不自动等于根因。

三支均完成 80 次更新且未记录非有限；原故障未复现，根因和修复效果仍未知。裁剪分支仅为诊断，不构成性能或科学等价证据。日志保留在同名远端 run root，未读取 query、truth，也未产生预测。
