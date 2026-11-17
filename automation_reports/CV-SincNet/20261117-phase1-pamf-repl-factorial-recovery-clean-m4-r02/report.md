# 六源因子矩阵补跑：共享显卡调度

替代等待空卡的本对话 r01 dispatcher；按用户 2026-11-17 要求尽快启动补跑及新的改进实验。仅在确认 r01 仍是无 worker 的 `SMOKE_WAITING_CAPACITY` 且确认具体 PID、CWD 和命令后停止该等待控制器；不停止其他 owner。原 r01 的队列及失败 run 的所有产物保留。

原 20261116 六源权重 3 因子矩阵四条未评分行的范围、合同、E20、seed 和配方完全不变；两条精确匹配且已冻结 E50 的行只执行 clean 预测及独立评分，另两条自原合法 E20 自身训练 E21–E50。所有 U 样本逐曝光盲态记录，仅各行 E50 冻结后独立 U 真值诊断；只测 clean 的 cross_date/cross_rx/cross_both，预测固定后 truth-last 评分。主 clean 合并 cross_rx 与 cross_both；target 和 seed 已曝光，全部为探索结论。

本 run 使用 GPU 0、2、4、6，改进矩阵使用另四张卡；每卡本 run 至多一 worker，全部 owner 和待初始化进程合计不超过 5，并要求不低于 12000 MiB 空闲。若没有满足条件的显卡则自动等待，不为了启动而停止他人工作。真实 E20 无 query smoke 是启动首项；协议、非有限或系统技术故障停止新调度。矩阵及输出路径见 `experiment.json`。发布和运行以独立产物为准。
