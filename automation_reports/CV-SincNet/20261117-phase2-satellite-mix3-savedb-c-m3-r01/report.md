# Phase2 星地共用 B 的 LoRa 和 Orthogonal 注册

本 run 接续 `20261117-phase2-satellite-mix3-m3-r01` 的五种域适应主干：ProtoNet-CDA、MRIOR-SDA、Two-stage、POSTER-FT 和 Tweak M1。每种 189 个三池／MIX01–03／七接收机／三个模型种子任务，每任务 K=5/10、N=5/10，对已保存且经绑定核对的同一个 B 状态分别运行 LoRa 平衡样本库和 Orthogonal 注册；不重新训练 B。Tweak 查询严格每次一条。新组合是移植注册机制，不能称原作者完整源模型。

Orthogonal 使用目标旧类 support 在实际 B 特征空间中的原型作为旧类权重，记为 `orthogonal_support_prototypes_v1`；该桥会改变 C 入口旧类判决，必须在结果中披露，不能声称等价于 B 原判决或已证明能提高性能。ProtoNet 从实际 B 模型及其旧类原型重建原始 ProtoClassifier，不将 MoPC affine 特征参数化误标为相同 Orthogonal 桥。LoRa 保持新旧每类 K 条的平衡样本库。

使用既有 `p2_min_v1`／`VALIDATED_ONCE` 星地接收数据及固定 split，query 不参与训练或参数选择；每任务预测固定后独立 truth-last 评分。前序真实非有限隔离项无完整 B 时保留缺项，不重发或覆盖旧产物。其他系统性技术错误停止新增派发并排空，记录未评分预测。只允许单一 owner，物理 GPU 0–7 每卡最多 6 个本任务进程；由于旧 owner 不支持原子共享准入，新队列等旧 owner GPU 队列派发完并排空后自动接续，不等待其 CPU 评分，也不干预其他 owner。

执行目录 `releases/20261117-phase2-satellite-mix3-savedb-c-m3-r01-release`，配置 `preparations/20261117-phase2-satellite-mix3-savedb-c-m3-r01/scheduler.json`，不可覆盖输出 `runs/20261117-phase2-satellite-mix3-savedb-c-m3-r01`，代码版本为本报告所在提交。N607 的 CVS-RFFI Python 通过 `python -m experiments.pamf_phase2_practical.satellite_saved_b_c dispatch --config <scheduler.json>` 执行；首批实际 B 做无 query 的旧、新 support 烟雾检查。预期产物为各模型和 K/N 的冻结预测、独立评分及调度状态。
