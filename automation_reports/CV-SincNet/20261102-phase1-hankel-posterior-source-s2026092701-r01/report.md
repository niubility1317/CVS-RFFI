# Phase1 物理后验与自适应 Hankel 低秩诊断

## 目的与权限
探索只依赖单条当前窗口的时间结构：将盲物理后验均值构成重叠 Hankel 矩阵，依据盲估观测噪声强度做固定奇异值软收缩，反对角线平均后与原后验以固定比例融合。窗口 32、阈值系数 0.1、混合比例 0.5，均在运行前固定；观测噪声只是尺度代理，不声称是精确后验方差。不加载 DURA 生成候选，不用外部设备模板，不训练参数或读取 TX 身份。相位旋转等变、零噪声恒等、静默输入、批量一致及非法噪声输入已做本机检查。

源物理后验和门使用既有合法盲估计、冻结基座伪类别物理先验；此先验属于 F-guided 依赖。固定与 DURA 后验均值已预测的同一 672 条源记录及七景、RX4/6 与 RX8 分层，只复用其物理 ID、RX/场景划分；DURA 预测只用于固定后的源诊断对照。候选在 CPU 计算而 DURA 对照来自 GPU；逐分组记录 raw-F CPU/GPU 分类不一致数，raw 一致也不能保证 DURA 输出的硬件数值完全一致。训练/预测不读取 TX 真值、U_s 真值、V 或目标测试。预测固定后才连接合法 L_s 身份评分，同时单列冻结 clean-F 伪标签；另用独立排序物理 ID 连接和集合计数复核。源记录此前多次探索，源结果不是盲测或地面测试。此阶段为未训练的源域机制诊断，不视为已完成新恢复网络的正式七景地面评测。

## 执行
代码/config 以本报告所在 Git commit 为准。N607 普通用户，项目 Python、CPU 2 线程、CUDA 禁用，CPU 负载超过 72 等待。入口 `python -X utf8 -u -m experiments.phase1_hankel_posterior.source_controller --commit <本报告所在 commit>`；先 28 条实际源模型检查，再固定 672 条源预测、truth-last 评分与独立复核。不可覆盖 run 根分别为 `20261102-phase1-hankel-posterior-smoke-s2026092701-r01`、`20261102-phase1-hankel-posterior-source-s2026092701-r01`、`20261102-phase1-hankel-posterior-score-s2026092701-r01`；预期结果为 fixed predictions、score 与 audit。仅协议越界、数值非有限、无法完成或输出冲突时停止并保留产物；低性能不停止。
