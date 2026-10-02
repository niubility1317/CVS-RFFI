# D92-GroupBalancedSupportMetric-GGN1-LocalRidge support 诊断预登记

当前为 RUNNING，已发布并独立核实实际进程、命令、版本和日志。两行正在 support 训练、两行排队；尚无完整拟合或评分结果，不能据此声称性能改善。

本轮仅把 C 阶段 gate 的逐训练样本损失权重改为 `N/(2*C_group*n_class)`。每组总权重为 N/2，组内每类等权，总权重仍为 N；保留 λ=1 的和式尺度、自由截距、全部旧类训练样本与新增类的严格 margin barrier，以及实际 B→C 继承。B 的算法和全部数值常数保持不变。权重只读当前合法训练 support 标签，不读取 held/query 的类别比例。

依据见[数学推导](../../../docs/D92_GROUP_BALANCED_SUPPORT_METRIC_DERIVATION_20261002.md)。这一 gate 子问题目标不等于 H；`q<6` 时新组每样本更重，`q>6` 时旧组更重，不能保证新类一定提高。

固定 Phase1 为已核实 source-only scratch final200 的两个模型，星地信道为 practical residual/post_sync/noeq/25 MHz。只使用已有合法 support 缓存、原始模型与冻结地面汇总；不传输源域样本或逐样本特征，不复用任何历史适应状态。本轮不访问 query。

矩阵为两个模型 seed × rx3/rx1，共 4 行、160 个物理 parent、1800 条路径。K=1/5/10/20，旧类 6 个，新增类 0/2/5/10/20；K=1 无独立 held 记 N/A。OOF 与单 anchor proxy 分开解释；A/R0 的独立拟合与本方法的顺序 B→C 明确区分。

下表为未评分占位，准确率单位为 %，差值单位为百分点；正式报告按同一 parent 旧类 query（本诊断为外层 held support）配对，先逐 parent 计算 H 与绝对差，再汇总。

| K | 旧类数 | 新增类数 | A 旧类 | B 旧类 | C 旧类 | C 新类 | 适应提升 | 旧类下降 | 新旧绝对差 | H |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 1 | 6 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 1 | 6 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 1 | 6 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 1 | 6 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 5 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 5 | 6 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 5 | 6 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 5 | 6 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 5 | 6 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 10 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 10 | 6 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 10 | 6 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 10 | 6 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 10 | 6 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 20 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 20 | 6 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 20 | 6 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 20 | 6 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 20 | 6 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

根 agent 为唯一 launch owner；CPU 两条 lane、每条 BLAS 线程 2，CUDA 关闭。资源上限沿用当前合规 support 诊断。技术失败只使所属行失败，不自动重试，不因低性能停止健康任务。

本轮将测量可训练参数、拟合/准备/推理时间、进程峰值 RSS、常驻数值状态、已有地面汇总 payload 和新增传输字节；未测量的星载硬件、能耗、显存和真实 wire 字节写 N/A。权重构造为 O(N)，门控矩阵求解的主复杂度没有因此消失。

配置：[冻结声明](../../../configs/d92_group_balanced_support_metric_joint_support_20261002.json)；实际生效配置、源版本、PID、完整日志、状态归档和独立读回随运行补充。

验证：新增方法共 86 个不同病例通过。首次核心为 18 PASS/1 FAIL；保留失败后，仅修复未接受试探步的有限数值/域错误缩步，gate 10 个病例通过，未改 joint 数学或算法常数。入口与控制 66 个病例通过；46 个明确运行源文件的隔离导入和配置逐项一致性通过。一次 P0/P1 审查与该问题一次定点复审均完成，结论 NO_UNRESOLVED_P0_P1。输入检查仅核对现有 VALIDATED_ONCE 缓存、合规来源与路径，不重验数据。

[完整验证及原失败记录](evidence/implementation_validation_20261002.json)。本地 Torch/NumPy 二进制 API 有已记录警告，本次没有执行骨干训练或 tensor/NumPy 转换，不将这些合成结果视为星载性能测量。

运行读回：2026-10-02 01:56:46 UTC，固定源版本 `b9bb1ab04bf16bf3940026b23cf1a96a81a26e05`，普通 N607 用户，supervisor PID 1391208，两个工作 PID 1391273/1391274；CPU 两 lane、每 lane BLAS 2、CUDA 关闭。配置保持原冻结声明，实际 runtime OID 另记，不修改运行中的 spec。独立读回 scope 仅为启动与当前健康运行。

[发布读回](evidence/publication/publication.json) · [实际进程与日志读回](evidence/support_runtime_1790906267136715700.json)。最终同 parent 的 A/B/C、新旧差距、H 及成本仍为 N/A，待完整产物后独立分析。
