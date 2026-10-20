# R34：不依赖 F 的源域解析指纹归属诊断

R25 已完成跨记录／跨 RX／跨日的同 TX 特征对比，但使用同一个训练后的 F；R31 使用人为施加的小信号方向，也不等于真实 TX 细节。R34 补充一个独立于 F 权重、嵌入和 logits 的诊断，检查已有恢复 IQ 的解析观测量是否仍接近不同 RX、不同日期的同 TX clean 记录。

仅用源域已选中且有正收益的 R17 `unroll_unshared` 已完成七视图恢复 IQ，以及原 R07 相同的 1,800 条源验证 IQ。没有新增恢复器/F 推理、权重加载、分类头、分类判决、优化、目标样本或目标成绩。R32 的健康训练和固定源资格规则完全独立，R34 不构成新的晋级门槛。

## 固定观测量和校准

复用 R22 的公开固定数学：正负频谱 log-ratio/asymmetry 64 维、谱边缘／PA／圆度统计 4 维、L-STF 相对复响应残差实虚部 24 维；移除参考响应的 4 个拟合质量坐标。另将三组拼成 92 维：各组先标准化，再除以 `sqrt(3*维数)`，使每组尺度贡献等权。

只在原 R07 训练缓存前 8,192 条 LU clean 上估计每坐标均值及总体标准差，标准差下限 0.05；FP64 累加。明确报告这次额外无标签校准的实际 L/U 数量。它是训练域统计估计，不是分类器拟合，也不使用 V、TX 标签或 F。原始解析计算含固定模板搜索与 RX/信道混杂，不预先称为 TX 专属硬件参数。

独立 CPU `extract` 先固定七视图 × raw/restored × 四组共 56 个解析特征单元，验证有限值、形状与原物理 ID 后，另一 CPU `score` 才打开源 V 标签。统计使用标签作分析分组，不能把整条诊断称为无标签。

## 对比和负控制

参考池沿用 R25：每个 TX×RX×day 格按 opaque physical ID 排序固定取前 2 条 clean。对每个 query，只比较 RX 和 day 都不同的参考格；每格须有全部 6 个 TX 的非零参考。格内计算同 TX 平均余弦与其他 5 个 TX 等权平均余弦之差，再对格等权。缺格、零向量、共同有效数、正间隔比例均保留；这些不是分类准确率。

另外固定 seed 2026102034，做 199 次 RX×day 格内 query 标签置换，参考池保持不变。报告 observed、负控制均值和 5%/95% 分位数；它是给定参考池的描述性负控制，不是 p 值、独立样本显著性或性能选择规则。clean 自身若不高于负控制，不能把该组恢复间隔解释为 TX 身份保持。每组同时报告恢复/raw 对同包 clean 的标准化解析误差，combined 使用等权三组误差。

七个场景逐项保留三郊区、三 urban、clean，并按 TX/RX/day 报告。单 seed、同一源域、相关视图；跨记录不等于已解码验证 payload 独立。解析特征只覆盖部分 IQ 信息，即使结果为正也不能证明每个生成细节属于输入 TX，不替代原冻结 F 的正式 70%／5 个百分点目标或真实在轨、新 TX 验证。

## 执行边界

run 为 `20261020-diagnostic-restoration-independent-analytic-TX-s2026092701-r34`，模块 `experiments.pamf_analytic_identity_probe.runner`。纯 CPU、2 线程、0 GPU；独立不可覆盖输出，普通 N607 账户。精确 commit、命令、输入路径、环境/CWD 和逐行矩阵见对应预登记。

本地聚焦协议负测和数学检查、一次独立 P0/P1、一个代码 commit、最小预登记、一次源句柄预检、单归档 SHA 比对和远端编译，随后实际 PID/CWD/argv/日志和产物读回。不加载 checkpoint，因此不增加真实 checkpoint smoke；不重新验证未变数据，不增加签名或哈希链。输入、权限、非有限值或输出冲突等技术错误保留产物，不自动重试，不改变健康 R32 或其他任务。

预期产物：`launch.json`、`resolved_config.json`、`calibration.json`、`analytic_features.npz`、`physical_ids.npy`、`feature_complete.json`、两阶段启动记录和完整日志、`summary.json`、`complete.json`。只提交方法、登记和公开汇总；校准状态、特征、物理 ID、标签和原始日志留在私有产物。

实际完成：SOURCE_ANALYTIC_ASSOCIATION_PARTLY_SUPPORTED；420条分组及84条负控制独立复核，CPU 8.602s，0新神经预测/F/target。谱统计 clean 未高于负控制，保留负结果；未证明逐细节真实性。见[完整报告](../automation_reports/CV-SincNet/20261020-diagnostic-restoration-independent-analytic-TX-s2026092701-r34/report.md)。
