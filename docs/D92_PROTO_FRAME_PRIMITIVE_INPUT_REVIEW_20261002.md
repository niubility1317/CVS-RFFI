# ProtoFrame 数学 primitives 与输入控制源码审查

日期：2026-10-02。状态：`NO_UNRESOLVED_P0_P1_WITHIN_REVIEWED_SOURCE_SCOPE`。本次审查文档已冻结。

本次是一次有界 P0/P1 源码审查。审查者是新 `d92_proto_frame_joint_local_ridge.py` 的作者，因此**不把该 joint core 纳入自己的独立审查结论**。本次只检查其他作者的 primitives、ground geometry bridge、supervisor、preflight、publisher 及其直接接口。没有修改这些实现，没有读取实际 run、原型包、模型、缓存、预测、truth、指标、登记索引或交接，没有执行数值、Conda、Git、SSH 或发布。

## 发现及局部处理

### P1：失败时距离 pair 计数包含尚未执行的批次——源码已解决

原 `interaction_distances` 在循环开始前填写完整 `n(n−1)/2` 或 `nm`。中途 QR 或距离计算失败时，失败账本会包含尚未调用的后续 pair，与实际工作量口径不符。

作者已改为每次实际 `_pair_distance` 调用前累加本批 `count`，完整成功时总数不变。已只读核对 [primitives](../code/cvsrffi/d92_proto_frame_primitives.py) 的 `interaction_distances` 和新增 `test_interrupted_distance_counts_only_attempted_pair_batches`：4 个物理点计划 6 对，在第 2 批失败时只记录实际尝试的 3+2=5 对，并保留未执行位置的 partial distance 状态。本审查未执行该回归。

### P1：Ridge JVP 失败缺少已完成 forward 和当次 RHS——源码已解决

`fit_free_intercept_ridge` 在 forward 解已经得到 `F/z/s/alpha/intercept` 后，仍直到全部 JVP 成功才更新相应 snapshot。JVP 求解失败时，现有异常保留 `K/Y/L/chol` 和已消耗计数，但不保留已完成的 forward 解及实际失败 `jvp_rhs`。这不改变成功数学结果或计费；它限制了失败档案对本次求解状态的直接复核。

根任务调度后，作者仅前移 snapshot 保存：forward 求解前保存 `combined_rhs`，完成后及时保存 `F/z/s/alpha/intercept/scores/train_scores`，JVP 求解前保存实际 `jvp_rhs`。已只读核对该局部变更，以及 `test_jvp_second_triangle_failure_preserves_completed_forward_and_actual_rhs`：第二个 JVP triangular 验证失败仍保留完整 forward、当次 RHS 和已发生的 4 次 triangular 调用；测试用独立 saddle 方程核验 partial 解，没有伪造未完成的 Jacobian。数学、运算顺序、资源参数和已有方法未改。本审查未执行该回归。

## 已检查范围

| 范围 | 源码核对结果 |
|---|---|
| 固定字典与切向变换 | `build_proto_frame_dictionary` 先按物理类名排序、单位化，再以最后一类构造全部 5 列差向量；不按 support 选择方向。`transform_z_id` 的饱和切向变换、完整归一化 Jacobian、零行规则及零参数下仍保留真实 Jacobian，对应数学稿式 (6)、(7)、(20)。 |
| 分支、interaction 与核 | 原分支归一化、FFT 权重 4 和辅助块尺度保持；隐式 interaction 距离包含两个端点的全部导数。半原距离 Gaussian 对应式 (8)；零核、原始 exact-equivalence 分支没有虚构连续梯度，也没有展开完整 interaction 特征。 |
| 完整 Ridge | 使用 `I+K` 一次 Cholesky 和 `[Y,1]`，保留自由截距及不平衡标签均值。五方向共享同一因子，包含 `dF/dz/db/dalpha` 和 train/held 两端，符合式 (12)、(13)。PSD 检查、因子、实际 triangular RHS、谱检查均有独立实际计数；无 inverse-K、pinv 或 jitter。 |
| RMSCE 与 GGN | 类内先求物理 CE 均值，再求类 RMS。梯度与 GGN 包括加权 softmax 曲率和 RMS 跨类曲率；固定 `I5` damping。五维 SPD hard-ball 子问题保留全部坐标，以确定性的 secular 求解处理 active 球面，检查原 KKT 残差。并未把 GGN 当完整 nonlinear Hessian。 |
| 实际参数与接受接口 | supervisor 比较完整冻结 `FROZEN_CONFIG`，独立限制每个有 OOF 的 stage 最多 1 次更新、12 个试点、5 个参数方向；这些是结构上限，不是实际 Newton/因子工作量。式 (25) 的真实目标回读在 joint core 中，由其他审查者核验，本次不自审该实现。 |
| ground bridge | [bridge](../code/cvsrffi/d92_proto_frame_ground_geometry.py) 只调用既有 source-only aggregate reader 和固定 center 解码；不重建 residual domain，不读源逐样本数据，不加载 checkpoint，不把原型变为标签或教师。既有 reader 继续核 checkpoint、完整 schema/member allowlist、source-only 结论和原始类顺序。零新增统计字节、组件文件字节、native wire 未测量各自说明。 |
| metadata 与监督器 | [preflight](../tools/preflight_d92_proto_frame_joint_probe.py) 仅读显式声明 metadata/文件存在性；数值 aggregate 的完整检验仍由既有 reader 执行。[supervisor](../tools/run_d92_proto_frame_joint_probe.py) 固定 2 模型×2 cohort 的 4 row、每 row 40 个唯一 parent、合计 160 parent，并绑定物理 support、6 类 seed 角色、resolved config、实际 PID/runtime commit。各 row 独占输出；失败不重试，健康 row 保留；失败成本不冒充完整 run 成本。 |
| SUM/MAX 与未测量项 | primitives 的 `AUDIT_SUM_KEYS` 包含谱、QR、因子、真实 RHS 和秒数；返回数值对象字节不是进程峰值。supervisor 使用明确的 peak 集合，factor-buffer 限制仅属于 gate 显式因子 buffer，不是全训练或进程内存上限。 |
| 发布控制与源白名单 | [publisher](../tools/publish_d92_proto_frame_joint_probe.py) 先核本地相关路径已提交且当前 branch OID 与远端一致，再归档明确源码闭包和显式 spec。独立临时目录 import 检查禁止 `np.load/torch.load` 和 native encoder 模块；远端核 archive 内容及独占 release/run。唯一 root supervisor 首次写 actual runtime commit，发布后独立读回 startup；异常保留并要求只读 reconcile，没有自动再次发布或启动。包中不包含训练数据、checkpoint、缓存或实验结果。 |

数学依据：[冻结设计稿](D92_LOCAL_RIDGE_JOINT_GENERALIZATION_COST_NOTE_20261002.md) 的式 (6)–(8)、(12)–(13)、(20)–(25)。合成测试源码包含独立有限特征 Ridge oracle、score Hessian oracle、完整方向差分、无信息/退化输入和资源失败检查；读到测试源码不等于测试通过。

## 结论边界

根任务确认已在实际激活的 `ssr-gpu`、native Windows 环境串行执行本轮 4 组合成测试，**86 PASS**，证据前缀 `pytest_native_activation_1790881160942242100`。上述两条 failure 修复回归包含在该次执行内并通过。先前 collect 阶段的 tools 导入路径失败 `1790881026017521400` 已保留，由根任务修正测试导入；该层失败不属于方法数值失败。以上是根任务提供的执行证据，本审查者没有自行执行或读取真实产物。

两项 P1 已在源码局部关闭，本次已检范围没有可证实的未解决 P0/P1。本审查不评定 query 泛化、准确率、实际设备成本、实际 ground 包合规性或部署完成状态。文档 UTF-8 与局部源码静态读取已完成。未增加数据重验、性能门槛、审批、receipt 链或完整矩阵早期门槛。局部问题关闭后仅核对应变更，至此冻结本次文档，不重复全项目审查。
