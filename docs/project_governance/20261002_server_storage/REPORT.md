# N607 CVS 磁盘存储检查

状态：VERIFIED。用户澄清检查对象为存储，本报告以磁盘占用为准。检查时间为 2026-10-02 00:00 至 00:09（Asia/Hong_Kong），远端身份 `szu2070436088`，主机 `dell-DSS8440`。本次只读检查，未清理、压缩、覆盖产物或修改活动实验。

## 容量结论

`/home` 总容量 10.83 TiB，已用 4.27 TiB，可用 6.01 TiB，使用率 42%。inode 使用率 3%，没有磁盘容量或 inode 即将耗尽的证据。

CVS 项目根 `/home/szu2070436088/2510044040/CV-SincNet` 实际磁盘占用 1.487 TiB。主要目录：

|目录|磁盘占用|说明|
|---|---:|---|
|`runs`|1.308 TiB|实验输出，含日志、预测、逐步数值状态和模型|
|`releases`|107.49 GiB|源码发布及历史材料副本|
|`logs`|10.42 GiB|外部日志目录；不包含 `runs` 内的大日志|
|`datasets`|23.84 GiB|数据目录|
|`paper_reproduction`|15.05 GiB|复现实验资料|

容量采用 `du -x -B1` 的实际分配块；重点文件同时记录逻辑字节与分配字节。各次采样相隔数分钟，活动输出会增长；`du` 对硬链接的去重范围也随单次命令变化，不能机械相加跨次结果。GiB=2^30 字节，TiB=2^40 字节。

## 已定位的记录膨胀

七个重点 D92 目录完整枚举结果如下。NPZ 是保存的逐步数值状态，不等同于无用缓存；JSONL 包含训练事件、完整轨迹、拟合阶段及部分预测记录，不能把全部 JSONL 都视作可删除日志。

|实验目录（均在 `runs/` 下）|文件数|全部 GiB|NPZ GiB|JSONL GiB|文本日志 GiB|
|---|---:|---:|---:|---:|---:|
|20261001-phase2-d92-margin-joint-repeat-m2-r01|100428|152.12|135.06|15.44|0.13|
|20261001-phase2-d92-affine-joint-support-m2-r01|56444|53.46|35.91|11.89|3.40|
|20260929-phase2-d92-branch-local-margin-support-m4-r01|109|53.21|0.00|22.67|25.47|
|20261001-phase2-d92-anchor-joint-support-m2-r01|57212|52.30|36.51|10.74|3.02|
|20261001-phase2-d92-margin-joint-support-m2-r01|40932|37.67|24.81|8.67|2.44|
|20261001-phase2-d92-conditional-joint-support-m2-r01|40932|34.73|25.49|6.04|1.83|
|20261001-phase2-d92-group-barrier-joint-support-m2-r01|44447|34.56|24.71|6.23|2.18|

明确的重复输出路径：

1. **BranchLocalMargin 文本输出两份。** `20260929-phase2-d92-branch-local-margin-support-m4-r01` 的 `probe.log` 合计 13.01 GiB，`solver_sweeps.log` 合计 12.46 GiB；`fit_trace.jsonl` 合计 11.48 GiB，`solver_sweeps.jsonl` 合计 10.66 GiB。源码 `tools/evaluate_d92_branch_local_margin_probe.py:133` 保存完整 JSONL，143 行写入文本文件，145 行将相同文本打印至外层日志。因此逐步文本存在两个输出渠道。没有做文件逐字节相等检查，不宣称整个文件完全相同。
2. **AffineJoint 多层事件复制。** `20261001-phase2-d92-affine-joint-support-m2-r01` 的四份 `fit_trace.jsonl` 合计 5.65 GiB，`training_events.jsonl` 合计 3.49 GiB，`training.log` 与 `probe.log` 各约 1.70 GiB。源码 `tools/evaluate_d92_affine_joint_probe.py:354` 将事件加入完整轨迹结构，532 行另存全量事件，535 行同时写文本文件和 stdout，548 行保存完整轨迹。这些文件是不同视图，但重复携带事件与引用。
3. **“compact” 文件也很大。** AffineJoint 的 `training_events_compact.csv` 为 1.58 GiB，`training_events_compact.jsonl` 为 1.47 GiB。`compact_event` 仍递归保存完整 NPZ 引用、objective 和 inner_folds；它避免复制矩阵，但并非仅含几列标量的短记录。完整保留两种全量格式进一步增加占用。
4. **最大目录主要是逐步状态。** MarginJoint repeat 占 152.12 GiB，其中 NPZ 135.06 GiB、100351 个文件；JSONL 15.44 GiB。仅精简文本日志无法解决这个目录的主要占用。需要先确认每类状态的复现、审计与报告引用，再决定未来存档粒度或无损压缩策略。
5. **发布包携带历史资料。** 历史 `releases` 中多个独立版本携带同一旧报告相对路径与相同大小的结果 JSON，例如 `automation_reports/CV-SincNet/phase2_adv3b02_stage2c_rescue_veto_sweep_20260707/remote_artifacts/detail/norm_e95_any_m1_k10.json`（16534157 字节）。这是重复打包线索；本次仅查元数据，未读取目标预测或进行内容哈希，未证明可删除。当前活动 Simplex release 仅约 17.35 MiB，已经明显较小。

七个目录文本日志合计 38.46 GiB；这个数是日志总量，**不是已确认可回收空间**。模型、预测、合法审计状态及唯一失败证据继续保留。

上述日志写入语句已从各自原始远端 release 独立读回核实；原始发布源码 SHA-256 保存在结构化汇总中。当前本地源码整文件与历史 release 的字节并不一致，本结论仅使用已核实的一致日志写入段。

## 占用最大的实验目录

|目录|实际占用 GiB|
|---|---:|
|`20261001-phase2-d92-margin-joint-repeat-m2-r01`|152.13|
|`phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r5`|65.23|
|`20261001-phase2-d92-affine-joint-support-m2-r01`|53.46|
|`20260929-phase2-d92-branch-local-margin-support-m4-r01`|53.21|
|`20261001-phase2-d92-anchor-joint-support-m2-r01`|52.30|
|`20261001-phase2-d92-margin-joint-support-m2-r01`|37.67|
|`20261001-phase2-d92-conditional-joint-support-m2-r01`|34.73|
|`20261001-phase2-d92-group-barrier-joint-support-m2-r01`|34.56|
|`phase1_native_response_matrix_20260914_r1`|33.13|
|`a1_mechanism_periodic_s392005_20260910_r1`|30.52|

## 处理优先级

1. 后续新发布保留完整结构化训练记录；stdout 输出启动配置、阶段/epoch 汇总和错误，不再镜像每个 solver sweep 或每个训练事件。保留 CVS 详细日志要求中的实际损失、权重、学习率、梯度和耗时，不削减科学训练预算。
2. 全量事件只保存在一个结构化流；完整轨迹用稳定事件/状态索引引用。紧凑 CSV/JSONL 保留按 epoch 或阶段的短标量汇总，完整引用映射另存一次。此项需同步兼容既有独立审计和分析程序。
3. 新 release 按源码、所需配置和依赖白名单打包，排除历史 `runs`、原始日志及大报告结果副本；版本追溯通过 commit 与原路径引用保存。
4. 对已完成历史 run，可先制定逐路径的无损压缩/归档清单，核对活动引用及保留副本，读回验证后再按明确授权释放原副本。当前没有历史清理授权，未执行删除或压缩。
5. 容量预警建议同时看剩余字节和短期增长率，例如 `/home` 使用率达到 80% 或可用空间低于 1 TiB 时提示；阈值只作提醒，不自动停止实验。此处为建议，未新建监控或改变调度器。

## 活动实验与检查边界

检查时活动 CVS 为 Simplex 源训练八行、一个 dispatcher。当前逐步日志约 516 至 530 字节/行、每行固定 10000 步预算；全部逐步日志预计约 40 MiB，实际阶段/epoch JSONL 与 CSV 加上外层日志后仍为数十 MiB 量级。现有已记录行未发现重复 step/epoch、大数组或无效完整 JSON 行。它不是当前存储膨胀的主要来源。

项目根、`runs`、`logs`、`releases` 的 `du` 目录统计完整成功，无权限错误。重点七个 D92 run 的文件元数据枚举完整成功。广泛 Python 文件扫描分别受 45 秒上限限制，项目扫描 540340 个文件、runs 扫描 519322 个文件，均是部分清单；不宣称已逐文件审查全部历史文件。未读取数据集、query truth、预测内容或按历史目标结果做方法选择。

证据见 [结构化汇总](summary.json)、[完整目录统计](directory_sizes.json)、[项目根统计与 inode](project_sizes.json)、[七个 D92 目录明细](focused_d92.json) 和 [检查脚本](audit_source.py)。原始快照保存在本机 `E:/type10-7/local_artifacts/resource_audit/20261002`；远端没有新增检查文件。远端状态 VERIFIED 来自实际目录/文件分配字节及独立进程、文件描述符与 GPU 读回，不能解释为“存储优化已实施”。
