# CVS Slow-Fast影子门控V2实验报告

- run ID：`cvs_slow_fast_shadow_gate_v2_s392002_20260825_r1`
- 当前状态：`ANALYZED / SCIENTIFIC_SIGNAL_NO_PROMOTION`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 代码与配置提交：`d26bc3428d61e5ad4d18261c368278b15bfbbb66`；GitHub远端OID已独立回读一致。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`
- N607预检：2026-08-25 20:05 CST直连成功；项目根可见；8张RTX3090均为0%利用率、1MiB显存占用；预注册GPU1。

## 方法、矩阵与协议

- 复用`p2_min_v1`、`VALIDATED_ONCE`、capsule=`d18-reuse-validated-once-rx20-1-seed713101-m7282101-k10-new10`、split=`p2_min_v1-rx20-1-m7282101-s7282201-q7282301-d7282401-k10-new10`，不重新验证数据。
- 固定receiver=`20-1`、seed=`392002`、`K10/new10`和`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`，三候选共9行。
- `COMMON_SHIFT_R4`每行输出DA0、5个固定lambda影子、旧门控和新cross-fit门控，共8个状态。
- `FAST_FILM_R8`与`FAST_LOWRANK_R8`每行输出DA0、`J={1,3,5,10}`×步长倍率`{0.5,1,2,4}`×5个lambda、旧门控和新门控，共83个状态。
- 所有状态在query truth未知时冻结；query逐样本只读且只提取一次；评分完成后的最佳影子状态只用于诊断，不反馈重跑或选择。
- `REJECTED_EXTRA_GATE`：不采用逐代码文件、逐feature或额外seal哈希；Git提交固定代码/config，发布只比较一次release归档SHA。

## 已完成改动与本地验证

- 统一Phase1.5、Phase2快更新与support门控的余弦logit scale；prediction显式保存raw cosine。
- `FAST_LOWRANK_R8`改为零中心有符号tanh门控，并对旧V1 bundle做前向等价迁移。
- 修复`COMMON_SHIFT_R4`强度重复缩放，lambda只由`rho`表达。
- 新增分层重复cross-fit、连续风险最小化、完整逐lambda trace、尝试／提交更新量与legacy门控对照。
- 新增truth-last多状态scorer，先校验全部prediction和opaque query ID，再首次连接truth。
- `ssr-gpu`相关回归36项通过；正式矩阵9行配置解析通过；唯一一次独立P0/P1审查为`NO_FINDINGS`。

## 路径、命令与预期artifact

- 本地CWD：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\meta-adapter-tri-r4-v1-20260824`
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_slow_fast_shadow_gate_v2_s392002_20260825_r1/checkout`
- smoke：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_slow_fast_shadow_gate_v2_s392002_20260825_r1_smoke.json`
- prediction：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_slow_fast_shadow_gate_v2_s392002_20260825_r1`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_slow_fast_shadow_gate_v2_s392002_20260825_r1.out`
- GPU：物理GPU1；进程内`cuda:0`。
- smoke命令：`CUDA_VISIBLE_DEVICES=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/smoke_slow_fast_no_query.py --config configs/stage2_slow_fast_smoke_common_s392002_20260825.json --output /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_slow_fast_shadow_gate_v2_s392002_20260825_r1_smoke.json --device cuda:0`
- prediction命令：`CUDA_VISIBLE_DEVICES=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_slow_fast_matrix.py --config configs/stage2_slow_fast_shadow_diag9_s392002_20260825.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_slow_fast_shadow_gate_v2_s392002_20260825_r1 --device cuda:0`
- 预期：一个smoke receipt、一个matrix receipt、9个row receipt、522个prediction NPZ；随后由独立scorer生成9个score和一个汇总。

## 停止与晋级规则

- 只因协议/query越权、错误stage/receiver/seed/K/scene/split、错误checkout、输出覆盖、进程归属不清、无法启动、prediction不完整或同一确定性pre-prediction异常至少出现两行而停止；不得因低性能停止。
- truth-last评分后，正式`DA1_GATE_CF_REG0`或预注册固定影子若旧类均值至少`+1.0pp`、floor至少`+0.5pp`且任一旧类下降不超过`5pp`，才进入Target25确认。
- 若存在非零query上界但support gate选不到，下一步只优化gate／步数／步长；若所有非零状态都无上界，才触发P1慢基重训；P1仍失败才考虑P2中间层Adapter。

## N607发布与执行闭合

- release归档本地／远端唯一一次SHA256均为`7ddb130241c035ebc78d3a0d6480d485a971f115f17bce32f6f18def2b27f689`；远端编译为`REMOTE_COMPILE_PASS`。
- 真实checkpoint无query smoke为`SMOKE_PASS`：60个target support物理样本，`query_input_capability=false`、`query_truth_opened=false`、`query_role_opened=false`、`source_opened=false`。
- prediction启动PID为`3078859`。首次启动后检查时进程已自然结束，但更高等级artifact证据已闭合：9／9行、522份prediction、9份row receipt和matrix receipt齐全；`truth_opened=false`、`source_opened=false`。
- scorer命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/score_stage2_slow_fast_matrix.py --matrix-config configs/stage2_slow_fast_shadow_diag9_s392002_20260825.json --prediction-root /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_slow_fast_shadow_gate_v2_s392002_20260825_r1 --truth-map configs/stage2_slow_fast_truth_map_s392002_20260825.json`。
- truth-last评分生成9份score和`diag9_score_summary.json`；`status=ANALYZED`、`truth_opened_after_predictions_complete=true`、`truth_last_selection_reused_for_adaptation=false`。

## 实验结果

三候选的`DA0_REG0`跨三场景聚合旧类均值均为66.67%，floor为38.33%。新cross-fit门控和legacy门控在9行都选择lambda=0，因此门控状态与基线相同，均未晋级。

|候选|代表性最轻非零状态|聚合均值变化|聚合floor变化|最差旧类变化|结论|
|---|---|---:|---:|---:|---|
|`COMMON_SHIFT_R4`|`DA1_L0125_REG0`|-1.67pp|0.00pp|-6.67pp|无上界|
|`FAST_FILM_R8`|`DA1_J01_A050_L0125_REG0`|-0.28pp|0.00pp|-3.33pp|局部信号、聚合失败|
|`FAST_LOWRANK_R8`|`DA1_J01_A050_L0125_REG0`|-0.56pp|0.00pp|-5.00pp|局部信号、聚合失败|

|候选|`leo_clear_weak`均值变化／决策变化数|`leo_low_elev_weak`均值变化／决策变化数|`leo_rain_weak`均值变化／决策变化数|
|---|---:|---:|---:|
|`COMMON_SHIFT_R4`|-1.67pp／7|-0.83pp／8|-2.50pp／10|
|`FAST_FILM_R8`|+0.83pp／4|-0.83pp／3|-0.83pp／3|
|`FAST_LOWRANK_R8`|+0.83pp／3|-0.83pp／2|-1.67pp／4|

## 门控诊断与结论

- 影子状态证明三类Adapter都会实际改变query判决，排除了“代码路径未生效”。`COMMON_SHIFT_R4`随lambda增大持续退化，不再作为优先方向。
- 两个FAST候选在`leo_clear_weak`出现`+0.83pp`局部收益，但没有达到预注册`+1.0pp`门槛，且在另外两个场景反向；不存在跨三场景稳定的非零query上界。
- FAST在lambda=0.125时的support cross-fit风险增益为0.123～0.200，但最大特征移动为0.151～0.277，超过固定trust radius=0.15。每行完成6次cross-fit拟合和21次尝试梯度更新，提交更新为0；当前门控主要被trust约束卡住，不是连续风险没有改善。
- 最合理的下一步仍属于P0：在不查看query的前提下，用source receiver-held-out episode预标定相对trust尺度或加入更小lambda，并用新的冻结seed／receiver验证。不能根据本轮truth直接挑选lambda重跑同一query。
- 因为已有局部FAST信号且失败机制指向trust校准，暂不触发P1慢基／paired operator重训，更不进入P2中间层Adapter。最终结论为`SCIENTIFIC_SIGNAL_NO_PROMOTION`，不进入Target25。
