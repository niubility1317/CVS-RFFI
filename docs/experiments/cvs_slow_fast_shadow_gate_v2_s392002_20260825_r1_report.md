# CVS Slow-Fast影子门控V2实验报告

- run ID：`cvs_slow_fast_shadow_gate_v2_s392002_20260825_r1`
- 当前状态：`LOCAL_VERIFIED`
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

