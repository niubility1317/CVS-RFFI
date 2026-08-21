# ADV3B02 SID-FFT96 Phase1实验报告

## 当前结论

- 状态：`LOCAL_VERIFICATION_IN_PROGRESS`。
- 本轮仅发布前端频谱可辨识性路线P0与S0–S3单seed矩阵；尚无N607性能结果，不声明优于CORE90。
- 基座固定为`ADV3B02_CORE90_SOFT_E200`，不修改Phase2/Phase3，不使用target/query，不与NTRS或CRRA组合。

## 最小预登记

- run ID：`phase1_advb02_sidfft96_leo_weak_20260821_v1`
- 实现基线提交：`e139e8078b6e3bc39f80432751601d6d5f683d1d`
- Git分支：`codex/advb02-ntrs-v2-recovery-20260820`
- 单seed：`392002`
- Phase1源角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`
- source/target接收机：`R_s∩R_t=∅`
- 训练场景：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`
- Core90卫星策略：`concat_sat_ce_only=true`、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`、卫星监督E80开始；E1–40仅clear且`p=0.30`，E41–90为low_elev+rain且`p=0.60`，E91–200为三场景且`p=0.80`。
- 最终评测：clean及三种LEO_WEAK逐场景，禁止只用聚合均值替代逐场景结果。

## 实验矩阵

|row|候选|输入掩码|训练策略|用途|
|---|---|---|---|---|
|P0|`P0_SPECTRAL_AUDIT`|无|只遍历`L_s`，统计TX散度、RX/day/LEO散度与噪声|生成固定center/phase/SID掩码|
|S0|`S0_FROZEN_CORE90`|无|不训练，仅冻结checkpoint评测|同checkpoint基线|
|S1|`S1_CENTER`|P0固定中心带|冻结ADV3B02，仅训练零初始化SID投影|中心频带对照|
|S2|`S2_PHASE`|P0固定相位带|冻结ADV3B02，仅训练零初始化SID投影|相位证据对照|
|S3|`S3_SIDFFT96`|P0多频带J-score掩码|冻结ADV3B02，仅训练零初始化SID投影|完整96维幅相SID候选|

## 环境、输入与输出

- 本地CWD：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\advb02-ntrs-leo-weak-20260820`
- N607项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 源数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 基座checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_sidfft96_leo_weak_20260821_v1`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_sidfft96_leo_weak_20260821_v1`
- launcher：`code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh`

## 精确命令

矩阵dry-run：

```bash
bash code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh --dry-run
```

独立row发布形式：

```bash
bash code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh --only=P0
bash code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh --only=S0
bash code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh --only=S1
bash code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh --only=S2
bash code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh --only=S3
```

GPU由发布时只读preflight后通过`GPU_P0/GPU_S0/GPU_S1/GPU_S2/GPU_S3`记录；每GPU训练进程不超过2个。

## 科学停止与晋级规则

- 技术停止只允许：协议/数据角色/seed/场景错误、错误release或CWD、输出碰撞、进程归属不清、确定性同类预prediction异常至少重复两次、无法产生checkpoint或独立prediction闭合。
- 低性能不停止实验，只进入同row分析。
- S3晋级门槛：clean下降不超过0.3pp；LEO均值至少提升1pp；Strict UDU至少提升1pp；LEO floor至少提升0.5pp；`rescued>harmed`；RX probe相对下降至少20%且TX probe不下降。
- 在真实prediction与独立scorer闭合前，不把本次发布描述为性能改善。

## 预期artifact

- P0：`spectral_identifiability.json`、`spectral_identifiability.csv`、`spectral_identifiability.png`、`sid_mask.npz`
- S0：`final_eval.json`、`final_eval.txt`
- S1–S3：`final_ssdg.pth`、`metrics_epoch.csv`、`metrics_epoch.jsonl`、`phase1_terminal_status.json`、`independent_final_eval/final_eval.json`、`independent_final_eval/final_eval.txt`
- 所有完成训练的候选必须保留checkpoint身份、clean和三种LEO_WEAK逐场景结果。

## 发布后待追加

- 本地聚焦验证与真实checkpoint无query smoke
- 一次独立P0/P1正确性审查
- release本地/远端单次SHA比较及远端编译/dry-run
- P0完成证据、各row PID/CWD/cmdline/GPU/log增长证据
- prediction闭合后的同row指标、异常、解释与下一候选决定
