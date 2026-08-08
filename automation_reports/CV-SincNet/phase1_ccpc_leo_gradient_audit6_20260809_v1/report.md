# Phase1 CCPC-LEO未缩放梯度6折one-shot技术审计

目标模式：`GOAL_MODE=ACTIVE`

状态：`LOCAL_VERIFIED / PREREGISTERED / NOT_LAUNCHED`

用途：`TECHNICAL_ONLY / NO_PERFORMANCE_RESULT / NO_POSTFREEZE`

## 1.目标与边界

实验ID：`phase1_ccpc_leo_gradient_audit6_20260809_v1`。时间：2026-08-09。operator：Codex主控；N607唯一runner：专属实验子代理。

v3错误地审计GradScaler放大后的非参数中间梯度；`unscale_(optimizer)`不会还原它，因此scaled overflow不能判定CCPC失败。本one-shot只验证真实checkpoint下的未缩放CCPC专属梯度、参数梯度和optimizer step。它不比较C/G性能、不访问heldout scorer、不执行postfreeze、不允许晋级。

方法、数据和优化不变：每fold从原GeoSat-C C checkpoint严格warm-start；同physical clean/LEO，detached clean bank，同TX正例、batch全TX clean分母；`T=0.12、lambda=0.02`；原AMP/AdamW/seed/sampler。唯一变化是健康审计在scaled backward前计算`autograd.grad(lambda*L_CCPC,z_leo)`。

## 2.冻结6fold G-only矩阵

| Fold | train TX | known-validation TX | proxy TX | GPU |
|---|---|---|---|---|
| F1 | 20-15,20-19,6-15,8-20 | 14-7 | 14-10 | 0 |
| F2 | 14-10,20-19,6-15,8-20 | 20-15 | 14-7 | 1 |
| F3 | 14-10,14-7,6-15,8-20 | 20-19 | 20-15 | 2 |
| F4 | 14-10,14-7,20-15,8-20 | 6-15 | 20-19 | 3 |
| F5 | 14-10,14-7,20-15,20-19 | 8-20 | 6-15 | 4 |
| F6 | 14-7,20-15,20-19,6-15 | 14-10 | 8-20 | 5 |

固定15epoch、seed=7281105、sat_view_seed=9281105、AMP=true、final-only；GPU0–5各1任务，GPU6–7空闲。

## 3.版本与本地验证

implementation commit：`753161c9127f72498507c8bbf4d7994bc4b7e698`。

| 文件 | SHA256 |
|---|---|
| `code/SSDG/train_ssdg.py` | `d8b23ae46a94e9c2f04130a7bcbfad5acd9255d020fe2f49762052228c6590b9` |
| `code/cvsrffi/phase1_ccpc_leo.py` | `88273c14ebd6579261a32697bb9d4e1a5626b2dd526f49bdbcc3c94a36679472` |
| `code/tests/test_phase1_ccpc_leo.py` | `92f948bee824f3ecd8f3a45085e089831e42d4a78cab47513479e0c094a9e7f3` |
| `code/scripts/launch_phase1_ccpc_leo_gradient_audit6_20260809.sh` | `289ab4201602aee4a13b65357db7631d2dd2281b87d2f0b105225317de499edc` |
| `analysis/phase1_ccpc_leo_design_20260809.md` | `9a7b01866f23595cf9475de65c8a24f6eff77701471167d965b61ad778d9b490` |

`ssr-gpu`验证：py_compile通过；focused pytest=24 passed；bash-n通过；dry-run=6条；git diff check通过。独立复核：`APPROVE / Critical=0 / Important=0`。

## 4.N607路径与唯一命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo_gradient_audit6_20260809_v1_753161c9`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo_gradient_audit6_20260809_v1`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo_gradient_audit6_20260809_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo_gradient_audit6_20260809_v1.launch.out`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

```text
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo_gradient_audit6_20260809_v1_753161c9/code && nohup setsid env RUN_ID=phase1_ccpc_leo_gradient_audit6_20260809_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo_gradient_audit6_20260809_v1_753161c9/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl GEOSAT_CKPT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1 RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo_gradient_audit6_20260809_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo_gradient_audit6_20260809_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo_gradient_audit6_20260809_v1_753161c9/code/scripts/launch_phase1_ccpc_leo_gradient_audit6_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo_gradient_audit6_20260809_v1.launch.out 2>&1 < /dev/null & echo $!
```

## 5.技术成功与停止规则

每fold成功必须15/15、terminal=`TECHNICAL_AUDIT_COMPLETE`、raw CCPC nonzero>=1、raw CCPC nonfinite=0、finite parameter-grad>=1、optimizer step>=1；`frozen_phase1_heldout_eval.json`必须为`SKIPPED_TECHNICAL_AUDIT/NO_PERFORMANCE_RESULT`，terminal与CCPC receipt均`technical_only=true、promotion_ready=false`。

N/A占位与有限零梯度不停止；GradScaler合法skip可计数。raw未缩放梯度None/nonfinite、真实loss/model故障、路径/hash/overwrite错误、OOM/CUDA或至少2fold同一确定性异常才触发系统停止。只停止精确run树，保留partial；不重启、不调参、不远端改码，retry=`NO`。只回收小receipt、metrics和日志，不下载checkpoint，不读取/解释任何性能。

