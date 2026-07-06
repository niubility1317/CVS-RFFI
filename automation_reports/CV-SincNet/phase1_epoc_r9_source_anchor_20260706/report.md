# Phase1 EPOC R9 Source Anchor N607 report

| field | value |
|---|---|
| experiment ID | `phase1_epoc_r9_source_anchor_20260706` |
| timestamp | 2026-07-06 08:19 CST |
| operator | Codex |
| objective | 在`ADV3B02_CORE90_SOFT_E200`基础上启动下一条source-only特征几何修复路线，避免继续只调协同阈值/riskgate。 |
| base teacher | `runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| algorithm label | `ADV3B02_SOURCE_ANCHOR` |
| route | `source_only_teacher_anchor_stable_feature_repair` |
| scope | Phase1训练；后续必须再做Stage2-C qknn8 M=1..all target receivers评估。 |
| verdict scope before launch | `NOT_EVALUATED`，不得写成Stage2-C成功或部署成功。 |

## Hypothesis

R7/R8已经显示协同投票和riskgate不能弥补当前LEO target特征几何不足。R9改为source-only old-class geometry repair：用`ADV3B02_CORE90_SOFT_E200` teacher KL/MSE保护旧类身份空间，加强ManySig旧类类内收缩、源域episode半径约束和LEO view一致性；unknown拒识只通过旧类包络外的virtual negatives形成风险边界，地面训练阶段不接触真实未知类。

## Comparison target

| prior route | evidence | limitation |
|---|---|---|
| R7 base qknn8 collaboration | M=1..5下old最高约21.53%，seen-new为0%，unknown reject随M增加下降。 | 协同数量增加不解决特征不可分。 |
| R8 current-best qknn8 riskgate | unknown reject可到约95%，但old约16%-31%，seen-new约0%-10%。 | 拒识换覆盖，不是可部署open-set识别。 |
| R8 live Phase1 proxy metrics | proxy AUC约0.38-0.40，virtual accept约0.81，nonfinite_grad记录为1.0。 | 强proxy shell压主干不稳定。 |

## Protocol boundary

| item | value |
|---|---|
| phase1 dataset | `ManySig.pkl` only |
| real unknown classes in ground training | 0 |
| target receiver samples in ground training | 0 |
| target unknown training count | 0 |
| ManyTx in training | 0 |
| proxy unknown real TX calibration | 0 |
| virtual unknown only | 1 |
| threshold selection label scope | `support_or_source_old_only` |
| Stage2 unknown query | eval-only |
| Stage2 success claim | 0 |
| deployment success claim | 0 |
| qknn deployment head | qknn8 after prototype export |
| collaboration counts for later eval | M=1..all target receivers |
| target channel view for later eval | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` |

## Candidate design

| candidate | GPU | seed | mechanism | expected diagnostic |
|---|---:|---:|---|---|
| `EPOC_R9_ANCHOR_NOPROXY` | 2 | 706901 | No real/proxy unknown pressure；stronger teacher clean/sat KL and source episode geometry. | Checks whether protecting old geometry alone restores downstream qknn8 old/seen-new separation. |
| `EPOC_R9_GENTLE_VIRTUAL_LATE` | 3 | 706911 | Delayed weak virtual-only boundary；`lambda_proxy_unknown=0.0020` from epoch90, no real unknown IDs. | Checks whether a light late virtual shell improves unknown reject without collapsing old/seen-new. |

## Local changes

| file | purpose |
|---|---|
| `code/scripts/launch_phase1_epoc_r9_source_anchor_20260706.sh` | N607 launcher for two R9 source-only Phase1 candidates. |
| `code/tests/test_phase1_epoc_r9_source_anchor_launcher.py` | Focused launcher tests for source-only protocol fields, two-candidate GPU allocation, and fail-closed forbidden training inputs. |
| `automation_reports/CV-SincNet/phase1_epoc_r9_source_anchor_20260706/report.md` | This experiment report. |
| `code/snapshots/phase1_epoc_r9_source_anchor_20260706/launch_phase1_epoc_r9_source_anchor_20260706.sh` | Local snapshot before N607 sync. |
| `code/SYNC_MANIFEST.txt` | Local-to-remote mapping and verification trail. |

## Verification plan

| command | expected |
|---|---|
| `bash -n code/scripts/launch_phase1_epoc_r9_source_anchor_20260706.sh` | PASS |
| `bash code/scripts/launch_phase1_epoc_r9_source_anchor_20260706.sh --dry-run --only=EPOC_R9_ANCHOR_NOPROXY` | PASS；prints source-only fields and no `ManyTx.pkl`/`--proxy_unknown_tx_ids` |
| `WISIG_PKL=/tmp/ManyTx.pkl bash code/scripts/launch_phase1_epoc_r9_source_anchor_20260706.sh --dry-run --only=EPOC_R9_ANCHOR_NOPROXY` | fail closed |
| `conda run -n ssr-gpu python -m pytest code/tests/test_phase1_epoc_r9_source_anchor_launcher.py -q` | PASS |
| `conda run -n ssr-gpu python -m py_compile code/tests/test_phase1_epoc_r9_source_anchor_launcher.py code/SSDG/train_ssdg.py` | PASS |

## N607 launch plan

| field | value |
|---|---|
| remote cwd | `/home/szu2070436088/2510044040/CV-SincNet` |
| server command | `cd /home/szu2070436088/2510044040/CV-SincNet; bash code/scripts/launch_phase1_epoc_r9_source_anchor_20260706.sh` |
| environment | remote Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` from launcher |
| run root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_epoc_r9_source_anchor_20260706` |
| log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_epoc_r9_source_anchor_20260706` |
| candidate logs | `EPOC_R9_ANCHOR_NOPROXY.out`;`EPOC_R9_GENTLE_VIRTUAL_LATE.out` |
| expected outputs | `best_joint_safe_ssdg.pth`,`latest_safe_ssdg.pth`,`latest_ssdg.pth`,`phase2_zid_prototypes.pt`,`phase2_zid_prototypes.json`,`metrics_epoch.csv` per candidate |
| GPU allocation | GPU2/GPU3, selected because prior monitor showed GPU0/GPU1 occupied by R8 and GPU2/GPU3 low-memory after R8 riskgate finished. |
| max active per GPU | launcher default `MAX_ACTIVE_PER_GPU=2` |

## Metrics to watch

| metric group | watch item |
|---|---|
| source old stability | `best_test_tx`,`best_val_tx`,`train_tx`,`joint_safe` |
| numerical stability | `train_skipped_nonfinite_grad`, log scan for Traceback/RuntimeError/CUDA-OOM/NaN/unrecognized/Killed |
| open-set proxy diagnostic | proxy AUC, virtual accept, proxy accept, source episode overflow; these are diagnostic only, not real unknown training evidence |
| prototype readiness | `phase2_zid_prototypes.pt` and `.json` export |
| downstream gate | after export, run qknn8 Stage2-C with M=1..5 target receivers and true unknown eval-only |

## Risks

| risk | mitigation |
|---|---|
| Virtual negatives may still not match real LEO unknowns. | Treat Phase1 as representation repair only；prove with Stage2-C eval-only unknown query later. |
| No-proxy candidate may protect old classes but not improve unknown reject. | Compare with gentle late virtual candidate using same qknn8 evaluator. |
| Gentle virtual candidate may repeat R8 collapse if even weak virtual shell perturbs old geometry. | Candidate A remains no-proxy control；monitor nonfinite gradients and old test accuracy. |
| Exact resource design file was not found locally. | Report only proxy/offline resource metrics until the real document is located. |

## Initial local verification

| check | result |
|---|---|
| `bash -n` | PASS at 2026-07-06 08:19 CST |
| dry-run `EPOC_R9_ANCHOR_NOPROXY` | PASS；prints ManySig-only, no real unknown, no `--proxy_unknown_tx_ids` |
| `WISIG_PKL=/tmp/ManyTx.pkl` guard | PASS；launcher exits nonzero with fail-closed error |
| full dry-run all candidates | PASS；prints `CUDA_VISIBLE_DEVICES=2` and `CUDA_VISIBLE_DEVICES=3` |
| pytest | PASS；`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n ssr-gpu python -m pytest code\tests\test_phase1_epoc_r9_source_anchor_launcher.py -q` => 3 passed with only `.pytest_cache` permission warning |
| py_compile | PASS；`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n ssr-gpu python -m py_compile code\tests\test_phase1_epoc_r9_source_anchor_launcher.py code\SSDG\train_ssdg.py` |
| environment note | First parallel `conda run` hit the known Windows temp-file lock and one non-UTF-8 retry hit conda GBK output encoding；serial UTF-8 rerun passed. |

## Remote sync and launch

| item | result |
|---|---|
| N607 preflight | PASS at 2026-07-06 08:21 CST；project root visible；GPU0/1 occupied by R8 at about2527/2447MiB；GPU2/3 idle at about10MiB. |
| remote hash verify | PASS；launcher `9dc39562dfaa8a1a07a492250a59fc508a8939f15e8bf58d55135d1ab50529dd`，test `89776390d528d70b13367ca36bd88c114625e6d25d3603d08c8e25b43cf1eabb`，snapshot same as launcher，report `551332664a8a5d0707404af2f6b56fdca616ceca926d9c17f070641eae631139`. |
| remote syntax/dry-run | PASS；remote `bash -n` PASS；dry-run prints `phase1_dataset=ManySig_only`、`virtual_unknown_only=1`、`stage2_success_claim=0`、`deployment_success_claim=0`. |
| remote fail-closed guard | PASS；`WISIG_PKL=/tmp/ManyTx.pkl` exits nonzero and reports refusing non-source Phase1 input. |
| remote py_compile | PASS；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/tests/test_phase1_epoc_r9_source_anchor_launcher.py code/SSDG/train_ssdg.py`. |
| launch command | `cd /home/szu2070436088/2510044040/CV-SincNet; nohup bash code/scripts/launch_phase1_epoc_r9_source_anchor_20260706.sh > logs/phase1_epoc_r9_source_anchor_20260706/driver.out 2>&1 &` |
| driver PID | `3320947`，driver submitted both candidates and exited after submit. |
| candidate PIDs | `EPOC_R9_ANCHOR_NOPROXY` main PID `3320959` on GPU2；`EPOC_R9_GENTLE_VIRTUAL_LATE` main PID `3321795` on GPU3. |
| logs | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_epoc_r9_source_anchor_20260706/EPOC_R9_ANCHOR_NOPROXY.out`；`.../EPOC_R9_GENTLE_VIRTUAL_LATE.out`. |
| SSH cleanup | After sync, verify, launch, and monitor commands: no local `ssh.exe` and no N607/bridge ESTABLISHED port22 connections. |

## Startup health

| timestamp | status |
|---|---|
| 2026-07-06 08:26 CST | Both candidates launched；GPU2 about1815MiB, GPU3 log just created；error scan empty. |
| 2026-07-06 08:28 CST | Both candidates reached epoch loop；GPU2/GPU3 about2249/2225MiB；logs contain `EPOCH-BEGIN`/`EPOCH-END`; error scan empty. |
| 2026-07-06 08:30 CST | Both candidates reached E009/200；GPU2/GPU3 about2243/2245MiB；`latest_safe_ssdg.pth` and `latest_ssdg.pth` exist for both candidates；error scan empty for Traceback/RuntimeError/CUDA-OOM/out-of-memory/unrecognized/Killed. |

Current claim boundary: R9 has only startup-health evidence. It is not Stage2-C evidence and not a deployment success claim. Stage2-C qknn8 collaborative inference with M=1..all target receivers remains pending until Phase1 completes and exports `phase2_zid_prototypes.pt`.

## 2026-07-06 08:34 CST训练中监控

N607只读预检PASS。GPU2/GPU3分别约2501/2381MiB，R9两个候选仍在运行并写入safe/latest checkpoint，但尚未导出`phase2_zid_prototypes.pt`或`.json`，因此不能启动Stage2-C qknn8协同评估。

|candidate|epoch|best_epoch|best_score|best_test_tx|train_tx_acc|source_overflow|skipped_nonfinite_grad|prototype|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|`EPOC_R9_ANCHOR_NOPROXY`|18/200|10|85.4510|89.9225|94.1106|0.9354|0.0000|absent|训练继续；当前仍是无proxy阶段，旧类稳定性略高于R8早期，但未到可评估阶段。|
|`EPOC_R9_GENTLE_VIRTUAL_LATE`|17/200|10|84.1287|89.7529|95.1923|0.9016|0.0000|absent|训练继续；proxy计划E90后才激活，当前不能判断unknown拒识。|

错误扫描：`Traceback`、`RuntimeError`、`CUDA out of memory`、`out-of-memory`、`unrecognized arguments`、`Killed`命中数为0。SSH清理已确认本地无`ssh.exe`且无N607/bridge 22端口ESTABLISHED连接。

边界：R9仍只是Phase1训练中证据，不是Stage2-C证据，也不能声明部署成功。下一步必须等待训练完成并导出prototype后，再用qknn8协同推理`M=1..all target receivers`做同row旧类、seen-new和真实unknown eval-only复评。

## 2026-07-06 08:39 CST训练中监控

N607只读预检PASS。GPU2/GPU3分别约2521/2403MiB，两个R9候选仍在运行；`best_joint_safe_ssdg.pth`、`latest_safe_ssdg.pth`、`latest_ssdg.pth`存在，但尚未导出`phase2_zid_prototypes.pt/json`，因此不能启动Stage2-C qknn8协同评估。

|candidate|epoch|best_epoch|best_score|best_test_tx|train_tx_acc|source_overflow|skipped_nonfinite_grad|prototype|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|`EPOC_R9_ANCHOR_NOPROXY`|22/200|10|85.4510|89.9225|93.3774|0.9329|0.0154|absent|训练继续；无proxy主线仍处早期，轻微非有限跳过需继续观察。|
|`EPOC_R9_GENTLE_VIRTUAL_LATE`|21/200|20|85.0565|89.7819|92.2596|0.9180|0.0000|absent|训练继续；proxy计划E90后激活，当前不能判断unknown拒识。|

错误扫描：`Traceback`、`RuntimeError`、`CUDA out of memory`、`out-of-memory`、`unrecognized arguments`、`Killed`命中数为0。SSH清理已确认本地无`ssh.exe`且无N607/bridge 22端口ESTABLISHED连接。

边界：R9仍是Phase1训练中证据；最终目标需要等prototype导出后再做qknn8协同推理`M=1..all target receivers`同row复评。
