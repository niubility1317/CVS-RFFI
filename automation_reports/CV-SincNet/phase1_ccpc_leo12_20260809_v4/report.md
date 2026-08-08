# Phase1 CCPC-LEO六折C/G v4完整训练报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`LOCAL_VERIFIED / PREREGISTERED / NOT_LAUNCHED`

证据边界：`PHASE1_SOURCE_ONLY_OPEN_WORLD_READY_REPRESENTATION_NON_CONFIRMATORY`

## 1.发布依据与假设

实验ID：`phase1_ccpc_leo12_20260809_v4`。时间：2026-08-09。operator：Codex主控；N607唯一runner：专属实验子代理。

G相对C只增加CCPC：LEO anchor对detached clean bank做class-conditional paired contrastive，同TX clean为正例、batch全部TX clean为分母；固定`T=0.12、lambda=0.02`。禁止RX/domain标签、GRL、MMD/CORAL、proxy/held训练、teacher、拒识head、阈值和扫参。

v3因审计scaled retained intermediate gradient而误停；该值不会被`GradScaler.unscale_(optimizer)`还原。修复后6fold G-only one-shot `phase1_ccpc_leo_gradient_audit6_20260809_v1`已在真实checkpoint/AMP下全部完成15epoch：每折raw未缩放CCPC梯度nonzero=450、nonfinite=0；finite parameter-grad与optimizer step均>=446；heldout跳过、promotion=false。v4从零重跑完整C/G，审计`autograd.grad(lambda*L_CCPC,z_leo)`，主loss/AMP/optimizer不变。

## 2.冻结矩阵

| Fold | train TX | known-validation TX | proxy TX | C/G GPU |
|---|---|---|---|---|
| F1 | 20-15,20-19,6-15,8-20 | 14-7 | 14-10 | 0/1 |
| F2 | 14-10,20-19,6-15,8-20 | 20-15 | 14-7 | 2/3 |
| F3 | 14-10,14-7,6-15,8-20 | 20-19 | 20-15 | 4/5 |
| F4 | 14-10,14-7,20-15,8-20 | 6-15 | 20-19 | 6/7 |
| F5 | 14-10,14-7,20-15,20-19 | 8-20 | 6-15 | 1/0 |
| F6 | 14-7,20-15,20-19,6-15 | 14-10 | 8-20 | 3/2 |

GPU0:F1C+F5G；GPU1:F1G+F5C；GPU2:F2C+F6G；GPU3:F2G+F6C；GPU4:F3C；GPU5:F3G；GPU6:F4C；GPU7:F4G。每卡最多2任务。固定40epoch、seed=7281105、sat_view_seed=9281105、AMP、AdamW、final-only；C/G按fold严格warm-start同一GeoSat-C C checkpoint并新建optimizer/AMP/RNG状态。

## 3.版本与本地验证

release commit：`ad261d2887d867c1993bca2f993f2d7b969000e6`；训练审计implementation commit：`753161c9127f72498507c8bbf4d7994bc4b7e698`。

| 文件 | SHA256 |
|---|---|
| `code/SSDG/train_ssdg.py` | `d8b23ae46a94e9c2f04130a7bcbfad5acd9255d020fe2f49762052228c6590b9` |
| `code/cvsrffi/phase1_ccpc_leo.py` | `88273c14ebd6579261a32697bb9d4e1a5626b2dd526f49bdbcc3c94a36679472` |
| `code/tests/test_phase1_ccpc_leo.py` | `92f948bee824f3ecd8f3a45085e089831e42d4a78cab47513479e0c094a9e7f3` |
| `code/scripts/launch_phase1_ccpc_leo12_20260809.sh` | `e4d39695f171e1449cddf090e91b47f30b28bc692c4a23eb89aac9c39bf4469e` |
| `analysis/phase1_ccpc_leo_design_20260809.md` | `9a7b01866f23595cf9475de65c8a24f6eff77701471167d965b61ad778d9b490` |

训练路径本地：py_compile通过；focused pytest=24 passed；full launcher bash-n与dry-run=12。one-shot和postfreeze实现均经独立复核`APPROVE / Critical=0 / Important=0`；postfreeze 12 tests、bash-n、dry-run=42。

## 4.N607路径与唯一命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v4_ad261d28`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v4`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v4`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v4.launch.out`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

```text
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v4_ad261d28/code && nohup setsid env RUN_ID=phase1_ccpc_leo12_20260809_v4 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v4_ad261d28/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl GEOSAT_CKPT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1 RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v4 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v4 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v4_ad261d28/code/scripts/launch_phase1_ccpc_leo12_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v4.launch.out 2>&1 < /dev/null & echo $!
```

## 5.健康、artifact与性能门

启动核launcher/CWD/cmdline、12 child、GPU映射、日志增长、CONFIG/E001。N/A占位、finite-zero和GradScaler合法skip不停止。raw未缩放CCPC梯度None/nonfinite、真实loss/model异常、OOM/CUDA、路径/P0、至少2任务同一异常或无进展才停止；只停本run，保留partial，retry=`NO`。

成功须12×E40、final checkpoint、metrics/config/terminal/heldout/resource receipts；G每折raw nonzero>=1、raw nonfinite=0、finite param-grad>=1、optimizer step>=1。训练完整后才运行已冻结42步postfreeze：12 clean导出+12 LEO导出+12 proxy连续评分+6 pair评分；proxy/held/LEO零fit、零校准、零选参。

五项非补偿门：技术健康；clean known 6/6的overall/minclass/minRX/minday G-C均>=-2pp；LEO 18/18四项G-C均>=-2pp且总体改善；source proxy连续排序相对C同向；真实checkpoint与artifact闭环。任一失败即`REJECT_CCPC_LEO_NO_RETRY`，不进入Phase3。

