# Phase1 WRC-NCT三场景LEO floor v2实验报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`LOCAL_VERIFIED / PREREGISTERED`

证据边界：`PHASE1_SOURCE_ONLY_LEO_PRESSURE_PROXY_NON_CONFIRMATORY`

## 1.目标与v1技术修复

run ID：`phase1_wrc_nct_leo6x3_20260809_v2`。时间：2026-08-09。

v1的6条export在产生NPZ前全部因通用exporter强制要求`target_old_tx_ids`而退出；score未启动，状态为`NO_PERFORMANCE_RESULT`。v2新增显式`--source_only_export`：在任何数据加载前禁止target-old/new/unknown/proxy TX参数，跳过new数据文件读取，并在manifest写入source-only与target禁用状态。方法、checkpoint、三场景、seed、batch32、GPU映射、固定readout和score矩阵均不变；v1路径与artifact不复用。

## 2.冻结矩阵与协议

| Fold | checkpoint | GPU | clean readout/GI |
|---|---|---:|---|
| F1 | F1C_ManyTxRealOE12 | 0 | F1 |
| F2 | F2C_ManyTxRealOE12 | 1 | F2 |
| F3 | F3C_ManyTxRealOE12 | 2 | F3 |
| F4 | F4C_ManyTxRealOE12 | 3 | F4 |
| F5 | F5C_ManyTxRealOE12 | 4 | F5 |
| F6 | F6C_ManyTxRealOE12 | 5 | F6 |

场景固定为`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`和`simplified_leo_residual`。每fold一次2000行source-only export；每physical一行、一次LEO叠加、一个scenario，三scenario物理集合互斥。LEO physical全集必须等于clean source全集；clean CSV只继承R/C/E和同physical paired baseline。固定clean readout/tau，不fit、不calibrate。

## 3.版本、验证与复核

实现commit：`ff1f295ef3407d5d5190a2a6af3313c31b0cddda`。

| 文件 | SHA256 |
|---|---|
| `code/export_spaceborne_features.py` | `7021a1c8a2ed4891d0a4f06f92892535b8b35d3562805fd1fb3fdec5701c0b38` |
| `code/scripts/eval_phase1_wrc_nct_leo.py` | `1175a0f4ee94c43b60608ad90ff5c84aa4bcc8b8b9fc1f4ee1d0119341304635` |
| `code/scripts/launch_phase1_wrc_nct_leo6x3_20260809.sh` | `6755d83b3523f09e811c3afb11839b7cc5286b62573eacd1cd6a8d56e8464da9` |
| `code/tests/test_export_spaceborne_features_source_only.py` | `b63d8f49a72d242e68acac28ec90e6047a9a2cfe888a74584eed50fec9dd1dc4` |
| `code/tests/test_phase1_wrc_nct_leo.py` | `3d0ee95f67ef8e160439c758e21846224892cec9a9efbc63ae9fbbae76389a41` |

本地`ssr-gpu`：联合26/26通过；py_compile、bash-n通过；dry-run精确6 export+6 score，6 export均带source-only锁。

独立复核：`VERDICT=APPROVE / P0=0 / P1=0 / ALLOW_NEW_RUN_ID=YES`；确认是纯技术修复，未改方法、阈值或矩阵。

## 4.N607冻结路径与命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_wrc_nct_leo6x3_20260809_v2_ff1f295e`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_wrc_nct_leo6x3_20260809_v2`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_wrc_nct_leo6x3_20260809_v2`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_wrc_nct_leo6x3_20260809_v2.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- TRAIN_ROOT/CLEAN_ROOT/GI_ROOT：沿用v1报告冻结值。

```text
cd <release>/code && nohup setsid env RUN_ID=phase1_wrc_nct_leo6x3_20260809_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python TRAIN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_20260808_v2 CLEAN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_wrc_nct_clean6_20260809_v2 GI_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gi_epior_clean6_20260809_v3 bash <release>/code/scripts/launch_phase1_wrc_nct_leo6x3_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_wrc_nct_leo6x3_20260809_v2.launch.out 2>&1 < /dev/null & echo $!
```

## 5.闭环与裁决门

每fold预期NPZ2000行、metrics JSON、scores CSV2001行；6 export与6 score必须全部exit0。只回收metrics、scores、日志、completion和manifest，不下载NPZ/checkpoint/GI/readout/runtime。技术失败才停止；启动一次，`retry=NO`。

18个fold×scenario原子格的两类overall/min-class/min-RX/min-day下降均须≤2个百分点：一类是WRC相对同LEO closed的附加下降；另一类是LEO fixed-WRC相对同physical paired-clean fixed-WRC的下降。任一格失败即`REJECT_LEO_FLOOR`；18格全过才完成Phase1五门。
