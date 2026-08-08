# Phase1 WRC-NCT三场景LEO floor v2实验报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`ANALYZED / REJECT_LEO_FLOOR / PHASE1_FIVE_GATES_NOT_COMPLETE`

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

## 6.技术终态（2026-08-09）

状态：`ARTIFACTS_COMPLETE / NO_PERFORMANCE_INTERPRETATION`。本run唯一启动，launcher PID=`3960893`；6条GPU export（F1→GPU0、F2→GPU1、F3→GPU2、F4→GPU3、F5→GPU4、F6→GPU5）和6条CPU score均exit=0，未执行重试。archive为commit `ff1f295ef3407d5d5190a2a6af3313c31b0cddda`的no-prefix tar，SHA256=`043d02170d752223856d089bcdfa104b6c63fcbd86119e1bfe8aaabfa769ba89`；远端archive/LF与Windows/worktree CRLF字节口径差异已在artifact manifest记录，未远端改码。

每fold NPZ结构只读核验：2000行、2000个unique physical key、role仅`source`；manifest `source_only_export=true`且target-old/new/unknown禁用；三场景计数`leo_clear_weak=672`、`leo_low_elev_weak=672`、`leo_rain_weak=656`，场景集合互斥，每场景5个TX/6个RX，NPZ physical key集合与clean source CSV逐行相等。未读取或回收IQ/features数组。

每fold `leo_scores.csv`为2001行（header+2000）；6 metrics JSON、6 score CSV、12 stdout、`export_completion.tsv`、`score_completion.tsv`及结构manifest已回收到：`E:\type10-7\automation_reports\CV-SincNet\phase1_wrc_nct_leo6x3_20260809_v2\artifacts`。共27个小文件；最终`manifest.json`为10178字节，SHA256=`77a37aaafd170cf04a70cd7b91e4279e01cd1af2e74fd9621a8b5b1719946628`，逐文件SHA/大小均记录其中。未回收NPZ、checkpoint、GI bundle、clean readout或runtime。远端launcher/exporter/scorer均已退出，GPU0-7均0%/1MiB，临时archive已删除，SSH/SCP连接已清理；本节不构成性能、LEO floor或Phase1五门结论。

## 7.十八格同physical配对结果

表中`WRC附加下降`依次为overall/min-class/min-RX/min-day，比较固定WRC拒识后的full与同一LEO观测的closed结果；`LEO配对下降`使用同一physical ID、同一冻结R/C/E定义和同一固定WRC读出，比较LEO fixed-WRC与paired-clean fixed-WRC。下降均以百分点计，任一分量大于2即该原子格失败。

| Fold | LEO场景 | E行数 | LEO closed | WRC full | WRC附加下降（四项） | LEO配对下降（四项） | 原子格结论 |
|---|---|---:|---:|---:|---|---|---|
| F1 | clear | 159 | 82.39% | 81.76% | 0.63/0.00/2.26/0.34 | 16.35/33.33/23.31/16.09 | FAIL |
| F1 | low-elev | 176 | 76.14% | 76.14% | 0.00/0.00/0.00/0.00 | 21.59/32.57/39.29/25.68 | FAIL |
| F1 | rain | 165 | 80.61% | 79.39% | 1.21/0.00/0.00/2.17 | 19.39/50.00/21.38/23.91 | FAIL |
| F2 | clear | 159 | 86.79% | 85.53% | 1.26/0.00/6.06/1.43 | 14.47/28.21/30.30/21.43 | FAIL |
| F2 | low-elev | 179 | 85.47% | 84.36% | 1.12/0.00/3.33/1.32 | 15.08/23.70/23.10/17.45 | FAIL |
| F2 | rain | 162 | 85.80% | 84.57% | 1.23/0.00/3.59/1.32 | 15.43/48.15/26.92/15.79 | FAIL |
| F3 | clear | 160 | 80.62% | 76.88% | 3.75/3.98/8.70/2.82 | 21.88/31.56/36.10/22.54 | FAIL |
| F3 | low-elev | 181 | 70.17% | 66.30% | 3.87/0.00/3.45/2.60 | 30.94/57.03/48.28/33.48 | FAIL |
| F3 | rain | 159 | 70.44% | 67.30% | 3.14/0.00/3.85/2.41 | 31.45/63.33/41.81/36.14 | FAIL |
| F4 | clear | 155 | 89.68% | 88.39% | 1.29/0.00/0.00/1.20 | 10.97/25.00/15.83/11.20 | FAIL |
| F4 | low-elev | 182 | 89.01% | 84.62% | 4.40/0.00/11.54/6.76 | 14.29/26.47/30.77/17.57 | FAIL |
| F4 | rain | 163 | 84.66% | 82.21% | 2.45/0.00/6.90/1.25 | 16.56/38.24/30.32/23.75 | FAIL |
| F5 | clear | 161 | 88.20% | 86.96% | 1.24/0.00/3.12/0.00 | 13.04/21.88/28.12/13.85 | FAIL |
| F5 | low-elev | 180 | 82.22% | 80.00% | 2.22/8.82/3.85/0.91 | 17.22/41.18/26.92/16.36 | FAIL |
| F5 | rain | 159 | 72.33% | 67.92% | 4.40/7.41/4.17/1.25 | 30.19/57.08/36.40/33.75 | FAIL |
| F6 | clear | 161 | 63.35% | 59.63% | 3.73/15.62/0.00/8.06 | 37.27/62.30/46.84/50.00 | FAIL |
| F6 | low-elev | 179 | 67.60% | 64.25% | 3.35/14.71/0.00/3.62 | 34.08/67.65/64.00/34.59 | FAIL |
| F6 | rain | 160 | 60.00% | 58.75% | 1.25/5.88/1.85/2.53 | 39.38/69.83/46.85/48.10 | FAIL |

汇总：WRC拒识相对同LEO closed的附加floor门仅2/18格全过、16/18格失败；同physical paired-clean LEO floor门为0/18格通过、18/18格失败。WRC附加下降的最坏值为overall 4.40、min-class 15.62、min-RX 11.54、min-day 8.06个百分点；LEO配对下降的最坏值为overall 39.38、min-class 69.83、min-RX 64.00、min-day 50.00个百分点。

## 8.Phase1五门裁决

| Phase1晋级门 | 证据 | 裁决 |
|---|---|---|
| 1.模型健康 | 6/6 export与6/6 score均exit0；结构、物理ID、场景互斥和日志闭环 | PASS |
| 2.已知类跨接收机无明显退化 | clean v2六折overall/min-class/min-RX/min-day相对同E closed下降均为0 | PASS |
| 3.最低类别与LEO floor无严重下降 | 16/18格WRC附加门失败，18/18格paired-clean LEO门失败 | **FAIL** |
| 4.source proxy unknown有明确正信号 | clean v2 proxy AUROC均值0.9307，但FAR均值96.14%，仅构成排序正信号 | PASS（弱，仅Phase1 proxy） |
| 5.真实checkpoint可导出deployment bundle | 六折固定readout/runtime和source-only LEO证据均成功导出并通过parity | PASS |

最终裁决：`REJECT_LEO_FLOOR / PHASE1_FIVE_GATES_NOT_COMPLETE / CLOSE_Q98_NCT_FAMILY / NO_PHASE3_RELEASE`。

WRC-NCT证明相对NCT几何在clean source proxy上具有排序信号，但固定source calibration阈值在真实LEO弱信道下既不能保护最低类别、接收机与日期floor，也不能把paired-clean性能迁移到LEO观测。clean proxy信号不得补偿第3门失败，更不得写成Phase3真实unknown结果。按预注册边界，本结果关闭当前source-held Q98/NCT阈值族：不再扫描`alpha`、分位数、RX聚合或场景阈值，不以局部fold/场景挑选替代完整18格结论。

Phase3本轮不启动。下一轮Phase1若继续，应转向会改变底层表征本身的训练方法，直接优化跨RX和LEO类条件稳定性，而不是继续给冻结表征叠加source-only阈值读出。
