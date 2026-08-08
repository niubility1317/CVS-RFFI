# Phase1 WRC-NCT三场景LEO floor实验报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`LOCAL_VERIFIED / PREREGISTERED`

证据边界：`PHASE1_SOURCE_ONLY_LEO_PRESSURE_PROXY_NON_CONFIRMATORY`

## 1.目标与前置结论

run ID：`phase1_wrc_nct_leo6x3_20260809_v1`。时间：2026-08-09。主Agent负责设计与五门裁决；唯一N607 runner只负责技术落地、运行和小artifact回收。

WRC-NCT clean v2已6/6通过：known overall/min-class/min-RX/min-day下降均为0；proxy FAR六折均低于100%且AUROC均高于0.926；readout/runtime闭环。该信号仍很弱，不等于真实unknown能力。

本run只回答：冻结clean readout在三种LEO物理压力代理下是否保留已知类overall和floor。LEO阶段不fit、不calibrate、不改阈值，不评价proxy/held。

## 2.数据、方法与矩阵

- 场景：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；实现：`simplified_leo_residual`。
- 每fold一次source-only export，共2000行；`shuffle=false,batch_size=32`，三个场景按batch确定性轮换。
- 每个physical ID只出现一行、接受一次LEO叠加并绑定一个scenario；三scenario physical集合两两互斥。
- LEO source physical全集必须等于clean source全集。
- clean CSV只继承冻结R/C/E membership并提供同physical paired-clean baseline；不能用于阈值或参数选择。
- 同fold clean readout的`tau`及GI geometry完全冻结。
- 每scenario的E必须覆盖全部5个source TX和6个source RX；known reject计错。

| Fold | checkpoint | GPU | clean readout/GI |
|---|---|---:|---|
| F1 | F1C_ManyTxRealOE12 | 0 | F1 |
| F2 | F2C_ManyTxRealOE12 | 1 | F2 |
| F3 | F3C_ManyTxRealOE12 | 2 | F3 |
| F4 | F4C_ManyTxRealOE12 | 3 | F4 |
| F5 | F5C_ManyTxRealOE12 | 4 | F5 |
| F6 | F6C_ManyTxRealOE12 | 5 | F6 |

## 3.实现、验证与独立复核

实现commit：`02e8fbe73d9e3d438963b7c2c6c3a50b28ee7cb4`。

| 文件 | SHA256 |
|---|---|
| `analysis/phase1_wrc_nct_leo_design_20260809.md` | `28fd2a1efcc0a9251322b35bdb538c36fa57f3b68e6e07350d84146dd17711b0` |
| `code/scripts/eval_phase1_wrc_nct_leo.py` | `1175a0f4ee94c43b60608ad90ff5c84aa4bcc8b8b9fc1f4ee1d0119341304635` |
| `code/scripts/launch_phase1_wrc_nct_leo6x3_20260809.sh` | `a93004246d05abab8baf743910ab381d7881f517bc82d2edde507a6011461d94` |
| `code/tests/test_phase1_wrc_nct_leo.py` | `3d0ee95f67ef8e160439c758e21846224892cec9a9efbc63ae9fbbae76389a41` |

本地`ssr-gpu`：py_compile通过；WRC-LEO/WRC/GI联合21/21通过；launcher `bash -n`通过；dry-run精确6 export+6 score。

独立复核：`P0=0 / P1=0 / ALLOW_RELEASE=YES`。复核确认physical/scenario互斥、固定readout、paired-clean同physical比较、每scenario E全TX/RX覆盖、GPU0～5各一任务及不覆盖路径。

## 4.N607冻结路径与命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_wrc_nct_leo6x3_20260809_v1_02e8fbe7`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_wrc_nct_leo6x3_20260809_v1`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_wrc_nct_leo6x3_20260809_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_wrc_nct_leo6x3_20260809_v1.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- TRAIN_ROOT：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_20260808_v2`
- CLEAN_ROOT：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_wrc_nct_clean6_20260809_v2`
- GI_ROOT：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gi_epior_clean6_20260809_v3`

```text
cd <release>/code && nohup setsid env RUN_ID=phase1_wrc_nct_leo6x3_20260809_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python TRAIN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_20260808_v2 CLEAN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_wrc_nct_clean6_20260809_v2 GI_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gi_epior_clean6_20260809_v3 bash <release>/code/scripts/launch_phase1_wrc_nct_leo6x3_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_wrc_nct_leo6x3_20260809_v1.launch.out 2>&1 < /dev/null & echo $!
```

## 5.预期artifact、健康门与裁决

每fold预期：远端`leo_features.npz`、`leo_metrics.json`、`leo_scores.csv`；log根6 export日志、6 score日志、两个completion。只回收metrics、scores、日志、completion和manifest；不下载NPZ、checkpoint、GI bundle、readout或runtime。

技术停止：协议/路径/覆盖风险、缺闭环、launcher-wide错误或至少2fold同一确定性异常；不得按性能停止。唯一启动一次，`retry=NO`。

18个fold×scenario原子格均需同时满足：

1.WRC相对同LEO closed的overall/min-class/min-RX/min-day附加下降均≤2个百分点。
2.LEO fixed-WRC相对同physical paired-clean fixed-WRC的overall/min-class/min-RX/min-day下降均≤2个百分点。
3.physical/scenario/R/C/E闭包与GI/WRC parity≤`1e-5`。

任一原子格任一floor失败即`REJECT_LEO_FLOOR`且不得进入Phase3。18格全过才可标记Phase1五门完成；这仍只是WiSig上的LEO压力代理，不是真实卫星或真实unknown结果。

## 6.运行器技术终态（2026-08-09）

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；本节不作性能解释。固定commit`02e8fbe73d9e3d438963b7c2c6c3a50b28ee7cb4`已无prefix归档至release，archive SHA256=`0bb55f4bdc5e099f69fff850f7872d5c7a4c16e872a1b84c5749ddb22e0ae1ac`。远端archive代码SHA与给定Windows工作树SHA的差异仅为LF/CRLF口径，launcher一致；远端结构、`py_compile`、exporter/evaluator `--help`、`bash -n`和精确6 export+6 score dry-run均通过。

唯一冻结命令已执行一次，launcher PID=`3954381`；6 export子PID=`3954384,3954385,3954386,3954388,3954389,3954390`，GPU映射0–5。6条export均exit=1，score阶段未启动、NPZ未生成。六fold产生同一确定性`ValueError: --target_old_tx_ids is required when --new_tx_ids is omitted`（`export_spaceborne_features.py:850`），按预注册规则停止；未重试、未修改release/run。

小artifact已回收到`E:\type10-7\automation_reports\CV-SincNet\phase1_wrc_nct_leo6x3_20260809_v1\artifacts`，包含6 export日志、`export_completion.tsv`、outer launch log和`manifest.json`；未下载NPZ、checkpoint、GI bundle、readout、runtime、metrics或scores。完成后无run-owned进程，8卡均0%/1MiB，SSH客户端/TCP22均清理；`retry=NO`。
