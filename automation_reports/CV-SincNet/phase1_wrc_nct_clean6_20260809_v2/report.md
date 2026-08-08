# Phase1 WRC-NCT六折clean v2实验报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`LOCAL_VERIFIED / PREREGISTERED`

证据边界：`PHASE1_SOURCE_ONLY_DEVELOPMENT_NON_CONFIRMATORY`

## 1.目标与修复范围

run ID：`phase1_wrc_nct_clean6_20260809_v2`。时间：2026-08-09。主Agent负责方法冻结与最终裁决；唯一N607 runner只负责技术运行和artifact回收。

v1在F3/F4/F5的GI/WRC parity门失败，最大差值分别为`5.865097e-4`、`2.441406e-4`、`6.103516e-5`，因此被标记为`NO_PERFORMANCE_RESULT`。根因是WRC runtime对GI runtime已经归一化的float32 prototype再次归一化，末位变化被极小MAD尺度放大。

v2只做一个技术修复：验证上游prototype为单位范数后原值注册，不再二次归一化。方法公式、`alpha=.02`、finite Q98、`tau=max_r tau_r`、R/C/E划分、六fold矩阵、输入和晋级门均不变；v1路径与部分artifact不复用。

## 2.冻结矩阵与方法

| Fold | feature候选 | source TX | held TX |
|---|---|---|---|
| F1 | F1C_ManyTxRealOE12 | 14-7,20-15,20-19,6-15,8-20 | 14-10 |
| F2 | F2C_ManyTxRealOE12 | 14-10,20-15,20-19,6-15,8-20 | 14-7 |
| F3 | F3C_ManyTxRealOE12 | 14-10,14-7,20-19,6-15,8-20 | 20-15 |
| F4 | F4C_ManyTxRealOE12 | 14-10,14-7,20-15,6-15,8-20 | 20-19 |
| F5 | F5C_ManyTxRealOE12 | 14-10,14-7,20-15,20-19,8-20 | 6-15 |
| F6 | F6C_ManyTxRealOE12 | 14-10,14-7,20-15,20-19,6-15 | 8-20 |

R/C/E仍为每TX 50/25/25物理互斥；每source RX的C至少50行；proxy与outer-held零fit、零校准、零选参；known reject计错；本轮为6条CPU命令，不扫任何参数。

## 3.版本、验证与独立复核

实现commit：`59492f1b68e50a5181f2e753eb0dc7db3ab7945b`。

| 文件 | SHA256 |
|---|---|
| `code/cvsrffi/phase1_wrc_nct.py` | `1408adff6256ba0bf76a64b89b799a44f1a8e2b9f675633bcdae61cc10b3077f` |
| `code/scripts/eval_phase1_wrc_nct.py` | `c66126abadd789066ba494174b7ff1102483281998a229a1f8a699b6a1926337` |
| `code/scripts/launch_phase1_wrc_nct_clean6_20260809.sh` | `714fecd304bec8886db7b2d71bb1b670b4635c628ac1704725403c8ad750d363` |
| `code/tests/test_phase1_wrc_nct.py` | `b67d4dbdac8b8b032698bb31cf829b267bf8eaee924d8c95c6d50f745b1362f2` |

本地`ssr-gpu`验证：`py_compile`通过；WRC+GI联合18/18通过；新增回归使用`torch.equal`验证GI/WRC prototype、d1、d2、ratio逐位一致；launcher `bash -n`通过；dry-run为6条。

独立复核：`VERDICT=APPROVE / P0=0 / P1=0 / ALLOW_NEW_RUN_ID=YES`，明确属于不改方法、阈值或矩阵的技术parity修复。

## 4.N607冻结路径与命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_wrc_nct_clean6_20260809_v2_59492f1b`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_wrc_nct_clean6_20260809_v2`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_wrc_nct_clean6_20260809_v2`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_wrc_nct_clean6_20260809_v2.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- INPUT_ROOT：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1`
- BUNDLE_ROOT：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gi_epior_clean6_20260809_v3`

```text
cd <release>/code && nohup setsid env RUN_ID=phase1_wrc_nct_clean6_20260809_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python INPUT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1 BUNDLE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gi_epior_clean6_20260809_v3 bash <release>/code/scripts/launch_phase1_wrc_nct_clean6_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_wrc_nct_clean6_20260809_v2.launch.out 2>&1 < /dev/null & echo $!
```

## 5.技术闭环与五项门

每fold必须生成readout JSON、runtime TS、metrics JSON、10401行scores CSV；log根生成6份stdout和completion。只回收小artifact，不下载runtime、feature、GI bundle或checkpoint。技术失败才停止；不得按性能停止。启动一次，`retry=NO`。

正式分析要求6/6完整后统一执行：

1.模型/数据/输出闭环健康。
2.每fold同一E子集known overall和min-RX下降不超过2个百分点。
3.每fold min-class和min-day下降不超过2个百分点；通过后才发布三种LEO视图。
4.每fold proxy FAR低于100%且AUROC高于0.5。
5.每fold不可变readout/runtime导出成功，GI/WRC及eager/TS parity不超过`1e-5`。

任一门失败即`REJECT`并关闭Q98/NCT阈值族；outer-held只作方向诊断，不得称为Phase3真实unknown。
