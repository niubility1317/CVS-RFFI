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

## 6.运行器技术终态（2026-08-09）

状态：`ARTIFACTS_COMPLETE`；边界：`NO_PERFORMANCE_INTERPRETATION`。固定commit`59492f1b68e50a5181f2e753eb0dc7db3ab7945b`已无prefix归档至release，archive SHA256=`62e7e253783bdb07fbb3786081e612d961789759313fff2be1eb85d2fa650dd3`。远端archive代码SHA与给定Windows工作树SHA的差异仅为LF/CRLF口径，launcher两者一致；结构、`py_compile`、`eval --help`、`bash -n`和v2精确6条dry-run均通过。

唯一冻结命令已执行一次，launcher PID=`3938012`；fold子PID=`F1:3938016,F2:3938017,F3:3938018,F4:3938019,F5:3938021,F6:3938023`。completion共6行且6条exit均为0；F1–F6均生成readout、runtime、metrics、scores，CSV均10401行。日志未发现Traceback、WRCNCTError、RuntimeError、TypeError、OOM等错误指纹。

只读结构核验：六fold parity、eager/TorchScript接受一致性、`outer_used_for_fit_or_calibration=false`、outer rows=0、每RX calibration计数≥50、R/C/E行闭合、immutable和prototype不二次归一化（`new_geometry=false`）均通过。小artifact已回收至`E:\type10-7\automation_reports\CV-SincNet\phase1_wrc_nct_clean6_20260809_v2\artifacts`，包含6 readout、6 metrics、6 scores、6日志、`completion.tsv`和`manifest.json`；未下载runtime/NPZ/GI bundle/checkpoint。完成后无run-owned进程，8卡均0%/1MiB，SSH客户端/TCP22均清理；`retry=NO`。

## 7.六折clean同行结果与裁决

下表只使用v2完整矩阵。下降值均为同fold、同一E子集上“无拒识closed−WRC完整判决full”；正值代表WRC造成退化。held仅为跨TX方向诊断，不参与晋级。

| Fold | known closed/full | overall下降(pp) | min-class下降 | min-RX下降 | min-day下降 | proxy FAR | proxy AUROC | held FAR | held AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | 98.20%/98.20% | 0.00 | 0.00 | 0.00 | 0.00 | 98.56% | 0.9260 | 100.00% | 0.9172 |
| F2 | 99.80%/99.80% | 0.00 | 0.00 | 0.00 | 0.00 | 96.48% | 0.9399 | 99.75% | 0.4878 |
| F3 | 98.20%/98.20% | 0.00 | 0.00 | 0.00 | 0.00 | 96.04% | 0.9270 | 93.75% | 0.9561 |
| F4 | 99.00%/99.00% | 0.00 | 0.00 | 0.00 | 0.00 | 92.94% | 0.9375 | 97.25% | 0.6688 |
| F5 | 98.40%/98.40% | 0.00 | 0.00 | 0.00 | 0.00 | 95.38% | 0.9271 | 97.25% | 0.8528 |
| F6 | 97.80%/97.80% | 0.00 | 0.00 | 0.00 | 0.00 | 97.48% | 0.9267 | 99.00% | 0.8861 |
| 平均 | 98.57%/98.57% | 0.00 | 0.00 | 0.00 | 0.00 | 96.14% | 0.9307 | 97.83% | 0.7948 |

五项门裁决：

1.模型与数据健康：通过，6/6完整且无错误。
2.已知跨接收机性能：通过，overall与min-RX六折下降均为0。
3.clean最低类别/最低day：通过，六折下降均为0；LEO floor仍待三场景实验。
4.source proxy正信号：通过，6/6 FAR低于100%且AUROC高于0.926。FAR仍高，说明只是明确但很弱的拒识出口，不能称为真实unknown能力。
5.deployment bundle：通过，6/6 readout/runtime导出且parity为0。

clean裁决：`CLEAN_PASS / RELEASE_THREE_LEO_FLOOR_VIEWS`。WRC-NCT保留了NCT排序，但worst-RX上包络非常保守：平均只拒绝3.86%的proxy。该结果满足本轮预注册的“明确正信号”门，不满足也不冒充Phase3的unknown FAR≤5%目标。下一步只把同一clean readout冻结应用到`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`三种物理样本互斥切片；不得在LEO上重新校准阈值。
