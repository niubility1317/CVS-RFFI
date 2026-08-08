# Phase1 WRC-NCT六折clean实验报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`LOCAL_VERIFIED / PREREGISTERED`

证据边界：`PHASE1_SOURCE_ONLY_DEVELOPMENT_NON_CONFIRMATORY`

## 1.实验目标与假设

run ID：`phase1_wrc_nct_clean6_20260809_v1`。时间：2026-08-09。执行角色：主Agent负责设计、实现与最终裁决；唯一N607 runner只负责落地、运行和回收技术证据。

本实验检验WRC-NCT是否能保留GI-EpiOR连续NCT排序信号，同时避免其二元head造成的source-known最低RX退化。WRC-NCT不重训backbone、不新增head或域对齐，只复用同fold GI v3 bundle的class geometry，使用source-only R/C/E物理互斥划分和worst-source-RX有限Q98阈值。

比较目标是同一E子集上的冻结C-arm `tx_logits argmax`无拒识结果。GI-EpiOR只作为上游几何来源和历史反例，不作为本轮阈值或性能选择依据。

## 2.冻结方法与矩阵

- R/C/E：沿用GI R/Q，再把每TX的Q按`SHA256(physical_id)`确定性平分为C/E，形成50/25/25。
- C中每个source RX至少50行，否则fail-closed。
- `k_r=min(n_r,ceil(0.98*(n_r+1)))`，`tau_r=s_(k_r)`，唯一阈值`tau=max_r tau_r`。
- `accept iff NCT<=tau`；所有known reject计错。
- proxy与outer-held均为零fit、零校准、零选参；outer-held只作方向诊断。
- 不扫alpha、分位数、阈值、fold、seed或聚合方式。

| Fold | feature候选 | source TX | held TX | 计算 |
|---|---|---|---|---|
| F1 | F1C_ManyTxRealOE12 | 14-7,20-15,20-19,6-15,8-20 | 14-10 | CPU |
| F2 | F2C_ManyTxRealOE12 | 14-10,20-15,20-19,6-15,8-20 | 14-7 | CPU |
| F3 | F3C_ManyTxRealOE12 | 14-10,14-7,20-19,6-15,8-20 | 20-15 | CPU |
| F4 | F4C_ManyTxRealOE12 | 14-10,14-7,20-15,6-15,8-20 | 20-19 | CPU |
| F5 | F5C_ManyTxRealOE12 | 14-10,14-7,20-15,20-19,8-20 | 6-15 | CPU |
| F6 | F6C_ManyTxRealOE12 | 14-10,14-7,20-15,20-19,6-15 | 8-20 | CPU |

## 3.本地实现、验证与版本

实现commit：`5f892f49d31653f02ac3ca31dc5bb8fc200eb450`。

| 文件 | 用途 | SHA256 |
|---|---|---|
| `analysis/phase1_wrc_nct_design_20260809.md` | 冻结设计 | `137575756bfacd031a5f633fb983d6f23d163332ac3afd89c2db434f0e6509b2` |
| `code/cvsrffi/phase1_wrc_nct.py` | R/C/E、Q98、runtime | `a384e75071194f16b36ecdb9d1ea291c050d2f6056b668747e7553b07283982d` |
| `code/scripts/eval_phase1_wrc_nct.py` | 单fold评估入口 | `c66126abadd789066ba494174b7ff1102483281998a229a1f8a699b6a1926337` |
| `code/scripts/launch_phase1_wrc_nct_clean6_20260809.sh` | 六折CPU launcher | `714fecd304bec8886db7b2d71bb1b670b4635c628ac1704725403c8ad750d363` |
| `code/tests/test_phase1_wrc_nct.py` | 协议与闭环测试 | `33d870f207c0fdec3a2ca60d0bdabc8783f281dd497b656717471ff150ff9ff5` |

`ssr-gpu`本地验证：`py_compile`通过；WRC+GI相关测试17/17通过；launcher `bash -n`通过；dry-run精确输出6条WRC命令。真实metadata回归测试确认只有TX ID规范化，`2021_03_01`、RX、role等字段原样保留。

独立复核：`P0=0 / P1=0 / ALLOW_RELEASE=YES`。

## 4.N607冻结路径与命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_wrc_nct_clean6_20260809_v1_5f892f49`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_wrc_nct_clean6_20260809_v1`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_wrc_nct_clean6_20260809_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_wrc_nct_clean6_20260809_v1.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- feature根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1`
- GI bundle根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gi_epior_clean6_20260809_v3`

```text
cd <release>/code && nohup setsid env RUN_ID=phase1_wrc_nct_clean6_20260809_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python INPUT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1 BUNDLE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gi_epior_clean6_20260809_v3 bash <release>/code/scripts/launch_phase1_wrc_nct_clean6_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_wrc_nct_clean6_20260809_v1.launch.out 2>&1 < /dev/null & echo $!
```

## 5.预期artifact、健康门与晋级标准

每fold预期生成`wrc_nct_readout.json`、`wrc_nct_runtime.ts`、`clean_metrics.json`、`clean_scores.csv`；log根生成6份stdout和`completion.tsv`。只回收小型readout、metrics、scores、日志、completion与manifest；不下载feature NPZ、GI bundle或checkpoint。

技术停止仅限协议/路径/覆盖风险、确定性异常、缺失输出闭环或两fold同一异常指纹；不得按性能提前停止。启动一次，`retry=NO`。

五项Phase1门：

1.六折模型、输入、readout/runtime和score闭环健康。
2.每折同一E子集known overall与min-RX相对无拒识下降均不超过2个百分点。
3.每折min-class与min-day下降均不超过2个百分点；通过clean后才允许发布三种LEO视图验证LEO floor。
4.六折proxy FAR均低于100%且AUROC均高于0.5，作为source proxy正信号。
5.六折不可变readout与TorchScript runtime导出成功，GI/WRC及eager/TS parity不超过`1e-5`。

任一门失败即`REJECT`并关闭当前Q98/NCT阈值族；不得用proxy改善补偿known floor，也不得把held诊断写成Phase3真实unknown性能。当前无best epoch/checkpoint选择，因为本实验不训练模型。
