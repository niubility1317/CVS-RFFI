# Phase1 GI-EpiOR六折score-only one-shot报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`ANALYZED / REJECT_CLEAN_KNOWN_RX_FLOOR`

证据边界：`PHASE1_SOURCE_ONLY_DEVELOPMENT_NON_CONFIRMATORY`

## 1.目标与输入继承

run ID：`phase1_gi_epior_score6_oneshot_20260809_v1`。该one-shot不训练、不重跑backbone，仅读取v3已经6/6 fit成功并生成的不可变GI-EpiOR bundle，对原六份C-arm NPZ执行6次clean score。v3的score没有输出任何metrics/scores，因此本run不存在基于性能的选择或重试。

v3 bundle根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gi_epior_clean6_20260809_v3`。feature根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1`。runner必须在启动前记录六份bundle与六份feature NPZ的SHA256，且不得修改它们。

## 2.最小修复与验证

N607的Torch→NumPy桥接在v3评分布尔组合处产生非标准dtype。score-only实现将Tensor通过Python list恢复为显式`np.float32`，并把接受门与closed-set正确标记显式恢复为`bool`；公式、阈值`0.5`、source/held/proxy角色和行集合不变。

实现commit：`eabb50afe37627555da7ce69d55a7ef7b18d551c`。worktree SHA256：evaluator=`a1563db36fbda673c5b51b48b940790e78fd077e21870636c6eca418c6751b18`；test=`c7627585f5a0a6861c6024cf41da06556f3bf0b8014642fd83a47e8cb0391161`；launcher=`a21c23126481b33828dbde26d5c6569df460d8ea734f05f11b7f64355a26c049`。

本地`ssr-gpu`验证：专项10/10、相关组合37/37、`py_compile`、`bash -n`通过；dry-run精确6条score、0条fit。独立复核：`VERDICT=APPROVE / P0=0 / P1=0 / ALLOW_SCORE_ONLY_ONESHOT=YES`。

## 3.N607冻结路径与命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gi_epior_score6_oneshot_20260809_v1_eabb50af`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gi_epior_score6_oneshot_20260809_v1`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gi_epior_score6_oneshot_20260809_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gi_epior_score6_oneshot_20260809_v1.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

archive固定来自commit`eabb50afe37627555da7ce69d55a7ef7b18d551c`，不带prefix并解包到release根。

```text
cd <release>/code && nohup setsid env RUN_ID=phase1_gi_epior_score6_oneshot_20260809_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python INPUT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1 BUNDLE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gi_epior_clean6_20260809_v3 bash <release>/code/scripts/launch_phase1_gi_epior_score6_oneshot_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gi_epior_score6_oneshot_20260809_v1.launch.out 2>&1 < /dev/null & echo $!
```

## 4.闭环与边界

预期每fold生成`clean_metrics.json`与`clean_scores.csv`，并产生6份score stdout及`score_completion.tsv`。只允许单次启动；技术失败即停止，不再修复或重试。不得按性能停止或调参。只回收metrics、scores、日志、completion与manifest，不下载bundle、runtime、feature NPZ或checkpoint。

主Agent将把本run的score与v3同fold fit bundle按hash连接后分析。clean阶段只判断模型健康、known跨接收机无明显退化、source proxy明确正信号和真实bundle闭环；外层held仅作诊断。clean通过才发布三种LEO视图。

## 5.运行器技术终态（2026-08-09）

状态：`ARTIFACTS_COMPLETE`；边界：`NO_PERFORMANCE_INTERPRETATION`。固定commit`eabb50afe37627555da7ce69d55a7ef7b18d551c`已无prefix归档至release，archive SHA256=`259faa7aabaa30fa07ed3e5ef8455fa2d04de27c86d564eacfa3c835229b93a1`。远端archive字节下evaluator/test分别为`1be2780ff1d9cc412c42619cd2a5c688582eb43e5f91d9f4aa0206020ec08b9b`/`623988152db037e515c4c6316b8636d1d31592649eff7b99496cc48ab5ad65dc`；报告给定的`a1563db...`/`c762758...`为Windows工作树CRLF口径，launcher两口径均为`a21c23126481b33828dbde26d5c6569df460d8ea734f05f11b7f64355a26c049`。远端结构、`py_compile`、`score --help`、`bash -n`和6条dry-run均通过。

唯一冻结命令已执行一次，launcher PID=`3912447`，score子PID=`F1:3912450,F2:3912451,F3:3912452,F4:3912453,F5:3912454,F6:3912457`。`score_completion.tsv`共6行且6条exit均为0；F1–F6各生成`clean_metrics.json`、`clean_scores.csv`和score日志，CSV均为10401行。未发现Traceback/RuntimeError/TypeError/OOM等错误指纹；完成后无run-owned进程，8卡均为0%/1MiB。

小artifact已回收至`E:\type10-7\automation_reports\CV-SincNet\phase1_gi_epior_score6_oneshot_20260809_v1\artifacts`，包含6 metrics、6 scores、6 logs、`score_completion.tsv`与`manifest.json`，逐项远端/本地SHA一致；未下载bundle、runtime、feature NPZ或checkpoint。SSH客户端与TCP22连接均已清理；`retry=NO`。

## 6.六折clean同行结果

冻结C没有unknown出口，因此其proxy FAR基线为100%。表中known下降、最低类下降、最低RX下降和最低day下降均以同fold、同一known query的“GI-EpiOR固定0.5拒识后－冻结C不拒识”计算；负值表示退化。外层held只作跨TX诊断，不参与本轮晋级。

| Fold | known closed/full | Δknown(pp) | Δmin-class(pp) | Δmin-RX(pp) | Δmin-day(pp) | proxy FAR | proxy AUROC | held FAR | held AUROC | clean裁决 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F1 | 97.90%/97.00% | -0.90 | -1.00 | -3.70 | -1.01 | 62.16% | 0.8045 | 84.50% | 0.6626 | FAIL：min-RX |
| F2 | 99.40%/99.00% | -0.40 | -0.50 | -1.16 | -0.23 | 74.62% | 0.6303 | 96.00% | 0.5160 | PASS |
| F3 | 97.60%/96.50% | -1.10 | -1.00 | -4.27 | -0.80 | 48.41% | 0.8520 | 51.25% | 0.8838 | FAIL：min-RX |
| F4 | 99.00%/98.60% | -0.40 | -0.50 | -1.32 | -0.21 | 66.92% | 0.7267 | 88.50% | 0.5742 | PASS |
| F5 | 97.90%/96.40% | -1.50 | -2.50 | -5.92 | -1.96 | 59.33% | 0.6700 | 78.25% | 0.4629 | FAIL：min-class/min-RX |
| F6 | 97.50%/96.60% | -0.90 | -2.50 | -4.02 | -1.41 | 53.84% | 0.8581 | 65.50% | 0.7842 | FAIL：min-class/min-RX |
| 平均 | 98.22%/97.35% | -0.87 | -1.33 | -3.40 | -0.93 | 60.88% | 0.7569 | 77.33% | 0.6473 | 2/6 known-floor PASS |

## 7.五项晋级裁决

1.模型健康：通过。六个fit与六个score均闭环；identity梯度恒为0、head梯度非零，物理split overlap为0。
2.已知类跨接收机性能：失败。known overall六折仅下降0.40--1.50个百分点，但最低RX在F1/F3/F5/F6分别下降3.70/4.27/5.92/4.02个百分点，超过2个百分点边界。
3.最低类别与LEO弱信道floor：clean最低类别在F5/F6各下降2.50个百分点，已足以阻止进入LEO；三种LEO实验不发布。
4.source proxy unknown正信号：通过。六折proxy FAR均低于100%基线，平均降至60.88%；六折AUROC均高于0.63，平均0.7569。
5.真实bundle导出：通过。v3六折均生成不可变bundle、TorchScript runtime和fit receipt；score-only run以远端hash连接这些bundle。

最终裁决：`REJECT_CLEAN_KNOWN_RX_FLOOR`。GI-EpiOR证明整TX episodic相对几何能够产生proxy分离，但固定0.5的小head把拒识错误集中到部分source RX和F5/F6最低类。proxy改善不能补偿known floor退化，因此不发布LEO视图，也不进入Phase3本地证据替换。

无阈值NCT ratio的平均proxy/held AUROC为0.9290/0.7955，说明相对几何排序本身比当前二元head稳定；该值只作为下一候选的连续证据，不把它追溯改写为本轮通过结果。
