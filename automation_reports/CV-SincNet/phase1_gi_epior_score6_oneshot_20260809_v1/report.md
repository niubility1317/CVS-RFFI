# Phase1 GI-EpiOR六折score-only one-shot报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`PREREGISTERED / SCORE_ONLY_ONESHOT`

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
