# Phase1 GI-EpiOR六折clean发布修复报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`PREREGISTERED / RELEASE_REPAIR_1_OF_2`

证据边界：`PHASE1_SOURCE_ONLY_DEVELOPMENT_NON_CONFIRMATORY`

## 1.目标与修复范围

run ID：`phase1_gi_epior_clean6_20260809_v2`。本次只修复v1的Git archive多余`--prefix=code/`落地路径；GI-EpiOR方法、固定阈值、六折矩阵、seed、输入NPZ和Phase1五项晋级条件均不改变。v1没有启动任何fit/score，不包含性能结果。

冻结archive来源：commit`01a2b7734c92ea7ae1d5cd8a4afde2c71b9e0ad9`；implementation commit`d3b5b610987f5ce8f38262875b5bb7ace1ba3143`。archive必须不带prefix并解包到release根，解包后必须满足`<release>/code/scripts/launch_phase1_gi_epior_clean6_20260808.sh`存在，且不得出现`<release>/code/code`。

## 2.输入与矩阵

- 输入根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1`
- 输入候选：`F1C_ManyTxRealOE12`至`F6C_ManyTxRealOE12`的六份`features.npz`。
- 每fold：冻结GeoSat-C特征＋一个`3->8->1`GI-EpiOR head；6次fit全部成功后再并行6次score。
- 训练：seed`7281105`、Adam`lr=1e-2`、`weight_decay=1e-3`、200 epoch、固定边界`e_epi>=0.5`。
- 运行资源：CPU-only；每子进程`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`；不重复运行backbone。

## 3.N607冻结路径与命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gi_epior_clean6_20260809_v2_01a2b773`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gi_epior_clean6_20260809_v2`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gi_epior_clean6_20260809_v2`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gi_epior_clean6_20260809_v2.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

精确入口：

```text
cd <release>/code && nohup setsid env RUN_ID=phase1_gi_epior_clean6_20260809_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python INPUT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1 bash <release>/code/scripts/launch_phase1_gi_epior_clean6_20260808.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gi_epior_clean6_20260809_v2.launch.out 2>&1 < /dev/null & echo $!
```

## 4.预期证据与停止规则

每fold必须产生`gi_epior_bundle.npz`、`gi_epior_runtime.ts`、`fit_receipt.json`、`clean_metrics.json`、`clean_scores.csv`；日志根还需`fit_completion.tsv`、`score_completion.tsv`及12份stdout。

停止仅限协议错误、覆盖风险、checkout/hash错误、确定性异常或零输出；不得按性能停止。任一fit失败不启动score。本v2是第一次发布工程修复，方法不重试、不调参、不扫阈值。只回收小receipt、metrics、scores、completion、日志与manifest，不下载输入NPZ、checkpoint、runtime或bundle。

## 5.晋级口径

clean阶段关闭：模型健康、known跨接收机性能无明显退化、source proxy相对C产生明确正信号、真实bundle导出。外层held-TX完整报告但不作为额外5%强门。上述clean四项通过后立即发布三种LEO视图，以验证最低类别和LEO弱信道floor无严重下降。
