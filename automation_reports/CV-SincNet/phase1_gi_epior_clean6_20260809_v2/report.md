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

## 6.运行器技术终态（2026-08-09）

- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。v2无prefix归档落地结构正确，未触碰v1 partial release。
- 归档SHA256=`c4d358299a5f8e4b1b9e9a6e1f4156f2a4ee3f15613181fb6c4b363b3103a7c8`（260956160 bytes）。规范哈希：design=`4788033aa429b32954f886b7414a0d7f0f1cd03c27f6be390890c64d91ff005c`、core=`c24575d68eabdf7c20fbc6df0c6dcbe83d0bc3c384a48e96b7130ea898ad955a`、eval=`00e9715538f545dba9633e7e92d51f7feabbd17e15f378650d68907c3a15f454`、test=`8f3332eaac32f8c9896fc1952db4c6e55eff3c6d6e9a53ada1e4cb8ff9386b33`、launcher=`a18bfa57e2807a4b77624371408cc678d9b2f2a0fc7d8e0ebc61b6b44ad83477`；工作树哈希按预注册记录，CRLF/LF差异仅记录。
- launcher PID=`3894460`；F1--F6 fit child PID=`3894466,3894467,3894468,3894469,3894471,3894473`，均exit=1。统一指纹：`RuntimeError: Could not infer dtype of numpy.int64`，调用位于`eval_phase1_gi_epior.py:166`与`cvsrffi/phase1_gi_epior.py:250`。score未启动。
- 远端bundle/runtime/fit_receipt/metrics/scores均0；fit_completion.tsv=6行且exit1=6；score_completion.tsv不存在；GPU0--7=0%/1MiB，run-owned进程退出，SSH清理完成。
- 本地artifact：`E:\type10-7\automation_reports\CV-SincNet\phase1_gi_epior_clean6_20260809_v2\artifacts`，已回收6份fit stdout、fit_completion.tsv、outer log和manifest；不下载NPZ/checkpoint/runtime/bundle，不解读性能，不重试。
