# Phase1 GI-EpiOR六折clean最终发布修复报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`PREREGISTERED / FINAL_RELEASE_REPAIR_2_OF_2`

证据边界：`PHASE1_SOURCE_ONLY_DEVELOPMENT_NON_CONFIRMATORY`

## 1.目标与唯一修复

run ID：`phase1_gi_epior_clean6_20260809_v3`。v2在六个fit上共同触发`RuntimeError: Could not infer dtype of numpy.int64`，且未产生bundle、runtime、receipt、metrics或scores。根因是N607的Torch/NumPy组合不能隐式推断`np.flatnonzero`索引dtype。

本次唯一代码修复将episode正负行号显式转换为`torch.long`；索引集合、顺序、标签、几何、head、loss、阈值、seed、输入与矩阵均不改变。新增回归测试主动拒绝无dtype的NumPy整数索引。本地`ssr-gpu`专项9/9、相关组合36/36、`py_compile`及`bash -n`通过；独立复核`VERDICT=APPROVE / P0=0 / P1=0 / ALLOW_FINAL_REPAIR_RELEASE=YES`。

代码修复commit：`0eb544ad59eb98351d322d44a6c668c777dd2b26`。worktree SHA256：core=`5fb4d93f2d1c5ecf74e42825997a6f2438fb44348f0718d2efe46dd5c493dfac`；test=`40fcb0226fe856ac3c1b856dc5eda7b173e2385de7eddf43c744491526459c1b`。

## 2.冻结输入与方法

- 输入根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1`
- 输入：`F1C_ManyTxRealOE12`至`F6C_ManyTxRealOE12`六份`features.npz`。
- 每fold：冻结GeoSat-C特征＋一个`3->8->1`GI-EpiOR head；六fit全成功后六score并行。
- seed`7281105`、Adam`lr=1e-2`、`weight_decay=1e-3`、200 epoch、固定边界`e_epi>=0.5`。
- CPU-only；每子进程`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`；backbone调用为0。

## 3.N607路径与落地约束

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gi_epior_clean6_20260809_v3_0eb544ad`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gi_epior_clean6_20260809_v3`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gi_epior_clean6_20260809_v3`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gi_epior_clean6_20260809_v3.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

archive固定来自commit`0eb544ad59eb98351d322d44a6c668c777dd2b26`，不得使用`--prefix`；解包到release根后必须存在`<release>/code/scripts/launch_phase1_gi_epior_clean6_20260808.sh`且`<release>/code/code`不存在。

精确入口：

```text
cd <release>/code && nohup setsid env RUN_ID=phase1_gi_epior_clean6_20260809_v3 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python INPUT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1 bash <release>/code/scripts/launch_phase1_gi_epior_clean6_20260808.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gi_epior_clean6_20260809_v3.launch.out 2>&1 < /dev/null & echo $!
```

## 4.闭环、停止与回收

每fold应产生`gi_epior_bundle.npz`、`gi_epior_runtime.ts`、`fit_receipt.json`、`clean_metrics.json`、`clean_scores.csv`；日志根应包含fit/score completion及12份stdout。停止仅限协议错误、覆盖风险、checkout/hash错误、确定性异常或零输出；不得按性能停止。任一fit失败不启动score。

这是发布工程修复第2/2轮；若同一v3仍发生确定性非科学缺陷，停止继续修复并保留证据。只回收小receipt、metrics、scores、completion、日志与manifest，不下载输入NPZ、checkpoint、runtime或bundle。

## 5.Phase1晋级口径

clean结果关闭模型健康、known跨接收机性能无明显退化、source proxy相对C明确正信号及真实bundle导出。外层held-TX完整报告但不设置额外5%强门。clean四项通过后立即发布三种LEO视图，验证最低类别和LEO弱信道floor。
