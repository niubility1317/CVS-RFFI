# Phase1 ManyTx RealOE12 Physical-RX v2 Repair实验报告

目标模式：`GOAL_MODE=ACTIVE`

## 1.状态

- 实验ID：`phase1_manytx_realoe12_physrx_v2_20260808_v2`
- 状态：`LOCAL_VERIFIED / PREREGISTERED`
- 日期：2026-08-08
- 主控：`/root`；唯一N607运行器：`/root/n607_geosat_lite_runner`
- 标签：`DEVELOPMENT_SOURCE_ONLY_NON_CONFIRMATORY`
- 科学实现父提交：`fc322b598232b6329b9c6965023bcb7052baf1d6`
- 唯一repair提交：`c4210a994ca8be8629a79190213b1a8e9e6cc01e`
- v1技术终态提交：`23a2a73d76c144fea7d8ad54342c24c3868ce8de`

根目录`E:\type10-7`不是Git仓库；本报告同时保存在Git工作树和`E:\type10-7\automation_reports\CV-SincNet\phase1_manytx_realoe12_physrx_v2_20260808_v2\report.md`。

## 2.repair边界

v1在训练telemetry之前12/12统一失败：通用CSV解析器利用Python`int`时把physical day`2021_03_01`转为`20210301`，随后无法匹配ManySig日期。v1保留为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不覆盖、不恢复、不重跑。

v2仅新增WiSig轴解析器：只有可选符号加纯十进制数字才转为index；`2021_03_01`与`1-1`保持physical label。四个train/test day/RX入口改用该解析器。损失、分区、数据、fold、seed、epoch、阈值、GPU矩阵与postfreeze命令完全不变。

独立复核：31 focused tests passed；physical day/RX到实际轴解析smoke通过；`py_compile`、`bash -n`与`git diff --check`通过；结论`P0=0`、`P1=0`、`ALLOW_REPAIR_V2_RELEASE=YES`。

## 3.冻结方法与数据

比较GeoSat-C与RealOE-G，6个leave-one-known-TX folds，共12任务。G相对C只增加ManyTx真实source OE与energy ranking loss：`lambda=0.02`、epoch61开始、warmup10、`T=margin=tau=1`、每known batch均衡16个OE TX×8样本。known anchor停止梯度；OE标签强制`-1`且`y_tx=None`。不使用receiver alignment、VOS、虚拟proxy、batch轮换known proxy或Q98调参。

- ManySig SHA256：`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。
- ManyTx SHA256：`c0319174d40eb64bc49f201743941ebedc5cc0ced284c655cab798b2bdd44275`。
- source physical RX：`1-1,1-19,14-7,18-2,19-2,2-1`；days：`2021_03_01,2021_03_08`；`equalized=1`。
- held target RX：`20-1,3-19,7-14,7-7,8-8`，与source无交。
- ManyTx分区：OE80/proxy20/reserve16/locked target_new20；root=`ca3ed65a533359d2abb022fa513c49101ad93235738a39b362b5cdd15879c3d1`。
- OE门：每TX至少400条、两个source day、至少两个共同physical RX。
- 训练：`lite_d`、from-scratch、120epoch、seed`7281105`、sat seed`9281105`、`final_only`、clean加三种`leo_*_weak`。

## 4.冻结矩阵与资源

|Fold|held TX|C GPU|G GPU|
|---|---|---:|---:|
|F1|14-10|0|1|
|F2|14-7|2|3|
|F3|20-15|4|5|
|F4|20-19|6|7|
|F5|6-15|1|0|
|F6|8-20|3|2|

每卡最多两个进程；子进程以`CUDA_VISIBLE_DEVICES=<physical>`和`--device cuda:0`运行。launcher及所有训练参数沿用冻结脚本`code/scripts/launch_phase1_manytx_realoe12_20260808.sh`。

## 5.文件与hash

|文件|SHA256|
|---|---|
|`code/SSDG/train_ssdg.py`|`4a8d3c4d17f1d6fb9b751fa42146af450a1bf021647518909f47842b952d4378`|
|`code/cvsrffi/losses.py`|`29b417274ff18bd9b816cda2ba4b353a98820ac17ce439b0bac1f709c7421d21`|
|`code/tests/test_phase1_manytx_realoe.py`|`6dcfd2e67450d3a4e813d277734c3f4fb0da31ecb7d57cb7e655512f86981f9e`|
|`code/scripts/launch_phase1_manytx_realoe12_20260808.sh`|`3215778c4a87d46b383ac5f81edd55ef0dd1ffcb159d0eabdba662819f0f9a19`|
|`analysis/phase1_manytx_realoe_design_20260808.md`|`e4a6db0b874cc9cc04f74862fcc83e71420c013787738ae12dff30a6cdaaeae9`|

## 6.N607路径与唯一启动

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_manytx_realoe12_physrx_v2_20260808_v2_<COMMIT8>`。
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_20260808_v2`。
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_manytx_realoe12_physrx_v2_20260808_v2`。
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_manytx_realoe12_physrx_v2_20260808_v2.launch.out`。
- CWD：`<release>/code`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。

```bash
cd <release>/code && nohup setsid env RUN_ID=phase1_manytx_realoe12_physrx_v2_20260808_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python bash <release>/code/scripts/launch_phase1_manytx_realoe12_20260808.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_manytx_realoe12_physrx_v2_20260808_v2.launch.out 2>&1 < /dev/null & echo $!
```

只启动一次，`retry=NO`。预期根artifact为`pids.tsv`、`completion.tsv`和12份stdout；每臂预期`final_ssdg.pth`、metrics CSV/JSONL与terminal/split/resource receipts。显式`NON_PROMOTABLE_P0_DISABLED/exit8`且artifact完整是现有训练脚本的预期技术终态，不等于系统异常。

## 7.健康、评估与结论边界

技术停止只允许路径/hash/覆盖/协议错误、overwrite、OOM、确定性异常、无进展或至少两个任务同一异常；不得读取中途accuracy、loss、FAR或AUROC来停止/调参。失败不自动重试。

12任务全部E120并闭合后，按v1报告§8已冻结命令各执行一次：每candidate导出10400行clean证据（source2000、fold-held400、proxy8000），再执行CPU proxy energy-only与fold-held energy-only评分；source正确样本Q0.95校准energy，confidence/margin gate关闭，`unknown_far_target=0.05`。每条命令一次，禁止选择性重跑；只回收小JSON/CSV/log/manifest，不下载checkpoint/NPZ。

known保护按同fold同一行比较G-C的clean、三种LEO、min-class与min-receiver；任何预注册项下降超过2pp则该G fold拒绝。energy AUROC从逐样本energy重新计算，不能把旧脚本的confidence AUROC冒充energy AUROC。所有结果仅为Phase1 source-only开发证据，不构成K-shot、target unknown或Phase3正式性能。

## 8.待回填

运行器回填精确release commit/archive SHA、PID/GPU、epoch/terminal/completion、artifact hash、异常与资源清理；主控仅在完整12臂和24条postfreeze评分返回后作paired分析。

## 9.N607终态（2026-08-08）

- 训练`STATUS=ARTIFACTS_COMPLETE`：release=`.../phase1_manytx_realoe12_physrx_v2_20260808_v2_eaf24235`，commit=`eaf2423524b0b4fc4025968c3acc0a866aa2691d`，implementation parent=`23a2a73d76c144fea7d8ad54342c24c3868ce8de`，archive tar SHA256=`af6e1125b84c833c5bffc18a463c4a77258b052126c0a58148427174fdc75cd3`。
- launcher PID=`3785839`，12任务固定GPU映射；12/12均E120、metrics CSV=121行/JSONL=120行、`final_ssdg.pth`与terminal receipt齐全；`completion.tsv` 12行全`exit_code=8`为预期技术终态；无训练异常，GPU收口0%/1MiB。
- postfreeze按冻结命令仅执行12次export，各PID已退出，NPZ=0/12，未启动任何score。12条日志同一确定性异常：`ValueError: cannot resolve 20210301 from ['2021_03_01', '2021_03_08', '2021_03_15', '2021_03_23']`，位置为`export_spaceborne_features.py:_resolve_indices`；因此`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，retry=NO。
- postfreeze小证据已回收至`E:\type10-7\automation_reports\CV-SincNet\phase1_manytx_realoe12_physrx_v2_20260808_v2\artifacts`：audit logs、export_pids/completion、训练metrics/receipt/log小归档；未下载checkpoint/NPZ。SSH/SCP/TCP22均清理。
