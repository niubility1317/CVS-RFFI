# Phase1 CAGM 12臂正式训练v2预登记报告

状态：`LOCAL_VERIFIED / PREREGISTERED_NOT_LAUNCHED / NO_PERFORMANCE_RESULT`

日期：2026-08-10

## 1.目标与Revision2原因

|字段|冻结值|
|---|---|
|run ID|`phase1_cagm12_20260810_v2`|
|目标|完成6个LOTO fold×C/G、40E、final-only共12臂训练，并生成可由CAGM postfreeze v2逐字段验证的终态收据|
|C|GeoSat-C continuation；共同clean＋单次LEO forward、`lambda_sat_cons=.10`|
|G|C上唯一增加`lambda_cagm=.02`的Clean-Anchored Class Geometry Matching|
|实现commit|`0ba9675e6fff859aea78319941ab68335c744cc9`|
|独立复核|Revision2训练＋postfreeze定向复审：`P0=0、P1=0、ALLOW`|
|当前声明|只有本地实现、窄回归和发布前字节证据；无v2性能结果|

v1已完成12臂技术训练，但其`cvs.phase1.cagm_receipt.v1`没有把`joint_zero_mask_aux_only`和真实optimizer类型持久化，postfreeze pair也未逐字段比较C/G共同训练binding。v1保留为`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`技术证据，不补写、不覆盖、不进入性能评价。Revision2不改变CAGM公式、loss、forward、数据、seed、矩阵、资源或评价门，仅将receipt schema升级为`cvs.phase1.cagm_receipt.v2`并闭合以下证据：

- G每batch与terminal严格`joint_zero_mask_aux_only=true`，C明确为false/N/A；
- 真实optimizer必须为`AdamW`，initial state empty与SHA写入并终态校验；
- postfreeze current pair和F6 raw recompute逐字段比较C/G共同checkpoint、class/order、source split、optimizer与batch/scene序列。

冻结损失仍为：

```text
L_CAGM = [sum_c(r_c^leo-sg(r_c^clean))^2
          + sum_c<d(g_cd^leo-sg(g_cd^clean))^2] / 10
L_G = L_base + 0.02 * L_CAGM
```

联合零掩码只作用于辅助项；精确零行仍进入共同base。clean统计完全detach，辅助梯度仅进入LEO侧共享encoder，exact classifier head辅助梯度为None或零。

## 2.版本、归档与本地验证

完整无prefix归档：

|字段|值|
|---|---|
|本地路径|`E:\type10-7\automation_reports\CV-SincNet\phase1_cagm12_20260810_v2\artifacts\phase1_cagm12_20260810_v2_0ba9675e_fulltree.tar`|
|SHA256|`2cb14fc5689c9de1fd450edf3286508d016ebbcb8d6712f2c7733f88f7767e44`|
|大小|`260669440`字节|
|members|`4912`，无prefix，预期`code/code=0`|

|文件|commit归档SHA256|作用|
|---|---|---|
|`analysis/phase1_cagm_design_20260810.md`|`66736d4295c06ed297f618bb8ede47e3a0f834334c1b3960d1ba0a222df0745b`|训练Revision2追踪|
|`analysis/phase1_cagm_postfreeze_design_20260810.md`|`0e9c3195196c8a0021b3579afd103ce697fdac3c61deef21dfa3d68ec67ef51f`|统一后冻结合同|
|`code/cvsrffi/phase1_cagm.py`|`ff84a35f9b8e1068de07580539cc6561b0f8fe19ecba37cb4ae75acd80ffeda3`|公式、receipt v2、VJP与terminal|
|`code/SSDG/train_ssdg.py`|`c842c0d830e7c39d5005ab6dcccbf3b580848f79574f787eaa6c900e60c432b5`|共同训练主循环，本Revision未改|
|`code/tests/test_phase1_cagm.py`|`24f9272dea167e3302e10ddad09a72bbeee6dcb55056646cdcb468dbbd1989a0`|训练负例与smoke|
|`code/scripts/launch_phase1_cagm12_20260810.sh`|`428415d61d1424e04fbcede5efc5d1656371a50f7b78318f4ee7a3cdeb225c74`|v2 12臂launcher；Git mode100755|

本地`ssr-gpu`验证：

- 训练focused：`11 passed`；CAGM＋GD＋ICMT＋CB＋CP：`51 passed`；
- postfreeze focused：`38 passed`；ICMT模板：`31 passed`；合计`69 passed`；
- `py_compile`、两个launcher的`bash -n`和`git diff --check`通过；
- 训练dry-run严格12臂且全为v2；postfreeze dry-run严格`42=12+12+12+6`且全为v2；
- 定向reviewer用旧schema、joint-mask缺失/False、optimizer漂移、G序列/SHA/rows/scenario漂移反例复验，结论`P0=0、P1=0、ALLOW`。

## 3.数据、权限与训练合同

ManySig=`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。split=`tx_rx_day_1_6_3`、seed=`7281105`、sat seed=`9281105`、L/U/V=`.07/.63/.30`、batch=`128`、lr=`2e-4`、weight decay=`1e-4`、label smoothing=`.01`、epochs=`40`、checkpoint selection=`final_only`。

更新严格L-only：U不迭代、不forward；V仅为C/G共同source-validation诊断，零loss/backward/optimizer/校准/选模；proxy/held零训练loader/forward。C/G使用相同L-batch、共同clean与单次LEO forward，场景按`(epoch+batch_idx-2)%3`轮换`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。

|fold|warm-start SHA256|train TX|known V|proxy TX|
|---:|---|---|---|---|
|1|`4d515204f2cea62c5b82313a01b722b3b3d13a3e4fe647ff4b723b69e8a0c040`|20-15,20-19,6-15,8-20|14-7|14-10|
|2|`29c7d7ca31d80d90d7c0235fa234707b05866914dc0acdae5c44505af1bbd76`|14-10,20-19,6-15,8-20|20-15|14-7|
|3|`39c6cdd65aade504efdea956db02cc5e762aee299a9e9319c07ed6fb839434b7`|14-10,14-7,6-15,8-20|20-19|20-15|
|4|`32d956f44f60844471ba2ef04526c5f40cad0f8bc8acb7249be6035aa85005e4`|14-10,14-7,20-15,8-20|6-15|20-19|
|5|`2b9381546878b19e7e8e2106a82b0d0a4672a3012ef79bd7f28eadfd03b75a9f`|14-10,14-7,20-15,20-19|8-20|6-15|
|6|`573ca9d039a8c854f9c0927b5b5c303ab8eeaf527ccd42cd0d764b81e630de6f`|14-7,20-15,20-19,6-15|14-10|8-20|

## 4.矩阵与GPU

|GPU|候选|
|---:|---|
|0|F1C、F5G|
|1|F1G、F5C|
|2|F2C、F6G|
|3|F2G、F6C|
|4|F3C|
|5|F3G|
|6|F4C|
|7|F4G；若SCB v4仍运行，则GPU7总计算进程不得超过2|

## 5.N607不可覆盖路径与唯一启动

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cagm12_20260810_v2_0ba9675e`|
|CWD|`<release>/code`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cagm12_20260810_v2`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cagm12_20260810_v2`|
|outer|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cagm12_20260810_v2_launcher.out`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|retry|NO；调用异常先只读确认landed，禁止重复launch|

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cagm12_20260810_v2_0ba9675e/code && nohup env RUN_ID=phase1_cagm12_20260810_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cagm12_20260810_v2_0ba9675e/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cagm12_20260810_v2_0ba9675e/code/scripts/launch_phase1_cagm12_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cagm12_20260810_v2_launcher.out 2>&1 < /dev/null &
```

## 6.健康停止、预期工件与后续

唯一Runner只做一次必要preflight、release/input/path/GPU核验、远端`py_compile`/help/`bash -n`/dry-run12和§5唯一启动。不得按accuracy、loss或耗时停止。仅路径/hash/覆盖、协议漂移、Traceback/OOM/CUDA、CAGM finite/coverage/VJP/receipt v2失败、两个distinct arm同一确定性预测前异常或必要工件缺失可触发本run树技术停止；停止后保留partial、`NO_PERFORMANCE_RESULT`、不重试。

预期每臂生成final checkpoint、metrics、config、training completion、terminal、resource、heldout；G必须CAGM receipt v2 terminal pass，C必须control pass。`NON_PROMOTABLE_P0_DISABLED/exit8`为预期终态。只回收小工件，不下载`.pth/.npz`，不读性能。

v2训练技术完成后立即执行已提交的CAGM postfreeze v2 42步；评价核固定为source-L-only float64 totalized-L2 Gaussian-NLL、固定400 proxy、三场景LEO、F6原始工件重算与非补偿门。任一完整门失败即`REJECT_P1_CAGM_PERMANENT`。

