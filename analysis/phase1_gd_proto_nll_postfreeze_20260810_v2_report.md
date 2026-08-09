# Phase1 GD-ProtoNLL postfreeze v2报告

状态：`PREREGISTERED_TOTALIZED_L2_REPAIR / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`

日期：2026-08-10

## 1.目标、唯一修复与证据

|字段|冻结值|
|---|---|
|run ID|`phase1_gd_proto_nll_postfreeze_20260810_v2`|
|目标|对同一GD-ProtoNLL v3 12个final checkpoint完整重算42步并形成6折aggregate|
|v1终态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；12 GD-clean、12 LEO、12 proxy完成，F1–F5 pair生成，F6在零范数几何处停止；旧pair不得沿用|
|权威诊断|253440行clean feature无nonfinite；唯一zero为F6C `source_validation_known` 1/16800；所有L与proxy zero=0|
|唯一数学修复|totalized L2：正范数`z/||z||₂`，精确零范数映射为零向量；全部L/V/proxy行保留，nonfinite仍fatal|
|不变项|exporter、L-only Gaussian、ddof1、class-equal pooled variance、.9/.1 shrink、1e-6 floor、完整NLL、u、分类/proxy非补偿门、训练root和42步矩阵|
|实现commit|`62d0cf905785c4fac7dff98bf824ff28de20d523`|
|独立复核|`P0=0、P1=0、ALLOW`；pycompile、focused10、bash-n、dry-run42、diff-check及zero-L/zero-V手算均通过|

文件SHA：evaluator=`0612acea58d61d1bca40d54ae1a404df700900f4ecf9900f191104dfd1627409`；tests=`d7367c6525127e315785c79790eb22d7942558f77f95f3fca26439387f6b4b9d`；launcher=`52d50aaab5292840b94faf4d6c7f687315e7ee65a42b0953e0da4711404ee7f3`；design=`30e4bd1ecfb920efc7bb8426ee5238badb28159adba19a5f6d7b7426cb48195c`。exporter沿用v1字节`c21d84a76c5448e0c8414389222ad764575820f011b05784982d313499f71580`。

zero的u完全由L-only Gaussian决定，不人为奖励、删除或施加固定惩罚，也不预判方向。每折receipt须封存C/G×L/V/proxy的total/positive/zero/nonfinite/retained/dropped与L逐类计数，且F6拒绝v1 schema、跨root或旧prior。

## 2.固定数据、门与42步

训练root只读：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll12_20260809_v3`；ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。专用clean严格重建L/U/V，只forward L、V、proxy，U forward=0；clean分类V、Gaussian fit L、known u V、unknown u proxy。LEO为source-only 1600行三场景。

硬门保持：clean 6/6四floor≥C−2pp；LEO 18格四floor≥C−2pp；每折三场景等权overall Δ≥0；18格等权overall Δ≥0；逐折6/6同时满足ΔAUROC>0与Δ(proxy mean u−V mean u)>0。任一失败即`REJECT_GD_PROTO_NLL_PERMANENT`，不重试、不调参。

42步固定为12 GD-clean＋12 LEO＋12 proxy＋6 pair；GPU矩阵0=`F1C＋F5G`、1=`F1G＋F5C`、2=`F2C＋F6G`、3=`F2G＋F6C`、4=`F3C`、5=`F3G`、6=`F4C`、7=`F4G`。

## 3.N607路径与唯一命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v2_62d0cf90`|
|CWD|`<release>/code`|
|postfreeze root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll_postfreeze_20260810_v2`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll_postfreeze_20260810_v2`|
|outer|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll_postfreeze_20260810_v2_launcher.out`|
|retry|NO|

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v2_62d0cf90/code && nohup env POSTFREEZE_RUN_ID=phase1_gd_proto_nll_postfreeze_20260810_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v2_62d0cf90/code TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll12_20260809_v3 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v2_62d0cf90/code/scripts/launch_phase1_gd_proto_nll_postfreeze_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll_postfreeze_20260810_v2_launcher.out 2>&1 < /dev/null &
```

## 4.发布与停止

唯一runner先direct preflight；新release/run/log/outer必须ABSENT，v1只读保留；核ManySig、12 final、archive/member、远端pycompile/help/bash-n/dry-run42和每卡≤2后唯一启动。caller超时仅只读确认，不重发。

只因路径/hash/覆盖、split/head/checkpoint/arm/root、Traceback/OOM/CUDA、确定性异常、缺输出或零pair停止，绝不按性能数值提前停。完成后核12 clean NPZ、12 LEO NPZ、12 proxy JSON/CSV、6个v2 pair JSON与F6 aggregate；不下载NPZ/checkpoint，只回收小artifact。runner只记录原始verdict，不解释性能。
