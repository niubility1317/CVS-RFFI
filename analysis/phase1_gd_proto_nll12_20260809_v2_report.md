# Phase1 GD-ProtoNLL 12臂正式训练v2报告

状态：`PREREGISTERED_MECHANICAL_LAUNCH_REPAIR / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`

日期：2026-08-09

## 1.目标、差异与版本

|字段|冻结值|
|---|---|
|run ID|`phase1_gd_proto_nll12_20260809_v2`|
|目标|执行与v1完全相同的P1-GD-ProtoNLL 6折C/G、40E、final-only 12臂矩阵|
|v1终态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；唯一命令因launcher文件mode 100644而`Permission denied`，0 child、0 run/log root|
|唯一修复|新命令显式使用`bash launch_phase1_gd_proto_nll12_20260809.sh`；不改代码、参数、数据、矩阵、GPU、停止门或结果门|
|实现commit|`6465a7f33abb730ae58de4f6e0bec5181f128d0a`|
|Git工作树|`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`|
|科学复核|沿用冻结实现的`P0=0、P1=0、ALLOW`；本轮是机械启动修复，不重开科学评审|
|声明|本run只完成训练技术闭环，不读取性能，不作晋级结论；postfreeze另行执行|

实现与本地验证保持不变：core SHA`71DCFFC9D1BED35BB746C67EB25A9A7C79F31C1DDFA0289E0CC803F17FBEF57D`；train`815545AA383C4E666EB9295B4A147B8870DB122DEFCA0F84A1F11B48D5250A46`；test`45EB9972674E9307646EEA054E38D29B4A46689D85FBE81A0B656E73D767EDC5`；launcher`50F2472C2F71A9FC15DFDD02729107FACFCB68F165FB5CAC872386EFB65C5601`；design`F28A2C8E20C8D4E91B010FD4B5852C3A73C0AF5CC2C7334A4A7E7EFF3B705F35`。本地`py_compile`、29 focused、lite_d no-query、`bash -n`、dry-run12及diff-check均PASS。

## 2.冻结数据、方法与矩阵

ManySig=`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，SHA`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；6个GeoSat-C final checkpoint及SHA、每折TX映射、L/U/V、seed和非补偿门与[v1报告](E:/type10-7/automation_reports/CV-SincNet/phase1_gd_proto_nll12_20260809_v1/report.md)完全相同。C/G只差G的`.10*L_GD`；每批local4、三场景共同序列、12格EMA/coverage、首批raw gradient合同不变。

GPU矩阵固定：0=`F1C＋F5G`；1=`F1G＋F5C`；2=`F2C＋F6G`；3=`F2G＋F6C`；4=`F3C`；5=`F3G`；6=`F4C`；7=`F4G`。preflight须记录SCB v3在GPU7的既有进程；加入F4G后任一卡计算进程不得超过2。

## 3.N607路径与唯一命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v2_6465a7f3`，启动前ABSENT|
|CWD|`<release>/code`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll12_20260809_v2`，启动前ABSENT|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v2`，启动前ABSENT|
|outer|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v2_launcher.out`，启动前ABSENT|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|retry|NO；caller超时后只读核验landed，禁止重复launch|

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v2_6465a7f3/code && nohup env RUN_ID=phase1_gd_proto_nll12_20260809_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v2_6465a7f3/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v2_6465a7f3/code/scripts/launch_phase1_gd_proto_nll12_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v2_launcher.out 2>&1 < /dev/null &
```

## 4.执行、健康门与工件

唯一runner重新执行direct preflight；仅direct TCP／SSH不可用时使用一次verified bridge。核新release/run/log/outer均ABSENT、ManySig和6 checkpoint SHA、archive/member SHA、SCB进程及每卡≤2、远端`py_compile`／help／`bash -n`／dry-run12后唯一启动。v1路径与outer只读保留，不覆盖、不续跑。

立即记录launcher及12 child的PID/CWD/cmdline/GPU。停止仅依据路径/hash/覆盖、协议/类序、Traceback/OOM/CUDA/nonfinite/failure receipt、两行同一异常、零checkpoint或成员不全；绝不依据性能。预期每臂E40、final checkpoint、40行JSONL/41行CSV及config/GD terminal/training/terminal/resource/heldout receipt；C为`CONTROL_ARM_NOT_APPLICABLE`，G需1200 batches、153600 rows、12 cells、EMA逐批、raw gradient和terminal pass。最终`NON_PROMOTABLE_P0_DISABLED/exit8`是预期P0 gate。

成功后只回收小日志、JSON/CSV、PID、completion和manifest，不下载checkpoint/NPZ；更新root与Git镜像报告，清SSH/SCP/TCP22。任何技术失败标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`并且不重试。

