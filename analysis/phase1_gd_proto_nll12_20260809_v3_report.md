# Phase1 GD-ProtoNLL 12臂正式训练v3报告

状态：`PREREGISTERED_ZERO_FEATURE_REPAIR / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`

日期：2026-08-09

## 1.目标与唯一方法变化

|字段|冻结值|
|---|---|
|run ID|`phase1_gd_proto_nll12_20260809_v3`|
|目标|完成P1-GD-ProtoNLL 6折C/G、40E、final-only 12臂矩阵，生成一次性postfreeze所需的12个最终checkpoint|
|v2终态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；F1G在E009由合并错误`feature/head contains non-finite or zero L2 norm`停止，通用fingerprint不能判定具体分支|
|v3唯一科学修复|GD辅助项只过滤精确有限零范数feature行；feature非有限、head非有限或零范数仍fail-closed；过滤后每个local4类至少1个valid行|
|不变部分|C路径、完整batch的base与共同`sat_cons=.10`、四类等权、`3Σqℓ`、旧q反传后EMA与12格softmax、数据、seed、40E、矩阵、final-only|
|实现commit|`10ead8b299f26ac793941180bee1bc87c45aa4df`|
|独立复核|科学摘要与实际diff均为`P0=0、P1=0、ALLOW`|
|声明|本run不以训练中性能停止，不在训练阶段读取或解释性能；完整postfreeze前不作晋级结论|

本地冻结SHA：core=`F2AAB82148F882B2BF7C6DEC0886396157720BFB720A41DF70552A84926632E4`；train=`FD571751B70EC74ACF38A04E5288092C085A3ECE97F8E99FD79EF7D3C7E1F67F`；test=`3E1A0EE72DC10800493C7AA782521F5A2C562299BFB3D45DC17E7CE35ECB026A`；launcher=`BE109FF1F209CC928E0C437354860A95C99D1926D609DECF2E4B5609C50E05A1`；design=`6CB62DBF9AC3C10236FBAF655A239ACCE5E96989CC0911335B96420176BF5649`。`py_compile`、GD+CB+CP 31项、3个lite_d no-query smoke、`bash -n`、dry-run12和`git diff --check`均通过。

## 2.数据、矩阵与合同

ManySig=`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。split=`tx_rx_day_1_6_3`、seed=`7281105`、L/U/V=`.07/.63/.30`；GD只读L，U关闭，V仅final-only评分。6个GeoSat-C基线路径位于`runs/phase1_loto_clsgeo12_20260808_v1/F{1..6}C_LOTO_CLSGeo12/final_ssdg.pth`，SHA依次为`4d515204...c040`、`29c7d7ca...76d`、`39c6cdd6...4b7`、`32d956f4...5e4`、`2b938154...a9f`、`573ca9d0...de6f`；runner须在启动前核全长SHA。

GPU矩阵固定：0=`F1C＋F5G`；1=`F1G＋F5C`；2=`F2C＋F6G`；3=`F2G＋F6C`；4=`F3C`；5=`F3G`；6=`F4C`；7=`F4G`。GPU7已有SCB v3构建时允许加入F4G，但任一卡不得超过2个计算进程。

G的receipt逐batch及12格记录`total_rows=valid_rows+zero_rows`、valid loss、有限性和解析非零logit-gradient见证；过滤前后均需local4。zero计数只作诊断，不参与阈值、选择或调参。C为`CONTROL_ARM_NOT_APPLICABLE`。

## 3.N607不可覆盖路径与唯一命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v3_10ead8b2`|
|CWD|`<release>/code`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll12_20260809_v3`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v3`|
|outer|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v3_launcher.out`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|retry|NO；caller超时后只读核验landed，禁止重复launch|

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v3_10ead8b2/code && nohup env RUN_ID=phase1_gd_proto_nll12_20260809_v3 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v3_10ead8b2/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v3_10ead8b2/code/scripts/launch_phase1_gd_proto_nll12_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v3_launcher.out 2>&1 < /dev/null &
```

## 4.发布、健康停止与预期工件

唯一runner先执行direct preflight；失败时只用verified bridge。启动前核release/run/log/outer均ABSENT、commit/archive/member、ManySig及6个基线SHA、SCB进程和每卡≤2；远端运行`py_compile`、help、`bash -n`和dry-run12后只启动一次。

启动后记录launcher/12 child的PID、CWD、cmdline、GPU与日志增长。只按路径/hash/覆盖错误、协议或类序漂移、Traceback/OOM/CUDA、拆分后的`FEATURE_NONFINITE`、`HEAD_NONFINITE_OR_ZERO`、`FEATURE_ZERO_FILTERED_INVALID_COVERAGE`、两行同一确定性异常、零checkpoint或成员缺失停止；不得按accuracy、loss趋势或任何性能值停止。触发时只终止严格绑定本run的进程树，保留partial，标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不重试。

成功预期12臂E40、final checkpoint、metrics/config、GD terminal/training/terminal/resource/heldout receipts。G应为1200 attempted batches，`state_update_batches=batches`，全部batch过滤后local4有效，12格计数闭合且终态通过；最终`NON_PROMOTABLE_P0_DISABLED/exit8`为预期训练P0门。只回收小日志、JSON/CSV、PID、completion和manifest，不下载checkpoint/NPZ；完整postfreeze另行执行。
