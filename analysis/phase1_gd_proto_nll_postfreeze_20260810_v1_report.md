# Phase1 GD-ProtoNLL v3一次性postfreeze报告

状态：`PREREGISTERED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`

日期：2026-08-10

## 1.目标与冻结边界

|字段|冻结值|
|---|---|
|run ID|`phase1_gd_proto_nll_postfreeze_20260810_v1`|
|目标|对已完成的GD-ProtoNLL v3 6折C/G final checkpoint执行一次性42步postfreeze，机械产生唯一接受或永久淘汰结论|
|训练输入|`phase1_gd_proto_nll12_20260809_v3`的12个`F{fold}{C/G}_GD_PROTO_NLL12/final_ssdg.pth`，训练合同已完整通过|
|实现commit|`a3a97d882a4bede06e638c9cb1a5697b8e3f459d`|
|独立复核|`P0=0、P1=0、ALLOW`；本地pycompile、focused 9、bash-n、dry-run42、diff-check全部通过|
|禁止|不训练、不校准阈值、不选择checkpoint/折/场景；失败不调参、不重试、不以proxy补偿分类门|

冻结文件SHA256：pair evaluator=`9daa7df9d94cdafdaa648ce9ba5fb03a88847a4686fcfe6daa9a9e895b3d3b76`；GD clean exporter=`c21d84a76c5448e0c8414389222ad764575820f011b05784982d313499f71580`；launcher=`c6ea80a3112c235761c94ab26be03cd2dfd9e872ba0390684300313838490e30`；test=`0cd3d64da0f9a70c90f8ea6b01f5bb19bab25eec77170af0958b2cc5b6652b30`；design=`91b8800bed75f012dd23d21ad37c5e8c643b4bc774b1f72241c7c12d3f5eccff`。

## 2.42步数据与判定

每个候选一次GD-clean导出严格重建训练同一local4、seed=`7281105`、split=`tx_rx_day_1_6_3`、L/U/V=`.07/.63/.30`：只forward L、V与proxy，U forward/persist均为0；ManySig实际SHA必须为`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。clean分类只用checkpoint未见V；Gaussian只用L拟合，known u只用V，unknown u只用proxy；无阈值。

LEO继续使用通用source-only导出：local4×400=1600行，三场景`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`，runtime view=`single`、source profile=`satellite`、TTA=`none`、seed=`7281718`。clean V和LEO是两个明确不同的固定诊断slice，各自C/G物理行严格配对，不混写。

判定为非补偿硬门：clean 6/6四floor均不低于C−2pp；LEO 18格四floor均不低于C−2pp；每折三场景等权overall Δ≥0；18格等权overall Δ≥0；逐折6/6同时满足GD连续u的ΔAUROC>0与Δ(proxy mean u−V-known mean u)>0。任一失败即`REJECT_GD_PROTO_NLL_PERMANENT`。

## 3.N607路径、矩阵与唯一命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v1_a3a97d88`|
|CWD|`<release>/code`|
|training root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll12_20260809_v3`，只读|
|postfreeze root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll_postfreeze_20260810_v1`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll_postfreeze_20260810_v1`|
|outer|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll_postfreeze_20260810_v1_launcher.out`|
|矩阵|0=`F1C＋F5G`；1=`F1G＋F5C`；2=`F2C＋F6G`；3=`F2G＋F6C`；4=`F3C`；5=`F3G`；6=`F4C`；7=`F4G`|
|步骤|12 GD-clean＋12 LEO＋12 proxy＋6 pair=42|
|retry|NO|

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v1_a3a97d88/code && nohup env POSTFREEZE_RUN_ID=phase1_gd_proto_nll_postfreeze_20260810_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v1_a3a97d88/code TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll12_20260809_v3 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v1_a3a97d88/code/scripts/launch_phase1_gd_proto_nll_postfreeze_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll_postfreeze_20260810_v1_launcher.out 2>&1 < /dev/null &
```

## 4.发布与健康门

唯一runner先direct preflight；启动前核release/run/log/outer均ABSENT、训练root12 checkpoint存在、ManySig与archive/member SHA、远端pycompile/help/bash-n/dry-run42、GPU每卡≤2。只启动一次；caller超时后只读核验，不重发。

停止只依据路径/hash/覆盖、split/head/checkpoint/arm/root绑定、Traceback/OOM/CUDA、确定性异常、缺失输出或零pair；不得依据任何性能数值提前停。结束后需有12 GD-clean NPZ、12 LEO NPZ、12 proxy JSON/CSV、6 pair JSON及日志；只回收小JSON/CSV/log/manifest/completion，不下载NPZ/checkpoint。最终由F6 aggregate原样给出结论，runner只作技术核验。
