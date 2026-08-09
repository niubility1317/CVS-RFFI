# Phase1单读出local4控制bundle v1真实构建v2报告

状态：`PREREGISTERED_LOCAL_REVISION10_VERIFIED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`

日期：2026-08-09

## 1.目标与版本

|字段|冻结值|
|---|---|
|run ID|`phase1_single_control_bundle_v1_build_20260809_v2`|
|目标|从冻结F1C checkpoint和ManySig构建不可变local4技术bundle，完成真实IQ六字段parity、状态零更新、资源门及CARE N=1闭环|
|比较|v1在resolved config审计处技术停止；v2仅修复九个真实模型默认字段的presence-aware闭合|
|实现commit|`ef9cd58a1931258b808a0401b0a560e7683046f9`|
|Git工作树|`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`，分支`codex/phase3-responsibility-20260807`|
|操作|主控负责方法、版本与结论；唯一runner负责N607落地、一次启动、监控、回收|
|声明|技术构建，不训练、不评测性能、不声明Phase1晋级、真实unknown、多星协同、注册或在轨性能|

v1完整失败证据见`phase1_single_control_bundle_v1_build_20260809_v1/report.md`。v1唯一指纹为resolved namespace缺失`dom_feature_key`；只读checkpoint审计确认九个字段由`build_baseline_model`的真实`getattr`默认决定。Revision10只有checkpoint args与final namespace双ABSENT时才物化这九个精确默认；显式`None`、错类型、单边缺失和非白名单缺失仍fail-closed。算法、数据、descriptor、bundle schema及资源门零变化。

## 2.冻结文件与验证

|文件|SHA256|
|---|---|
|`code/cvsrffi/phase1_single_control_bundle_v1.py`|`FB53408B6D82BC907D73027BC39FAD53DA2BFC5D324ABC4FCE89AFB3DA3D7627`|
|`code/scripts/build_phase1_single_control_bundle_v1.py`|`E116788F0B13884C3E3C2AE0F2376CA242BD9D8731C95DE677D6E1C7D11D162E`|
|`code/tests/test_phase1_single_control_bundle_v1.py`|`92E6ED282750DBFE3367FCD2B483168B16EEE2BFC6A1A3530ECB85E6E782A72D`|
|`analysis/phase1_single_control_bundle_v1_design_20260809.md`|`30974517ECDE36497225F4467876C35A8ABCCA53C6D93C28D6EA95575231D592`|
|`analysis/phase3_final_goal_traceability_20260809.md`|`7015FEAE796FC45323211DA107197E5C3311488A7D44FB3B6A9E10364FFB70F0`|

本地`ssr-gpu`：`py_compile`PASS；focused SCB`27 passed`；CLI`--help`及fixture build＋external-root verify PASS；`git diff --check`PASS。独立复审：`P0=0、P1=0、ALLOW`。不再增加发布前审查。

## 3.冻结输入

|输入|远端路径|SHA256|
|---|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/final_ssdg.pth`|`0b1e1d24621f5c044b0a77f30915ec1f67342e6132fba8df28f21b43ad6b2ab8`|
|completion|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_training_completion_receipt.json`|`c31edd31f1ec322615b4d0647cfcb9ece4e8ef5c3940d54aaa89c85c60f4431c`|
|terminal|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_terminal_status.json`|`0575ed6ee778e5b7b94e1e5b842e9ff24bf32496b05d36f82f658117a791c3a2`|
|CP terminal|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_cp_sfce_terminal_receipt.json`|`5a9677d6eab883f221ceb5c544f8e0bf6bcdb26479bba326766494bb7ce482e0`|
|ManySig|`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`|`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`|

split仍为seed`7281105`、`tx_rx_day_1_6_3`、L／U／V=`3920／35280／16800`、local4顺序`20-15／20-19／6-15／8-20`、source days`0,1`／RX`0..6`、equalized`1`、三场景`clear／low-elev／rain weak`。不读取proxy、held、target或query进入模型。

## 4.N607路径、资源与唯一命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_single_control_bundle_v1_build_20260809_v2_ef9cd58a`|
|CWD|release根|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_single_control_bundle_v1_build_20260809_v2`，启动前必须ABSENT|
|log|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_single_control_bundle_v1_build_20260809_v2/build.out`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|GPU|物理GPU7；若会超过2进程／卡则停止，不换卡|
|retry|NO；caller超时后只读确认landed，禁止重复launch|

```bash
CUDA_VISIBLE_DEVICES=7 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python   code/scripts/build_phase1_single_control_bundle_v1.py   --real-build   --project-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_single_control_bundle_v1_build_20260809_v2_ef9cd58a   --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/final_ssdg.pth   --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl   --completion-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_training_completion_receipt.json   --terminal-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_terminal_status.json   --cp-terminal-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_cp_sfce_terminal_receipt.json   --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_single_control_bundle_v1_build_20260809_v2   --device cuda
```

runner先direct preflight；只有TCP／SSH不可用才使用一次verified bridge。核release／output／log／staging均ABSENT、输入SHA、archive/member SHA、无`code/code`、远端`py_compile`和`--help`后，以detached方式唯一启动并记录PID／CWD／cmdline／GPU。

## 5.成功门、停止门与工件

成功输出严格为9 payload＋`manifest.json`：`runtime/local_evidence.ts`、3个state NPZ、5个locks JSON和manifest。用manifest`content_root`执行一次同CLI external-root verify。

|技术门|要求|
|---|---|
|成员|严格10文件、无extra／missing／symlink，member SHA和外部content root闭合|
|状态|`TECHNICAL_LOCAL4_CONTROL_BUNDLE`、`performance_promoted=false`|
|parity|真实IQ六字段／decision一致，runtime和state前后digest不变|
|资源|bundle≤32MiB；evidence≤64KiB；CPU RSS≤512MiB；CPU p99≤250ms；CUDA VRAM≤256MiB|
|CARE|N=1规范化后`p_local／decision／label／reason／evidence_hash`恒等|
|结论|`NO_PERFORMANCE_RESULT`，不读取或解释accuracy／AUROC／FAR|

错误commit／SHA／路径、覆盖风险、receipt／split／class／scenario／strict-load漂移、Traceback／OOM／nonfinite／parity／state／resource／loader失败、非零exit或成员不全，均立即标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，保留partial，不重试、不调门。

只回收log、manifest和JSON小receipt／PID／exit；不下载ManySig、源checkpoint、embedded-weight runtime或NPZ。更新本报告根与Git镜像，记录archive、PID、content root、resource/parity、artifact和清理证据。

