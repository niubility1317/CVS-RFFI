# Phase1单读出local4控制bundle v1真实构建v4报告

状态：`PREREGISTERED_REVISION12 / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`

日期：2026-08-10

## 1.目标与唯一修复

|字段|冻结值|
|---|---|
|run ID|`phase1_single_control_bundle_v1_build_20260810_v4`|
|目标|从冻结F1C checkpoint与ManySig生成10成员技术bundle，完成真实IQ六字段parity、状态零更新、资源门与CARE N=1恒等闭环|
|v3终态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；唯一PID持续CPU约4小时35分后无可观察exit消失，未生成output；无Traceback、OOM或kernel kill证据|
|v4唯一修复|首次加载的完整ManySig对象原样返回并复用于末段proxy／held／target排除审计，删除第二次完整PKL加载|
|明确不变|split、索引、view seed、L/U/V样本、geometry、descriptor、tail、runtime、bundle schema/content root、resource、CARE语义|
|实现commit|`c3949740356d478459cd4a3c30094b7bcab025b7`|
|独立复核|`P0=0、P1=0、ALLOW`|
|声明|Revision12只降低重复加载和峰值风险；不宣称已证明v3根因或OOM，也不构成真实build、性能或Phase1晋级结论|

## 2.冻结文件与本地验证

|文件|SHA256|
|---|---|
|`code/cvsrffi/phase1_single_control_bundle_v1.py`|`278f402ba183431092bdfc9e5a8d5dcf73c4db1162f42d98232ced2116d0efe2`|
|`code/scripts/build_phase1_single_control_bundle_v1.py`|`e116788f0b13884c3e3c2ae0f2376ca242bd9d8731c95de677d6e1c7d11d162e`|
|`code/tests/test_phase1_single_control_bundle_v1.py`|`9949e30fa1c478bff26a3aa7a9d594137f6e2ee2d267b18e39c8d842c31de031`|
|`analysis/phase1_single_control_bundle_v1_design_20260809.md`|`f897f63454f6e3f86cffa095d19901b71b37f3edb02266903722a21d194131da`|
|`analysis/phase3_final_goal_traceability_20260809.md`|`5bd512fec04d5797e79596feaa2fa2404c0b323faecdfae023e0dd5b5d0e0d4a`|

`ssr-gpu`验证：`py_compile`通过；Revision12定向测试`2 passed`；完整focused SCB回归`30 passed`；公开build CLI`--help`通过；`git diff --check`通过。core模块本身没有公开CLI，其`--help`按内部worker合同预期拒绝，本轮不把它列为验证项。

生产调用链的冻结结构为：helper内loader调用恰好1次；builder调用helper恰好1次、直接loader为0次；proxy、held、target三次排除枚举的首参均为首次加载的同一完整6TX对象。local4 source view继续浅引用原始前4个TX，不复制IQ数组。

## 3.冻结输入与数据轴

|输入|远端路径|SHA256|
|---|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/final_ssdg.pth`|`0b1e1d24621f5c044b0a77f30915ec1f67342e6132fba8df28f21b43ad6b2ab8`|
|completion|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_training_completion_receipt.json`|`c31edd31f1ec322615b4d0647cfcb9ece4e8ef5c3940d54aaa89c85c60f4431c`|
|terminal|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_terminal_status.json`|`0575ed6ee778e5b7b94e1e5b842e9ff24bf32496b05d36f82f658117a791c3a2`|
|CP terminal|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_cp_sfce_terminal_receipt.json`|`5a9677d6eab883f221ceb5c544f8e0bf6bcdb26479bba326766494bb7ce482e0`|
|ManySig|`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`|`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`|

固定split：seed=`7281105`；mode=`tx_rx_day_1_6_3`；L/U/V=`3920/35280/16800`；source day轴=`0,1`、RX轴=`0..6`；target day轴=`2,3`、RX轴=`10,11,7,8,9`；equalized=`1`；local4顺序=`20-15/20-19/6-15/8-20`。所有receipt、checkpoint、数据和场景SHA必须在构建前逐项闭合。

## 4.N607路径、资源与唯一命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_single_control_bundle_v1_build_20260810_v4_c3949740_full`|
|CWD|release根|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_single_control_bundle_v1_build_20260810_v4`，启动前必须ABSENT|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_single_control_bundle_v1_build_20260810_v4`|
|GPU|物理GPU7；每卡计算进程不得超过2|
|retry|NO；调用端超时先只读确认是否landed，禁止重复launch|

Runner在确认log root为新路径后先创建该目录，再从release根仅执行下列detached命令一次；wrapper在`build.exit`保留真实退出码，外层PID写入`build.pid`：

```bash
nohup bash -lc 'CUDA_VISIBLE_DEVICES=7 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/build_phase1_single_control_bundle_v1.py --real-build --project-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_single_control_bundle_v1_build_20260810_v4_c3949740_full --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/final_ssdg.pth --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --completion-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_training_completion_receipt.json --terminal-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_terminal_status.json --cp-terminal-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_cp_sfce_terminal_receipt.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_single_control_bundle_v1_build_20260810_v4 --device cuda; rc=$?; printf "%s\n" "$rc" > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_single_control_bundle_v1_build_20260810_v4/build.exit; exit "$rc"' > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_single_control_bundle_v1_build_20260810_v4/build.out 2>&1 < /dev/null & printf '%s\n' "$!" > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_single_control_bundle_v1_build_20260810_v4/build.pid
```

## 5.发布、健康监控与终态

预发布修复记录：首个仅含5个冻结文件的partial release`phase1_single_control_bundle_v1_build_20260810_v4_c3949740`已落地并保留；远端`py_compile`通过，但公开build CLI`--help`在导入阶段失败，指纹为`ModuleNotFoundError: cvsrffi.checkpoint_loading`。本run未启动，output/log未创建；partial release及小归档作为失败证据保留。经授权仅新增同一commit的全依赖树release`phase1_single_control_bundle_v1_build_20260810_v4_c3949740_full`，不覆盖或清理partial。

唯一Runner先执行direct N607 preflight；启动前核release/output/log/staging/temp均ABSENT、输入SHA、commit/archive/member SHA、无`code/code`、GPU进程数。落地后运行`py_compile`、公开build CLI`--help`和focused Revision12定向测试；通过后唯一启动。

启动后核PID、CWD、cmdline、GPU、CPU时间、日志增长、output/staging/resource worker。只因路径/hash/覆盖、receipt/split/class/scenario/strict-load漂移、Traceback/OOM/nonfinite/parity/state/resource/loader错误、非零exit或成员不全而停止；不得因耗时、静默、accuracy、loss或其它性能值停止。若技术停止，只终止已证明属于本run的进程树，保留partial，标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不重试。

成功输出严格为9 payload＋`manifest.json`，随后以外部content root执行同一公开CLI verify。技术门：10文件及SHA/root闭合；真实IQ六字段/decision parity；state digest不变；bundle≤32MiB；evidence≤64KiB；CPU RSS≤512MiB；CPU p99≤250ms；CUDA VRAM≤256MiB；CARE N=1规范化恒等。只回收日志、manifest和JSON小receipt；不下载ManySig、源checkpoint、embedded runtime或NPZ。

当前终态：`NOT_LAUNCHED / NO_PERFORMANCE_RESULT`。
