# Phase1单读出local4控制bundle v1真实构建v5报告

状态：`LOCAL_VERIFIED / PREREGISTERED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`

日期：2026-08-10

## 1.目标与唯一修复

|字段|冻结值|
|---|---|
|run ID|`phase1_single_control_bundle_v1_build_20260810_v5`|
|目标|从冻结F1C checkpoint与ManySig生成10成员技术bundle，完成真实IQ六字段parity、状态零更新、资源门与CARE N=1恒等闭环|
|v4终态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；`build.exit=139`且Python进程明确`Segmentation fault (core dumped)`，output/staging/resource均未生成|
|v5唯一修复|ManySig严格路径只执行一次`pickle.load`；L/U descriptor、L/V runtime和V descriptor改由固定512条、fresh-exec、单线程worker执行；worker非零/139/乱序/缺行一次性fail-closed且不可进入`build_bundle`|
|明确不变|checkpoint、ManySig、split、索引、view seed、L/U/V样本、geometry、descriptor、median/MAD、tail、runtime、bundle schema/10成员、resource门、CARE语义|
|实现commit|`728b4029507ce5bf5a79002999f709778fe3ff53`|
|独立复核|`P0=0、P1=0、ALLOW`，仅本地技术发布签字|
|声明|v5降低长生命周期native状态与隐藏双反序列化风险，不把v3/v4归因为OOM，也不预先声称真实build成功、性能改善或Phase1晋级|

## 2.冻结文件与本地验证

|文件|SHA256|
|---|---|
|`code/cvsrffi/phase1_single_control_bundle_v1.py`|`1d36b95beeae9044d44b311c982deddb9ad7a0f6b5112094486a28c51c80ead1`|
|`code/scripts/build_phase1_single_control_bundle_v1.py`|`e116788f0b13884c3e3c2ae0f2376ca242bd9d8731c95de677d6e1c7d11d162e`|
|`code/tests/test_phase1_single_control_bundle_v1.py`|`99e61e6e294f5802cb4f1e5d69d600e679eae080cdf7da1d48f6880e9cd57a15`|
|`analysis/phase1_single_control_bundle_v1_design_20260809.md`|`6a8fb4e4bf18008f5283f172606b3ce69bc09b6d1309202e09133461b462cd59`|
|`analysis/phase1_single_control_bundle_v1_v5_native_isolation_20260810.md`|`2b5821b3bb16f301c0fcf4c18c319799a17a530958e281dd7b95677bde392ed4`|
|`analysis/phase3_final_goal_traceability_20260809.md`|`5bd512fec04d5797e79596feaa2fa2404c0b323faecdfae023e0dd5b5d0e0d4a`|

`ssr-gpu`串行验证：`py_compile`通过；完整focused SCB回归`37 passed`；公开build CLI`--help`通过；`git diff --check`通过。新增直接证据：真实小PKL恰好一次`pickle.load`且历史loader不可达；L/U/V descriptor多chunk与单进程reference为`np.array_equal`；L/V TorchScript runtime同样等价；U IPC输入/输出的label、physical key和hash字段均拒绝；exit139无重试且output/staging不存在；乱序、缺行与IPC超限在worker/bundle前fail-closed；既有10成员、schema、parity、resource和CARE回归通过。

## 3.冻结输入与数据轴

|输入|远端路径|SHA256|
|---|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/final_ssdg.pth`|`0b1e1d24621f5c044b0a77f30915ec1f67342e6132fba8df28f21b43ad6b2ab8`|
|completion|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_training_completion_receipt.json`|`c31edd31f1ec322615b4d0647cfcb9ece4e8ef5c3940d54aaa89c85c60f4431c`|
|terminal|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_terminal_status.json`|`0575ed6ee778e5b7b94e1e5b842e9ff24bf32496b05d36f82f658117a791c3a2`|
|CP terminal|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_cp_sfce_terminal_receipt.json`|`5a9677d6eab883f221ceb5c544f8e0bf6bcdb26479bba326766494bb7ce482e0`|
|ManySig|`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`|`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`|

固定split：seed=`7281105`；mode=`tx_rx_day_1_6_3`；L/U/V=`3920/35280/16800`；source day轴=`0,1`、RX轴=`0..6`；target day轴=`2,3`、RX轴=`10,11,7,8,9`；equalized=`1`；local4顺序=`20-15/20-19/6-15/8-20`。worker chunk=`512`是固定工程常量，不是方法超参或可调资源旋钮。

## 4.N607路径、资源与唯一命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_single_control_bundle_v1_build_20260810_v5_728b4029_full`|
|CWD|release根|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_single_control_bundle_v1_build_20260810_v5`，启动前必须ABSENT|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_single_control_bundle_v1_build_20260810_v5`，启动前必须ABSENT|
|GPU|物理GPU7；每卡计算进程不得超过2|
|retry|`NO`；调用端超时先清理本地SSH并只读确认是否landed，禁止重复launch|

Runner必须从commit`728b4029507ce5bf5a79002999f709778fe3ff53`生成完整、无prefix的LF归档，不再尝试仅含目标文件的partial release。完成archive/member/mode、输入SHA、远端`py_compile`和公开CLI`--help`后，先创建新的log root，再从release根仅执行下列detached命令一次：

```bash
nohup bash -lc 'CUDA_VISIBLE_DEVICES=7 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/build_phase1_single_control_bundle_v1.py --real-build --project-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_single_control_bundle_v1_build_20260810_v5_728b4029_full --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/final_ssdg.pth --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --completion-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_training_completion_receipt.json --terminal-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_terminal_status.json --cp-terminal-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_cp_sfce_terminal_receipt.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_single_control_bundle_v1_build_20260810_v5 --device cuda; rc=$?; printf "%s\n" "$rc" > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_single_control_bundle_v1_build_20260810_v5/build.exit; exit "$rc"' > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_single_control_bundle_v1_build_20260810_v5/build.out 2>&1 < /dev/null & printf '%s\n' "$!" > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_single_control_bundle_v1_build_20260810_v5/build.pid
```

## 5.健康控制、成功工件与分析边界

唯一Runner先执行direct N607 preflight并核：release/output/log/staging/temp均ABSENT；输入SHA逐项匹配；GPU7现有计算进程数；完整archive无`code/code`且目标成员SHA/mode闭合。任何SSH/SCP超时或引号错误均先清理本地`ssh.exe`/TCP22并只读核远端状态，不把调用端失败直接当作实验失败或成功。

启动后立即核wrapper/Python PID、CWD、cmdline、output/log绑定、GPU7和日志。监控额外记录fresh-exec worker的role/chunk、主进程CPU/RSS、GPU显存、build.exit及output/staging状态；短命worker自然出现/退出不视为异常。不得因耗时、日志静默、CPU/GPU利用率、accuracy、loss或任何性能值停止。

技术停止仅限：错误checkout/hash/覆盖、receipt/split/class/scenario/strict-load漂移、Traceback/OOM/nonfinite/parity/state/resource/loader错误、任一native worker非零或signal退出、主build非零退出、成员不全或协议/P0违反。停止前先绑定精确run-owned PID/CWD/cmdline，只停止本run树并保留partial；标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不重试、不重启。

成功输出严格为9 payload＋`manifest.json`，随后从外部content root运行同一公开CLI verify。成功门：10文件及SHA/root闭合；真实IQ六字段/decision parity；state digest不变；bundle≤32MiB；evidence≤64KiB；CPU RSS≤512MiB；CPU p99≤250ms；CUDA VRAM≤256MiB；CARE N=1规范化恒等。只回收日志、manifest和JSON小receipt；不下载ManySig、源checkpoint、embedded runtime或NPZ。

本run是Phase3前置技术构建，不读取分类性能，不提供unknown、多卫星、协同收益或Phase1晋级结论。若真实bundle成功，下一步仅执行已实现的真实N=1 bridge；N=2..5仍须真实同事件多接收机输入或明确标记proxy，不能由本run推断。
