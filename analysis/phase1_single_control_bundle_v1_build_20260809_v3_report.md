# Phase1单读出local4控制bundle v1真实构建v3报告

状态：`PREREGISTERED_LOCAL_REVISION11_VERIFIED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`

日期：2026-08-09

## 1.目标与冻结版本

|字段|冻结值|
|---|---|
|run ID|`phase1_single_control_bundle_v1_build_20260809_v3`|
|目标|从冻结F1C checkpoint和ManySig构建不可变local4技术bundle，完成真实IQ六字段parity、状态零更新、资源门及CARE N=1闭环|
|与v2差异|仅把receipt中的ASCII数字day／RX值按训练轴索引解析，并在source构建及proxy／held／target排除枚举中恢复真实物理标签；不改模型、数据、公式、descriptor、schema或资源门|
|实现commit|`3d6b739a5f3d2811b39e28247afddff491e35e48`|
|Git工作树|`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`，分支`codex/phase3-responsibility-20260807`|
|声明|技术构建，不训练、不读取性能、不声明Phase1晋级、真实unknown、多星协同、注册或在轨性能|

v1因resolved config缺失真实默认字段停止；v2通过该点后，因字符串化的day索引`0`被误作ManySig日期标签而停止。两次均为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。Revision11只修复该确定性接口错误。

## 2.冻结文件与本地验证

|文件|SHA256|
|---|---|
|`code/cvsrffi/phase1_single_control_bundle_v1.py`|`EF1C6D3F3975D2DD40EDF6566E4234FE2C44B6A40A0453A52891E7A6E7C2787C`|
|`code/scripts/build_phase1_single_control_bundle_v1.py`|`E116788F0B13884C3E3C2AE0F2376CA242BD9D8731C95DE677D6E1C7D11D162E`|
|`code/tests/test_phase1_single_control_bundle_v1.py`|`3AFE74695296D12CDA892E575D2347424DAB4D63D8008E1E605652622AB5E0B5`|
|`analysis/phase1_single_control_bundle_v1_design_20260809.md`|`C1B02F2CF9B7A0841E78366456909B822F9407FF9B569E447988DDCC1BE4BD1B`|
|`analysis/phase3_final_goal_traceability_20260809.md`|`5BD512FEC04D5797E79596FEAA2FA2404C0B323FAECDFAE023E0DD5B5D0E0D4A`|

`ssr-gpu`本地验证：`py_compile`PASS；focused SCB`28 passed`；CLI`--help`PASS；`git diff --check`PASS。独立窄复核：`P0=0、P1=0、ALLOW`。不增加P2审查或新数据对齐层。

## 3.冻结输入与数据轴

|输入|远端路径|SHA256|
|---|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/final_ssdg.pth`|`0b1e1d24621f5c044b0a77f30915ec1f67342e6132fba8df28f21b43ad6b2ab8`|
|completion|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_training_completion_receipt.json`|`c31edd31f1ec322615b4d0647cfcb9ece4e8ef5c3940d54aaa89c85c60f4431c`|
|terminal|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_terminal_status.json`|`0575ed6ee778e5b7b94e1e5b842e9ff24bf32496b05d36f82f658117a791c3a2`|
|CP terminal|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_cp_sfce_terminal_receipt.json`|`5a9677d6eab883f221ceb5c544f8e0bf6bcdb26479bba326766494bb7ce482e0`|
|ManySig|`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`|`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`|

固定split：seed`7281105`；模式`tx_rx_day_1_6_3`；L／U／V=`3920／35280／16800`；source day轴`0,1`、RX轴`0..6`；target day轴`2,3`、RX轴`10,11,7,8,9`；equalized=`1`；local4顺序`20-15／20-19／6-15／8-20`。轴索引必须经训练同一resolver映射到ManySig真实日期／RX标签。

## 4.N607路径、资源与唯一命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_single_control_bundle_v1_build_20260809_v3_3d6b739a`|
|CWD|release根|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_single_control_bundle_v1_build_20260809_v3`，启动前必须ABSENT|
|log|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_single_control_bundle_v1_build_20260809_v3/build.out`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|GPU|物理GPU7；不得超过2个训练／计算进程每卡|
|retry|NO；caller超时后只读确认landed，禁止重复launch|

```bash
CUDA_VISIBLE_DEVICES=7 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/build_phase1_single_control_bundle_v1.py --real-build --project-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_single_control_bundle_v1_build_20260809_v3_3d6b739a --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/final_ssdg.pth --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --completion-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_training_completion_receipt.json --terminal-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_terminal_status.json --cp-terminal-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_cp_sfce_terminal_receipt.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_single_control_bundle_v1_build_20260809_v3 --device cuda
```

唯一runner先执行direct只读preflight；只有direct TCP／SSH不可用才使用一次verified bridge。核release／output／log／staging均ABSENT、输入SHA、archive/member SHA、无`code/code`、远端`py_compile`和`--help`后唯一启动。

## 5.成功、停止与工件

成功输出严格为9 payload＋`manifest.json`：`runtime/local_evidence.ts`、3个state NPZ、5个locks JSON和manifest；随后用外部content root执行一次同CLI verify。技术门为：严格10文件、SHA/content root、真实IQ六字段／decision parity、state digest不变、bundle≤32MiB、evidence≤64KiB、CPU RSS≤512MiB、CPU p99≤250ms、CUDA VRAM≤256MiB及CARE N=1规范化恒等。

若出现错误commit／SHA／路径、覆盖风险、receipt／split／class／scenario／strict-load漂移、Traceback／OOM／nonfinite／parity／state／resource／loader失败、非零exit或成员不全，标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，保留partial，不重试、不调门。只回收log、manifest及JSON小receipt／PID／exit；不下载ManySig、源checkpoint、embedded runtime或NPZ。

## 6.运行终态（2026-08-10）

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；`retry=NO`。唯一detached进程PID`420208`在约4小时35分钟CPU密集计算后退出；未观察到输出根、manifest、payload或worker。退出码因nohup detached启动未保留wait状态，诚实记录为`null`，归因指纹为`UNOBSERVED_PROCESS_EXIT_AFTER_SUSTAINED_CPU`，不猜测OOM或其他原因。

|证据|结果|
|---|---|
|远端archive SHA256|`faef637b2d9af836eeb932baf1d5200006d4a69344a86681104f3a6bd6f7c8ff`|
|远端archive成员SHA256（LF）|core=`9788cfec06715def97a7461141331e3e8be6f04ad845e52b8b89df1bde192ef2`；CLI=`c843a49ec152105f7841e941c4415d80f9a4e7e47c88a3cfec2e39a844ad151b`；test=`edaea6a756ca9b882991beef6cf5aba836b2dd1b115c99f734e06213bc580455`；design=`58bdaabe047b130db02cec105b31a8d530d09ea44dc674dc9dfb9bd54a1cf75f`；trace=`243d54783bc387aa0697ae3cd4eca3c3635ed6e5506145704486e4329b901125`|
|PID/CWD/GPU|启动时PID`420208`、CWD为release根、GPU7；终态PID不存在，GPU7释放至1MiB级别|
|运行观测|CPU≈100%；04:15:58时cpu_ticks=1,536,591、rchar=9,486,939,700、syscr=10,940、read_bytes=0、write_bytes=5,582,848；04:30:53时cpu_ticks=1,626,141（约5分钟增29,742ticks）；GPU7瞬时util=0%、mem=498MiB（run compute=488MiB）|
|技术输出|output根ABSENT；build.out 4,747 bytes，内容仅PyTorch`TracerWarning`；无Traceback/OOM/nonfinite；无子worker、`--resource-probe`、`.scb-n1-staging`或`/tmp/scb1-resource-*`|
|系统归因核验|目标日志无wait/exit记录；最近6小时`journalctl -k`/`dmesg`无OOM、out-of-memory、killed-process或kill命中；标记`UNOBSERVED_PROCESS_EXIT_AFTER_SUSTAINED_CPU`|

远端小证据：`logs/phase1_single_control_bundle_v1_build_20260809_v3/build.out` SHA256=`9ea28e8e36899d495f3733a6548416ca4482c739117657b2ed14d7a7e94c98eb`；`build_pid_exit.json` SHA256=`753a0fba035b7ade34c481ce275101457a92462f364b10b4581770eb95caec1b`（495 bytes）。本地回收目录`E:\type10-7\automation_reports\CV-SincNet\phase1_single_control_bundle_v1_build_20260809_v3\artifacts`中两文件与远端bytes/SHA逐项一致。SSH/SCP/TCP22、run进程及GPU均已清理；release保留且未覆盖。无任何性能结果或bundle晋级结论。
