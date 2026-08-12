# D92 CSOAS G0 v2 N607 runner handoff

## STATUS

`ARTIFACTS_COMPLETE / G0_MECHANISM_RESOURCE_PASS / NO_PERFORMANCE_RESULT`

唯一run：`d92_e0_full_csoas_g0_k10_20260812_v2`。科学commit：`b8ebd4f4522fcc3e9e6b7dd18d722c329021f181`；release仓库`E:\type10-7\code\snapshots\d92_125wt`在`9b881ea4a853234a6720e1cb5716ad7999bba32d`且clean。P0=0、P1=0、APPROVE；fresh retry=`false`。

## PRECHECK

2026-08-12 19:56:45 CST执行`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`：direct普通账户、项目根目录和8张RTX3090均通过，GPU为0%利用率/1MiB占用。本地每次SSH/SCP后均无`ssh.exe`残留、无N607 TCP22连接。launch前远端source/output/logs均`ABSENT`且无同run进程；只创建source目录，未预建code/output/logs。

## SYNC/HASH

按冻结顺序SCP archive→launch：

|artifact|size|SHA256|结果|
|---|---:|---|---|
|`d92_csoas_g0_runtime_b8ebd4f4.tar.gz`|6,137,499|`4b0b434a26b47511cb0ddeb9f2455bc81964d8fcef312e75e57879547b631ca5`|远端匹配|
|`launch.sh`|10,309|`d60520af5c0e9d7019894361cc8ab81fade8c0f3ec4957fa5bb01b7f09dff3f7`|远端匹配|

远端landing核验：archive 1458 members、无绝对路径/`..`路径/link成员、required entries齐全；`bash -n`通过；Python 3.10.19、`CUDA_AVAILABLE=True`、device_count=8；四份seal及ground manifest SHA全部匹配。

## COMMAND

按冻结命令执行一次，SSH返回`SSH_EXIT=0`，无重试、无重启、无换命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_b8ebd4f4_20260812_v2 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

## PID-CWD-CMDLINE-GPU

任务在首次完整探针前已完成，因此没有可持久读取的run PID；固定CWD为上述source目录，固定cmdline为上述唯一命令。最终跳过探针自身PID后`RUN_ACTIVE=0`，`nvidia-smi --query-compute-apps`为空，GPU已释放。一次`/proc/<pid>/cmdline: No such file`是读取已退出进程的竞态，不是run异常。

## G0三场景技术表

`logs/g0_validation.json`：schema=`cvs.phase2.d92_csoas.truth_free_g0_validation.v2`，marker=`D92_CSOAS_G0_ACTIVE_NON_E0_RESOURCE_PASS`。14个query禁用字段在三场景均为false；不运行scorer、不读取性能。

|scene|active|fallback|candidate/ref fit|FULL actual/total|wall|peak bytes|paired E0 peak|结果|
|---|---|---|---:|---:|---:|---:|---:|---|
|`leo_clear_weak`|true|false|1/0|1/2|12.965436ms|876,544|1,327,104|PASS|
|`leo_low_elev_weak`|true|false|1/0|1/2|12.726085ms|581,632|1,060,864|PASS|
|`leo_rain_weak`|true|false|1/0|1/2|14.268484ms|458,752|57,344|PASS|

wall P90 nearest-rank=`14.268484ms`；target120ms和hard150ms均通过。paired query-token/scenario SHA匹配；candidate prediction SHA=`dd25a86a6b080eb1f30b7d3bf5b19857c23e0870e9f0b121e95e24925738eb05`，不同于paired E0=`d539001ece0319b967023ea05dea7764264a731daa44bf7882a45660ba183cc0`。这只是机制/资源和artifact identity证据。

## ARTIFACTS/PATHS

远端保留并完整取回：

- source：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_b8ebd4f4_20260812_v2`
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_csoas_g0_k10_20260812_v2`
- logs：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_csoas_g0_k10_20260812_v2`
- local retrieval：`E:\type10-7\local_artifacts\d92_e0_full_csoas_g0_k10_20260812_v2\{source,output,logs}`

|root|files|bytes|tree SHA256|
|---|---:|---:|---|
|source|1429|73,632,132|`5814b80c56e397da21ff51626be482c4f2f2fede20dfbfe902461f9f22ac8f5e`|
|output|10|1,018,135|`afb51a16be729a46639a7b6d20d2cd383de89b88ecbcde49cf20310df68d774a`|
|logs|5|2,597|`b5243b2a679fae2f014e3c47b05362cf2ff333d3f5a920e695fbe87f82eae1a6`|

本地与远端count、bytes、逐文件SHA和tree SHA一致。source tree SHA按远端`find -type f -printf '%P\\n' | sort`顺序，以`relative_path  file_sha256\\n`作为规范输入计算。

## EXCEPTIONS/CLEANUP

`launch_driver.err`、`import_closure.err`、`prediction.err`为空；无P0/P1、无确定性异常指纹、无覆盖风险、无retry。最终run/GPU/SSH/SCP/TCP22均清零。远端source/output/logs未删除。

## NEXT

主agent可基于本handoff与`report.md`纳入G0技术分析；不得把本run写成准确率、H、BA、floor、forgetting、unknown拒识、Phase3协同或真实在轨性能结果。runner不改变方法、阈值、矩阵，不启动任何后续run。
