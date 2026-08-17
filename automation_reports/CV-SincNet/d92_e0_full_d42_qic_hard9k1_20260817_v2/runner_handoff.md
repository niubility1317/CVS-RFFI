# Runner handoff: d92_e0_full_d42_qic_hard9k1_20260817_v2

## STATUS

`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

Sole runner=Luna/max。G0 v1/v2、CCOC及Hard9 v1未触碰。本run的冻结launch只执行1次，未重试、未重启、未运行analyzer、未读取性能值。

## PRECHECK / SYNC

- direct ordinary `N607` preflight通过；项目根目录、冻结Python和8张GPU可见。
- 同run远端archive、driver、source、output、logs、driver out/err及进程在launch前均ABSENT。
- archive顺序SCP成功：320955B，SHA256 `46cd187aeb902eaa5adc8ab79777e6f2a9b94bf0aa85009e85cadd65593175e1`，50 members。
- driver顺序SCP成功：6692B，SHA256 `57c787f0f06968bf565d34d02b526af50bac2d3dd3df09005e758c817313c025`；远端`bash -n`通过；archive embedded config SHA与manifest/Git均为`6a8fc7942a2820b5779e5cf74e1222df22d800791864ce312db5708d7a47e8a8`。

## COMMAND / FAILURE

唯一命令为报告第4节冻结detached command：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_qic_hard9_k1_driver_d92_e0_full_d42_qic_hard9k1_20260817_v2.sh >./d92_qic_hard9_k1_driver_d92_e0_full_d42_qic_hard9k1_20260817_v2.out 2>./d92_qic_hard9_k1_driver_d92_e0_full_d42_qic_hard9k1_20260817_v2.err </dev/null &
```

prepare通过并生成`matrix_manifest.json`后，truth-free smoke失败。`smoke.err`原文为：

```text
D92 QIC Hard9+K1 failed: candidate fit is not finite
```

driver随后退出，8 shards未启动；未生成smoke receipt、job receipt、score或truth sidecar。

## TECHNICAL COUNTS / BOUNDARY

|项|结果|
|---|---:|
|formal prediction before/after|0/0|
|formal COMMIT/fit/resource/execution|0/0/0/0|
|job_receipt/score/shard_summary|0/0/0|
|prepare matrix manifest|1；job_count=10、scene_arm_count=30|
|smoke technical before/after closure|各1组；各含COMMIT、execution、fit、resource、prediction artifact|
|smoke marker/receipt|ABSENT|
|truth sidecars|0生成、0取回|
|performance/analyzer|未读取、未运行|

## ARTIFACTS / CLEANUP

本地取回根目录：`E:/type10-7/local_artifacts/d92_e0_full_d42_qic_hard9k1_20260817_v2`。包含94个source文件、13个output文件、8个技术日志、driver out/err和输入archive/launch。driver out为2506B，driver err为0B。远端source、output、logs、driver out/err保留。收尾核验同run PID=0、GPU compute为空、本地N607 TCP22无ESTABLISHED连接。

## NEXT_ACTION

主代理应在本地定位candidate fit非有限问题，完成窄验证及独立P0/P1复核后，使用新的不可覆盖run ID重新注册。`fresh_run_retry=false`；不得覆盖或重试本run。
