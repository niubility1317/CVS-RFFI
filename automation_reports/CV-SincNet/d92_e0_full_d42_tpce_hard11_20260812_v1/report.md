# D92 E0 FULL D42 TPCE Tail-Pair Code Exchange Hard11本地发布准备

## 状态

|字段|冻结值|
|---|---|
|run ID|`d92_e0_full_d42_tpce_hard11_20260812_v1`|
|候选|`E0_FULL_D42_TAIL_PAIR_CODE_EXCHANGE`；`d92_e0_full_d42_tail_pair_code_exchange`|
|矩阵|10 performance+1 K1 liveness、11 jobs、33 scene-arm、1 arm、8 shards|
|协议|`p2_min_v1`；复用`VALIDATED_ONCE`，不重复验证|
|状态|`LOCAL_VERIFIED / READY_TO_LAND`；未SSH、未启动|
|HEAD|`ecae572d071e5607c650aef0b3dcdde0fae9ff37`|
|环境|本地`ssr-gpu`；远端预注册`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|

## 本地交付

|交付物|本地路径|远端immutable目标|
|---|---|---|
|runtime archive|`E:\\type10-7\\automation_reports\\CV-SincNet\\d92_e0_full_d42_tpce_hard11_20260812_v1\\d92_tpce_runtime_closure_ecae572d_r2.tar.gz`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_tpce_source_snapshot_ecae572d_20260812_v1/d92_tpce_runtime_closure_ecae572d_r2.tar.gz`|
|method lock|`code/configs/stage2_d92_full_d42_tpce_hard11_v1.json`|`.../runs/d92_tpce_source_snapshot_20260812_v1/configs/stage2_d92_full_d42_tpce_hard11_v1.json`|
|launch|同目录`launch.sh`|`.../runs/d92_tpce_source_snapshot_20260812_v1/launch.sh`|
|外部镜像|`E:\\type10-7\\automation_reports\\CV-SincNet\\d92_e0_full_d42_tpce_hard11_20260812_v1\\`|仅供发布交接；本轮不执行同步|

runtime archive仅包含当前HEAD的Git跟踪`code/`树；不含数据集、checkpoint、结果或凭据。启动器仅在K>2 truth-free smoke满足`active=true`、`fallback=false`、fit=2/1、query零访问和state receipt闭合后才进入8 shard；`fresh_run_retry=false`。

## 冻结命令与路径

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_tpce_source_snapshot_ecae572d_20260812_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_tpce_hard11_20260812_v1`；日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_d42_tpce_hard11_20260812_v1`。GPU绑定为8个shard分别使用`CUDA_VISIBLE_DEVICES=0..7`并在进程内使用`cuda:0`。

健康停止仅限协议/安全违规、错误checkout/hash、覆盖风险、prediction闭包失败、launcher确定性技术故障或两个不同outer在prediction前出现同一异常指纹；不得按性能指标停止。异常run保留artifact并标记`NO_PERFORMANCE_RESULT`，不在同一run ID重启。

## 验证记录

已计划并本地执行：`pytest` TPCE Hard11机械测试、4个TPCE生产文件`py_compile`、config JSON解析、runner/analyzer `--help`、`git diff --check`、`bash -n launch.sh`、runtime archive解包后的import closure。详细hash与输出见下表；本轮不SSH、不启动、不创建远端路径。

|项目|结果|
|---|---|
|commit/hash|`ecae572d071e5607c650aef0b3dcdde0fae9ff37`|
|runtime archive|5,095,542B；SHA256=`a527062a64be9b68307164b77f793e25dea9c6c786cf056730c0ec84ef9abb14`；1317个Git归档成员|
|method lock SHA256|`58dabf7ed4510c74aa2beff4031a2bbe745be940d2dc1b8361300ecf07f7f23c`|
|外部launch SHA256|3,474B；`9e5e8f86ff47842352e2de4bd037668caae9f385a3b9c54745d27e02bd696aef`|
|本地测试|TPCE+相邻Pareto聚焦回归117项通过；7个生产模块`py_compile`通过；config/CLI/selection identity通过|
|独立复核|P0=0，P1=0，APPROVE|
|本地Git状态|科学与机械实现已提交；本报告和启动器将在发布交接提交中纳入Git|

真实K10 checkpoint smoke是8个shard的硬前置：仅当TPCE三场景均`active=true`、`fallback=false`、fit=2/1、D42 state闭合且所有query访问为false时启动Hard10。任何K>2 exact-E0 fallback均记技术停止且不生成性能结论。
