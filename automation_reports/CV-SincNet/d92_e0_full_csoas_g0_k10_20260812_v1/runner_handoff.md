# D92 CSOAS G0 N607 runner handoff

## STATUS

`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。唯一冻结run ID为`d92_e0_full_csoas_g0_k10_20260812_v1`，未重试、未重启、未换run ID，`fresh_run_retry=false`。prediction阶段产生了truth-free artifact，但closure脚本因`KeyError: 'final_coefficient_quantization_error_max'`未生成`g0_validation.json`及成功marker`D92_CSOAS_G0_ACTIVE_NON_E0_RESOURCE_PASS`。

## PRECHECK

- direct普通账号preflight通过：`N607`普通账号`szu2070436088`，项目根`/home/szu2070436088/2510044040/CV-SincNet`可见。
- 8×RTX3090可见且pre-launch空闲；未使用`N607-admin`、bridge或其它relay。
- release repo=`E:\type10-7\code\snapshots\d92_125wt`，启动前tracked clean，HEAD=`10c1107645d61c696cd8ac9894152da787bca1fd`；科学commit=`b8ebd4f4522fcc3e9e6b7dd18d722c329021f181`。
- 远端source/output/logs及同run进程启动前均ABSENT/0；本地retrieval root启动前ABSENT。
- archive成员`1458`；绝对路径、`../`、`code/code`均0；`bash -n`通过。Python=`3.10.19`，torch=`2.1.0+cu121`，CUDA available，device_count=8。四份seal及ground manifest hash匹配冻结值。

## SYNC/HASH

依次SCP archive→`launch.sh`，每次后SSH/TCP22清零。

|file|size|SHA256|
|---|---:|---|
|`d92_csoas_g0_runtime_b8ebd4f4.tar.gz`|6137499|`4b0b434a26b47511cb0ddeb9f2455bc81964d8fcef312e75e57879547b631ca5`|
|`launch.sh`|8902|`c49450d7e82b9fb3c927493dbb5e1f5a9935d3cf0cbd38a4e4e055efcb2b7374`|

远端source：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_b8ebd4f4_20260812_v1`。只创建source root；output/logs/code由launch按冻结脚本处理。

## COMMAND

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_b8ebd4f4_20260812_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

SSH launch exit=`0`，exact command只执行一次。

## PID/CWD/CMDLINE/GPU

run自然快速完成，未捕获可持续PID；即时和终态均无匹配`bash ./launch.sh`或`run_d92_e0d_prediction.py`进程。冻结CWD为`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_b8ebd4f4_20260812_v1`，GPU绑定为`CUDA_VISIBLE_DEVICES=0`、device=`cuda:0`。终态GPU0=`0%`、`1MiB/24576MiB`。

## G0三场景技术表（不得解释为性能）

|scene|active|fallback|candidate/ref fit|FULL actual/total|wall|incremental peak|query禁用字段|
|---|---:|---:|---:|---:|---:|---:|---|
|`leo_clear_weak`|true|false|1/0|1/2|13.129212ms|868352 bytes|fit/update/selection/truth/role/quota/global均false|
|`leo_low_elev_weak`|true|false|1/0|1/2|12.966224ms|692224 bytes|fit/update/selection/truth/role/quota/global均false|
|`leo_rain_weak`|true|false|1/0|1/2|14.481507ms|425984 bytes|fit/update/selection/truth/role/quota/global均false|

wall P90（3场景nearest-rank=max）=`14.481507ms`；query MAC=`3168=11×288`；`query_decision_policy=per_sample_all_registered_classes`。`prediction.out`状态为`D92_E0D_TRUTH_FREE_PREDICTIONS_COMPLETE`，但closure随后失败，所以不能写G0 pass。

## ARTIFACT COUNTS

完整递归取回到`E:\type10-7\local_artifacts\d92_e0_full_csoas_g0_k10_20260812_v1`，远端保留未删除。canonical tree hash（相对路径、size、file SHA256，Ordinal排序）远端=本地：

|root|files|bytes|tree SHA256|
|---|---:|---:|---|
|source|1429|73628803|`3966300b82263e01401ec0c905d6506f01947a9bdfd848835706fcb11a81aef1`|
|output|10|1018139|`ee10d95dbfc2ebac906a1575068f474a5724105f18cc58db3dbafa07ba3d62e6`|
|logs|4|399|`c590aaa74575d482edeca90561a2ff7202fc5c98210c5ca98aea6dd45e193094`|

关键异常：`source/launch_driver.err`为closure `KeyError: 'final_coefficient_quantization_error_max'`；`logs/prediction.err`为空；未生成`logs/g0_validation.json`。未运行scorer/analyzer，未读取任何性能指标。

## REMOTE/LOCAL PATHS

- source：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_b8ebd4f4_20260812_v1`
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_csoas_g0_k10_20260812_v1`
- logs：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_csoas_g0_k10_20260812_v1`
- local retrieval：`E:\type10-7\local_artifacts\d92_e0_full_csoas_g0_k10_20260812_v1`

## SSH CLEANUP

所有bounded SSH/SCP客户端均已退出；本地无`ssh.exe`/`scp.exe`残留，N607及bridge（未使用）TCP22均无ESTABLISHED连接；远端同run进程为0，GPU已释放。

## NEXT_ACTION

主agent仅可在本地审查并修复closure字段/测试，完成新的独立review与commit后另行决定是否创建全新run ID。当前run禁止修复、重启、重试、覆盖或性能分析。
