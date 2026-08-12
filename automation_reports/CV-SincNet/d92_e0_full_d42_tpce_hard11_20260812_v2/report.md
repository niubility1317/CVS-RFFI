# D92 E0 FULL D42 TPCE Hard11 v2发布预注册

|字段|冻结值|
|---|---|
|run ID|`d92_e0_full_d42_tpce_hard11_20260812_v2`|
|目标|10个最难performance outer上相对E0_FULL_ONLY八指标严格Pareto筛选；另含1个K1 liveness|
|候选|`E0_FULL_D42_TAIL_PAIR_CODE_EXCHANGE`；`d92_e0_full_d42_tail_pair_code_exchange`|
|科学commit|`4768c6b2613ba3764bb9fbee95c9a7ce3561220f`|
|矩阵|11 jobs；33 scene-arm；8 shards；`p2_min_v1`|
|状态|`LOCAL_VERIFIED / READY_TO_LAND`|
|独立复核|递归修复后P0=0，P1=0，APPROVE|

v1在K10 truth-free smoke的manifest artifact校验中发生确定性递归，尚未进入TPCE fit、未生成active/fallback字段、未启动shard，已标`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。v2只修复原始artifact validator的冻结引用；方法、矩阵、阈值、路径语义均未变。

发布输入：runtime archive`d92_tpce_runtime_closure_4768c6b2.tar.gz`，5,095,342B，SHA256=`e775b66584a83756e11834dfcf2908398c9707b783e2c1fe71628f6944f388a7`；method lock SHA256=`58dabf7ed4510c74aa2beff4031a2bbe745be940d2dc1b8361300ecf07f7f23c`；launch 3,764B，SHA256=`d8ef3c0ecd73bcc8b69671030e8d44fdb2daf82500a897855d7bbcdcd00e0ed2`。递归专项与TPCE机械聚焦16项通过，生产模块`py_compile`通过。

远端source/output/log分别为：

- `/home/szu2070436088/2510044040/CV-SincNet/runs/d92_tpce_source_snapshot_4768c6b2_20260812_v2`
- `/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_tpce_hard11_20260812_v2`
- `/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_d42_tpce_hard11_20260812_v2`

唯一启动命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_tpce_source_snapshot_4768c6b2_20260812_v2 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

真实K10 checkpoint smoke仍是8shard前置硬门：三场景必须TPCE active、无fallback、fit=2/1、D42 state/prediction closure闭合、所有query访问为false。`fresh_run_retry=false`；仅按协议/技术健康规则停止，绝不按性能停止。
